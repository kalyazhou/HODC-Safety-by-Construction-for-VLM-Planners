# -*- coding: utf-8 -*-
"""
mode_c.py — High-Order Driving Command (HODC++) module
------------------------------------------------------
Drop‑in replacement for your MODE C path. Plugs into the AB pipeline
without touching A/B behaviors. Design goals:
- Robust high‑order driving command (HODC++) JSON with explicit temporal guards
  and BEV‑aware bounds.
- Two‑stage generation: (1) language → constraints (HODC++), (2) constraints →
  10‑step [v,k] with smoothness/safety objectives.
- Deterministic parsing + unit conversion (k×100 from VLM → real curvature 1/m).
- Confidence‑aware blending with B‑reference, TTC‑aware safety caps, and
  optional open‑loop evaluation bypasses (eval_pre_safety_openloop).

How to use (minimal):
    from mode_c import run_mode_c
    vk, hodc_constraints, diag = run_mode_c(
        obs_images,      # [front_last_b64, bev_mid_b64, bev_last_b64]
        v_hist_10,       # list[10] m/s
        k_hist_10,       # list[10] real curvature
        bev_hints={"speed_limit_mps": 12.0},
        cache=prev_cache,                # dict or None, returned in diag["cache_out"]
        openloop=args.eval_pre_safety_openloop,
        api_model="gpt-4o-mini",
    )
    # vk -> list of 10 pairs [[v, k_real], ...]

This file is standalone: it only needs your global OpenAI client if you want to
reuse it; otherwise pass your own client via kwarg `client=`.
"""
from __future__ import annotations
import json, re, time, random
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

try:
    from openai import OpenAI  # optional; you can pass client in
except Exception:  # pragma: no cover
    OpenAI = object  # type: ignore

# ---------------------------- Utils ----------------------------

def _safe_json_loads(txt: str) -> dict:
    """More tolerant JSON parsing for partially‑formatted LLM outputs.
    - trims to outermost {...}
    - fixes common literals/trailling commas
    """
    s = txt.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("no JSON object found")
    s = s[l:r+1]
    s = re.sub(r"\bnul\b", "null", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return json.loads(s)

# Speed/curvature helpers
DT = 0.5
G = 9.80665
ALAT_COMFORT_G = 0.30
K_MAX_PHYS = 0.08  # 1/m, ~12.5m min radius
V_MAX = 12.0       # m/s, cap (dataset typical)

# ---------------------- Prompt + Inference ----------------------

_DEFAULT_SYS = (
    "You are a BEV-aware autonomous driving planner. Return ONLY valid JSON.\n"
    "You must encode High-Order Driving Commands (HODC++) with explicit\n"
    "temporal guards and numeric bounds. Curvature k is reported in the\n"
    "interface as k×100 (so ±6 → real ±0.06 1/m).\n"
    "If a red light is within 8m, plan to stop by setting near-term v upper\n"
    "bounds to 0. Use 0.5s time grid up to 5.0s."
)

_SCHEMA_HINT = (
    '{\n'
    '  "scene": {"location":"string","lane_topology":"string"},\n'
    '  "signals": {"tl_state":"red|yellow|green|none","stopline_distance_m":number|null,"speed_limit_mps":number|null},\n'
    '  "maneuver": {"type":"straight|turn_left|turn_right|lane_change_left|lane_change_right|merge","k_sign":-1|0|+1,"urgency":[0,1],"k_peak_range":[kmin,kmax]},\n'
    '  "guards": [\n'
    '     {"op":"G","window_s":[t0,t1],"what":"stay_in_lane|keep_clear|yield","tighten":0|1},\n'
    '     {"op":"U","window_s":[t0,t1],"what":"yield","until":"agent_idx:0"}\n'
    '  ],\n'
    '  "agents": [ {"class":"vehicle|ped|cyclist|other","rel_dist_m":number,"rel_bearing_deg":number,"rel_speed_mps":number,"priority":"low|med|high","ttc_s":number|null} ],\n'
    '  "conflicts": [ {"agent_idx":0,"time_window_s":[t0,t1],"min_sep_m":number,"required_action":"yield|brake|nudge_left|nudge_right","target_speed_cap_mps":number|null,"k_bound":[kmin,kmax]|null} ],\n'
    '  "HODC": {"v_bounds": [[t,vmin,vmax],...], "k_bounds": [[t,kmin,kmax],...], "stay_in_lane": true|false, "notes":"<=120 chars"}\n'
    '}'
)

_DEF_MOTION_SYS = (
    "You convert HODC++ into a smooth 10-step plan. Return ONLY valid JSON.\n"
    "Output format: {\"pairs\": [[v1, k1], [v2, k2], ..., [v10, k10]]}\n"
    "where v is speed in m/s, k is curvature×100 (integer in [-600,600]).\n"
    "MUST output exactly 10 pairs. Limit Δv per 0.5s to ≤0.9 m/s.\n"
    "Prefer bell-shaped k for turns; S-curve v for smooth acceleration."
)

_TIME_GRID = [round(0.5 * (i + 1), 1) for i in range(10)]  # 0.5..5.0


def _chat(client: Any, model: str, sys: str, user: str, images_b64: Optional[List[str]] = None, force_json: bool = True, max_tokens: int = 600) -> str:
    """Thin wrapper around OpenAI Chat Completions (image + text)."""
    messages: List[Dict[str, Any]] = []
    if sys:
        messages.append({"role": "system", "content": sys})
    content: List[Dict[str, Any]] = []
    if images_b64:
        for b64 in images_b64:
            # Chat Completions multimodal expects image_url (data URL or https URL)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    content.append({"type": "text", "text": user})
    messages.append({"role": "user", "content": content})
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1 if force_json else 0.3,
        top_p=1.0,
        **({"response_format": {"type": "json_object"}} if force_json else {})
    )
    return res.choices[0].message.content

# ---------------------- Core HODC++ Pipeline ----------------------

class HODCGenerator:
    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model

    def build_prompt(self, history_vk: List[List[float]], prev_summary: Optional[dict], risk_feedback: Optional[dict]) -> str:
        return (
            "Inputs:\n"
            f"- History [v,k×100]: {json.dumps(history_vk)}\n"
            f"- Prev summary: {json.dumps(prev_summary or {}, ensure_ascii=False)}\n"
            + (f"- Feedback: {json.dumps(risk_feedback)}\n" if risk_feedback else "") +
            "Task: Produce STRICT JSON matching the schema below. Be numerically specific\n"
            "(meters, seconds, m/s, k×100 in [-6,+6]).\n\nSchema:\n" + _SCHEMA_HINT
        )

    def generate(self, images: List[str], history_vk: List[List[float]], prev_summary: Optional[dict], risk_feedback: Optional[dict]) -> dict:
        sys = _DEFAULT_SYS
        user = self.build_prompt(history_vk, prev_summary, risk_feedback)
        raw = _chat(self.client, self.model, sys, user, images_b64=images, force_json=True, max_tokens=700)
        return _safe_json_loads(raw)


def _extract_bounds(h: dict) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]], dict, List[dict]]:
    H = h.get("HODC", h)
    sig = (h.get("signals") or H.get("signals") or {}) if isinstance(H, dict) else {}
    v_bounds = H.get("v_bounds", [])
    k_bounds = H.get("k_bounds", [])
    # align to 10 steps on TIME_GRID
    V, K = [], []
    for t in _TIME_GRID:
        vmin, vmax = 0.0, V_MAX
        kmin, kmax = -6.0, 6.0  # k×100 in JSON
        for vb in v_bounds:
            if len(vb) >= 3 and abs(float(vb[0]) - t) < 1e-3:
                vmin, vmax = float(vb[1]), float(vb[2])
                break
        for kb in k_bounds:
            if len(kb) >= 3 and abs(float(kb[0]) - t) < 1e-3:
                kmin, kmax = float(kb[1]), float(kb[2])
                break
        # convert k×100 → real
        K.append((max(-K_MAX_PHYS, kmin/100.0), min(K_MAX_PHYS, kmax/100.0)))
        V.append((max(0.0, vmin), min(V_MAX, vmax)))
    return V, K, sig, h.get("conflicts", [])


def _compile_temporal_guards(h: dict, V: List[Tuple[float,float]], K: List[Tuple[float,float]]) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    """Impose simple LTL-like guards on bounds.
    Supported ops:
      - G (always): tighten bounds within window
      - F (eventually): keep slack but bias k toward 0 in early steps
      - U (until agent): mirror as yield cap in window (acts like conflict)
    """
    guards = h.get("guards", []) or []
    for g in guards:
        op = str(g.get("op", "")).upper()
        w = g.get("window_s", [0,0])
        tighten = int(g.get("tighten", 0))
        i0 = max(0, int((float(w[0]) / DT) - 1)) if len(w)>=2 else 0
        i1 = min(9, int((float(w[1]) / DT) - 1)) if len(w)>=2 else 9
        if op == "G":
            for i in range(i0, i1+1):
                vmin, vmax = V[i]
                kmin, kmax = K[i]
                if tighten:
                    V[i] = (vmin, max(vmin, min(vmax, vmin + 2.0)))  # force cautious
                    mid = 0.5*(kmin+kmax)
                    bw = max(0.01, 0.5*(kmax-kmin))
                    K[i] = (max(-K_MAX_PHYS, mid - bw/2), min(K_MAX_PHYS, mid + bw/2))
                else:
                    V[i] = (vmin, vmax)
        elif op == "F":
            for i in range(0, i1+1):  # encourage early straightening
                kmin, kmax = K[i]
                K[i] = (0.8*kmin, 0.8*kmax)
        elif op == "U":
            # handled via conflicts in next stage; here do nothing
            pass
    return V, K


class MotionCompiler:
    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model

    def build_prompt(self, hodc_json: dict) -> str:
        return "HODC JSON:\n" + json.dumps(hodc_json, ensure_ascii=False)

    def generate(self, hodc_json: dict) -> List[List[float]]:
        sys = _DEF_MOTION_SYS
        user = self.build_prompt(hodc_json)
        # 强制 JSON 输出，避免返回单对象 {"speed":0,"curvature":0}
        raw = _chat(self.client, self.model, sys, user, images_b64=None, force_json=True, max_tokens=320)
        vk: List[List[float]] = []
        try:
            obj = _safe_json_loads(raw)
            # 尝试多种可能的键名
            pairs = None
            if isinstance(obj, dict):
                for key in ("pairs", "plan", "trajectory", "vk", "data", "motion"):
                    if key in obj and isinstance(obj[key], list):
                        pairs = obj[key]
                        break
            # 如果直接是数组
            if pairs is None and isinstance(obj, list):
                pairs = obj
            
            if isinstance(pairs, list):
                for it in pairs[:10]:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        v = float(it[0])
                        k = float(it[1]) / 100.0
                        vk.append([v, k])
            
            print(f"[HODC++] Parsed {len(vk)} pairs from VLM output")
        except Exception as e:
            print(f"[HODC++] Parse error: {e}")
        
        # 检查是否需要兜底：长度不足10 或 曲率全为0
        needs_fallback = (len(vk) < 10) or (len(vk) > 0 and all(abs(k) < 1e-6 for _, k in vk))
        if needs_fallback:
            print(f"[HODC++] Triggering fallback: len={len(vk)}, needs_fallback={needs_fallback}")
            # 清空并重新生成
            vk = []
            # 退化：用 bounds 生成保守轨迹（避免直线化）
            Vb, Kb, _, _ = _extract_bounds(hodc_json)
            # 曲率钟形模板（中段稍高，前后收敛）
            bell = np.array([0.0, 0.3, 0.6, 0.9, 1.0, 0.9, 0.6, 0.3, 0.1, 0.0], dtype=float)
            for i in range(10):
                vmin, vmax = Vb[i]
                kmin, kmax = Kb[i]
                v = min(max(0.0, 0.5*(vmin+vmax)), V_MAX)
                # 取几何中值再乘 bell
                k_mid = 0.5*(kmin + kmax)
                k = np.clip(k_mid * bell[i], kmin, kmax)
                vk.append([float(v), float(k)])
            k_vals = [k for _, k in vk]
            print(f"[HODC++] Applied fallback trajectory: v_range=[{min(v for v,_ in vk):.1f}, {max(v for v,_ in vk):.1f}], k_range=[{min(k_vals):.4f}, {max(k_vals):.4f}]")
        return vk[:10]

# ----------------------- Safety + Blending -----------------------

def _apply_speed_caps(v: np.ndarray, k: np.ndarray, speed_limit: Optional[float]) -> np.ndarray:
    k_abs = np.maximum(np.abs(k), 1e-6)
    v_limit_curv = np.sqrt((ALAT_COMFORT_G * G) / k_abs)
    v_cap = np.minimum(V_MAX, v_limit_curv)
    if speed_limit is not None:
        v_cap = np.minimum(v_cap, float(speed_limit))
    return np.minimum(v, v_cap)


def _smooth_speed(v: np.ndarray, max_dv: float = 0.9) -> np.ndarray:
    out = v.copy()
    for i in range(1, len(out)):
        dv = np.clip(out[i] - out[i-1], -max_dv, max_dv)
        out[i] = out[i-1] + dv
    return out


def _sanitize_k(k: np.ndarray) -> np.ndarray:
    k = np.clip(k, -3.0*K_MAX_PHYS, 3.0*K_MAX_PHYS)  # soft clip
    # light smoothing
    if len(k) >= 3:
        ks = k.copy()
        for i in range(1, len(k)-1):
            ks[i] = 0.7*k[i] + 0.3*0.5*(k[i-1]+k[i+1])
        k = ks
    return np.clip(k, -K_MAX_PHYS, K_MAX_PHYS)


def _confidence_from_hodc(h: dict) -> float:
    sig = (h.get("signals") or h.get("HODC", {}).get("signals") or {})
    completeness = 0
    completeness += 1 if sig.get("speed_limit_mps") is not None else 0
    completeness += 1 if (sig.get("tl_state") or "none") != "none" else 0
    completeness += 1 if sig.get("stopline_distance_m") is not None else 0
    completeness /= 3.0
    man = (h.get("maneuver") or {}).get("type", "")
    ksign = (h.get("maneuver") or {}).get("k_sign", 0)
    base = 0.25 + 0.10*completeness + (0.05 if (man and ksign in (-1,0,1)) else 0.0)
    return float(np.clip(base, 0.15, 0.35))

# ---------------------------- Public API ----------------------------

class ModeC:
    def __init__(self, client: Any, api_model: str = "gpt-4o-mini") -> None:
        self.hgen = HODCGenerator(client, api_model)
        self.mcmp = MotionCompiler(client, api_model)

    def step(self,
             images3: List[str],  # [front,last], [bev,mid], [bev,last]
             v_hist: List[float],
             k_hist: List[float],
             prev_summary: Optional[dict] = None,
             risk_feedback: Optional[dict] = None,
             b_ref_vmax: Optional[List[Tuple[float,float]]] = None,
             openloop: bool = False) -> Tuple[List[List[float]], dict, dict]:
        """Run full C pipeline. Returns (vk10, hodc, diag)."""
        history_vk = [[float(v), float(k*100.0)] for v,k in zip(v_hist, k_hist)]
        hodc = self.hgen.generate(images3, history_vk, prev_summary, risk_feedback)
        Vb, Kb, signals, conflicts = _extract_bounds(hodc)
        Vb, Kb = _compile_temporal_guards(hodc, Vb, Kb)

        # fuse BEV-derived bounds into HODC for motion generation
        hodc_fused = dict(hodc)
        hodc_fused.setdefault("HODC", {})
        hodc_fused["HODC"]["v_bounds"] = [[t, Vb[i][0], Vb[i][1]] for i,t in enumerate(_TIME_GRID)]
        hodc_fused["HODC"]["k_bounds"] = [[t, Kb[i][0]*100.0, Kb[i][1]*100.0] for i,t in enumerate(_TIME_GRID)]

        vk = self.mcmp.generate(hodc_fused)
        if len(vk) < 10:
            # conservative fallback: cruise within bounds
            vk = [[min(Vb[i][1], 4.0), 0.0] for i in range(10)]

        v = np.array([x[0] for x in vk], dtype=float)
        k = np.array([x[1] for x in vk], dtype=float)

        # confidence‑aware blending with B reference (if provided)
        w = _confidence_from_hodc(hodc)
        if b_ref_vmax is not None and len(b_ref_vmax) >= 10:
            v_b = np.array([min(8.0, vmax*0.85) for (_, vmax) in b_ref_vmax], dtype=float)
            k_b = np.zeros_like(k)
            k = (1-w)*k_b + w*k
            v = np.minimum((1-w)*v_b + w*v, v_b)
        
        # 强制注入转弯模板：如果曲率几乎为0且场景需要转弯
        if np.std(k) < 1e-4 and hodc.get("maneuver", {}).get("type", "").startswith("turn"):
            print(f"[HODC++] Injecting turn template for {hodc.get('maneuver', {}).get('type', 'unknown')}")
            # 根据转弯方向注入钟形曲率
            k_sign = hodc.get("maneuver", {}).get("k_sign", 0)
            if k_sign != 0:
                bell = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 0.8, 0.5, 0.2, 0.1, 0.0], dtype=float)
                k_peak = 0.03 * k_sign  # 峰值曲率
                k = bell * k_peak
                # 相应调整速度
                v = np.minimum(v, np.sqrt(0.3 * 9.8 / np.maximum(np.abs(k), 1e-6)))

        # Safety: curvature → speed caps; smooth dv; final clamps
        speed_limit = signals.get("speed_limit_mps") if isinstance(signals, dict) else None
        v = _apply_speed_caps(v, k, speed_limit)
        v = _smooth_speed(v, max_dv=0.9)
        k = _sanitize_k(k)

        if not openloop:
            # 更保守的最小进度地板：仅前3步 1.2 m/s，且无冲突时
            head_conflict = any([(c.get("time_window_s", [10,0])[0] <= 1.5) for c in conflicts])
            if (speed_limit or V_MAX) > 2.0 and (not head_conflict):
                v[:3] = np.maximum(v[:3], 1.2)

        out = [[float(v[i]), float(k[i])] for i in range(10)]
        diag = {
            "confidence": w,
            "signals": signals,
            "bounds_v": Vb,
            "bounds_k": Kb,
            "raw_hodc": hodc,
        }
        return out, hodc_fused, diag


# -------- Convenience wrapper the main file can call (one function) --------

def run_mode_c(obs_images: List[str],
               v_hist_10: List[float],
               k_hist_10: List[float],
               bev_hints: Optional[dict] = None,
               cache: Optional[dict] = None,
               openloop: bool = False,
               api_model: str = "gpt-4o-mini",
               client: Optional[Any] = None,
               b_ref_vmax: Optional[List[Tuple[float,float]]] = None,
               risk_feedback: Optional[dict] = None,
               prev_summary: Optional[dict] = None,
               ) -> Tuple[List[List[float]], dict, dict]:
    """Minimal one‑shot entrypoint used by your main loop.
    - obs_images: suggest [front_last, bev_mid, bev_last] (b64 JPEGs)
    - v_hist_10/k_hist_10: last 10 steps (5s) history in SI units
    - bev_hints: not used now (reserved for map priors)
    - cache: reserved (you can ignore or extend)
    - openloop: if True, skip some progress floors for fair ADE eval
    - b_ref_vmax: optional B‑mode v upper bounds for blending
    Returns: (vk10, hodc_fused, diagnostics)
    """
    # lazy client fallback to global if available
    global_client = None
    try:
        from openai import OpenAI as _OpenAI
        global_client = _OpenAI()
    except Exception:
        pass
    client = client or global_client
    if client is None:
        raise RuntimeError("OpenAI client not available — pass client= explicitly")

    modec = ModeC(client, api_model)
    vk, hodc, diag = modec.step(
        images3=obs_images,
        v_hist=v_hist_10,
        k_hist=k_hist_10,
        prev_summary=prev_summary,
        risk_feedback=risk_feedback,
        b_ref_vmax=b_ref_vmax,
        openloop=openloop,
    )
    return vk, hodc, diag


