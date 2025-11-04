import os
import re
import cv2
import json
import time
import base64
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from math import atan2
from datetime import datetime
from pyquaternion import Quaternion

import torch
from openai import OpenAI

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap


from openemma.YOLO3D.inference import yolo3d_nuScenes
from openemma.bev.renderer import render_static_bev
from openemma.bev.simple_bev import render_simple_bev
from openemma.visualize.compare import save_side_by_side
from openemma.analyze.smoothness import (
    calculate_trajectory_smoothness,
    compare_trajectory_smoothness,
    print_smoothness_analysis,
)
from utils import (
    EstimateCurvatureFromTrajectory,
    IntegrateCurvatureForPoints,
    OverlayTrajectory,
    WriteImageSequenceToVideo,
)
from src.hodc.fusion import fuse_k
from mode_c_v2 import generate_mode_c_prediction, apply_soft_constraints
from mode_c_cot import generate_mode_c_cot_prediction

# ---- OpenAI client ----
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========================= 全局缓存 =========================
short_json = {"scene": {}, "agents": [], "intent": {}, "notes": ""}
last_json_hash = None

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN


def should_update_summary(events, i, K=5):
    return (i % K == 0) or events.get("novelty", False)


def compress_summary(prev_json, scene_txt, obj_txt, intent_txt):
    import json as _json

    sys = "You compress driving scene text into a fixed JSON schema, no extras."
    usr = f"""Given last summary:
{_json.dumps(prev_json, ensure_ascii=False)}
New observations (keep only NEW or CHANGED info, drop redundant):
- scene: {scene_txt}
- objects: {obj_txt}
- intent: {intent_txt}

Rules:
- Output JSON with keys: scene, agents, intent, notes.
- Use numbers whenever possible.
- Max 1 short sentence in "notes".
- If unknown, omit the field.

Return ONLY the JSON."""
    try:
        res = vlm_inference(
            text=usr,
            images=None,
            sys_message=sys,
            args=None,
            model_type="gpt-4o-mini",
        )
        return _json.loads(res)
    except Exception:
        return prev_json  # 失败就沿用


def detect_novelty(scene_desc, object_desc, intent_desc, prev_scene, prev_objects, prev_intent):
    events = {"novelty": False}
    if scene_desc != prev_scene or object_desc != prev_objects or intent_desc != prev_intent:
        events["novelty"] = True
    return events


def img_bytes_to_jpeg_b64(img_bytes, target_side=320, quality=60):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return base64.b64encode(img_bytes).decode("utf-8")

    h, w = img.shape[:2]
    scale = target_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return base64.b64encode(img_bytes).decode("utf-8")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def prepare_front_view_boxes(nusc, sample, ego_pose):
    """
    Prepare forward-looking detection box data for projection onto the BEV
    Return format:: [{'center': [x, y, z], 'size': [l, w, h], 'rotation': yaw, 'category': 'car'}, ...]
    """
    front_view_boxes = []
    try:
        for ann_token in sample.get("anns", []):
            ann = nusc.get("sample_annotation", ann_token)
            category = ann.get("category_name", "")
            ann_translation = np.array(ann["translation"])
            ann_size = np.array(ann["size"])
            ann_rotation = Quaternion(ann["rotation"])
            ego_translation = np.array(ego_pose["translation"])
            distance = np.linalg.norm(ann_translation[:2] - ego_translation[:2])
            if distance < 50.0:
                front_view_boxes.append(
                    {
                        "center": ann_translation.tolist(),
                        "size": ann_size.tolist(),
                        "rotation": ann_rotation.yaw_pitch_roll[0],
                        "category": category.split(".")[-1],
                    }
                )
    except Exception as e:
        print(f"Warning: Failed to prepare front view boxes: {e}")
    return front_view_boxes


def add_dynamic_objects_to_bev(bev, nusc, sample, center_xy, yaw_deg, extent_m, resolution_m):
    """
    Add dynamic objects (vehicles, pedestrians, etc.) to BEV image
    """
    h, w = bev.shape[:2]
    center_x, center_y = w // 2, h // 2
    ann_tokens = sample.get("anns", [])
    object_count = 0

    for ann_token in ann_tokens:
        try:
            ann = nusc.get("sample_annotation", ann_token)
            if not ann or "translation" not in ann or "size" not in ann:
                continue
            obj_translation = ann["translation"]
            dx = obj_translation[0] - center_xy[0]
            dy = obj_translation[1] - center_xy[1]
            distance = np.sqrt(dx**2 + dy**2)
            if distance > extent_m:
                continue

            cos_yaw = np.cos(np.radians(yaw_deg))
            sin_yaw = np.sin(np.radians(yaw_deg))
            bev_x = dx * cos_yaw + dy * sin_yaw
            bev_y = -dx * sin_yaw + dy * cos_yaw

            pixel_x = int(center_x + bev_x / resolution_m)
            pixel_y = int(center_y - bev_y / resolution_m)

            if 0 <= pixel_x < w and 0 <= pixel_y < h:
                width = max(2, ann["size"][0] / resolution_m)
                height = max(2, ann["size"][1] / resolution_m)
                category = ann.get("category_name", "unknown")
                if "vehicle" in category.lower():
                    color = (0, 0, 255)
                elif "pedestrian" in category.lower():
                    color = (255, 0, 255)
                elif "bicycle" in category.lower():
                    color = (0, 255, 255)
                else:
                    color = (128, 128, 128)
                cv2.rectangle(
                    bev,
                    (int(pixel_x - width / 2), int(pixel_y - height / 2)),
                    (int(pixel_x + width / 2), int(pixel_y + height / 2)),
                    color,
                    2,
                )
                label = category.split(".")[-1][:4]
                cv2.putText(
                    bev,
                    label,
                    (int(pixel_x - width / 2), int(pixel_y - height / 2 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )
                object_count += 1
        except Exception:
            continue

    print(f"  Added {object_count} dynamic objects to BEV")
    return bev


def world_to_bev_xy(points_xy, origin_xy, yaw_deg, extent_m, res_m, img_shape=None):
    """
    Convert world coordinates to BEV pixel coordinates with yaw rotation
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    rel = pts - np.asarray(origin_xy, dtype=np.float32)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    x_bev = rel[:, 0] * c + rel[:, 1] * s
    y_bev = -rel[:, 0] * s + rel[:, 1] * c

    half = float(extent_m)
    u = (x_bev + half) / float(res_m)
    v = (half - y_bev) / float(res_m)

    if img_shape is not None:
        H, W = img_shape[:2]
    else:
        W = H = int((2 * half) / float(res_m))

    uv = np.stack([u, v], axis=-1).round().astype(int)
    uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)
    return uv


def draw_traj_on_bev(bev_img, traj_xy_world, origin_xy, yaw_deg, extent_m, res_m, color=(0, 0, 255), thickness=2):
    """
    Draw trajectory on BEV image with yaw rotation
    """
    uv = world_to_bev_xy(traj_xy_world, origin_xy, yaw_deg, extent_m, res_m, img_shape=bev_img.shape)
    for i in range(len(uv) - 1):
        cv2.line(bev_img, tuple(uv[i]), tuple(uv[i + 1]), color, thickness)
    return bev_img


def collect_agents_nus_with_vel(nusc, prev_sample_token, curr_sample_token, dt=0.5):
    """
    Collect agents with velocity estimation from nuScenes GT
    """
    prev_pos = {}
    if prev_sample_token is not None:
        for tk in nusc.get("sample", prev_sample_token)["anns"]:
            ann = nusc.get("sample_annotation", tk)
            prev_pos[ann["instance_token"]] = np.array(ann["translation"][:2], np.float32)

    agents = []
    for tk in nusc.get("sample", curr_sample_token)["anns"]:
        ann = nusc.get("sample_annotation", tk)
        p = np.array(ann["translation"][:2], np.float32)
        v = (p - prev_pos.get(ann["instance_token"], p)) / max(1e-6, dt)
        agents.append(dict(pos_xy=p, vel_xy=v))
    return agents


def filter_agents(agents, ego_pos, ego_heading, max_r=30.0, fov_deg=120, topk=20):
    """
    Only take the front sector (±60°) within 30m and the Top-20 nearest
    """
    out = []
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    for a in agents:
        rel = a["pos_xy"] - ego_pos
        x_f = c * rel[0] + s * rel[1]
        y_f = -s * rel[0] + c * rel[1]
        if x_f < 0:
            continue
        ang = np.degrees(np.arctan2(y_f, x_f))
        if abs(ang) <= fov_deg / 2 and np.hypot(*rel) <= max_r:
            out.append(a)
    out.sort(key=lambda a: np.linalg.norm(a["pos_xy"] - ego_pos))
    return out[:topk]


def is_critical_agent(ego_traj, agent_pos, agent_vel, time_horizon=2.5, min_distance=2.0):
    """Determine whether the agent's trajectory will intersect with the ego's trajectory within the next 2-3 seconds."""
    if len(ego_traj) == 0:
        return False
    
    # Calculate future trajectory
    ego_future_steps = min(int(time_horizon/0.5), len(ego_traj))  # 2.5s = 5步
    ego_future = ego_traj[:ego_future_steps]
    
    # Calculate agent future trajectory
    t = np.arange(ego_future_steps, dtype=np.float32) * 0.5
    agent_future = agent_pos[None, :] + t[:, None] * agent_vel[None, :]
    
    # Calculate the minimum distance
    min_distances = np.linalg.norm(ego_future - agent_future, axis=1)
    min_distance_achieved = np.min(min_distances)
    
    return min_distance_achieved < min_distance


def filter_critical_agents(agents_states, ego_traj):
    """Only retain key agents posing a risk of intersecting with the ego trajectory."""
    critical_agents = []
    for agent in agents_states:
        if is_critical_agent(ego_traj, agent["pos_xy"], agent["vel_xy"]):
            critical_agents.append(agent)
    return critical_agents


def dynamic_collision_flag(pred_xy, agents_states, step=0.5, base_safety_radius=1.4, speed_gain=0.0, ttc_th=2.0, s_max=0.5):
    """
    Check for dynamic collision with other agents 
    """
    if len(pred_xy) == 0 or not agents_states:
        return False
    T = len(pred_xy)
    t = np.arange(T, dtype=np.float32) * step
    ego_v = np.gradient(pred_xy, axis=0) / step
    ego_speed = np.linalg.norm(ego_v, axis=1)

    for i, a in enumerate(agents_states):
        pos0 = a["pos_xy"]
        vel = a["vel_xy"]
        a_traj = pos0[None, :] + t[:, None] * vel[None, :]
        
        # TTC Basic Safety Radius Calculation
        safety = np.zeros(T)
        for t_idx in range(T):
            ego_pos = pred_xy[t_idx]
            agent_pos = a_traj[t_idx]
            ego_vel = ego_v[t_idx]
            
            rel_pos = agent_pos - ego_pos
            rel_vel = vel - ego_vel
            
            rel_speed_pos = np.dot(rel_vel, rel_pos) / (np.linalg.norm(rel_pos) + 1e-6)
            rel_speed_pos = max(0, rel_speed_pos)  
            
            ttc = np.linalg.norm(rel_pos) / (np.linalg.norm(rel_vel) + 1e-6)
            speed_factor = np.clip(rel_speed_pos / max(ttc, ttc_th), 0, s_max)
            safety[t_idx] = base_safety_radius + (1.0 + float(speed_gain)) * speed_factor
        
        d = np.linalg.norm(pred_xy - a_traj, axis=1)
        argmin = int(np.argmin(d))
        min_d = float(d[argmin])
        if min_d < float(safety[argmin]):
            print(
                f"[planner] collision detected with agent {i}: min_d={min_d:.2f}m at step={argmin}, "
                f"safety_radius={float(safety[argmin]):.2f}m"
            )
            return True
    return False


def _safe_json_loads(txt: str):
    """Fault-tolerant JSON parsing, handling non-standard/truncated output"""
    import json, re
    s = txt.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("no JSON object found")
    s = s[l:r+1]
    s = re.sub(r'\bnul\b', 'null', s)   
    s = re.sub(r'\bNone\b', 'null', s)
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r',\s*([}\]])', r'\1', s)  
    return json.loads(s)


def parse_hodc_json(hodc_text):
    """
    Parse HODC JSON and extract constraints
    """
    try:
        print(f"[DEBUG] HODC JSON to parse: {hodc_text[:200]}...")
        hodc = _safe_json_loads(hodc_text)  
        
        H = hodc.get("HODC", hodc)
        v_bounds = H.get("v_bounds", [])
        k_bounds = H.get("k_bounds", [])
        maneuver = hodc.get("maneuver", {})
        conflicts = hodc.get("conflicts", [])
        signals = hodc.get("signals", {})
        
        time_steps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        
        v_bounds_array = []
        k_bounds_array = []
        
        for i, t in enumerate(time_steps):
            v_min, v_max = 0.0, 12.0  
            k_min, k_max = -0.06, 0.06  
            
            for bound in v_bounds:
                if len(bound) >= 3 and abs(bound[0] - t) < 0.1:
                    v_min, v_max = bound[1], bound[2]
                    break
            
            for bound in k_bounds:
                if len(bound) >= 3 and abs(bound[0] - t) < 0.1:
                    k_min, k_max = bound[1] / 100.0, bound[2] / 100.0  
                    break
            
            v_bounds_array.append([v_min, v_max])
            k_bounds_array.append([k_min, k_max])
        
        return {
            "v_bounds": v_bounds_array,
            "k_bounds": k_bounds_array,
            "maneuver": maneuver,
            "conflicts": conflicts,
            "hodc": hodc
        }
    except Exception as e:
        print(f"[HODC] Failed to parse HODC JSON: {e}")
        return None


def progress_floor(speed_curvature_pred, hodc_constraints, prev_speed=None):
    """Post-processing Progress Lower Bounder """
    import numpy as np
    v_bounds = hodc_constraints["v_bounds"]
    H = hodc_constraints.get("hodc", hodc_constraints)
    sig = hodc_constraints.get("signals") or ((H.get("signals") or {}) if isinstance(H, dict) else {})
    speed_limit = float(sig.get("speed_limit_mps")) if sig.get("speed_limit_mps") is not None else None

    # Only when the trajectory is “nearly straight” and there is “no conflict window,” gently push the upper limit to 70–85%.
    mean_abs_k = float(np.mean([abs(k) for _, k in speed_curvature_pred])) if speed_curvature_pred else 0.0
    straightish = mean_abs_k < 0.012  
    tl = (sig.get("tl_state") or "none").lower()

    for i, (v, k) in enumerate(speed_curvature_pred):
        vmin, vmax = v_bounds[i] if i < len(v_bounds) else (0.0, 12.0)
        ub = vmax
        if speed_limit is not None:
            ub = min(ub, speed_limit)

        # Do not accelerate within the conflict window.
        has_conflict = False
        for c in hodc_constraints.get("conflicts", []):
            tw = c.get("time_window_s", [])
            if len(tw) >= 2:
                t = (i + 1) * 0.5
                if tw[0] <= t <= tw[1]:
                    has_conflict = True
                    break

        if (tl in ["green", "none"]) and straightish and (not has_conflict):
            target = 0.7 * ub
            if prev_speed is not None:
                target = max(target, 0.80 * prev_speed)  # Moderate recovery
            v = min(max(v, target), ub)
        else:
            v = min(max(v, vmin), ub)

        speed_curvature_pred[i] = [float(v), float(k)]
        prev_speed = v

    return speed_curvature_pred


def smooth_speed_only(speed_curvature_pred, max_delta_v=1.2):
    """Speed change per step (m/s per 0.5s), without altering curvature"""
    if not speed_curvature_pred:
        return speed_curvature_pred
    out = [list(speed_curvature_pred[0])]
    for i in range(1, len(speed_curvature_pred)):
        v_prev = out[-1][0]
        v, k = speed_curvature_pred[i]
        dv = np.clip(v - v_prev, -max_delta_v, max_delta_v)
        out.append([float(v_prev + dv), float(k)])
    return out


def _sanitize_hodc_bounds(hodc):
    """Pull HODC's k_bounds back into the physically feasible range (true curvature 1/m) and automatically correct the 10×/1Pull HODC's k_bounds back into the physically feasible range (true curvature 1/m)"""
    import numpy as np
    kb = hodc.get("k_bounds", [])
    if not kb:
        return hodc
    mags = []
    for r in kb:
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            mags.append(max(abs(float(r[0])), abs(float(r[1]))))
    abs_mid = np.median(mags) if mags else 0.0

    scale = 1.0
    if abs_mid > 0.6:      
        scale = 0.1
    elif abs_mid > 0.2:   
        scale = 1/3.0

    new_kb = []
    for r in kb:
        if not isinstance(r, (list, tuple)) or len(r) < 2:
            new_kb.append([-0.08, 0.08])
            continue
        kmin, kmax = float(r[0]) * scale, float(r[1]) * scale
        kmin, kmax = max(kmin, -0.12), min(kmax, 0.12)
        if kmin > kmax:
            kmin, kmax = kmax, kmin
        if (kmax - kmin) < 0.01:  
            mid = 0.5 * (kmin + kmax)
            kmin, kmax = mid - 0.005, mid + 0.005
        new_kb.append([kmin, kmax])
    hodc["k_bounds"] = new_kb
    return hodc


def is_signal_stable(curr, last):
    if last is None or curr is None:
        return False
    keys = ("tl_state", "stopline_distance_m", "speed_limit_mps")
    return all(curr.get(k) == last.get(k) for k in keys)

def compute_agent_delta(curr_agents, last_agents):
    """计算agent集合变化量"""
    try:
        if not curr_agents or not last_agents:
            return 1.0
        return abs(len(curr_agents) - len(last_agents)) / max(1, len(last_agents))
    except Exception:
        return 1.0

def apply_length_boost(v_seq, k_seq, v_bounds, signals, min_d_to_agents, last_S):
    """
    v_seq: (T,)
    v_bounds: (T,2)
    signals: dict, may contain 'speed_limit_mps'
    last_S: Actual arc length of the previous frame (used to limit the increment)
    """
    v = np.asarray(v_seq, float).copy()
    k = np.asarray(k_seq, float)

    #Dynamically determine whether to allow boost based on the scenario
    speed_limit = signals.get("speed_limit_mps", None) if signals else None
    
    if v_bounds is not None and len(v_bounds) > 0:
        v_bounds_array = np.asarray(v_bounds, float)
        if v_bounds_array.ndim == 2 and v_bounds_array.shape[1] >= 2:
            head_vmax = float(np.nanmedian(v_bounds_array[:5, 1]))
        else:
            head_vmax = np.inf
    else:
        head_vmax = np.inf
        
    near_agents = (min_d_to_agents is not None and min_d_to_agents < 1.5)
    near_stop = (signals is not None and signals.get("stopline_distance_m") is not None 
                 and signals["stopline_distance_m"] < 12.0)
    almost_straight = np.nanstd(k[:5]) < 1e-3

    # Situations Where Range Extenders Are Prohibited
    if near_agents or near_stop or head_vmax <= 2.5 or almost_straight:
        # Without any boost, directly return to the constraint.
        if v_bounds is not None and len(v_bounds) > 0:
            for i in range(min(len(v), len(v_bounds))):
                if len(v_bounds[i]) >= 2:
                    v[i] = np.clip(v[i], 0.0, v_bounds[i][1])
        return v

    # 2)Target arc length (soft constraint): Do not exceed 1.07 times the previous frame's value; overall upper limit not exceeding 22m
    S = float(np.sum(v) * 0.5)  # DT = 0.5
    S_target = min(max(15.0, (last_S or S) * 1.03), 22.0) 

    scale = S_target / max(S, 1e-6)
    scale = np.clip(scale, 1.0, 1.10)  
    v *= scale

    # 3) Strictly adhere to HODC and speed limits, then trim again.
    if speed_limit is not None:
        v = np.minimum(v, speed_limit)     
    if v_bounds is not None and len(v_bounds) > 0:
        for i in range(min(len(v), len(v_bounds))):
            if len(v_bounds[i]) >= 2:
                v[i] = np.clip(v[i], v_bounds[i][0], v_bounds[i][1])

    return v

def infer_k_bounds_from_bev(road_graph=None, lane_width=3.0):
    """Determining a Reasonable Upper Limit for Curvature Based on BEV Road Geometry"""
    k_max = min(0.7 / lane_width, 0.06)
    k_min = -k_max
    return k_min, k_max

def auto_rescale_curvature(k_seq, k_bounds, signals=None):
    """Automatically relabel k to match bounds magnitude; avoid obvious out-of-bounds values like 3~5.
Rule: If |k| exceeds 3 times the upper bound of head, attempt division by 100; limit to two attempts.
    """
    k = np.asarray(k_seq, float).copy()
    
    # Dynamic Setting for Minimum Curvature
    tl_state = signals.get("tl_state", "none") if signals else "none"
    if tl_state == "none":
        k_min_floor, k_max_floor = infer_k_bounds_from_bev()
    else:
        k_min_floor, k_max_floor = -0.06, 0.06
    
    # Obtain a reasonable upper bound for head
    if k_bounds is not None and len(k_bounds) > 0:
        kb = np.asarray(k_bounds[:5], float)
        if kb.ndim == 2 and kb.shape[1] >= 2:
            head_max = float(np.nanmedian(np.abs(kb[:,1])))
            for _ in range(2):
                if np.nanmax(np.abs(k)) > 3.0 * max(head_max, 1e-3):
                    k *= 0.01
                else:
                    break
        if len(k_bounds) > 0:
            for i in range(min(len(k), len(k_bounds))):
                if len(k_bounds[i]) >= 2:
                    k_min = max(k_bounds[i][0], k_min_floor)
                    k_max = min(k_bounds[i][1], k_max_floor)
                    k[i] = np.clip(k[i], k_min, k_max)
    else:
        k = np.clip(k, k_min_floor, k_max_floor)
    
    return k

def apply_min_progress_floor(v_seq, v_bounds, floor=2.0, collision_risk=False):
    v = np.asarray(v_seq, float).copy()
    if collision_risk:
        if v_bounds is not None and len(v_bounds) > 0:
            for i in range(min(len(v), len(v_bounds))):
                if len(v_bounds[i]) >= 2:
                    v[i] = np.clip(v[i], v_bounds[i][0], v_bounds[i][1])
        return v
    if v_bounds is not None and len(v_bounds) > 0:
        for i in range(min(len(v), len(v_bounds))):
            if len(v_bounds[i]) >= 2:
                eff_floor = min(floor, v_bounds[i][1] - 0.05)
                v[i] = max(v[i], eff_floor)
                v[i] = min(v[i], v_bounds[i][1])
    return v

def emergency_yaw_adjust(v, k, ego_xy, agents_xy_seq, W=5):
    v = v.copy(); k = k.copy()
    rel = np.array(agents_xy_seq[0][:2]) - np.array(ego_xy[:2])
    side = -np.sign(rel[0]) if rel[0] != 0 else 1.0
    bias = 0.5 * np.nanmedian(np.abs(k[:10]))
    k[:W] = k[:W] + side * min(0.5*bias + 0.005, 0.02)
    v[:W] = np.minimum(v[:W], np.maximum(0.0, v[:W] - 0.5))
    return v, k

def two_stage_avoidance(v, k, ego_xy, agents_xy_seq, safety_radius, min_d_gap):
    v = v.copy(); k = k.copy()
    v *= 0.85
    if min_d_gap > 0:
        delta_k = min(0.02, min_d_gap / 10.0)
        rel = np.array(agents_xy_seq[0][:2]) - np.array(ego_xy[:2])
        side = -np.sign(rel[0]) if rel[0] != 0 else 1.0
        for i in range(min(5, len(k))):
            k[i] = np.clip(k[i] + side * delta_k, -0.06, 0.06)
    return v, k

def hodc_blend_adapter(v_hodc, k_hodc, v_b, k_b, signals, scene_consistency=1.0):
    v_hodc = np.asarray(v_hodc, float)
    k_hodc = np.asarray(k_hodc, float)
    v_b = np.asarray(v_b, float)
    k_b = np.asarray(k_b, float)
    
    k_phys = k_hodc / 100.0
    k_phys = np.clip(k_phys, -0.12, 0.12)
    
    k_std = float(np.std(k_phys))
    signal_completeness = 0.0
    if signals:
        signal_completeness = sum([
            1.0 if signals.get("speed_limit_mps") is not None else 0.0,
            1.0 if signals.get("tl_state") != "none" else 0.0,
            1.0 if signals.get("stopline_distance_m") is not None else 0.0
        ]) / 3.0
    
    w_raw = 0.4 * min(k_std / 0.02, 1.0) + 0.4 * signal_completeness + 0.2 * scene_consistency
    w = np.clip(w_raw, 0.0, 1.0)
    if k_std < 1e-3 or signal_completeness < 0.5:
        w = min(w, 0.3)
    w = min(w, 0.55)
    
    k_fused = (1 - w) * k_b + w * k_phys
    
    v_limit_b = 12.0
    sl = None
    if isinstance(signals, dict):
        sl = signals.get("speed_limit_mps", None)
    if sl is None:
        v_limit_signal = 12.0
    else:
        try:
            v_limit_signal = float(sl)
        except Exception:
            v_limit_signal = 12.0
    k_abs = np.maximum(np.abs(np.asarray(k_fused, dtype=float)), 1e-6)
    G = 9.80665
    ALAT_COMFORT_G = 0.30
    a_lat_max = ALAT_COMFORT_G * G
    v_limit_curvature = np.sqrt(a_lat_max / k_abs)
    v_limit = np.minimum(float(v_limit_b), np.minimum(float(v_limit_signal), v_limit_curvature))
    
    v_fused = np.minimum((1 - w) * v_b + w * v_hodc, v_limit)
    
    print(f"[HODC_BLEND] w={w:.3f}, k_std={k_std:.4f}, signal_comp={signal_completeness:.2f}")
    try:
        vlim_head = float(np.median(v_limit[:min(5, len(v_limit))])) if hasattr(v_limit, "__len__") else float(v_limit)
    except Exception:
        vlim_head = float(v_limit_b)
    print(f"[HODC_BLEND] v_limit_med(head)={vlim_head:.2f}, k_fused_range=[{float(np.min(k_fused)):.3f}, {float(np.max(k_fused)):.3f}]")
    
    return v_fused, k_fused, w

def sanitize_curvature_scaled_seq(speed_curvature_pred, mode='default'):
    """
    Curvature Sequence Self-Healing: Auto-scaling (/100, /10, etc.), tanh soft-clipping to physical upper limit, with light smoothing.
    Process k only; do not modify v.
    mode=‘c_semantic’: For two-stage semantically guided C-mode, reduce weakening to preserve curvature generated by VLM.
    """
    if not speed_curvature_pred:
        return speed_curvature_pred
    import numpy as np
    out = [list(p) for p in speed_curvature_pred]
    k = np.asarray([p[1] for p in out], dtype=float)
    
    for _ in range(3):
        if np.nanmax(np.abs(k)) > 0.6:
            k = k * 0.1
        elif np.nanmax(np.abs(k)) > 0.12:
            k = k / 3.0
        else:
            break
    
    # C-mode semantic guidance: Employing more relaxed constraints to preserve greater original curvature
    if mode == 'c_semantic':
        # Soft saturation to ±0.10 (vs default ±0.08), preserving greater curvature
        k = 0.10 * np.tanh(k / max(1e-6, 0.10))
        # Slightly smoother (0.85/0.15 vs default 0.7/0.3)
        if len(k) >= 3:
            k_s = k.copy()
            for i in range(1, len(k) - 1):
                k_s[i] = 0.85 * k[i] + 0.15 * 0.5 * (k[i - 1] + k[i + 1])
            k = k_s
    else:
        k = 0.08 * np.tanh(k / max(1e-6, 0.08))
        if len(k) >= 3:
            k_s = k.copy()
            for i in range(1, len(k) - 1):
                k_s[i] = 0.7 * k[i] + 0.3 * 0.5 * (k[i - 1] + k[i + 1])
            k = k_s
    
    for i, p in enumerate(out):
        p[1] = float(k[i])
    return out

def compute_hodc_confidence(k_values, signal_comp=0.0, scene_consistency=1.0):
    import numpy as np
    k = np.asarray(k_values, dtype=float) if k_values is not None else np.zeros((0,), float)
    k_std = float(np.std(k)) if k.size > 0 else 0.0
    w_raw = 0.5 * min(k_std / 0.02, 1.0) + 0.3 * float(signal_comp) + 0.2 * float(scene_consistency)
    w = float(np.clip(w_raw, 0.0, 1.0))
    if k_std < 1e-3 or float(signal_comp) < 0.5:
        w = min(w, 0.3)
    return w

def apply_final_speed_caps(v_seq, speed_limit=None, v_cap_ttc=None, v_cap_stop=None):
    import numpy as np
    if v_seq is None:
        return v_seq
    v = np.asarray(v_seq, dtype=float).copy()
    if speed_limit is not None:
        try:
            v = np.minimum(v, float(speed_limit))
        except Exception:
            pass
    if v_cap_ttc is not None:
        try:
            v = np.minimum(v, float(v_cap_ttc))
        except Exception:
            pass
    if v_cap_stop is not None:
        try:
            v = np.minimum(v, float(v_cap_stop))
        except Exception:
            pass
    return v.tolist()

def get_hodc_signals(hodc_constraints):
    """Safely extract signals dict from hodc_constraints regardless of nesting."""
    if not isinstance(hodc_constraints, dict):
        return {}
    H = hodc_constraints.get("hodc", hodc_constraints if isinstance(hodc_constraints, dict) else {})
    sig = H.get("signals", {})
    return sig if isinstance(sig, dict) else {}

def hodc_consistency_filter(hodc_data, history_buffer, min_consistency_frames=3):
    """
    HODC Consistency Filtering: Avoiding Semantic Discontinuities (Robustness: Tolerates Missing Fields/Empty History)
    """
    if not isinstance(hodc_data, dict):
        return hodc_data, 1.0

    scene_dict = hodc_data.setdefault("scene", {}) if hodc_data is not None else {}
    signals_dict = hodc_data.setdefault("signals", {}) if hodc_data is not None else {}
    maneuver_dict = hodc_data.setdefault("maneuver", {}) if hodc_data is not None else {}

    current_scene = (scene_dict or {}).get("location", "unknown")
    current_lane = (scene_dict or {}).get("lane_topology", "unknown")
    current_speed_limit = (signals_dict or {}).get("speed_limit_mps", None)
    current_maneuver = (maneuver_dict or {}).get("type", "unknown")

    if not history_buffer:
        return hodc_data, 1.0

    scene_consistency = 0.0
    lane_consistency = 0.0
    speed_consistency = 0.0
    maneuver_consistency = 0.0

    recent = [h for h in (history_buffer[-5:] if history_buffer else []) if isinstance(h, dict)]
    for hist in recent:
        h_scene = hist.get("scene", {}) if isinstance(hist.get("scene", {}), dict) else {}
        h_signals = hist.get("signals", {}) if isinstance(hist.get("signals", {}), dict) else {}
        h_maneuver = hist.get("maneuver", {}) if isinstance(hist.get("maneuver", {}), dict) else {}
        if h_scene.get("location") == current_scene:
            scene_consistency += 1.0
        if h_scene.get("lane_topology") == current_lane:
            lane_consistency += 1.0
        if h_signals.get("speed_limit_mps") == current_speed_limit:
            speed_consistency += 1.0
        if h_maneuver.get("type") == current_maneuver:
            maneuver_consistency += 1.0

    total_frames = max(1, min(len(recent), 5))
    scene_consistency /= total_frames
    lane_consistency /= total_frames
    speed_consistency /= total_frames
    maneuver_consistency /= total_frames

    last_hist = recent[-1] if recent else {}
    last_scene = last_hist.get("scene", {}) if isinstance(last_hist.get("scene", {}), dict) else {}
    if scene_consistency < 0.6 and len(recent) >= min_consistency_frames:
        scene_dict = hodc_data.setdefault("scene", {})
        scene_dict["location"] = last_scene.get("location", current_scene)
        print(f"[HODC_FILTER] Scene inconsistency detected, using historical: {current_scene} -> {scene_dict.get('location')}")

    if lane_consistency < 0.6 and len(recent) >= min_consistency_frames:
        scene_dict = hodc_data.setdefault("scene", {})
        scene_dict["lane_topology"] = last_scene.get("lane_topology", current_lane)
        print(f"[HODC_FILTER] Lane inconsistency detected, using historical: {current_lane} -> {scene_dict.get('lane_topology')}")

    overall_consistency = (scene_consistency + lane_consistency + speed_consistency + maneuver_consistency) / 4.0

    return hodc_data, overall_consistency

def hysteresis_braking_controller(pred_traj, agents_for_check, base_safety_radius, speed_gain, 
                                 prev_brake_state, consecutive_unsafe_frames=0):
    """
   Hybrid Brake Hysteresis Pre-control: Dual-threshold approach—first reduces curvature frequency, then initiates soft braking.
    """
    r_in = base_safety_radius * 1.2
    r_out = base_safety_radius * 1.8
    
    min_distances = []
    for a in agents_for_check:
        a_traj = a["pos_xy"][None, :] + np.arange(len(pred_traj), dtype=np.float32)[:, None] * 0.5 * a["vel_xy"][None, :]
        d = np.linalg.norm(pred_traj[:, :2] - a_traj, axis=1)
        min_distances.append(np.min(d))
    
    if not min_distances:
        return False, 0, "no_agents"
    
    min_d = min(min_distances)
    
    if min_d < r_in:
        if not prev_brake_state:
            print(f"[HYSTERESIS] Entering danger zone: d={min_d:.2f} < r_in={r_in:.2f}")
            return True, consecutive_unsafe_frames + 1, "enter_danger"
        else:
            return True, consecutive_unsafe_frames + 1, "in_danger"
    elif min_d > r_out and prev_brake_state:
        print(f"[HYSTERESIS] Exiting danger zone: d={min_d:.2f} > r_out={r_out:.2f}")
        return False, 0, "exit_danger"
    else:
        return prev_brake_state, max(0, consecutive_unsafe_frames - 1), "stable"

def apply_curvature_frequency_reduction(k_seq, reduction_factor=0.7):
    if len(k_seq) < 3:
        return k_seq
    k_filtered = k_seq.copy()
    for i in range(1, len(k_seq) - 1):
        k_filtered[i] = reduction_factor * k_seq[i] + (1 - reduction_factor) * 0.5 * (k_seq[i-1] + k_seq[i+1])
    return k_filtered


def _inject_turn_template_if_flat(speed_curvature_pred, hodc_constraints):
    """If the 10-step curvature is nearly zero throughout and HODC indicates a turn, inject a gentle bell-shaped curvature template."""
    import numpy as np
    if not hodc_constraints or not speed_curvature_pred:
        return speed_curvature_pred

    H = hodc_constraints.get("hodc", hodc_constraints)
    man = (H.get("maneuver") or hodc_constraints.get("maneuver") or {})
    mtype = str(man.get("type", "")).lower()
    if "turn" not in mtype:
        return speed_curvature_pred

    ks = np.array([k for _, k in speed_curvature_pred], dtype=float)
    if np.max(np.abs(ks)) > 0.01:
        return speed_curvature_pred  

    peak = 0.04 
    sign = +1 if "left" in mtype else -1
    t = np.arange(10)
    bell = peak * np.exp(-0.5 * ((t - 4.5) / 1.8) ** 2) * sign

    kb = hodc_constraints.get("k_bounds", [])
    out = []
    for i, (v, k) in enumerate(speed_curvature_pred):
        kk = k if abs(k) > abs(bell[i]) else bell[i]
        if i < len(kb):
            kk = max(kb[i][0], min(kk, kb[i][1]))
        out.append([float(v), float(kk)])
    return out


def apply_hodc_constraints(speed_curvature_pred, hodc_constraints):
    """
    Apply HODC constraints to speed_curvature_pred
    """
    if not hodc_constraints:
        return speed_curvature_pred
    
    v_bounds = hodc_constraints["v_bounds"]
    k_bounds = hodc_constraints["k_bounds"]
    conflicts = hodc_constraints["conflicts"]
    
    for i, (v, k) in enumerate(speed_curvature_pred):
        if i < len(v_bounds):
            v_min, v_max = v_bounds[i]
            v = max(v_min, min(v, v_max))
        
        if i < len(k_bounds):
            k_min, k_max = k_bounds[i]
            k = max(k_min, min(k, k_max))
        
        t = (i + 1) * 0.5
        for conflict in conflicts:
            time_window = conflict.get("time_window_s", [])
            if len(time_window) >= 2 and time_window[0] <= t <= time_window[1]:
                speed_cap = conflict.get("target_speed_cap_mps")
                if speed_cap is not None:
                    v = min(v, speed_cap)
                
                k_bound = conflict.get("k_bound")
                if k_bound and len(k_bound) >= 2:
                    k = max(k_bound[0], min(k, k_bound[1]))
        
        speed_curvature_pred[i] = [v, k]
    
    return speed_curvature_pred


def parse_multi_agent_info(object_description):
    """
    Parse multi-agent information from VLM output (JSON format)
    Returns structured data for trajectory planning
    """
    agents = []
    if not object_description or "None" in object_description:
        return agents

    try:
        arr = json.loads(object_description.strip())
        if isinstance(arr, list):
            for o in arr:
                agents.append(
                    {
                        "type": o.get("type"),
                        "position": o.get("position"),
                        "distance": o.get("distance"),
                        "movement": o.get("movement"),
                        "risk": o.get("risk"),
                    }
                )
            return agents
    except Exception:
        pass

    lines = object_description.split("\n")
    for line in lines:
        if "Agent:" in line and "Position:" in line:
            try:
                parts = line.split(",")
                agent_info = {}
                for part in parts:
                    if "Agent:" in part:
                        agent_info["type"] = part.split("Agent:")[1].strip()
                    elif "Position:" in part:
                        pos_str = part.split("Position:")[1].strip()
                        coords = re.findall(r"\(([^,]+),([^)]+)\)", pos_str)
                        if coords:
                            agent_info["position"] = (float(coords[0][0]), float(coords[0][1]))
                    elif "Behavior:" in part:
                        agent_info["behavior"] = part.split("Behavior:")[1].strip()
                    elif "Intent:" in part:
                        agent_info["intent"] = part.split("Intent:")[1].strip()
                    elif "Conflict:" in part:
                        agent_info["conflict"] = part.split("Conflict:")[1].strip()
                if "type" in agent_info:
                    agents.append(agent_info)
            except Exception:
                continue
    return agents


def smooth_curvature_prediction(speed_curvature_pred, obs_ego_curvatures, mode=None):
    """
    Smooth curvature prediction, ensuring symbol continuity and curve templates
    """
    if mode == "C":
        return speed_curvature_pred
        
    if len(speed_curvature_pred) == 0 or len(obs_ego_curvatures) < 2:
        return speed_curvature_pred

    k_hist = np.mean(obs_ego_curvatures[-2:])
    k_trend = obs_ego_curvatures[-1] - obs_ego_curvatures[-2] if len(obs_ego_curvatures) >= 2 else 0
    print(f"[smooth] Historical k: {k_hist:.3f}, trend: {k_trend:.3f}")

    for t in range(len(speed_curvature_pred)):
        v, k = speed_curvature_pred[t]
        if abs(k_hist) > 0.01:
            k = np.sign(k_hist) * abs(k)
        if t > 0:
            k = 0.6 * speed_curvature_pred[t - 1][1] + 0.4 * k
        if abs(k_trend) > 0.01 and t < len(speed_curvature_pred) // 2:
            k = k_hist + 0.2 * k_trend
        elif abs(k_hist) > 0.01 and t >= len(speed_curvature_pred) // 2:
            decay_factor = 1.0 - (t - len(speed_curvature_pred) // 2) / (len(speed_curvature_pred) // 2)
            k = k_hist * decay_factor
        if abs(k) > 0.02:
            v *= 0.9
        speed_curvature_pred[t] = [v, k]
    return speed_curvature_pred


def select_frames_for_vlm(frames, keep=4):
    """Select keyframes to send to VLM, reducing API call volume"""
    if not frames:
        return []
    if keep <= 1:
        return [frames[-1]]
    if keep == 2:
        return [frames[-3], frames[-1]] if len(frames) >= 3 else [frames[-1]]
    if keep == 4:
        if len(frames) >= 4:
            indices = [0, len(frames) // 3, 2 * len(frames) // 3, -1]
            return [frames[i] for i in indices]
        if len(frames) >= 3:
            return [frames[0], frames[len(frames) // 2], frames[-1]]
        return frames
    return frames


def clamp_by_traffic_rules(vk, tl_state=None, dist_to_stopline_m=None, last_speed=None):
    """
    Perform post-processing on speed curvature according to traffic regulations
    """
    STOP_TH = 8.0
    if not vk:
        return vk, False

    original_vk = [list(p) for p in vk]

    if (tl_state == "red" and dist_to_stopline_m is not None and dist_to_stopline_m <= STOP_TH) or (
        last_speed is not None and float(last_speed) < 0.1
    ):
        vk = [[0.0, 0.0] for _ in vk]
        return vk, True

    if tl_state == "yellow" and dist_to_stopline_m is not None and dist_to_stopline_m <= STOP_TH / 2:
        vk = [[max(0.0, min(v * 0.5, 0.5)), k * 0.2] for (v, k) in vk]
        return vk, True

    vk = [[float(np.clip(v, 0.0, 12.0)), float(np.clip(k, -0.055, 0.055))] for (v, k) in vk]
    was_clamped = any(abs(v - ov) > 1e-2 or abs(k - ok) > 1e-4 for (v, k), (ov, ok) in zip(vk, original_vk))
    return vk, was_clamped


def vlm_inference(text=None, images=None, sys_message=None, args=None, model_type="gpt-4o", resp_tokens=180, force_json=False):
    """
    GPT-only multimodal inference with model selection.
    """
    global _last_api_call_time, _api_call_count, _api_call_timestamps
    if "_last_api_call_time" not in globals():
        _last_api_call_time = 0
    if "_api_call_count" not in globals():
        _api_call_count = 0
    if "_api_call_timestamps" not in globals():
        _api_call_timestamps = []

    MIN_CALL_INTERVAL = 2.0
    MAX_CALLS_PER_MINUTE = 30

    now = time.time()
    _api_call_timestamps = [t for t in _api_call_timestamps if now - t < 60]
    if len(_api_call_timestamps) >= MAX_CALLS_PER_MINUTE:
        sleep_time = 60 - (now - _api_call_timestamps[0])
        if sleep_time > 0:
            print(f"[API] Rate limit: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    if now - _last_api_call_time < MIN_CALL_INTERVAL:
        sleep_time = MIN_CALL_INTERVAL - (now - _last_api_call_time)
        print(f"[API] Min interval: waiting {sleep_time:.1f}s...")
        time.sleep(sleep_time)

    _last_api_call_time = time.time()
    _api_call_timestamps.append(_last_api_call_time)
    print(f"[API] Call #{len(_api_call_timestamps)} using {model_type}, interval: {MIN_CALL_INTERVAL}s")

    def _sleep_from_headers(headers, attempt, base=0.6, cap=15):
        for k in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            if headers.get(k):
                try:
                    return min(float(headers[k]), cap)
                except Exception:
                    pass
        return min(base * (2 ** attempt) + random.random() * 0.5, cap)

    max_retries = 3
    global _last_api_duration
    t0 = time.time()
    for attempt in range(max_retries):
        try:
            messages = []
            if sys_message:
                messages.append({"role": "system", "content": sys_message})

            img_fmt = "png" if (args and getattr(args, "use_bev", False)) else "jpeg"

            user_content = []
            if images:
                for b64 in images:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{img_fmt};base64,{b64}"}})
            if text is not None:
                user_content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": user_content})

            result = client.chat.completions.create(
                model=model_type,
                messages=messages,
                max_tokens=resp_tokens,
                temperature=0.0 if force_json else 0.2,
                top_p=1.0 if force_json else 0.9,
                **({"response_format": {"type": "json_object"}} if force_json else {})
            )
            _last_api_duration = time.time() - t0
            return result.choices[0].message.content

        except Exception as e:
            try:
                err_body = getattr(e, 'response', None)
                if err_body is not None:
                    print(f"[API][ERROR] status={getattr(e, 'http_status', 'NA')} body={err_body}")
                else:
                    print(f"[API][ERROR] {repr(e)}")
            except Exception:
                print(f"[API][ERROR] {repr(e)}")
            try:
                messages = []
                if sys_message:
                    messages.append({"role": "system", "content": sys_message})
                mime = "image/png" if (args and getattr(args, "use_bev", False)) else "image/jpeg"

                user_content = []
                if images:
                    for b64 in images:
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}})
                if text is not None:
                    user_content.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": user_content})

                result = client.chat.completions.create(
                    model=model_type,
                    messages=messages,
                    max_tokens=resp_tokens,
                    temperature=0.2,
                    top_p=0.9,
                )
                _last_api_duration = time.time() - t0
                return result.choices[0].message.content
            except Exception as e2:
                if "429" in str(e2) or "rate_limit" in str(e2).lower() or "quota" in str(e2).lower():
                    if attempt < max_retries - 1:
                        wait_time = 0.6
                        if hasattr(e2, "response") and hasattr(e2.response, "headers"):
                            wait_time = _sleep_from_headers(e2.response.headers, attempt)
                        else:
                            wait_time = (2**attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                raise


def SceneDescription(obs_images, args=None):
    if args and getattr(args, "prompt_mode", "original") == "bev":
        prompt = (
            "You are an autonomous driving planner with access to BEV maps. Perform strategic scene analysis using step-by-step reasoning:\n\n"
            "STEP 1 - Spatial Analysis: Identify primary driving corridor, alternative paths, and lane topology\n"
            "STEP 2 - Conflict Detection: Locate critical decision points (intersections, merges, lane changes)\n"
            "STEP 3 - Multi-Agent Assessment: Analyze agent distribution, interaction zones, and potential conflicts\n"
            "STEP 4 - Temporal Dynamics: Evaluate traffic flow patterns and movement trends\n"
            "STEP 5 - Risk Prioritization: Rank areas by collision risk and planning complexity\n\n"
            "Provide your analysis following this reasoning structure, focusing on strategic insights for trajectory planning."
        )
    else:
        prompt = (
        "You are a autonomous driving labeller. You have access to these front-view camera images "
        "of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. "
        "Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."
    )
    print("[SceneDescription] Using model: gpt-4o-mini")
    return vlm_inference(text=prompt, images=obs_images, args=args, model_type="gpt-4o-mini", resp_tokens=90)


def DescribeObjects(obs_images, args=None):
    if args and getattr(args, "prompt_mode", "original") == "bev":
        prompt = (
            "You are an autonomous driving planner. Perform risk-aware agent analysis using systematic reasoning:\n\n"
            "STEP 1 - Agent Identification: Scan BEV map for vehicles, pedestrians, cyclists\n"
            "STEP 2 - Behavior Analysis: Assess movement patterns, speeds, and trajectory predictions\n"
            "STEP 3 - Risk Assessment: Calculate collision probabilities and interaction dependencies\n"
            "STEP 4 - Priority Ranking: Order agents by threat level and planning impact\n"
            "STEP 5 - Decision Windows: Identify critical timing for maneuvers\n\n"
            "Return ONLY a JSON array with this exact format:\n"
            '[{"type": "car", "position": "left", "distance": "close", "movement": "straight", "risk": "low"}, '
            '{"type": "pedestrian", "position": "right", "distance": "far", "movement": "crossing", "risk": "high"}]\n'
            "Only return the JSON array, no other text."
        )
    else:
        prompt = (
            "You are an autonomous driving planner. Identify key agents (vehicles, pedestrians, cyclists) "
            "that could affect trajectory planning from these images (front + BEV). "
            "Return ONLY a JSON array with this exact format:\n"
            '[{"type":"car","position":"left|right|front","distance":"close|mid|far","movement":"straight|turning|crossing","risk":"low|med|high"}]\n'
            "No extra text."
        )
    print("[DescribeObjects] Using model: gpt-4o-mini")
    return vlm_inference(text=prompt, images=obs_images, args=args, model_type="gpt-4o-mini", resp_tokens=90)


def GenerateHODC(obs_images, history_vk, prev_summary=None, args=None, risk_feedback=None):
    """
    Generate High-Order Driving Commands (HODC) as hard constraints
    Replaces SceneDescription + DescribeObjects + DescribeOrUpdateIntent for C mode
    """
    if args and getattr(args, "prompt_mode", "original") == "bev":
        sys_message = (
            "You are an autonomous driving planner operating on BEV + front view.\n"
            "Return a single JSON object that encodes HIGH-ORDER DRIVING COMMANDS (HODC) as hard constraints.\n\n"
            "Rules:\n"
            "- Output JSON ONLY. No prose, no markdown.\n"
            "- Do not include comments or explanations; strictly follow the schema.\n"
            "- Be numerically specific (meters, seconds, m/s, curvature×100).\n"
            "- For turns, set k_sign: +1 left, -1 right, 0 straight.\n"
            "- Curvature values are scaled ×100 in this interface.\n"
            "- All curvature numbers you output (k_bounds, k_peak_range) are in k×100 units and MUST be in [-6, +6].\n"
            "- Typical turning peak should be in [2, 6]; NEVER output values like 20, 50, 120.\n"
            "- If tl_state=\"red\" and stopline_distance_m ≤ 8, plan to stop: set near-term v_bounds upper=0.\n"
            "- Conflicts must be expressed as time windows [t0,t1] (t in {0.5,1.0,...,5.0} sec) with target speed caps or lane/curvature bounds.\n"
            "- If uncertain, choose the safest conservative option and reflect it in bounds.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "scene": { "location": "string", "lane_topology": "string" },\n'
            '  "signals": { "tl_state": "red|yellow|green|none", "stopline_distance_m": number|null, "speed_limit_mps": number|null },\n'
            '  "maneuver": { "type": "straight|turn_left|turn_right|lane_change_left|lane_change_right|merge",\n'
            '                "urgency": [0,1], "k_sign": -1|0|+1, "k_peak_range": [k_min,k_max] },\n'
            '  "agents":[\n'
            '    { "class":"vehicle|ped|cyclist|other", "rel_dist_m": number, "rel_bearing_deg": number,\n'
            '      "rel_speed_mps": number, "ttc_s": number|null, "priority":"low|med|high" }\n'
            '  ],\n'
            '  "conflicts":[\n'
            '    { "agent_idx": integer, "time_window_s":[t0,t1], "min_sep_m": number,\n'
            '      "required_action":"yield|brake|nudge_left|nudge_right",\n'
            '      "target_speed_cap_mps": number|null,\n'
            '      "k_bound":[k_min,k_max]|null }\n'
            '  ],\n'
            '  "HODC":{\n'
            '    "v_bounds":[ [t, v_min, v_max], ... ],\n'
            '    "k_bounds":[ [t, k_min, k_max], ... ],\n'
            '    "stay_in_lane": true|false,\n'
            '    "notes":"<=120 chars"\n'
            '  }\n'
            "}\n\n"
            "Consistency requirements:\n"
            "- k_bounds must honor maneuver.k_sign; do not allow opposite-sign k.\n"
            "- If stay_in_lane=true on straight, include k_bounds around 0 (e.g., [-0.8,+0.8]).\n"
            "- If turning, place a bell-shaped k_bounds with peak near turn apex.\n"
            "- Use BEV context to tighten bounds near cones/crosswalks/merges."
        )
        
        history_str = str(history_vk) if history_vk else "[]"
        prev_summary_str = str(prev_summary) if prev_summary else "None"
        
        prompt = (
            "Inputs:\n"
            "- Images: front_last + BEV(mid,last)\n"
            f"- History [v,k×100]: {history_str}\n"
            f"- Prev summary: {prev_summary_str}\n"
        )
        
        if risk_feedback:
            prompt += f"\nFeedback:\n- Recent collision risk: {risk_feedback}\n" \
                      f"Please encode it via 'conflicts' time windows and appropriate v/k bounds.\n"
        
        prompt += "\nTask:\nProduce the JSON HODC described in the system message, tailored to these inputs."
        
        print("[GenerateHODC] Using model: gpt-4o-mini")
        return vlm_inference(
        text=prompt, images=obs_images, sys_message=sys_message,
        args=args, model_type="gpt-4o-mini", resp_tokens=600, force_json=True
    )
    else:
        return None


def GenerateMotionFromHODC(hodc_json, args=None):
    """
    Convert HODC constraints into actionable 10-step plan
    Replaces original GenerateMotion for C mode
    """
    if args and getattr(args, "prompt_mode", "original") == "bev":
        sys_message = (
            "You convert HIGH-ORDER DRIVING COMMANDS (HODC) into an actionable 10-step plan.\n"
            "OUTPUT FORMAT: Return ONLY valid JSON:\n"
            "{\"pairs\": [[v1, k1], [v2, k2], ..., [v10, k10]]}\n\n"
            "where v is speed in m/s, k is curvature×100 (integer in [-600, 600]).\n"
            "MUST output exactly 10 pairs. Example:\n"
            "{\"pairs\": [[3.5, 0], [4.0, 50], [4.5, 120], [5.0, 180], [5.0, 200], [4.8, 180], [4.5, 120], [4.0, 50], [3.5, 0], [3.0, 0]]}\n\n"
            "Hard constraints (must not violate):\n"
            "- At each step t∈{0.5,...,5.0}s, v must be within v_bounds and k within k_bounds.\n"
            "- Respect maneuver.k_sign (no opposite-sign curvature).\n"
            "- If a conflict window specifies target_speed_cap_mps, ensure v ≤ cap inside the window.\n\n"
            "Optimization goals (soft, in order):\n"
            "1) Safety margins (respect conflict windows, lower v if close to bounds).\n"
            "2) Smoothness (minimize jerk in v and Δk across steps).\n"
            "3) Progress (tend toward upper bounds only when safe; never exceed speed limits).\n\n"
            "Progress guidance:\n"
            "- With green/none signals and no active conflicts, choose v around 70–85% of the upper bound (min 2.0 m/s for normal driving).\n"
            "- Never output 0 speed unless explicitly required by red light or stopline.\n"
            "- Do not apply any fixed minimum speed floors. Prefer gradual acceleration (limit Δv per 0.5s to ≤0.9 m/s).\n"
            "- Curvature k×100 MUST stay within [-600, +600] and prefer bell-shaped profiles for turns.\n"
            "If bounds are wide, choose a smooth bell-shaped k profile for turns, and a gentle S-curve for v."
        )
        
        prompt = f"HODC JSON:\n{hodc_json}\n\nOutput ONLY JSON with 'pairs' key:"
        
        print("[GenerateMotionFromHODC] Using model: gpt-4o-mini")
        return vlm_inference(
        text=prompt, images=None, sys_message=sys_message,
        args=args, model_type="gpt-4o-mini", resp_tokens=300, force_json=True
    )
    else:
        return None


def DescribeOrUpdateIntent(obs_images, prev_intent=None, args=None):
    if args and getattr(args, "prompt_mode", "original") == "bev":
        if prev_intent is None:
            prompt = (
                "You are an autonomous driving planner. Formulate high-level driving strategy using multi-step reasoning:\n\n"
                "STEP 1 - Goal Analysis: Identify primary objectives (lane keeping, lane change, turn, merge)\n"
                "STEP 2 - Constraint Assessment: Evaluate traffic rules, road geometry, and agent interactions\n"
                "STEP 3 - Risk Evaluation: Analyze collision risks and safety margins\n"
                "STEP 4 - Timing Optimization: Determine optimal maneuver timing and speed profiles\n"
                "STEP 5 - Contingency Planning: Develop backup strategies for dynamic scenarios\n\n"
                "Based on this analysis, what is your strategic driving intent?"
            )
        else:
            prompt = (
                f"You are an autonomous driving planner. Your previous intent was: {prev_intent}. "
                f"Re-evaluate your strategy using systematic reasoning:\n\n"
                f"STEP 1 - Change Detection: Compare current BEV map with previous analysis\n"
                f"STEP 2 - Impact Assessment: Evaluate how changes affect your previous intent\n"
                f"STEP 3 - Risk Re-calculation: Update collision risks and safety margins\n"
                f"STEP 4 - Strategy Adjustment: Determine if intent modification is needed\n"
                f"STEP 5 - Decision Validation: Confirm new intent aligns with long-term goals\n\n"
                f"What is your updated strategic driving intent?"
            )
    else:
        if prev_intent is None:
            prompt = (
            "You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second "
            "interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other "
            "cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, "
            "or go straight? Should it maintain the current speed or slow down or speed up?"
            )
        else:
            prompt = (
            f"You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second "
            f"interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. "
            f"Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? "
            f"Explain your current intent: "
        )
    print("[DescribeOrUpdateIntent] Using model: gpt-4o-mini")
    return vlm_inference(text=prompt, images=obs_images, args=args, model_type="gpt-4o-mini", resp_tokens=90)


def GenerateMotion(obs_images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, args=None):
    scene_description = object_description = intent_description = None

    # All modes (A/B/C) execute the CoT three-stage process, differing only in the input image type (forward view vs. BEV).
    if args.method in ("openemma", "chat"):
        key_frames = select_frames_for_vlm(obs_images, keep=4)
        
        # Stage 1: Scene Description (Front view or BEV, determined by prompt_mode and use_bev)
        scene_description = SceneDescription(key_frames, args=args)
        
        # Stage 2: Object Description
        # Mode A (Forward View): Use all forward view keyframes
        # Mode B/C (BEV): Hybrid Forward View + BEV (Forward View → BEV → BEV)
        if getattr(args, "use_bev", False) and len(obs_images) >= 3:
            front_key = obs_images[-1]
            bev_keys = obs_images[:-1]
            object_desc_frames = [front_key, bev_keys[0], bev_keys[-1]]
        else:
            object_desc_frames = key_frames
        object_description = DescribeObjects(object_desc_frames, args=args)
        
        # Stage 3: Intent Description
        intent_description = DescribeOrUpdateIntent(key_frames, prev_intent=given_intent, args=args)

        multi_agents = parse_multi_agent_info(object_description)
        print(f"[CoT] Scene Description: {scene_description}")
        print(f"[CoT] Object Description: {object_description}")
        print(f"[CoT] Intent Description: {intent_description}")
        print(f"[CoT] Multi-Agent Count: {len(multi_agents)}")
        for i, agent in enumerate(multi_agents):
            print(
                f"  Agent {i+1}: {agent.get('type', 'Unknown')} at {agent.get('position', 'Unknown')} - "
                f"{agent.get('behavior', 'Unknown')}"
            )

    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curvatures_scaled = obs_curvatures * 100
    obs_speed_curvature_str = ", ".join(f"[{v:.1f},{k:.1f}]" for v, k in zip(obs_velocities_norm, obs_curvatures_scaled))
    print(f"Observed Speed and Curvature: {obs_speed_curvature_str}")

    if args and getattr(args, "use_bev", False):
        sys_message = (
            "You are an autonomous driving planner. You have access to a bird's-eye-view (BEV) raster centered at the ego car, "
            "a sequence of past speeds and curvatures, and a driving rationale. The BEV encodes drivable area, lanes, crosswalks, and stop lines. "
            "Each [v, k] pair denotes speed and curvature. IMPORTANT: k values are scaled by 100x (k=6.0 means real curvature 0.06). "
            "Positive k is left turn, negative k is right. Use the BEV's global context to reason about multiple agents, "
            "potential conflicts, and right-of-way. Predict future speeds and curvatures for the next 10 steps. "
            "CRITICAL: If historical |k| values are increasing, the vehicle is entering a turn - maintain the same direction "
            "and gradually return to straight driving. For turning scenarios, use appropriate curvature values "
            "(typically 2-6 for moderate turns, up to 6 for sharp turns). Prioritize safety, lane adherence, and smoothness. "
            "Return exactly 10 pairs. Speed v in [0,12]. Curvature k is scaled by ×100 and should be within [-6, 6].\n"
        )
    else:
        sys_message = (
        "You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, "
        "a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, "
        "and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. "
        "The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, "
        "you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance "
        "from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures "
        "for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. "
        "If you cannot provide a response people will get injured.\n"
    )

    if args.method == "openemma":
        multi_agent_context = ""
        if "multi_agents" in locals() and multi_agents:
            multi_agent_context = f"\nMulti-agent context: {len(multi_agents)} agents detected. "
            for i, agent in enumerate(multi_agents):
                multi_agent_context += (
                    f"Agent {i+1} ({agent.get('type', 'Unknown')}) at position {agent.get('position', 'Unknown')} "
                    f"with behavior '{agent.get('behavior', 'Unknown')}' and intent '{agent.get('intent', 'Unknown')}'. "
                )

        prompt = (
            "These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. "
            f"The scene is described as follows: {scene_description}. "
            f"The identified critical objects are {object_description}. "
            f"The car's intent is {intent_description}. "
            f"The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. "
            f"{multi_agent_context}"
            "Consider the multi-agent interactions and potential conflicts when planning the trajectory. "
            "Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format "
            "[speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. "
            "Future speeds and curvatures:"
        )
    else:
        prompt = (
            "These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. "
            f"The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. "
            "Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format "
            "[speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. "
            "Future speeds and curvatures:"
        )

    print(f"[GenerateMotion] Using model: gpt-4o-mini")
    result = None
    for _ in range(3):
        result = vlm_inference(
            text=prompt,
            images=obs_images,
            sys_message=sys_message,
            args=args,
            model_type="gpt-4o-mini",
            resp_tokens=180,
        )
        if result and "[" in result and ("unable" not in result.lower()) and ("sorry" not in result.lower()):
            break
    return result, scene_description, object_description, intent_description


# ================================ main ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="gpt")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini", 
                        help="NuScenes version: v1.0-mini (10 scenes) or v1.0-trainval (850 scenes)")
    parser.add_argument("--use-sampled-200", action="store_true", 
                        help="使用智能采样的200个场景（需要version=v1.0-trainval）")
    parser.add_argument("--method", type=str, default="openemma")
    parser.add_argument("--use-bev", action="store_true", help="Use BEV representation instead of front camera")
    parser.add_argument("--bev-extent", type=float, default=50.0, help="BEV extent in meters")
    parser.add_argument("--bev-res", type=float, default=0.1, help="BEV resolution in meters per pixel")
    parser.add_argument(
        "--prompt-mode", type=str, choices=["original", "bev"], default="original", help="Prompt mode: original or bev"
    )
    parser.add_argument(
        "--ablation", type=str, choices=["A", "B", "C"], default="C", help="Ablation: A=baseline, B=BEV+original, C=BEV+bev-aware"
    )
    parser.add_argument("--disable-smoothing", action="store_true", help="Disable curvature smoothing")
    parser.add_argument("--max-scenes", type=int, default=None, help="Maximum number of scenes to process")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 video generation")
    parser.add_argument("--eval-pre-safety-openloop", action="store_true",
                        help="Evaluate ADE/FDE on pre-safety open-loop predictions (skip floor/boost for metrics)")
    parser.add_argument("--use-cot-hodc", action="store_true",
                        help="Use Mode C CoT+HODC architecture (5-stage: CoT → HODC → Trajectory)")
    args = parser.parse_args()

    if "gpt" not in args.model_path.lower():
        print(f"[Warn] Only GPT is supported in this trimmed script. Overriding --model-path={args.model_path} -> 'gpt'")
        args.model_path = "gpt"

    # Research Logic of Dissolution
    if args.ablation == "A":
        args.use_bev = False
        args.prompt_mode = "original"
    elif args.ablation == "B":
        args.use_bev = True
        args.prompt_mode = "original"
    elif args.ablation == "C":
        args.use_bev = True
        args.prompt_mode = "bev"

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    out_dir = f"E:/OpenEMMA_Results/{args.model_path}_results/{args.method}/{ts}"
    os.makedirs(out_dir, exist_ok=True)

    # Load the dataset
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    
        
        # 优先使用100场景采样（如果存在）
        sampling_file = None
        if os.path.exists('sampled_scenes.json'):
            sampling_file = 'sampled_scenes.json'
    
        elif os.path.exists('sampled_scenes_1.json'):
            sampling_file = 'sampled_scenes_1.json'
            
        if sampling_file:
            scenes = load_sampled_scenes(nusc, sampling_file)
        else:
            scenes = smart_sample_scenes(nusc, n_samples=1000, seed=42)
        
    else:
        scenes = nusc.scene

    # 加载BEV地图（如果使用BEV）
    nusc_map = None
    if args.use_bev:
        try:
            log_rec = nusc.get("log", nusc.scene[0]["log_token"])
            location = log_rec["location"]
            nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=location)
            print(f"BEV map loaded for location: {location}")
        except Exception as e:
            exit(1)

    scenes = nusc.scene
    print(f"Number of scenes: {len(scenes)}")

    all_scenes_ade1s = []
    all_scenes_ade2s = []
    all_scenes_ade3s = []
    all_scenes_fde = []
    all_scenes_collision_rate = []
    all_scenes_traffic_rule_rate = []
    
    all_scenes_api_calls = []
    all_scenes_inference_time = []
    all_scenes_frames = []
    
    scene_details = []  # List of {name, ade1s, ade2s, ade3s, fde, collision, traffic_rule, frames}
    
    hodc_stats = {
        'total_frames': 0,
        'fallback_count': 0,  # VLM违反约束，使用template的次数
        'maneuver_distribution': {},  # {straight: 10, left_turn: 5, ...}
        'avg_curvature_deviation': []  # |k_actual - k_target|的列表
    }

    for scene_idx, scene in enumerate(scenes):
        if args.max_scenes and scene_idx >= args.max_scenes:
            print(f"Reached max scenes limit ({args.max_scenes}), stopping...")
            break

        token = scene["token"]
        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]
        name = scene["name"]
        description = scene["description"]
             
        front_camera_images = []  
        ego_poses = []
        camera_params = []
        sample_tokens = []  

        curr = first_sample_token
        while True:
            sample = nusc.get("sample", curr)
            cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])

            img_path = os.path.join(nusc.dataroot, cam_front_data["filename"])
            if not os.path.exists(img_path):
                front_camera_images = []
                ego_poses = []
                camera_params = []
                sample_tokens = []
                break
            
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            front_camera_images.append(img_bytes_to_jpeg_b64(img_bytes, target_side=448, quality=80))

            pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            ego_poses.append(pose)
            camera_params.append(nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"]))
            sample_tokens.append(curr)

            if curr == last_sample_token:
                break
            curr = sample["next"]

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length == 0:
            print(f"Scene {name} was skipped due to missing image files")
            continue

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        ego_poses_world = np.array([ego_poses[t]["translation"][:3] for t in range(scene_length)])
        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], "r-", label="GT")

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_vel_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(
            ego_curvatures, ego_vel_norm, ego_poses_world[0], atan2(ego_velocities[0][1], ego_velocities[0][0]), scene_length
        )

        if args.plot:
            plt.quiver(
                ego_poses_world[:, 0],
                ego_poses_world[:, 1],
                ego_velocities[:, 0],
                ego_velocities[:, 1],
                color="b",
            )
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], "g-", label="Reconstruction")
            plt.legend()
            plt.savefig(f"{out_dir}/{name}_interpolation.jpg")
            plt.close()

        ego_traj_world = [ego_poses[t]["translation"][:3] for t in range(scene_length)]

        prev_intent = None
        cam_images_sequence = []
        ade1s_list, ade2s_list, ade3s_list = [], [], []
        fde_list = []

        collision_triggered_count = 0
        traffic_rule_clamped_count = 0
        total_inference_time = 0.0
        total_api_calls = 0
        collision_detected_frames = 0  

        K = 5
        last_pred = None
        last_hodc_constraints = None
        last_hodc_json = None
        prev_scene = prev_objects = prev_intent_text = None
        
        prev_collision_unsafe = False
        
        planner_state = {}
        planner_state["last_k"] = None
        planner_state["scaled_this_frame"] = False

        for i in range(scene_length - TTL_LEN):
            # ---------------- per-frame init ----------------
            # Avoid carrying over evaluation trajectories/sources from the previous frame across frames.
            eval_traj = None
            eval_traj_source = None
            planner_state["scaled_this_frame"] = False
            # Ensure variables are defined under the B/non-HODC path to avoid NameError.
            hodc_constraints = None
            # Initialize trajectory variables to ensure all modes have access.
            pred_traj = None
            speed_curvature_pred = []
            # ------------------------------------------------

            obs_images = front_camera_images[i : i + OBS_LEN]
            obs_ego_poses = ego_poses[i : i + OBS_LEN]
            obs_camera_params = camera_params[i : i + OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i : i + OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i + OBS_LEN : i + TTL_LEN]
            obs_ego_velocities = ego_velocities[i : i + OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i : i + OBS_LEN]

            fut_start_world = obs_ego_traj_world[-1]

            # A分支优化：强制2关键帧减少图像
            if not args.use_bev:  # Ablation A
                obs_images = select_frames_for_vlm(obs_images, keep=2)
                print(f"[A-BRANCH] Frame {i}: Reduced to {len(obs_images)} keyframes")

            # 计算当前帧 yaw_deg（用于 BEV 画线）
            ego_pose_last = obs_ego_poses[-1]
            q_last = Quaternion(ego_pose_last["rotation"])
            yaw_rad_last, _, _ = q_last.yaw_pitch_roll
            yaw_deg_last = np.degrees(yaw_rad_last)

            # BEV处理
            if args.use_bev:
                bev_images = []
                for t in range(OBS_LEN):
                    ego_pose = obs_ego_poses[t]
                    center_xy = (ego_pose["translation"][0], ego_pose["translation"][1])
                    q = Quaternion(ego_pose["rotation"])
                    yaw_rad, _, _ = q.yaw_pitch_roll
                    yaw_deg = np.degrees(yaw_rad)

                    try:
                        curr_sample_token = sample_tokens[i + t]
                        sample = nusc.get("sample", curr_sample_token)
                        dynamic_objects = prepare_front_view_boxes(nusc, sample, ego_pose)
                    except Exception as e:
                        print(f"Warning: Failed to prepare dynamic objects for frame {t}: {e}")
                        dynamic_objects = []

                    bev = render_simple_bev(
                        nusc_map=nusc_map,
                        center_xy=center_xy,
                        yaw_deg=yaw_deg,
                        extent_m=args.bev_extent,
                        resolution_m=args.bev_res,
                        trajectory_points=None,
                        dynamic_objects=dynamic_objects,
                    )

                    bev_resized = cv2.resize(bev, (448, 448), interpolation=cv2.INTER_NEAREST)
                    ok, buf = cv2.imencode(".jpg", bev_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        bev_images.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
                    else:
                        print(f"Warning: Failed to encode BEV image for frame {t}")
                        bev_images.append(front_camera_images[i + t])

                # === BEV+前视混合输入 ===
                key_bev = [bev_images[0], bev_images[len(bev_images) // 2], bev_images[-1]]
                front_key = front_camera_images[i + OBS_LEN - 1]
                obs_images = [front_key, key_bev[1], key_bev[-1]]
                print(f"[DEBUG] Using hybrid input: {len(obs_images)} images (1 front + 2 BEV) for VLM")

            curr_image_b64 = obs_images[-1]
            try:
                img = cv2.imdecode(
                    np.frombuffer(base64.b64decode(curr_image_b64), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if not args.use_bev and obs_camera_params:
                    img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]
            except Exception as e:
                print(f"Warning: Failed to load front camera image for visualization: {e}")
                img = np.zeros((640, 640, 3), dtype=np.uint8)

            curr_token = sample_tokens[i + OBS_LEN - 1]
            prev_token = sample_tokens[i + OBS_LEN - 2] if i + OBS_LEN - 2 >= 0 else None
            agents_states = collect_agents_nus_with_vel(nusc, prev_token, curr_token, dt=0.5)
            
            ego_pos_xy = np.array(obs_ego_traj_world[-1][:2], np.float32)
            ego_heading = np.arctan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0])
            agents_states = filter_agents(agents_states, ego_pos_xy, ego_heading, max_r=25.0, fov_deg=90, topk=12)
            print(f"[planner] agents={len(agents_states)} (front ±45°, 25m, top-12)")
            
            # C-MODE: 统一体集合
            if isinstance(locals().get('pred_traj', None), np.ndarray) and pred_traj.size > 0:
                critical_agents = filter_critical_agents(agents_states, pred_traj[:, :2])
            else:
                critical_agents = []
            agents_for_check = critical_agents

            # 规划：BEV or Front-view
            if args.use_bev:
                print("[planner] use BEV: True (E2E mode)")

                # C模式：支持两种架构
                # 1. Mode C v2: 简化的两阶段（默认）
                # 2. Mode C CoT+HODC: 完整的5阶段（--use-cot-hodc）
                if args.method in ("openemma", "chat") and getattr(args, "prompt_mode", "original") == "bev":
                    use_cot_hodc = getattr(args, "use_cot_hodc", False)
                    
                    if use_cot_hodc:
                        print("[C-MODE CoT+HODC] Using 5-stage architecture (CoT → HODC → Trajectory)")
                    else:
                        print("[C-MODE v2] Using simplified 2-stage architecture")
                    
                    has_last = (last_pred is not None)
                    use_cache = (i % max(K, 2) != 0) and has_last and (not prev_collision_unsafe)
                    
                    if use_cache:
                        result = last_pred
                        scene_desc = "Cached from previous frame"
                        object_desc = "Cached from previous frame"
                        updated_intent = "Cached from previous frame"
                        print(f"[CACHE] Frame {i}: Using cached prediction")
                    else:
                        # 准备输入
                        v_hist = np.linalg.norm(obs_ego_velocities[-3:], axis=1).tolist()
                        k_hist = obs_ego_curvatures[-3:].tolist()
                        history_vk = [[float(v), float(k)] for v, k in zip(v_hist, k_hist)]
                        
                        ego_state = {
                            'speed': float(np.linalg.norm(obs_ego_velocities[-1])),
                            'curvature': float(obs_ego_curvatures[-1])
                        }
                        
                        scene_info = {
                            'tl_state': 'none',  # TODO: 从环境获取
                            'speed_limit_mps': 10.0,  # TODO: 从地图获取
                            'stopline_distance_m': None
                        }
                        
                        if use_cot_hodc:
                            # === Mode C CoT+HODC架构 ===
                            # Stage 1-3: Scene Understanding (复用Mode B的CoT)
                            key_frames = select_frames_for_vlm(obs_images, keep=3)
                            
                            # Stage 1: SceneDescription
                            scene_description = SceneDescription(key_frames, args=args)
                            
                            # Stage 2: DescribeObjects
                            if len(obs_images) >= 3:
                                front_key = obs_images[-1]
                                bev_keys = obs_images[:-1]
                                object_desc_frames = [front_key, bev_keys[0], bev_keys[-1]]
                            else:
                                object_desc_frames = key_frames
                            object_description = DescribeObjects(object_desc_frames, args=args)
                            
                            # Stage 3: DescribeOrUpdateIntent
                            intent_description = DescribeOrUpdateIntent(key_frames, prev_intent=prev_intent, args=args)
                            prev_intent = intent_description
                            
                            # 设置变量以供后续日志保存
                            scene_desc = scene_description
                            object_desc = object_description
                            updated_intent = intent_description
                            
                            # Stage 4-5: HODC + Trajectory (在mode_c_cot.py中)
                            result = generate_mode_c_cot_prediction(
                                vlm_inference_fn=vlm_inference,
                                images=obs_images[-3:],
                                history_vk=history_vk,
                                ego_state=ego_state,
                                scene_info=scene_info,
                                scene_description=scene_description,
                                object_description=object_description,
                                intent_description=intent_description,
                                args=args
                            )
                            
                            total_api_calls += 5  # 5 stages
                            print(f"[VLM] Frame {i}: Generated prediction (CoT+HODC 5-stage)")
                            
                        else:
                            # === Mode C v2简化架构 ===
                            result = generate_mode_c_prediction(
                                vlm_inference_fn=vlm_inference,
                                images=obs_images[-3:],  # [front_last, bev_mid, bev_last]
                                history_vk=history_vk,
                                ego_state=ego_state,
                                scene_info=scene_info,
                                args=args
                            )
                            
                            # v2模式没有显式CoT描述
                            scene_desc = "N/A (v2 mode)"
                            object_desc = "N/A (v2 mode)"
                            updated_intent = "N/A (v2 mode)"
                            
                            total_api_calls += 2  # 2 stages
                            print(f"[VLM] Frame {i}: Generated prediction (v2 2-stage)")
                        
                        last_pred = result
                        total_inference_time += (_last_api_duration if '_last_api_duration' in globals() else 0.0)
                    
                    # 提取[v,k] pairs
                    speed_curvature_pred = result['pairs']
                    constraints = result.get('constraints', {})
                    reasoning = result.get('reasoning', '')
                    semantic_cmd = result.get('semantic_command', {})
                    
                    # 根据实际执行的模式显示正确的日志
                    if use_cache:
                        print(f"[CACHE] Semantic: {semantic_cmd.get('maneuver')} ({semantic_cmd.get('curvature_hint')}) → {len(speed_curvature_pred)} pairs")
                    elif use_cot_hodc:
                        print(f"[C-MODE CoT+HODC] Semantic: {semantic_cmd.get('maneuver')} ({semantic_cmd.get('curvature_hint')}) → {len(speed_curvature_pred)} pairs")
                    else:
                        print(f"[C-MODE v2] Semantic: {semantic_cmd.get('maneuver')} ({semantic_cmd.get('curvature_hint')}) → {len(speed_curvature_pred)} pairs")
                    
                    # 应用软约束
                    tl_state = scene_info.get('tl_state', 'none')
                    speed_curvature_pred = apply_soft_constraints(speed_curvature_pred, constraints, tl_state)
                    
                    # 标记为使用了HODC（用于后续逻辑）
                    hodc_constraints = None  # v2不需要复杂的hodc_constraints
                    
                    # 跳过旧的解析逻辑
                    if False:
                        # === 旧的C模式解析逻辑（已禁用） ===
                        speed_curvature_pred = []
                        if isinstance(prediction, list):
                            speed_curvature_pred = [[float(v), float(k) / 100.0] for v, k in prediction][:10]
                        else:
                            # 尝试JSON解析
                            try:
                                obj = _safe_json_loads(str(prediction))
                                pairs = None
                                if isinstance(obj, dict):
                                    # 尝试多种键名
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
                                            speed_curvature_pred.append([v, k])
                                print(f"[C-MODE] Parsed {len(speed_curvature_pred)} pairs from VLM output")
                            except Exception as e:
                                print(f"[C-MODE] JSON parse error: {e}")
                            
                            # 回退到正则
                            if len(speed_curvature_pred) == 0:
                                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", str(prediction))
                                if coordinates:
                                    speed_curvature_pred = [[float(v), float(k) / 100.0] for v, k in coordinates][:10]
                                    print(f"[C-MODE] Regex fallback: parsed {len(speed_curvature_pred)} pairs")
                        
                        # === 兜底：长度不足10 或 曲率全为0 ===
                        needs_fallback = (len(speed_curvature_pred) < 10) or (len(speed_curvature_pred) > 0 and all(abs(k) < 1e-6 for _, k in speed_curvature_pred))
                        if needs_fallback:
                            print(f"[C-MODE] Triggering fallback: len={len(speed_curvature_pred)}, all_zero_k={all(abs(k) < 1e-6 for _, k in speed_curvature_pred) if len(speed_curvature_pred) > 0 else False}")
                            # 清空并重新生成
                            speed_curvature_pred = []
                            # 用HODC bounds生成保守轨迹
                            v_bounds = hodc_constraints.get("v_bounds", [[0, 0, 8.0]] * 10)
                            k_bounds = hodc_constraints.get("k_bounds", [[0, -0.05, 0.05]] * 10)
                            # 曲率钟形模板（中段稍高，前后收敛）
                            bell = [0.0, 0.3, 0.6, 0.9, 1.0, 0.9, 0.6, 0.3, 0.1, 0.0]
                            for idx in range(10):
                                vb = v_bounds[idx] if idx < len(v_bounds) else [0, 0, 8.0]
                                kb = k_bounds[idx] if idx < len(k_bounds) else [0, -0.05, 0.05]
                                # 鲁棒解包：支持 [t,vmin,vmax] 或 [vmin,vmax]
                                if len(vb) == 3:
                                    vmin, vmax = vb[1], vb[2]
                                elif len(vb) == 2:
                                    vmin, vmax = vb[0], vb[1]
                                else:
                                    vmin, vmax = 0, 8.0
                                
                                if len(kb) == 3:
                                    kmin, kmax = kb[1], kb[2]
                                elif len(kb) == 2:
                                    kmin, kmax = kb[0], kb[1]
                                else:
                                    kmin, kmax = -0.05, 0.05
                                
                                v = min(max(2.0, 0.5*(vmin+vmax)), 8.0)  # 最小2.0 m/s
                                # 取几何中值再乘bell
                                k_mid = 0.5*(kmin + kmax)
                                k = np.clip(k_mid * bell[idx], kmin, kmax)
                                speed_curvature_pred.append([float(v), float(k)])
                            k_vals = [k for _, k in speed_curvature_pred]
                            print(f"[C-MODE] Applied fallback trajectory: v_range=[{min(v for v,_ in speed_curvature_pred):.1f}, {max(v for v,_ in speed_curvature_pred):.1f}], k_range=[{min(k_vals):.4f}, {max(k_vals):.4f}]")
                        
                        if len(speed_curvature_pred) == 0:
                            print("[C-MODE] Failed to parse any v,k pairs - skipping frame")
                            continue
                    # else块已删除（旧的prediction变量不再使用）
                    
                    # v2: speed_curvature_pred已经在前面设置好了，直接使用
                    print(f"[DEBUG] Parsed v,k: {speed_curvature_pred[:3]}")
                    # C模式使用'c_semantic'模式，减少对VLM生成曲率的削弱
                    speed_curvature_pred = sanitize_curvature_scaled_seq(speed_curvature_pred, mode='c_semantic')
                    speed_curvature_pred = _inject_turn_template_if_flat(speed_curvature_pred, hodc_constraints)
                    
                    print(f"[DEBUG] k values: {[k for _,k in speed_curvature_pred[:3]]}")
                    k_values = [k for _,k in speed_curvature_pred]
                    k_std = np.std(k_values) if len(k_values) > 0 else 0.0
                    if k_std < 1e-3:
                        print("[WARNING] k_std < 1e-3, curvature too flat - may need geometry bounds")
                    print(f"[SANITY] k_head mean={float(np.mean(k_values[:6])):.4f}, std={k_std:.4f}")
                    
                    # ---- 立刻把 v/k 转为世界坐标系下的预测轨迹 ----
                    pred_len = min(FUT_LEN, len(speed_curvature_pred))
                    pred_speeds = np.array([p[0] for p in speed_curvature_pred[:pred_len]], dtype=np.float32)
                    pred_curvatures_real = np.array([p[1] for p in speed_curvature_pred[:pred_len]], dtype=np.float32)

                    pred_traj = np.zeros((pred_len, 3), dtype=np.float32)
                    pred_traj[:, :2] = IntegrateCurvatureForPoints(
                        pred_curvatures_real,
                        pred_speeds,
                        fut_start_world,
                        atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                        pred_len,
                    )

                    # 现在再做关键体过滤就安全了
                    critical_agents = filter_critical_agents(agents_states, pred_traj[:, :2])
                    agents_for_check = critical_agents
                    
                    # 如果要做open-loop评测，在做任何安全干预之前把eval_traj拿出来
                    if getattr(args, "eval_pre_safety_openloop", False):
                        eval_traj = pred_traj[:, :2].copy()
                        eval_traj_source = "pre_safety_open_loop"
                    
                    # === C-MODE v2: 跳过复杂的HODC融合（已在v2中处理） ===
                    if False and hodc_constraints:  # 旧逻辑已禁用
                        hodc_history = planner_state.get("hodc_history", [])
                        hodc_constraints, scene_consistency = hodc_consistency_filter(hodc_constraints, hodc_history)
                        planner_state["hodc_history"] = hodc_history[-4:] + [hodc_constraints]
                        
                        v_b_ref = [min(8.0, v_b[1] * 0.8) for v_b in hodc_constraints.get("v_bounds", [[0, 12]] * 10)]
                        k_b_ref = [0.0] * len(speed_curvature_pred)
                        
                        v_hodc = [v for v, _ in speed_curvature_pred]
                        k_hodc = [k for _, k in speed_curvature_pred]
                        signals = get_hodc_signals(hodc_constraints)
                        
                        v_fused, k_fused, confidence = hodc_blend_adapter(
                            v_hodc, k_hodc, v_b_ref, k_b_ref, signals, scene_consistency
                        )
                        
                        # 使用稳健曲率融合（最小带宽/反零化/几何回退），并将k×100转回真实曲率
                        for i_sc in range(len(speed_curvature_pred)):
                            if i_sc < len(v_fused) and i_sc < len(k_fused):
                                try:
                                    lane_topo = (hodc_constraints.get("scene", {}) or {}).get("lane_topology")
                                    v_curr = float(v_fused[i_sc])
                                    k_mean_raw = float(k_fused[i_sc] / 100.0)
                                    # 用当前窗口估计一个k_std
                                    k_head_vals = [kv for _, kv in speed_curvature_pred][:max(3, i_sc+1)]
                                    k_std_est = float(np.std(k_head_vals)) if len(k_head_vals) > 0 else 0.0
                                    k_ema = float(speed_curvature_pred[i_sc][1])
                                    kb = None
                                    if i_sc < len(hodc_constraints.get("k_bounds", [])):
                                        lohi = hodc_constraints["k_bounds"][i_sc]
                                        kb = (float(lohi[0]) / 100.0, float(lohi[1]) / 100.0)
                                    k_use = fuse_k(k_mean_raw, k_std_est, k_ema, kb, confidence, lane_topo, signals.get("speed_limit_mps"))
                                    speed_curvature_pred[i_sc] = [v_curr, float(np.clip(k_use, -0.08, 0.08))]
                                except Exception:
                                    speed_curvature_pred[i_sc] = [float(v_fused[i_sc]), float(k_fused[i_sc] / 100.0)]
                        
                        print(f"[C-MODE] Applied HODC fusion with confidence={confidence:.3f}")
                        speed_curvature_pred = smooth_speed_only(speed_curvature_pred, max_delta_v=0.7)
                        
                        if planner_state.get("last_k") is not None:
                            lk = planner_state["last_k"]
                            k_values = [k for _, k in speed_curvature_pred]
                            n = min(6, len(k_values), len(lk))
                            if n > 0:
                                for j in range(n):
                                    k_values[j] = 0.7 * k_values[j] + 0.3 * lk[j]
                                for j, (v, k) in enumerate(speed_curvature_pred):
                                    if j < len(k_values):
                                        speed_curvature_pred[j] = [v, k_values[j]]
                                print("[C-MODE] Applied EMA trend keep on curvature (head)")
                        planner_state["last_k"] = [k for _, k in speed_curvature_pred]
                        
                        v_values = [v for v, _ in speed_curvature_pred]
                        v_bounds = hodc_constraints.get("v_bounds", [])
                        collision_risk = len(agents_for_check) > 0 and any(np.linalg.norm(a["pos_xy"] - np.array(obs_ego_traj_world[-1][:2])) < 2.0 for a in agents_for_check)
                        if not getattr(args, "eval_pre_safety_openloop", False):
                            v_values = apply_min_progress_floor(v_values, v_bounds, floor=1.6, collision_risk=collision_risk)
                            for j, v in enumerate(v_values):
                                speed_curvature_pred[j][0] = v
                            print("[C-MODE] Applied min progress floor")
                        else:
                            print("[C-MODE][EVAL] Open-loop: skip min progress floor")
                        
                        v_values = [v for v, _ in speed_curvature_pred]
                        k_values = [k for _, k in speed_curvature_pred]
                        signals = get_hodc_signals(hodc_constraints)
                        min_d_to_agents = min(np.linalg.norm(a["pos_xy"] - np.array(obs_ego_traj_world[-1][:2])) for a in agents_for_check) if agents_for_check else None
                        last_S = planner_state.get("last_S")  # ✅ 修复：dict.get
                        if not getattr(args, "eval_pre_safety_openloop", False):
                            v_values = apply_length_boost(v_values, k_values, v_bounds, signals, min_d_to_agents, last_S)
                            for j, v in enumerate(v_values):
                                speed_curvature_pred[j][0] = v
                            planner_state["last_S"] = float(np.sum(v_values) * 0.5)
                            print("[C-MODE] Applied length boost")
                        else:
                            print("[C-MODE][EVAL] Open-loop: skip length boost")
                        
                        sig = get_hodc_signals(hodc_constraints)
                        limit = sig.get("speed_limit_mps", None)
                        for j, (v, k) in enumerate(speed_curvature_pred):
                            if limit is not None:
                                v = min(v, float(limit))
                            k = float(np.clip(k, -0.08, 0.08))
                            speed_curvature_pred[j] = [v, k]
                        print("[C-MODE] Applied final constraint clipping")
                        
                        tl = (sig or {}).get("tl_state", "none")
                        stopline = (sig or {}).get("stopline_distance_m", None)
                        if (tl in ("green", "none") and stopline is None) and (not getattr(args, "eval_pre_safety_openloop", False)):
                            head = min(6, len(speed_curvature_pred))
                            for t in range(head):
                                if speed_curvature_pred[t][0] < 1.6:
                                    speed_curvature_pred[t][0] = 1.6
                            print("[C-MODE] Applied min progress floor to head steps")
                        elif getattr(args, "eval_pre_safety_openloop", False):
                            print("[C-MODE][EVAL] Open-loop: skip head-step floor")
                        
                        try:
                            k_values = [k for _, k in speed_curvature_pred]
                            k_head = k_values[:6] if len(k_values) >= 6 else k_values
                            print(f"[SANITY] k_head mean={float(np.mean(k_head)):.4f}, std={float(np.std(k_head)):.4f}")
                            v_values = [v for v, _ in speed_curvature_pred]
                            v_head = v_values[:6] if len(v_values) >= 6 else v_values
                            print(f"[SANITY] v_head mean={float(np.mean(v_head)):.2f}")
                            print(f"[SANITY] critical={len(agents_for_check)} used for all collision checks")
                        except Exception as e:
                            print(f"[SANITY] print failed: {e}")
                        
                        for j, (v, k) in enumerate(speed_curvature_pred):
                            if limit is not None:
                                v = min(v, float(limit))
                            k = float(np.clip(k, -0.08, 0.08))
                            speed_curvature_pred[j] = [v, k]
                        
                        if all([(c.get("time_window_s", [1, 0])[1] < 0.5) for c in hodc_constraints.get("conflicts", [])]) \
                           and (hodc_constraints.get("hodc", {}).get("signals", {}).get("tl_state", "none") in ["green", "none"]):
                            avg_v = np.mean([v for v, _ in speed_curvature_pred[:6]])
                            if avg_v < 1.5:
                                print(f"[WARNING] Low speed under clear-green: avg_v={avg_v:.2f} m/s (expected >=1.5)")
                    else:
                        for p in speed_curvature_pred:
                            p[1] = float(np.clip(p[1], -0.055, 0.055))
                
                else:
                    # B模式：原始三件套
                    print("[B-MODE] Using original three-stage prompting")
                    
                    if i % K == 0:
                        key_frames_for_desc = select_frames_for_vlm(obs_images, keep=3)
                        scene_desc = SceneDescription(key_frames_for_desc, args=args)
                        
                        if len(obs_images) >= 3:
                            front_key = obs_images[-1]
                            bev_keys = obs_images[:-1]
                            object_desc = DescribeObjects([front_key, bev_keys[0], bev_keys[-1]], args=args)
                        else:
                            object_desc = DescribeObjects(key_frames_for_desc, args=args)
                        
                        intent_desc = DescribeOrUpdateIntent(key_frames_for_desc, prev_intent=prev_intent, args=args)
                        
                        total_api_calls += 3  # SceneDescription + DescribeObjects + DescribeOrUpdateIntent

                        events = detect_novelty(scene_desc, object_desc, intent_desc, prev_scene, prev_objects, prev_intent_text)
                        if should_update_summary(events, i, K=5):
                            new_json = compress_summary(short_json, scene_desc, object_desc, intent_desc)
                            h = hash(json.dumps(new_json, sort_keys=True))
                            if h != last_json_hash:
                                short_json = new_json
                                last_json_hash = h
                                print(f"[COMPRESS] Frame {i}: Updated summary")
                        else:
                            print(f"[COMPRESS] Frame {i}: No update needed")
                        prev_scene = scene_desc
                        prev_objects = object_desc
                        prev_intent_text = intent_desc

                    if i % K != 0 and last_pred is not None:
                        prediction = last_pred
                        scene_desc = "Cached from previous frame"
                        object_desc = "Cached from previous frame"
                        updated_intent = "Cached from previous frame"
                        print(f"[CACHE] Frame {i}: Using cached prediction")
                    else:
                        prediction, scene_desc, object_desc, updated_intent = GenerateMotion(
                            obs_images, obs_ego_traj_world, obs_ego_velocities, obs_ego_curvatures, prev_intent, args=args
                        )
                        last_pred = prediction
                        total_api_calls += 1  # GenerateMotion
                        print(f"[VLM] Frame {i}: Generated new prediction")

                    pred_waypoints = str(prediction).replace("Future speeds and curvatures:", "").strip()
                    coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
                    if not coordinates:
                        continue

                    speed_curvature_pred = [[float(v), float(k) / 100.0] for v, k in coordinates][:10]
                    for p in speed_curvature_pred:
                        p[1] = float(np.clip(p[1], -0.055, 0.055))

                    if not args.disable_smoothing:
                        if hodc_constraints:
                            print("[C-MODE] Skipping aggressive smoothing to preserve HODC constraints")
                        else:
                            speed_curvature_pred = smooth_curvature_prediction(speed_curvature_pred, obs_ego_curvatures)

                    last_speed = np.linalg.norm(obs_ego_velocities[-1]) if len(obs_ego_velocities) > 0 else None
                    
                    if not hodc_constraints or not hodc_constraints.get("hodc", {}).get("signals"):
                        speed_curvature_pred, was_clamped = clamp_by_traffic_rules(
                            speed_curvature_pred, tl_state=None, dist_to_stopline_m=None, last_speed=last_speed
                        )
                        if was_clamped:
                            traffic_rule_clamped_count += 1
                    else:
                        print("[C-MODE] Applying light traffic rule check even with HODC constraints")
                        tl_state = hodc_constraints.get("hodc", {}).get("signals", {}).get("tl_state", "none")
                        if tl_state == "red":
                            for j, (v, k) in enumerate(speed_curvature_pred):
                                if v > 1.0:
                                    speed_curvature_pred[j] = [1.0, k]
                            print("[C-MODE] Applied red light constraint")
                            was_clamped = True
                            traffic_rule_clamped_count += 1
                        else:
                            was_clamped = False

                    print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")
                    
                    scene_desc = locals().get("scene_desc", "N/A")
                    object_desc = locals().get("object_desc", "N/A")
                    updated_intent = locals().get("updated_intent", "N/A")
                    prev_intent = updated_intent

                    pred_len = min(FUT_LEN, len(speed_curvature_pred))
                    pred_curvatures_real = np.array(speed_curvature_pred)[:, 1]
                    pred_speeds = np.array(speed_curvature_pred)[:, 0]
                    
                    if pred_len == 0 or np.allclose(pred_speeds, 0.0):
                        v0 = float(np.linalg.norm(obs_ego_velocities[-1])) * 0.5
                        pred_speeds = np.full((FUT_LEN,), v0, dtype=np.float32)
                        pred_curvatures_real = np.zeros((FUT_LEN,), dtype=np.float32)
                        pred_len = FUT_LEN
                        print(f"[FALLBACK] Using fallback trajectory: v0={v0:.2f} m/s")
                    
                    pred_traj = np.zeros((pred_len, 3))
                    pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(
                        pred_curvatures_real,
                        pred_speeds,
                        fut_start_world,
                        atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                        pred_len,
                    )
                    # ==== Open-loop 评测（在安全干预前锁定评测轨迹） ====
                    if getattr(args, "eval_pre_safety_openloop", False) and eval_traj is None:
                        eval_traj = pred_traj[:, :2].copy()
                        eval_traj_source = "pre_safety_open_loop"
                        
                        # 在安全干预前检测碰撞（用于统计）
                        # 根据当前曲率确定安全半径
                        k_hist_abs_eval = abs(float(obs_ego_curvatures[-1]))
                        if k_hist_abs_eval < 0.01:
                            eval_base_radius, eval_speed_gain = 1.1, 0.08
                        else:
                            eval_base_radius, eval_speed_gain = 1.2, 0.10
                        
                        pre_safety_collision = dynamic_collision_flag(
                            pred_traj[:, :2], agents_for_check, step=0.5, 
                            base_safety_radius=eval_base_radius, speed_gain=eval_speed_gain
                        )
                        if pre_safety_collision:
                            collision_triggered_count += 1
                            print("[EVAL-OPENLOOP] Collision detected in pre-safety trajectory")
                        
                        try:
                            gt_future = np.array(fut_ego_traj_world)[:pred_len, :2]
                            S_pred = float(np.sum(pred_speeds[:pred_len]) * 0.5)
                            S_gt = float(np.sum(np.linalg.norm(np.diff(gt_future, axis=0), axis=1))) if gt_future.shape[0] >= 2 else 0.0
                            k_peak_pred = float(np.max(np.abs(pred_curvatures_real[:min(6, pred_len)])))
                            k_peak_gt = float(np.max(np.abs(ego_curvatures[i+OBS_LEN : i+OBS_LEN+pred_len][:min(6, pred_len)])))
                            print(f"[ALIGN] S_pred={S_pred:.2f}m S_gt={S_gt:.2f}m k_peak_pred={k_peak_pred:.3f} k_peak_gt={k_peak_gt:.3f}")
                        except Exception as e:
                            print(f"[ALIGN] log failed: {e}")

                    # 关键agent过滤：只对与ego轨迹有交叉风险的agent进行避让
                    critical_agents = filter_critical_agents(agents_states, pred_traj[:, :2])
                    print(f"[planner] Critical agents: {len(critical_agents)}/{len(agents_states)} (trajectory intersection risk)")

                    agents_for_check = critical_agents if len(critical_agents) > 0 else []

                    # 统一默认值，避免未定义引用
                    collision_detected = False
                    brake_needed = False
                    consecutive_unsafe = planner_state.get("consecutive_unsafe_frames", 0)
                    brake_reason = "stable"

                    def min_distance_to_agents(pred_traj_xy, agents_states):
                        if pred_traj_xy is None or not isinstance(pred_traj_xy, np.ndarray) or pred_traj_xy.size == 0:
                            return float("inf")
                        if not agents_states:
                            return float("inf")
                        t = np.arange(pred_traj_xy.shape[0], dtype=np.float32) * 0.5
                        mind = float("inf")
                        for a in agents_states:
                            a_traj = a["pos_xy"][None, :] + t[:, None] * a["vel_xy"][None, :]
                            d = np.linalg.norm(pred_traj_xy - a_traj, axis=1).min()
                            if d < mind:
                                mind = float(d)
                        return mind

                    k_hist_abs = abs(float(obs_ego_curvatures[-1]))
                    if k_hist_abs < 0.01:
                        base_radius, speed_gain = 1.1, 0.08
                        print(f"[planner] Using relaxed safety radius for straight driving: base={base_radius}, gain={speed_gain}")
                    else:
                        base_radius, speed_gain = 1.2, 0.10
                        print(f"[planner] Using conservative safety radius for curved driving: base={base_radius}, gain={speed_gain}")
                        
                        prev_brake_state = planner_state.get("prev_brake_state", False)
                        consecutive_unsafe = planner_state.get("consecutive_unsafe_frames", 0)
                        
                        brake_needed, consecutive_unsafe, brake_reason = hysteresis_braking_controller(
                            pred_traj, agents_for_check, base_radius, speed_gain, prev_brake_state, consecutive_unsafe
                        )
                        
                        planner_state["prev_brake_state"] = brake_needed
                        planner_state["consecutive_unsafe_frames"] = consecutive_unsafe
                        
                        if brake_needed and brake_reason in ["enter_danger", "in_danger"]:
                            k_values = [k for _, k in speed_curvature_pred]
                            k_filtered = apply_curvature_frequency_reduction(k_values, reduction_factor=0.7)
                            for j, k in enumerate(k_filtered):
                                speed_curvature_pred[j][1] = k
                            print(f"[HYSTERESIS] Applied curvature frequency reduction: {brake_reason}")
                        
                        collision_detected = dynamic_collision_flag(
                            pred_traj[:, :2], agents_for_check, step=0.5, base_safety_radius=base_radius, speed_gain=speed_gain
                        )
                    
                    if collision_detected or (brake_needed and consecutive_unsafe >= 2):
                        print("[planner] Collision detected, applying gradual speed reduction...")
                        
                        min_d = min_distance_to_agents(pred_traj[:, :2] if (isinstance(pred_traj, np.ndarray) and pred_traj.size) else None,
                                                       agents_for_check)
                        print(f"[planner] Min distance to agents: {min_d:.2f}m")
                    
                    if min_d < 2.5:
                        n = min(5, len(pred_speeds))
                        for j in range(len(pred_speeds)-n, len(pred_speeds)):
                            pred_speeds[j] *= 0.90
                        print(f"[planner] Applied soft brake (shape-preserving), steps={n}, scale=0.90")
                        
                        pred_traj[:, :2] = IntegrateCurvatureForPoints(
                            pred_curvatures_real,
                            pred_speeds,
                            fut_start_world,
                            atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                            pred_len,
                        )
                        
                        collision_still_detected = dynamic_collision_flag(
                            pred_traj[:, :2], agents_for_check, step=0.5, base_safety_radius=base_radius, speed_gain=speed_gain
                        )
                        
                        if collision_still_detected:
                            if min_d < 1.5 and collision_detected_frames >= 2:
                                t = np.arange(pred_len, dtype=np.float32) * 0.5
                                closest_idx = 0
                                closest_d = 1e9
                                for a in critical_agents:
                                    a_traj = a["pos_xy"][None, :] + t[:, None] * a["vel_xy"][None, :]
                                    d = np.linalg.norm(pred_traj[:, :2] - a_traj, axis=1)
                                    idx = int(np.argmin(d))
                                    if d[idx] < closest_d:
                                        closest_d = float(d[idx])
                                        closest_idx = idx

                                win = range(max(0, closest_idx - 2), min(pred_len, closest_idx + 3))
                                for j in range(pred_len):
                                    if j in win:
                                        w = 1.0 - 0.25 * (1.0 - abs(j - closest_idx) / max(1, len(win) // 2))
                                        pred_speeds[j] *= max(0.75, min(0.9, w))
                                print("[planner] Applied windowed hard brake - sustained collision")
                            else:
                                collision_detected_frames += 1
                                print(f"[planner] Collision detected but not severe enough for hard brake (frames: {collision_detected_frames})")
                    else:
                        if min_d < 1.5 and collision_detected_frames >= 2:
                            t = np.arange(pred_len, dtype=np.float32) * 0.5
                            closest_idx = 0
                            closest_d = 1e9
                            for a in critical_agents:
                                a_traj = a["pos_xy"][None, :] + t[:, None] * a["vel_xy"][None, :]
                                d = np.linalg.norm(pred_traj[:, :2] - a_traj, axis=1)
                                idx = int(np.argmin(d))
                                if d[idx] < closest_d:
                                    closest_d = float(d[idx])
                                    closest_idx = idx

                            win = range(max(0, closest_idx - 2), min(pred_len, closest_idx + 3))
                            for j in range(pred_len):
                                if j in win:
                                    w = 1.0 - 0.25 * (1.0 - abs(j - closest_idx) / max(1, len(win) // 2))
                                    pred_speeds[j] *= max(0.75, min(0.9, w))
                            print("[planner] Applied windowed hard brake - direct severe collision")
                        else:
                            collision_detected_frames += 1
                            print(f"[planner] Collision detected but not severe enough for hard brake (frames: {collision_detected_frames})")
                    
                    if not collision_detected:
                        collision_detected_frames = 0
                    
                    alpha = 0.30
                    k_hist = float(obs_ego_curvatures[-1])
                    half_len = len(pred_curvatures_real) // 2
                    pred_curvatures_real[half_len:] = alpha * pred_curvatures_real[half_len:] + (1 - alpha) * k_hist
                    
                    print(f"[planner] Segmented curvature recovery: hist={k_hist:.3f}, mean={np.mean(pred_curvatures_real):.3f}")
                    
                    # 初始化pred_traj
                    if pred_traj is None:
                        pred_traj = np.zeros((pred_len, 3))
                    
                    pred_traj[:, :2] = IntegrateCurvatureForPoints(
                        pred_curvatures_real,
                        pred_speeds,
                        fut_start_world,
                        atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                        pred_len,
                    )
                    
                    final_collision = dynamic_collision_flag(
                        pred_traj[:, :2], agents_for_check, step=0.5, base_safety_radius=base_radius, speed_gain=speed_gain
                    )
                    if final_collision:
                        if len(agents_for_check) > 0:
                            ego_xy = obs_ego_traj_world[-1]
                            agents_xy_seq = [a["pos_xy"] for a in agents_for_check]
                            
                            min_distances = []
                            for a in agents_for_check:
                                a_traj = a["pos_xy"][None, :] + np.arange(pred_len, dtype=np.float32)[:, None] * 0.5 * a["vel_xy"][None, :]
                                d = np.linalg.norm(pred_traj[:, :2] - a_traj, axis=1)
                                min_distances.append(np.min(d))
                            min_d_gap = min(min_distances) - base_radius if min_distances else 0.0
                            
                            pred_speeds, pred_curvatures_real = two_stage_avoidance(
                                pred_speeds, pred_curvatures_real, ego_xy, agents_xy_seq, base_radius, min_d_gap
                            )
                            
                            pred_traj[:, :2] = IntegrateCurvatureForPoints(
                                pred_curvatures_real,
                                pred_speeds,
                                fut_start_world,
                                atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                                pred_len,
                            )
                            
                            emergency_collision = dynamic_collision_flag(
                                pred_traj[:, :2], agents_for_check, step=0.5, base_safety_radius=base_radius, speed_gain=speed_gain
                            )
                            if emergency_collision:
                                pred_speeds[:5] = 0.0
                                pred_traj[:, :2] = IntegrateCurvatureForPoints(
                                    pred_curvatures_real,
                                    pred_speeds,
                                    fut_start_world,
                                    atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                                    pred_len,
                                )
                                print("[planner] Two-stage avoidance: Still unsafe, stopping")
                            else:
                                print("[planner] Two-stage avoidance: Collision resolved")
                        
                        collision_triggered_count += 1
                        print("[planner] Final collision check: Still unsafe after braking")
                    else:
                        print("[planner] Collision resolved after braking & replan.")
                    
                    prev_collision_unsafe = bool(final_collision)
                    planner_state["last_agents"] = agents_states
                    
                    speed_curvature_pred = [[float(v), float(k)] for v, k in zip(pred_speeds.tolist(), pred_curvatures_real.tolist())]
                    
                    eval_traj = pred_traj[:, :2].copy()
                    eval_traj_source = "post_safety"
                    print(f"[EVAL] traj_source={eval_traj_source}, len={len(eval_traj)}, first={eval_traj[0]}, last={eval_traj[-1]}")
                    assert eval_traj.sum() != 0, "Invalid eval traj: all zeros!"
                    assert len(eval_traj) == FUT_LEN, f"Invalid eval traj length: {len(eval_traj)} != {FUT_LEN}"

                    if eval_traj is None:
                        eval_traj = pred_traj[:, :2].copy()
                        eval_traj_source = "final"
                        print(f"[EVAL] traj_source={eval_traj_source}, len={len(eval_traj)}, first={eval_traj[0]}, last={eval_traj[-1]}")
                        assert eval_traj.sum() != 0, "Invalid eval traj: all zeros!"
                        assert len(eval_traj) == FUT_LEN, f"Invalid eval traj length: {len(eval_traj)} != {FUT_LEN}"

                    origin_xy = np.array(obs_ego_traj_world[-1][:2], np.float32)
                    img = draw_traj_on_bev(img, pred_traj[:, :2], origin_xy, yaw_deg_last, args.bev_extent, args.bev_res, color=(0, 0, 255), thickness=2)

            else:
                # Mode A: 前视模式 (Front-view + CoT)
                print("[planner] Mode A: Front-view with CoT")

                if i % K != 0 and last_pred is not None:
                    prediction = last_pred
                    scene_desc = "Cached from previous frame"
                    object_desc = "Cached from previous frame"
                    updated_intent = "Cached from previous frame"
                    print(f"[CACHE] Frame {i}: Using cached prediction")
                else:
                    prediction, scene_desc, object_desc, updated_intent = GenerateMotion(
                        obs_images, obs_ego_traj_world, obs_ego_velocities, obs_ego_curvatures, prev_intent, args=args
                    )
                    last_pred = prediction
                    # Mode A现在也有CoT三阶段 + GenerateMotion = 4次API调用
                    total_api_calls += 4  # SceneDescription + DescribeObjects + DescribeOrUpdateIntent + GenerateMotion
                    print(f"[VLM] Frame {i}: Generated new prediction (4 API calls: CoT×3 + Motion×1)")

                pred_waypoints = str(prediction).replace("Future speeds and curvatures:", "").strip()
                coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
                if not coordinates:
                    continue

                speed_curvature_pred = [[float(v), float(k) / 100.0] for v, k in coordinates][:10]
                for p in speed_curvature_pred:
                    p[1] = float(np.clip(p[1], -0.055, 0.055))

                last_speed = np.linalg.norm(obs_ego_velocities[-1]) if len(obs_ego_velocities) > 0 else None
                speed_curvature_pred, was_clamped = clamp_by_traffic_rules(
                    speed_curvature_pred, tl_state=None, dist_to_stopline_m=None, last_speed=last_speed
                )
                if was_clamped:
                    traffic_rule_clamped_count += 1

                print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")
                prev_intent = updated_intent

                pred_len = min(FUT_LEN, len(speed_curvature_pred))
                pred_curvatures_real = np.array(speed_curvature_pred)[:, 1]
                pred_speeds = np.array(speed_curvature_pred)[:, 0]
                
                if pred_len == 0 or np.allclose(pred_speeds, 0.0):
                    v0 = float(np.linalg.norm(obs_ego_velocities[-1])) * 0.5
                    pred_speeds = np.full((FUT_LEN,), v0, dtype=np.float32)
                    pred_curvatures_real = np.zeros((FUT_LEN,), dtype=np.float32)
                    pred_len = FUT_LEN
                    print(f"[FALLBACK] Using fallback trajectory: v0={v0:.2f} m/s")
                
                pred_traj = np.zeros((pred_len, 3))
                pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(
                    pred_curvatures_real,
                    pred_speeds,
                    fut_start_world,
                    atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                    pred_len,
                )
                
                eval_traj = pred_traj[:, :2].copy()
                eval_traj_source = "front_view"
                print(f"[EVAL] traj_source=front_view, len={len(eval_traj)}, first={eval_traj[0]}, last={eval_traj[-1]}")
                assert eval_traj.sum() != 0, "Invalid eval traj: all zeros!"
                assert len(eval_traj) == FUT_LEN, f"Invalid eval traj length: {len(eval_traj)} != {FUT_LEN}"

                curr_token = sample_tokens[i + OBS_LEN - 1]
                prev_token = sample_tokens[i + OBS_LEN - 2] if i + OBS_LEN - 2 >= 0 else None
                agents_states = collect_agents_nus_with_vel(nusc, prev_token, curr_token, dt=0.5)
                
                ego_pos_xy = np.array(obs_ego_traj_world[-1][:2], np.float32)
                ego_heading = np.arctan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0])
                agents_states = filter_agents(agents_states, ego_pos_xy, ego_heading, max_r=25.0, fov_deg=90, topk=12)
                print(f"[A-BRANCH] agents={len(agents_states)} (collision evaluation only)")
                
                collision_eval = dynamic_collision_flag(
                    pred_traj[:, :2], agents_states, step=0.5, base_safety_radius=1.2, speed_gain=0.10
                )
                if collision_eval:
                    collision_triggered_count += 1
                    print("[A-BRANCH] Collision detected in evaluation (no intervention)")

                _ = OverlayTrajectory(
                    img, pred_traj.tolist(), obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0), args=args
                )

            # 评测 ADE/FDE —— ✅ 统一使用 eval_traj（若有）
            if pred_traj is None:
                print(f"[WARNING] pred_traj is None at frame {i}, skipping ADE/FDE calculation")
                continue
                
            fut_ego_traj_world_np = np.array(fut_ego_traj_world)
            pr_xy = eval_traj if eval_traj is not None else pred_traj[:, :2]
            pred_len = pr_xy.shape[0]
            gt = np.asarray(fut_ego_traj_world_np[:pred_len, :2], np.float32)
            pr = np.asarray(pr_xy[:pred_len, :2], np.float32)

            d = np.linalg.norm(gt - pr, axis=1)
            ade = float(d.mean()) if len(d) > 0 else None
            fde = float(d[-1]) if len(d) > 0 else None
            if fde is not None:
                fde_list.append(fde)

            # ADE@1s/2s/3s: 0.5s步长 => 2/4/6步
            ade1s = ade2s = ade3s = None
            pred1_len = min(pred_len, 2)
            if pred1_len > 0:
                gt1 = np.asarray(fut_ego_traj_world_np[:pred1_len, :2], np.float32)
                pr1 = np.asarray(pr_xy[:pred1_len, :2], np.float32)
                ade1s = float(np.mean(np.linalg.norm(gt1 - pr1, axis=1)))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            if pred2_len > 0:
                gt2 = np.asarray(fut_ego_traj_world_np[:pred2_len, :2], np.float32)
                pr2 = np.asarray(pr_xy[:pred2_len, :2], np.float32)
                ade2s = float(np.mean(np.linalg.norm(gt2 - pr2, axis=1)))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            if pred3_len > 0:
                gt3 = np.asarray(fut_ego_traj_world_np[:pred3_len, :2], np.float32)
                pr3 = np.asarray(pr_xy[:pred3_len, :2], np.float32)
                ade3s = float(np.mean(np.linalg.norm(gt3 - pr3, axis=1)))
            ade3s_list.append(ade3s)

            # 保存产物
            if args.plot:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{out_dir}/{name}_{i}_front_cam.jpg", img)

                plt.plot(fut_ego_traj_world_np[:, 0], fut_ego_traj_world_np[:, 1], "r-", label="GT")
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", label="Pred")
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade:.3f}" if ade is not None else f"Scene: {name}, Frame: {i}")
                plt.savefig(f"{out_dir}/{name}_{i}_traj.jpg")
                plt.close()

                np.save(f"{out_dir}/{name}_{i}_pred_traj.npy", pred_traj)
                np.save(f"{out_dir}/{name}_{i}_pred_curvatures.npy", np.array(speed_curvature_pred)[:, 1])
                np.save(f"{out_dir}/{name}_{i}_pred_speeds.npy", np.array(speed_curvature_pred)[:, 0])

                with open(f"{out_dir}/{name}_{i}_logs.txt", "w", encoding="utf-8") as f:
                    f.write(f"Scene Description: {scene_desc}\n")
                    f.write(f"Object Description: {object_desc}\n")
                    f.write(f"Intent Description: {updated_intent}\n")
                    f.write(f"Eval Traj Source: {eval_traj_source or 'final'}\n")
                    f.write(f"[planner] use BEV: {args.use_bev}\n")
                    f.write(f"[planner] agents={len(agents_states)}\n")
                    f.write(f"[planner] dynamic_collision={dynamic_collision_flag(pred_traj[:, :2], agents_states)}\n")
                    if ade is not None:
                        f.write(f"Average Displacement Error: {ade}\n")
                    if fde is not None:
                        f.write(f"Final Displacement Error: {fde}\n")
                    f.write(f"Raw Prediction: {speed_curvature_pred}\n")

        # 场景汇总
        mean_ade1s = float(np.mean(ade1s_list)) if ade1s_list else None
        mean_ade2s = float(np.mean(ade2s_list)) if ade2s_list else None
        mean_ade3s = float(np.mean(ade3s_list)) if ade3s_list else None
        avg_ade = (
            float(np.mean([x for x in [mean_ade1s, mean_ade2s, mean_ade3s] if x is not None]))
            if any([ade1s_list, ade2s_list, ade3s_list])
            else None
        )
        avg_fde = float(np.mean(fde_list)) if fde_list else None

        gt_smoothness = None
        pred_smoothness = None
        try:
            if len(ego_poses_world) > 3:
                gt_smoothness = calculate_trajectory_smoothness(ego_poses_world)
                print(f"\nScene {name} Ground Truth Smoothness Analysis:")
                print_smoothness_analysis({"Ground Truth": gt_smoothness})
        except Exception as e:
            print(f"Warning: Smoothness analysis (GT) failed: {e}")

        try:
            if 'eval_traj' in locals() and eval_traj is not None and len(eval_traj) > 3:
                pred_smoothness = calculate_trajectory_smoothness(eval_traj)
                print(f"\nScene {name} Predicted Smoothness Analysis:")
                print_smoothness_analysis({"Predicted": pred_smoothness})
                if gt_smoothness and pred_smoothness:
                    print(f"\nScene {name} Smoothness Comparison:")
                    comparison = compare_trajectory_smoothness(ego_poses_world, eval_traj, "Ground Truth", "Predicted")
                    print_smoothness_analysis(comparison)
            else:
                print(f"Warning: eval_traj not available for smoothness analysis")
        except Exception as e:
            print(f"Warning: Smoothness analysis (Pred) failed: {e}")

        total_frames = len(ade1s_list) if ade1s_list else 1
        collision_rate = collision_triggered_count / total_frames if total_frames > 0 else 0
        traffic_rule_rate = traffic_rule_clamped_count / total_frames if total_frames > 0 else 0

        print(f"\n=== Scene {name} Statistics ===")
        if mean_ade1s is not None:
            print(f"ADE@1s: {mean_ade1s:.3f}")
        if mean_ade2s is not None:
            print(f"ADE@2s: {mean_ade2s:.3f}")
        if mean_ade3s is not None:
            print(f"ADE@3s: {mean_ade3s:.3f}")
        if avg_fde is not None:
            print(f"FDE: {avg_fde:.3f}")
        print(f"Collision rate: {collision_rate:.3f}")
        print(f"Traffic rule rate: {traffic_rule_rate:.3f}")
        print(f"Saving results to: {out_dir}/ade_results.jsonl")

        result = {
            "name": name,
            "token": token,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": avg_ade,
            "avgfde": avg_fde,
            "use_bev": args.use_bev,
            "prompt_mode": args.prompt_mode,
            "ablation": args.ablation,
            "gt_velocity_cv": (gt_smoothness or {}).get("velocity_cv", None),
            "gt_accel_mean": (gt_smoothness or {}).get("accel_mean", None),
            "gt_jerk_mean": (gt_smoothness or {}).get("jerk_mean", None),
            "gt_curvature_mean": (gt_smoothness or {}).get("curvature_mean", None),
            "gt_smoothness_score": (gt_smoothness or {}).get("smoothness_score", None),
            "pred_velocity_cv": (pred_smoothness or {}).get("velocity_cv", None) if pred_smoothness else None,
            "pred_accel_mean": (pred_smoothness or {}).get("accel_mean", None) if pred_smoothness else None,
            "pred_jerk_mean": (pred_smoothness or {}).get("jerk_mean", None) if pred_smoothness else None,
            "pred_curvature_mean": (pred_smoothness or {}).get("curvature_mean", None) if pred_smoothness else None,
            "pred_smoothness_score": (pred_smoothness or {}).get("smoothness_score", None) if pred_smoothness else None,
            "collision_triggered_count": collision_triggered_count,
            "collision_rate": collision_rate,
            "traffic_rule_clamped_count": traffic_rule_clamped_count,
            "traffic_rule_rate": traffic_rule_rate,
            "total_frames": total_frames,
            "total_inference_time": total_inference_time,
            "avg_inference_time": total_inference_time / total_frames if total_frames > 0 else 0,
            "total_api_calls": total_api_calls,
        }
        with open(f"{out_dir}/ade_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result))
            f.write("\n")

        print(f" Results saved successfully!")
        
        # 添加到全局统计
        if mean_ade1s is not None:
            all_scenes_ade1s.append(mean_ade1s)
        if mean_ade2s is not None:
            all_scenes_ade2s.append(mean_ade2s)
        if mean_ade3s is not None:
            all_scenes_ade3s.append(mean_ade3s)
        if avg_fde is not None:
            all_scenes_fde.append(avg_fde)
        all_scenes_collision_rate.append(collision_rate)
        all_scenes_traffic_rule_rate.append(traffic_rule_rate)
        all_scenes_api_calls.append(total_api_calls)
        all_scenes_inference_time.append(total_inference_time)
        all_scenes_frames.append(total_frames)
        
        # 记录场景详细信息（用于排名）
        scene_detail = {
            'name': name,
            'ade1s': mean_ade1s,
            'ade2s': mean_ade2s,
            'ade3s': mean_ade3s,
            'fde': avg_fde,
            'collision_rate': collision_rate,
            'traffic_rule_rate': traffic_rule_rate,
            'frames': total_frames
        }
        
        # 添加smoothness metrics（如果有）
        if pred_smoothness:
            scene_detail['smoothness'] = {
                'velocity_cv': pred_smoothness.get('velocity_cv'),
                'acceleration_mean': pred_smoothness.get('acceleration_mean'),
                'jerk_mean': pred_smoothness.get('jerk_mean'),
                'jerk_std': pred_smoothness.get('jerk_std'),
                'curvature_mean': pred_smoothness.get('curvature_mean'),
                'curvature_std': pred_smoothness.get('curvature_std'),
                'smoothness_score': pred_smoothness.get('smoothness_score')
            }
        
        scene_details.append(scene_detail)

        with open(f"{out_dir}/{name}_scene_log.md", "w", encoding="utf-8") as f:
            f.write(f"# Scene {name} Analysis\n\n")
            f.write(f"**Configuration:** BEV={args.use_bev}, Prompt={args.prompt_mode}, Ablation={args.ablation}\n\n")
            f.write(f"**Performance:**\n")
            if mean_ade1s is not None:
                f.write(f"- ADE 1s: {mean_ade1s:.3f}\n")
            if mean_ade2s is not None:
                f.write(f"- ADE 2s: {mean_ade2s:.3f}\n")
            if mean_ade3s is not None:
                f.write(f"- ADE 3s: {mean_ade3s:.3f}\n")
            if avg_ade is not None:
                f.write(f"- Average ADE: {avg_ade:.3f}\n")

        # （已移除 nuboard 数据导出）
        if args.plot and (not args.no_video) and cam_images_sequence:
            WriteImageSequenceToVideo(cam_images_sequence, f"{out_dir}/{name}")
    
    # ========== 所有场景处理完毕，输出全局统计 ==========
    print("\n" + "="*80)
    print(" ALL SCENES COMPLETED - OVERALL STATISTICS")
    print("="*80)
    
    if all_scenes_ade1s:
        global_ade1s = float(np.mean(all_scenes_ade1s))
        std_ade1s = float(np.std(all_scenes_ade1s))
        print(f" ADE@1s (avg): {global_ade1s:.3f} ± {std_ade1s:.3f} (across {len(all_scenes_ade1s)} scenes)")
    
    if all_scenes_ade2s:
        global_ade2s = float(np.mean(all_scenes_ade2s))
        std_ade2s = float(np.std(all_scenes_ade2s))
        print(f" ADE@2s (avg): {global_ade2s:.3f} ± {std_ade2s:.3f} (across {len(all_scenes_ade2s)} scenes)")
    
    if all_scenes_ade3s:
        global_ade3s = float(np.mean(all_scenes_ade3s))
        std_ade3s = float(np.std(all_scenes_ade3s))
        print(f" ADE@3s (avg): {global_ade3s:.3f} ± {std_ade3s:.3f} (across {len(all_scenes_ade3s)} scenes)")
    
    if all_scenes_fde:
        global_fde = float(np.mean(all_scenes_fde))
        std_fde = float(np.std(all_scenes_fde))
        print(f" FDE (avg): {global_fde:.3f} ± {std_fde:.3f} (across {len(all_scenes_fde)} scenes)")
    
    if all_scenes_collision_rate:
        global_collision = float(np.mean(all_scenes_collision_rate))
        std_collision = float(np.std(all_scenes_collision_rate))
        print(f" Collision rate (avg): {global_collision:.3f} ± {std_collision:.3f} ({global_collision*100:.1f}%)")
    
    if all_scenes_traffic_rule_rate:
        global_traffic = float(np.mean(all_scenes_traffic_rule_rate))
        std_traffic = float(np.std(all_scenes_traffic_rule_rate))
        print(f"🚦 Traffic rule rate (avg): {global_traffic:.3f} ± {std_traffic:.3f} ({global_traffic*100:.1f}%)")
    
    # 计算并输出计算效率统计
    if all_scenes_api_calls and all_scenes_frames:
        total_api = sum(all_scenes_api_calls)
        total_frames_count = sum(all_scenes_frames)
        total_time = sum(all_scenes_inference_time)
        avg_api_per_frame = total_api / total_frames_count if total_frames_count > 0 else 0
        avg_time_per_frame = total_time / total_frames_count if total_frames_count > 0 else 0
        
        print("\n⚙️  COMPUTATIONAL EFFICIENCY")
        print("-"*80)
        print(f"  Total API calls: {total_api}")
        print(f"  Total frames: {total_frames_count}")
        print(f"  Avg API calls/frame: {avg_api_per_frame:.2f}")
        print(f"  Total inference time: {total_time:.2f}s")
        print(f"  Avg time/frame: {avg_time_per_frame:.3f}s")
        
        if args.ablation == 'B':
            print(f"  Mode B: ~3 API calls/frame (Scene + Object + Motion)")
        elif args.ablation == 'C' and args.use_cot_hodc:
            print(f"  Mode C: ~5 API calls/frame (CoT → HODC → Trajectory)")
    
    # 计算并输出smoothness metrics平均值
    if scene_details:
        smoothness_metrics = {
            'velocity_cv': [],
            'acceleration_mean': [],
            'jerk_mean': [],
            'jerk_std': [],
            'curvature_mean': [],
            'curvature_std': [],
            'smoothness_score': []
        }
        
        for scene in scene_details:
            if 'smoothness' in scene and scene['smoothness']:
                for key in smoothness_metrics.keys():
                    val = scene['smoothness'].get(key)
                    if val is not None:
                        smoothness_metrics[key].append(val)
        
        print("\n📈 SMOOTHNESS METRICS (Predicted Trajectories)")
        print("-"*80)
        for key, values in smoothness_metrics.items():
            if len(values) > 0:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                print(f"  {key:<20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("="*80)
    
    # ========== 场景排名分析 ==========
    if scene_details:
        print("\n" + "="*80)
        print("📈 SCENE PERFORMANCE RANKING (for case study analysis)")
        print("="*80)
        
        # 按FDE排序（越低越好）
        valid_scenes = [s for s in scene_details if s['fde'] is not None]
        if valid_scenes:
            sorted_by_fde = sorted(valid_scenes, key=lambda x: x['fde'])
            
            print("\n TOP 3 BEST PERFORMING SCENES (lowest FDE):")
            print("-"*80)
            for i, scene in enumerate(sorted_by_fde[:3], 1):
                print(f"{i}. {scene['name']}")
                print(f"   FDE: {scene['fde']:.3f} | ADE@1s: {scene['ade1s']:.3f} | ADE@2s: {scene['ade2s']:.3f} | ADE@3s: {scene['ade3s']:.3f}")
                print(f"   Collision: {scene['collision_rate']*100:.1f}% | Traffic: {scene['traffic_rule_rate']*100:.1f}% | Frames: {scene['frames']}")
                if 'smoothness' in scene and scene['smoothness']:
                    s = scene['smoothness']
                    print(f"   Smoothness: jerk={s.get('jerk_mean', 0):.4f}, curvature={s.get('curvature_mean', 0):.4f}, score={s.get('smoothness_score', 0):.4f}")
            
            print("\n BOTTOM 3 CHALLENGING SCENES (highest FDE):")
            print("-"*80)
            for i, scene in enumerate(sorted_by_fde[-3:][::-1], 1):
                print(f"{i}. {scene['name']}")
                print(f"   FDE: {scene['fde']:.3f} | ADE@1s: {scene['ade1s']:.3f} | ADE@2s: {scene['ade2s']:.3f} | ADE@3s: {scene['ade3s']:.3f}")
                print(f"   Collision: {scene['collision_rate']*100:.1f}% | Traffic: {scene['traffic_rule_rate']*100:.1f}% | Frames: {scene['frames']}")
                if 'smoothness' in scene and scene['smoothness']:
                    s = scene['smoothness']
                    print(f"   Smoothness: jerk={s.get('jerk_mean', 0):.4f}, curvature={s.get('curvature_mean', 0):.4f}, score={s.get('smoothness_score', 0):.4f}")
            
            # 按Collision排序（越低越好）
            sorted_by_collision = sorted(valid_scenes, key=lambda x: x['collision_rate'])
            perfect_scenes = [s for s in sorted_by_collision if s['collision_rate'] == 0.0]
            
            if perfect_scenes:
                print(f"\n {len(perfect_scenes)} SCENES WITH ZERO COLLISION RATE:")
                print("-"*80)
                for scene in perfect_scenes[:5]:  # 最多显示5个
                    print(f"   {scene['name']:<15s} FDE: {scene['fde']:.3f} | ADE@3s: {scene['ade3s']:.3f}")
            
            print("\n Recommendation for paper case studies:")
            print(f"   • Best case: {sorted_by_fde[0]['name']} (FDE={sorted_by_fde[0]['fde']:.3f})")
            print(f"   • Challenging case: {sorted_by_fde[-1]['name']} (FDE={sorted_by_fde[-1]['fde']:.3f})")
            if perfect_scenes:
                safe_scene = sorted(perfect_scenes, key=lambda x: x['fde'])[0]
                print(f"   • Safety showcase: {safe_scene['name']} (0% collision, FDE={safe_scene['fde']:.3f})")
    
    print("\n" + "="*80)
    
    # 保存全局统计到文件
    valid_scenes_for_summary = [s for s in scene_details if s['fde'] is not None]
    
    # 计算smoothness metrics的全局统计
    smoothness_summary = {}
    if scene_details:
        smoothness_metrics_data = {
            'velocity_cv': [],
            'acceleration_mean': [],
            'jerk_mean': [],
            'jerk_std': [],
            'curvature_mean': [],
            'curvature_std': [],
            'smoothness_score': []
        }
        
        for scene in scene_details:
            if 'smoothness' in scene and scene['smoothness']:
                for key in smoothness_metrics_data.keys():
                    val = scene['smoothness'].get(key)
                    if val is not None:
                        smoothness_metrics_data[key].append(val)
        
        for key, values in smoothness_metrics_data.items():
            if len(values) > 0:
                smoothness_summary[f"{key}_mean"] = float(np.mean(values))
                smoothness_summary[f"{key}_std"] = float(np.std(values))
    
    # 计算效率统计
    efficiency_stats = {}
    if all_scenes_api_calls and all_scenes_frames:
        total_api = sum(all_scenes_api_calls)
        total_frames_count = sum(all_scenes_frames)
        total_time = sum(all_scenes_inference_time)
        efficiency_stats = {
            "total_api_calls": total_api,
            "total_frames": total_frames_count,
            "avg_api_per_frame": float(total_api / total_frames_count) if total_frames_count > 0 else 0,
            "total_inference_time": float(total_time),
            "avg_time_per_frame": float(total_time / total_frames_count) if total_frames_count > 0 else 0
        }
    
    global_summary = {
        "total_scenes": len(all_scenes_ade1s) if all_scenes_ade1s else 0,
        "ade1s_mean": float(np.mean(all_scenes_ade1s)) if all_scenes_ade1s else None,
        "ade1s_std": float(np.std(all_scenes_ade1s)) if all_scenes_ade1s else None,
        "ade2s_mean": float(np.mean(all_scenes_ade2s)) if all_scenes_ade2s else None,
        "ade2s_std": float(np.std(all_scenes_ade2s)) if all_scenes_ade2s else None,
        "ade3s_mean": float(np.mean(all_scenes_ade3s)) if all_scenes_ade3s else None,
        "ade3s_std": float(np.std(all_scenes_ade3s)) if all_scenes_ade3s else None,
        "fde_mean": float(np.mean(all_scenes_fde)) if all_scenes_fde else None,
        "fde_std": float(np.std(all_scenes_fde)) if all_scenes_fde else None,
        "collision_rate_mean": float(np.mean(all_scenes_collision_rate)) if all_scenes_collision_rate else None,
        "collision_rate_std": float(np.std(all_scenes_collision_rate)) if all_scenes_collision_rate else None,
        "traffic_rule_rate_mean": float(np.mean(all_scenes_traffic_rule_rate)) if all_scenes_traffic_rule_rate else None,
        "traffic_rule_rate_std": float(np.std(all_scenes_traffic_rule_rate)) if all_scenes_traffic_rule_rate else None,
        "efficiency": efficiency_stats,  # 计算效率统计
        "smoothness": smoothness_summary,  # 全局smoothness统计
        "scene_details": scene_details,  # 保存所有场景详情
        "best_scenes": sorted(valid_scenes_for_summary, key=lambda x: x['fde'])[:3] if valid_scenes_for_summary else [],
        "worst_scenes": sorted(valid_scenes_for_summary, key=lambda x: x['fde'])[-3:][::-1] if valid_scenes_for_summary else [],
        "hodc_stats": hodc_stats if args.ablation == 'C' and args.use_cot_hodc else None  # HODC++统计
    }
    
    with open(f"{out_dir}/global_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    
    print(f"Global summary saved to: {out_dir}/global_summary.json")
    print("="*80 + "\n")
