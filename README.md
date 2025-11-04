Status (⚠️ Important): This repository contains research outcomes. To respect third-party copyrights and proprietary code, it includes only my original contributions. Upstream/private modules (such as openemma.*, src.hodc.*, utils.*, etc.) are not distributed here. Therefore, this repository is not ready for immediate use out of the box.
This repository focuses on the key reasoning and safety‑constraint components used in the thesis, specifically Mode C (BEV + HODC++) and the comparison baselines Mode A/B. It covers:

VLM inference interface: vlm_inference(...) (OpenAI GPT family; multi‑modal input; with rate‑limit handling & retries).

Three modes

A: Front‑view + CoT reasoning

B: BEV‑assisted + CoT reasoning

C: BEV + HODC++ 

Mode C routes

CoT + HODC (five stages): CoT scene → objects → intent → HODC constraints → trajectory

Simplified v2 (two stages)【testing】: semantics + geometry → constraint softening → [v, k] generation (engineering‑oriented refactor)

HODC++ parsing & constraints: parse_hodc_json(...), apply_hodc_constraints(...), hodc_consistency_filter(...).

Safety/comfort layer: collision pre‑check, two‑stage yielding/avoidance, braking hysteresis, curvature limiting/self‑healing, progress floor/arc‑length soft limits, etc.

Evaluation metrics: ADE@{1,2,3}s, FDE, Collision Rate, traffic‑rule constraint trigger rate, smoothness (jerk/curvature, etc.).

Visualization & logging: alignment plots, trajectory plots, JSONL results, global statistics aggregation.

Note on dependencies: Any references to modules like openemma.* are placeholders for upstream/private implementations.
