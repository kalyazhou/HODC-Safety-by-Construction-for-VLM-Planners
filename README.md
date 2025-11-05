Status (⚠️ Important): This repository contains research outcomes. To respect third-party copyrights and proprietary code, it includes only my original contributions. Upstream/private modules (such as openemma.*, utils.*, etc.) are not distributed here.
This repository focuses on the key reasoning and safety‑constraint components used in the thesis, specifically Mode C (BEV + HODC++) Mode B(BEV + CoT reasoning) and the comparison baselines Mode A. It covers:

VLM inference interface: vlm_inference(...) (OpenAI GPT; multi‑modal input; with rate‑limit handling & retries).

Three modes

A: Front‑view + CoT reasoning

B: BEV + CoT reasoning

C: BEV + HODC++ 

Mode C routes

CoT + HODC (five stages): CoT scene → objects → intent → HODC constraints → trajectory

HODC++ parsing & constraints: parse_hodc_json(...), apply_hodc_constraints(...), hodc_consistency_filter(...).

Safety/comfort layer: collision pre‑check, two‑stage yielding/avoidance, braking hysteresis, curvature limiting/self‑healing, progress floor/arc‑length soft limits, etc.

Evaluation metrics: ADE@{1,2,3}s, FDE, Collision Rate
Visualization & logging: alignment plots, trajectory plots, JSONL results, global statistics aggregation.
