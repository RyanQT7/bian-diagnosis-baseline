# BiAn v3 Diagnosis Baseline

This project adapts BiAn's hierarchical two-stage reasoning to case-level
optical-link diagnosis with `DeepSeek-R1-Distill-Qwen-32B` in both stages.
Stage 1 extracts structured local/remote physical evidence. Stage 2 applies
optical-link expert SOP knowledge as soft diagnostic priors, first predicts a
physical location (`local`, `remote`, or `fiber`), and then converts an endpoint
fault to the official class using observable endpoint rate.

The official labels are `l1`, `l2`, and `fiber`: `l1` corresponds to a 400G
fault and `l2` corresponds to a 200G fault. **l1/l2 do not correspond to
local/remote.** Local and remote are physical endpoint directions used only
inside diagnosis. The predicted physical endpoint is mapped to `l1` or `l2`
according to whether that endpoint is 400G or 200G.

Endpoint rate is read from `link_side_ip_interface_map`, whose endpoint strings
explicitly contain `400G` or `200G`. The alarm endpoint defines the local side
when available; otherwise JSON endpoint order supplies a deterministic physical
orientation. No true label is used for this mapping.

## Run

Install `requirements.txt` in a suitable environment. The data root contains
`data1/` and `data2/`.

```bash
python3 run_experiment.py \
  --data-root /path/to/data \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --output-dir /path/to/results/bian/v3_correct_labels \
  --gpu-ids 4,5 --smoke-test

python3 run_experiment.py \
  --data-root /path/to/data \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --output-dir /path/to/results/bian/v3_correct_labels \
  --gpu-ids 4,5
```

The run generates `bian_v3_results.csv`, `scores.txt`, metadata, and a local
inference audit. Re-score a result with:

```bash
python3 score.py --predictions /path/to/bian_v3_results.csv \
  --output /path/to/scores.txt
```

## Method and leakage controls

The inference loader removes the JSON `label` before constructing summaries or
prompts. It compresses telemetry into per-side metric, lane, temporal, status,
and cross-side statistics. Historical SOP thresholds are marked
`expert_reference_only`; they describe evidence and never directly determine a
class. Stage 2 compares both endpoints, temporal behavior, severity,
multi-metric consistency, and conflicting fiber evidence.

Predictions are finalized before the evaluator reopens JSON files for ground
truth. The evaluator treats the JSON `label` as the sole ground-truth source
and converts a labelled endpoint to the official class by its observable rate.
This also handles legacy endpoint identifiers while preserving the official
`l1`/`l2`/`fiber` scoring space.

This differs from the source paper in data, model, prompt, and final task.
Generated data, predictions, scores, audits, logs, caches, and model weights are
intentionally ignored and are not committed.
