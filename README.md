# BiAn Native-Endpoint Diagnosis Baseline

This project adapts BiAn's hierarchical two-stage reasoning to optical-link
cases using `DeepSeek-R1-Distill-Qwen-32B` for both stages. The candidate
diagnoses are defined independently for every case as:

```text
link_side_ip_interface_map.keys() + [fiber]
```

Endpoint identifiers such as `l1`, `l2`, `l3`, or `l4` are kept verbatim in
summaries and prompts. They are never renamed to local/remote and are never
derived from 400G/200G interface rates. Other telemetry fields are
cross-checked for endpoint coverage; missing endpoint telemetry remains
unavailable. Transmission keys retain native directions such as `l3-l4`.

Stage 1 extracts endpoint-specific, lane, temporal, directional, and conflict
evidence. Stage 2 applies same-end/opposite-end optical SOP knowledge relative
to each case's actual endpoint identifiers. The SOP and historical thresholds
are soft priors only. `alarm_ip_interface` is reported as evidence and does not
automatically determine the diagnosis.

## Run

```bash
python3 run_experiment.py \
  --data-root /path/to/data \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --output-dir /path/to/results/bian/native_endpoints \
  --gpu-ids 4,5 --smoke-test

python3 run_experiment.py \
  --data-root /path/to/data \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --output-dir /path/to/results/bian/native_endpoints \
  --gpu-ids 4,5
```

The full run writes `bian_results.csv` and `scores.txt`. The inference loader
removes JSON `label` before summary and prompt construction. Predictions are
fixed before the evaluator reopens each JSON and reads `label` as the sole
ground truth source.

Generated data, predictions, scores, logs, caches, and model weights are local
artifacts and are not committed.
