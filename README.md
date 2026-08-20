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

Stage 1 produces an objective factual summary of endpoint values, lane spread,
actual temporal observations, native transmission directions, statuses,
metadata, and missing data. It does not make a preliminary diagnosis and does
not create rule-derived threshold flags. Stage 2 independently compares both
endpoint hypotheses and fiber, recording supporting, contradictory, and
unexplained observations for each.

Expert SOP knowledge has the lowest evidence priority. It is optional,
potentially incomplete or inaccurate background and cannot override stronger
case observations or cross-metric/cross-endpoint consistency. No fixed
same/opposite mapping or implicit SOP decision tree is implemented.
`alarm_ip_interface` is evidence and does not automatically determine the
diagnosis.

## Run

```bash
python3 run_experiment.py \
  --data-root /path/to/data \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --output-dir /path/to/results/bian/fiber_aware \
  --gpu-ids 4,5 --smoke-test

python3 run_experiment.py \
  --data-root /path/to/data \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --output-dir /path/to/results/bian/fiber_aware \
  --gpu-ids 4,5
```

The full run writes `bian_results.csv` and `scores.txt`. The inference loader
removes JSON `label` before summary and prompt construction. Predictions are
fixed before the evaluator reopens each JSON and reads `label` as the sole
ground truth source.

Generated data, predictions, scores, logs, caches, and model weights are local
artifacts and are not committed.
