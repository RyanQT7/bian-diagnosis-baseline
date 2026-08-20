# BiAn data-driven native-endpoint baseline

This project adapts BiAn's hierarchical two-stage reasoning to the optical-link
case task with a local `DeepSeek-R1-Distill-Qwen-32B` model. Each case keeps the
native keys from `link_side_ip_interface_map` verbatim. Its candidates are
those keys plus `fiber`; endpoint names are not renamed and interface rates are
metadata only, never a label mapping.

The strategy is evidence-first. A programmatic summary retains lane-level
values, missingness, endpoint telemetry, native TxPower-to-destination-RxPower
relationships, transmission directions, and robust cross-endpoint comparisons.
Stage 1 extracts objective observations without an early diagnosis. Stage 2
compares every native endpoint candidate and `fiber` using supporting,
contradictory, and unexplained observations. The development analysis found
directional Tx/Rx/transmission consistency and cross-endpoint power/SNR
relationships more useful than isolated fields; these are inspected as
relationships, not converted into thresholds or rules. Any expert background
is only a weak, optional reference and cannot override observed evidence.

The lane-keyed values in the current data are spatial lane observations, not
time series, so no temporal trend is invented. Null values remain unavailable;
zero and `-40` remain observed values.

## Reproduce

Install the dependencies in an environment that can read the local model:

```bash
python -m pip install -r requirements.txt
```

First create the fixed seed-42, stratified 25% development / 75% holdout
manifest and the schema/development report:

```bash
python analyze_data.py \
  --data-root /path/to/NSDI26-baseline/data \
  --output-dir /path/to/results/bian/data_analyzed \
  --seed 42 --development-fraction 0.25
```

Run a small two-stage smoke test:

```bash
python run_experiment.py \
  --data-root /path/to/NSDI26-baseline/data \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --output-dir /path/to/results/bian/smoke \
  --gpu-ids 4,5 --smoke-test --smoke-cases 4
```

Run the frozen strategy on holdout cases:

```bash
python run_experiment.py \
  --data-root /path/to/NSDI26-baseline/data \
  --model-path /path/to/DeepSeek-R1-Distill-Qwen-32B \
  --output-dir /path/to/results/bian/data_analyzed \
  --split-manifest /path/to/results/bian/data_analyzed/split_manifest.json \
  --subset holdout \
  --results-name holdout_results.csv --scores-name holdout_scores.txt \
  --gpu-ids 4,5
```

After inference has finished, the evaluator reopens each JSON and reads only
`json["label"]` for scoring. The holdout prediction CSV is written before
ground truth is read. A full-dataset descriptive run uses `--subset all` and
should be reported separately from the held-out evaluation.

Both stages use the same 32B model. `--gpu-ids 4,5` gives tensor parallel size
2; paths, model location, output location, seed, and generation limits are all
CLI-configurable. Generated CSVs, scores, caches, logs, data, and model
weights are local artifacts and are not committed to this repository.

## Data contract

The input is `data1/` plus `data2/`, with JSON cases below one directory level.
Each JSON must contain a native endpoint map and a legal `label` equal to one
of the endpoint keys or `fiber`. The inference loader removes `label` before
constructing either prompt. No rate-to-label conversion is performed:
400G/200G strings may be shown as endpoint metadata, but native keys define the
candidate set.

## Provenance

This is a case-level adaptation of BiAn's two-stage/hierarchical reasoning,
not the original device Top-K RCA task. The input schema, candidate set, label
semantics, evidence representation, split protocol, and evaluation are
specific to this dataset. Reported experimental scores must be generated
locally from the prediction CSVs; no paper accuracy is claimed here.
