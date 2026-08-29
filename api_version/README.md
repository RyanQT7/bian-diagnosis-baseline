# BiAn API version

This directory is an API-backed version of the case-native endpoint BiAn
baseline. It keeps the two-stage preprocessing and structured diagnosis from
the native-endpoint implementation, while replacing local 32B inference with
the official OpenAI Python SDK Responses API.

Candidates are discovered independently for each case from
`link_side_ip_interface_map.keys()` plus `fiber`. Endpoint names are preserved
verbatim; no `local`/`remote` aliases or rate-to-label mapping is used. JSON
`label` is removed before inference and is reopened only after predictions are
fixed for evaluation.

## Install and configure

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="..."
export OPENAI_MODEL="<model-name>"
```

The key is read only from `OPENAI_API_KEY`; it is never written to source,
logs, or result files. Choose the model with `--model` or `OPENAI_MODEL`.

## Dry-run

No API request is made by dry-run:

```bash
python3 run_experiment.py \
  --data-root /path/to/data \
  --split-dir /path/to/results/time_split_20251001 \
  --subset test \
  --output-dir /path/to/results/bian-api/dry-run \
  --dry-run --limit 2
```

## Future API run

The fixed time split is `alarm_time < 2025-10-01 00:00:00` for
Train/Development and `>=` for Test. A future frozen Test inference can be
started with:

```bash
python3 run_experiment.py \
  --data-root /path/to/data \
  --split-dir /path/to/results/time_split_20251001 \
  --subset test \
  --model "$OPENAI_MODEL" \
  --output-dir /path/to/results/bian-api/test
```

`--max-output-tokens`, optional `--reasoning-effort`, bounded
`--max-retries` (capped at 2), `--resume`, and `--limit` are supported. A run
writes predictions, scores, resumable records, and actual API usage totals to
the selected output directory. Usage includes Stage 1, Stage 2, retries, and
overall input/output/total tokens.

This version is implementation-only in the current experiment; no full API
run is started automatically.
