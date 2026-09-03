# BiAn API version

This directory is an API-backed version of the case-native endpoint BiAn
baseline. It keeps the two-stage preprocessing and structured diagnosis from
the native-endpoint implementation, while replacing local 32B inference with
an OpenAI-compatible Chat Completions API.

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
export CHATANYWHERE_BASE_URL="https://api.chatanywhere.tech/v1"
```

The key is read only from `OPENAI_API_KEY`; it is never written to source,
logs, or result files. Choose the model with `--model` or `OPENAI_MODEL`.
The OpenAI-compatible host can be selected with `--base-url` or
`CHATANYWHERE_BASE_URL` (the two ChatAnywhere hosts are not hard-coded).

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
  --output-dir /path/to/results/bian-api/test \
  --base-url "$CHATANYWHERE_BASE_URL"
```

`--max-output-tokens`, `--connect-timeout-seconds`,
`--read-timeout-seconds`, `--write-timeout-seconds`,
`--pool-timeout-seconds`, bounded `--max-retries` (one transport and one
format retry at most), `--resume`, and `--limit` are supported. The client
uses one connection pool per run and disables SDK-level automatic retries.
`response_format=json_object` is the only structured-output request option;
schema validation and JSON recovery are local for provider compatibility.
The default timeout budget is connect 20s, read 150s, write 60s, and pool
30s. A run writes predictions, scores, resumable records, request metadata,
and actual API usage totals to the selected output directory. Usage includes
known Stage 1/Stage 2 tokens, retries, and requests whose usage is unknown
after a transport failure.

This version is implementation-only in the current experiment; no full API
run is started automatically.
