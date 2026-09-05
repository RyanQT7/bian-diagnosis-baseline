#!/usr/bin/env bash
set -euo pipefail

# Keep the parent Codex shell unchanged; only this BiAn child bypasses proxies.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset NO_PROXY no_proxy

python_bin="${BIAN_PYTHON:-python3}"
exec "$python_bin" "$(dirname "$0")/run_experiment.py" "$@"
