#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$project_root"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec python -m lerobot.async_inference.policy_server \
    --host="${POLICY_HOST:-127.0.0.1}" \
    --port="${POLICY_PORT:-5555}" \
    --fps=30 "$@"
