#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$project_root"

checkpoint="${POLICY_CHECKPOINT:-$project_root/pretrained_model}"
[[ -d "$checkpoint" ]] || { echo "Checkpoint directory missing: $checkpoint" >&2; exit 1; }

task="${INFERENCE_TASK:-task3}"
case "$task" in
    task1|task3) ;;
    *) echo "INFERENCE_TASK 仅支持 task1 或 task3，当前为：$task" >&2; exit 2 ;;
esac
rounds="${INFERENCE_ROUNDS:-1}"
episode_length_s="${INFERENCE_EPISODE_LENGTH_S:-60}"
sim_device="${SIM_DEVICE:-cpu}"
policy_type="${POLICY_TYPE:-lerobot-pi05}"
policy_action_horizon="${POLICY_ACTION_HORIZON:-24}"
policy_instruction="${POLICY_LANGUAGE_INSTRUCTION:-Insert the test tubes on the desktop into the rack.}"

exec python scripts/evaluation/policy_inference.py \
    --task="$task" --inference_rounds="$rounds" \
    --episode_length_s="$episode_length_s" --step_hz=30 \
    --policy_type="$policy_type" --policy_host="${POLICY_HOST:-127.0.0.1}" \
    --policy_port="${POLICY_PORT:-5555}" --policy_timeout_ms="${POLICY_TIMEOUT_MS:-15000}" \
    --policy_action_horizon="$policy_action_horizon" \
    --policy_language_instruction="$policy_instruction" \
    --device="$sim_device" --enable_cameras --policy_checkpoint_path="$checkpoint" \
    "$@"
