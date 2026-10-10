#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$project_root"

python_cmd="$(command -v python3 || command -v python)"
config_output="$($python_cmd -c '
import config
values = (
    config.teleop_mode, config.task, config.conda_env, config.device,
    config.num_envs, "" if config.seed is None else config.seed,
    config.step_hz, config.sensitivity, int(config.record),
    config.dataset_root, config.dataset_layout, int(config.resume),
    config.num_demos, int(config.enable_cameras), int(config.multi_view),
    int(config.quality), int(config.enable_task1_visualization),
    int(config.left_disabled), config.port, config.left_arm_port,
    config.right_arm_port, int(config.recalibrate),
)
print(*values, sep="\n")
')"
mapfile -t config_values <<< "$config_output"

teleop_mode="${config_values[0]:-}"
task="${config_values[1]:-}"
conda_env="${config_values[2]:-}"
device="${config_values[3]:-}"
num_envs="${config_values[4]:-}"
seed="${config_values[5]:-}"
step_hz="${config_values[6]:-}"
sensitivity="${config_values[7]:-}"
record="${config_values[8]:-0}"
dataset_root="${config_values[9]:-}"
dataset_layout="${config_values[10]:-}"
resume="${config_values[11]:-0}"
num_demos="${config_values[12]:-}"
enable_cameras="${config_values[13]:-0}"
multi_view="${config_values[14]:-0}"
quality="${config_values[15]:-0}"
enable_task1_visualization="${config_values[16]:-0}"
left_disabled="${config_values[17]:-0}"
port="${config_values[18]:-}"
left_arm_port="${config_values[19]:-}"
right_arm_port="${config_values[20]:-}"
recalibrate="${config_values[21]:-0}"

case "$teleop_mode" in
    xtrainerleader|xtrainer_vr|bi_keyboard) ;;
    *)
        echo "配置错误：teleop_mode 仅支持 xtrainerleader、xtrainer_vr 或 bi_keyboard，当前为：$teleop_mode" >&2
        exit 2
        ;;
esac

case "$task" in
    task1|task3) ;;
    *)
        echo "配置错误：选手包仅支持 task1 或 task3，当前为：$task" >&2
        exit 2
        ;;
esac

case "$dataset_layout" in
    per_episode|single_file) ;;
    *)
        echo "配置错误：dataset_layout 仅支持 per_episode 或 single_file。" >&2
        exit 2
        ;;
esac

[[ "$num_envs" =~ ^[1-9][0-9]*$ ]] || { echo "配置错误：num_envs 必须是正整数。" >&2; exit 2; }
[[ "$num_demos" =~ ^[0-9]+$ ]] || { echo "配置错误：num_demos 必须是非负整数。" >&2; exit 2; }
if [[ "$record" == "1" && "$dataset_layout" == "per_episode" && "$num_envs" != "1" ]]; then
    echo "配置错误：per_episode 数采要求 num_envs = 1。" >&2
    exit 2
fi

teleop_script="$project_root/scripts/environments/teleoperation/teleop_se3_agent.py"
scene_file="$project_root/assets/scenes/$task/training_env.usd"
if [[ "$dataset_root" != /* ]]; then
    dataset_root="$project_root/${dataset_root#./}"
fi
dataset_file="${dataset_root%/}/$task/dataset.hdf5"

[[ -f "$teleop_script" ]] || { echo "遥操入口不存在：$teleop_script" >&2; exit 2; }
[[ -f "$scene_file" ]] || { echo "场景资产不存在：$scene_file" >&2; exit 2; }

teleop_args=(
    "$teleop_script"
    "--task=$task"
    "--teleop_device=$teleop_mode"
    "--num_envs=$num_envs"
    "--device=$device"
    "--step_hz=$step_hz"
    "--sensitivity=$sensitivity"
    "--dataset_layout=$dataset_layout"
    "--dataset_file=$dataset_file"
    "--num_demos=$num_demos"
    "--port=$port"
    "--left_arm_port=$left_arm_port"
    "--right_arm_port=$right_arm_port"
)

[[ -n "$seed" ]] && teleop_args+=("--seed=$seed")
[[ "$record" == "1" ]] && teleop_args+=("--record")
[[ "$resume" == "1" ]] && teleop_args+=("--resume")
[[ "$enable_cameras" == "1" ]] && teleop_args+=("--enable_cameras")
[[ "$multi_view" == "1" ]] && teleop_args+=("--multi_view")
[[ "$quality" == "1" ]] && teleop_args+=("--quality")
[[ "$left_disabled" == "1" ]] && teleop_args+=("--left_disabled")
[[ "$recalibrate" == "1" ]] && teleop_args+=("--recalibrate")

if [[ "$task" == "task1" && "$enable_task1_visualization" == "1" ]]; then
    teleop_args+=("--enable_visualization")
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$conda_env" ]]; then
    command=(python "${teleop_args[@]}")
else
    conda_cmd="$(command -v conda || true)"
    [[ -n "$conda_cmd" ]] || {
        echo "未找到 conda，请先激活 $conda_env 环境。" >&2
        exit 2
    }
    command=("$conda_cmd" run --no-capture-output -n "$conda_env" python "${teleop_args[@]}")
fi

export PYTHONPATH="$project_root/source/leisaac${PYTHONPATH:+:$PYTHONPATH}"
export LEISAAC_ASSETS_ROOT="$project_root/assets"
export PYTHONUNBUFFERED=1

echo "任务：$task"
echo "遥操方式：$teleop_mode"
echo "是否录制：$([[ "$record" == "1" ]] && echo 是 || echo 否)"
echo "数据目录：$(dirname -- "$dataset_file")"
printf '启动命令：'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${1:-}" == "--dry-run" ]]; then
    exit 0
fi

if [[ "$record" == "1" ]]; then
    mkdir -p "$(dirname -- "$dataset_file")"
fi
exec "${command[@]}"
