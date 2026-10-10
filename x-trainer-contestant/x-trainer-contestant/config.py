"""遥操数采配置：直接修改下面的变量即可。"""

# 任务与遥操方式
teleop_mode = "bi_keyboard"  # 可选：xtrainerleader、xtrainer_vr、bi_keyboard
task = "task1"  # 可选：task1（物料分拣）、task3（试管插入）

# 运行环境
conda_env = "Dobot-Sim"  # environments/create_conda_env.sh 创建的默认环境名称
device = "cpu"  # 可选：cuda、cuda:0、cpu；RTX 5090 默认使用 cpu
num_envs = 1  # 正整数；per_episode 数采必须为 1
seed = None  # 可选：None（随机种子）或整数，例如 42
step_hz = 30  # 正整数，需与任务控制频率匹配
sensitivity = 1.0  # 正数；主要用于 bi_keyboard，越大动作越快

# 数采设置
record = False  # 可选：True（录制）、False（只遥操不保存）
dataset_root = "./datasets"  # 可填写相对路径或绝对路径
dataset_layout = "per_episode"  # per_episode 或 single_file
resume = False  # single_file 已存在时设 True；per_episode 会自动续接编号
num_demos = 0  # 可选：0（持续采集）或目标成功回合数

# 显示与任务设置
enable_cameras = True  # 是否启用仿真相机；数采建议 True
multi_view = True  # 是否显示顶部、左腕、右腕多视角窗口
quality = False  # 是否启用高质量渲染；开启后显存占用更高
enable_task1_visualization = False  # 是否显示 task1 检测区域；仅 task1 生效
left_disabled = False  # 是否禁用左臂；仅 xtrainerleader 生效

# 遥操设备设置
port = "/dev/ttyACM0"  # 单 SO101 主手串口，当前 task1/task3 不使用
left_arm_port = "/dev/ttyACM0"  # 双 SO101 左臂串口，当前任务不使用
right_arm_port = "/dev/ttyACM1"  # 双 SO101 右臂串口，当前任务不使用
recalibrate = False  # 是否重新标定 SO101，当前 XTrainer 遥操不使用
