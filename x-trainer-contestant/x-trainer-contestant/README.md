# X-Trainer 参赛选手包

本资料包仅包含两个竞赛任务：

- `task1`：将黑色方块放入红色框、白色方块放入蓝色框。
- `task3`：抓取桌面试管并插入试管架。

完整的环境安装、数据采集、训练、模型打包和提交说明见
[参赛选手完整教程](assets/docs/智能协作机器人入场景应用赛参赛选手完整教程.md)。

## 1. 创建环境

在本目录执行：

```bash
bash environments/create_conda_env.sh
```

脚本创建 `Dobot-Sim` 和 `lerobot` 两个 Conda 环境，并自动安装本目录中的
`source/leisaac`。

## 2. 启动仿真遥操

默认配置为 Task1、键盘遥操、CPU 仿真且不录制数据。直接执行：

```bash
bash start_teleop.sh
```

需要切换任务、遥操设备或开启录制时，编辑根目录的 `config.py`。选手包只接受
`task1` 和 `task3`。

## 3. 使用 X-Trainer Leader

首次使用主手必须先识别串口并完成本机标定：

```bash
conda activate Dobot-Sim
python scripts/find_port.py
python scripts/get_offset.py
python scripts/get_offset.py --calibrate --confirm
```

若当前用户没有串口权限，先加入 `dialout` 用户组并重新登录：

```bash
sudo usermod -aG dialout "$USER"
```

标定结果仅适用于当前设备，不要与其他队伍互相复制
`dobot_settings.ini`。纯仿真键盘或 VR 遥操不需要连接真实夹爪。

## 4. 使用 VR

安装 VR 依赖后，将 `config.py` 中的 `teleop_mode` 改为 `xtrainer_vr`：

```bash
python -m pip install -r source/leisaac/leisaac/xtrainer_utils/XLeVR/requirements.txt
bash start_teleop.sh
```

首次启动会在本机生成自签名 TLS 证书。终端会显示 VR 页面地址；在头显浏览器中
打开该地址并允许证书后，连接左右控制器。

## 5. 数据与模型目录

- 资料包不附带示例数据；选手自行采集的原始 HDF5 数据保存到 `datasets/task1/` 或 `datasets/task3/`。
- 数据转换脚本位于 `scripts/convert/`。
- 本地推理的完整 checkpoint 放入 `pretrained_model/`。

仓库不附带训练权重或评分程序。比赛成绩以赛事平台正式评测为准。
