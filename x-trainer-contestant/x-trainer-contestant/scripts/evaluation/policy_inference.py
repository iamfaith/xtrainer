"""Run a policy in the X-Trainer simulation without computing scores."""

import argparse
import multiprocessing
import time

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visual policy inference without scoring.")
parser.add_argument(
    "--task",
    choices=("task1", "task3"),
    required=True,
    help="Competition simulation task name.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Maximum duration of one inference episode.")
parser.add_argument("--inference_rounds", type=int, default=1, help="Number of inference episodes; 0 runs until the window closes.")
parser.add_argument("--policy_type", type=str, default="lerobot-pi05", help="Policy adapter, for example lerobot-pi05 or xtrainer_act.")
parser.add_argument("--policy_host", type=str, default="127.0.0.1", help="Policy server host.")
parser.add_argument("--policy_port", type=int, default=5555, help="Policy server port.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Policy server timeout in milliseconds.")
parser.add_argument("--policy_action_horizon", type=int, default=24, help="Number of actions executed from each policy chunk.")
parser.add_argument("--policy_language_instruction", type=str, required=True, help="Task instruction passed to the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, required=True, help="Checkpoint path visible to the policy server.")
parser.add_argument("--enable_visualization", action="store_true", help="Show optional task debug markers when supported.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import omni
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401
import leisaac.tasks  # noqa: F401
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type


class RateLimiter:
    def __init__(self, hz: int):
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env: ManagerBasedRLEnv) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        while self.last_time < time.time():
            self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        self.reset_state = False

    def close(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self) -> None:
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "R":
            self.reset_state = True
        return True


def preprocess_observation(observation: dict, model_type: str, instruction: str) -> dict:
    if model_type not in {"gr00tn1.5", "lerobot", "openpi", "xtrainer_act"}:
        raise ValueError(f"Policy model type {model_type} not supported")
    observation["task_description"] = instruction
    return observation


def cleanup_policy(policy) -> None:
    if policy is None:
        return
    channel = getattr(policy, "channel", None)
    if channel is not None:
        try:
            channel.close()
        except Exception:
            pass


def create_policy(env: ManagerBasedRLEnv, task_type: str):
    from isaaclab.sensors import Camera

    camera_infos = {
        key: sensor.image_shape
        for key, sensor in env.scene.sensors.items()
        if isinstance(sensor, Camera)
    }

    if args_cli.policy_type == "gr00tn1.5":
        from leisaac.policy import Gr00tServicePolicyClient

        if task_type != "so101leader":
            raise ValueError(f"Task type {task_type} is not supported by GR00T N1.5")
        return (
            Gr00tServicePolicyClient(
                host=args_cli.policy_host,
                port=args_cli.policy_port,
                timeout_ms=args_cli.policy_timeout_ms,
                camera_keys=list(camera_infos),
                modality_keys=["single_arm", "gripper"],
            ),
            "gr00tn1.5",
        )

    if args_cli.policy_type.startswith("lerobot-"):
        from leisaac.policy import LeRobotServicePolicyClient

        policy_name = args_cli.policy_type.split("-", 1)[1]
        if not policy_name:
            raise ValueError("lerobot policy type must include a model name")
        return (
            LeRobotServicePolicyClient(
                host=args_cli.policy_host,
                port=args_cli.policy_port,
                timeout_ms=args_cli.policy_timeout_ms,
                camera_infos=camera_infos,
                task_type=task_type,
                policy_type=policy_name,
                pretrained_name_or_path=args_cli.policy_checkpoint_path,
                actions_per_chunk=args_cli.policy_action_horizon,
                device=args_cli.device,
            ),
            "lerobot",
        )

    if args_cli.policy_type == "xtrainer_act":
        from leisaac.policy import LeRobotServicePolicyClient

        return (
            LeRobotServicePolicyClient(
                host=args_cli.policy_host,
                port=args_cli.policy_port,
                timeout_ms=args_cli.policy_timeout_ms,
                camera_infos=camera_infos,
                task_type=task_type,
                policy_type="act",
                pretrained_name_or_path=args_cli.policy_checkpoint_path,
                actions_per_chunk=args_cli.policy_action_horizon,
                device=args_cli.device,
            ),
            "xtrainer_act",
        )

    if args_cli.policy_type == "openpi":
        from leisaac.policy import OpenPIServicePolicyClient

        return (
            OpenPIServicePolicyClient(
                host=args_cli.policy_host,
                port=args_cli.policy_port,
                camera_keys=list(camera_infos),
                task_type=task_type,
            ),
            "openpi",
        )

    raise ValueError(f"Policy type {args_cli.policy_type} not supported")


def main() -> None:
    if args_cli.step_hz <= 0:
        raise ValueError("step_hz must be positive")
    if args_cli.inference_rounds < 0:
        raise ValueError("inference_rounds must be non-negative")
    if args_cli.episode_length_s <= 0:
        raise ValueError("episode_length_s must be positive")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = "xtrainerleader"
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.recorders = None
    if hasattr(env_cfg, "enable_visualization"):
        env_cfg.enable_visualization = args_cli.enable_visualization
    if args_cli.inference_rounds == 0 and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    controller = Controller()
    rate_limiter = RateLimiter(args_cli.step_hz)
    policy = None

    try:
        observation, _ = env.reset()
        completed_episodes = 0
        print("[Inference] Visual inference only. No score or result file will be produced.")

        while simulation_app.is_running() and (
            args_cli.inference_rounds == 0 or completed_episodes < args_cli.inference_rounds
        ):
            cleanup_policy(policy)
            policy, model_type = create_policy(env, task_type)
            episode_number = completed_episodes + 1
            print(f"[Inference] Starting episode {episode_number}. Press R to reset it.")
            success = False
            timed_out = False
            manual_reset = False

            while simulation_app.is_running():
                if controller.reset_state:
                    controller.reset()
                    manual_reset = True
                    break

                policy_observation = preprocess_observation(
                    observation["policy"], model_type, args_cli.policy_language_instruction
                )
                with torch.inference_mode():
                    actions = policy.get_action(policy_observation).to(env.device)

                for index in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    observation, _, terminated, truncated, _ = env.step(actions[index, :, :])
                    if terminated[0]:
                        success = True
                        break
                    if truncated[0]:
                        timed_out = True
                        break
                    rate_limiter.sleep(env)

                if success or timed_out:
                    break

            if not simulation_app.is_running():
                break
            if manual_reset:
                print(f"[Inference] Episode {episode_number} reset by user; no result recorded.")
                observation, _ = env.reset()
                continue

            completed_episodes += 1
            outcome = "task completed" if success else "time limit reached"
            print(f"[Inference] Episode {episode_number}: {outcome}. No score computed.")
            if args_cli.inference_rounds == 0 or completed_episodes < args_cli.inference_rounds:
                observation, _ = env.reset()
                controller.reset()

        print(f"[Inference] Finished {completed_episodes} episode(s). No score computed.")
    finally:
        cleanup_policy(policy)
        controller.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
