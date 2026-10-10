from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PACKAGE_ROOT = PROJECT_ROOT / "source" / "leisaac"
sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from leisaac.xtrainer_utils.dobot_control.agents.dobot_agent import DobotAgent  # noqa: E402
from leisaac.xtrainer_utils.utils.manipulate_utils import load_ini_data_hands  # noqa: E402

HAND_SECTIONS = {"left": "HAND_LEFT", "right": "HAND_RIGHT"}


@dataclass
class ButtonAState:
    locked: bool = True
    previous: int | None = None
    pressed_at: float | None = None

    def update(
        self, value: int, now: float, short_press_max_s: float
    ) -> tuple[bool, float] | None:
        value = int(value)
        if self.previous is None:
            self.previous = value
            return None
        if self.previous == 1 and value == 0:
            self.pressed_at = now
        elif self.previous == 0 and value == 1 and self.pressed_at is not None:
            duration = now - self.pressed_at
            self.pressed_at = None
            if duration <= short_press_max_s:
                self.locked = not self.locked
                self.previous = value
                return self.locked, duration
        self.previous = value
        return None


def selected_sections(hand: str) -> list[str]:
    if hand == "both":
        return ["HAND_LEFT", "HAND_RIGHT"]
    return [HAND_SECTIONS[hand]]


def format_six(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:7.1f}" for value in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: inspect leader poses and short-press A to lock or unlock each hand. "
            "This command never connects to a follower or starts Isaac Sim."
        )
    )
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 runs until Ctrl+C.")
    parser.add_argument("--home-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--log-every", type=float, default=0.25)
    parser.add_argument("--short-press-max", type=float, default=0.5)
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must be non-negative")
    if min(args.home_tolerance_deg, args.log_every, args.short_press_max) <= 0:
        parser.error("tolerance and timing values must be positive")

    _, hand_configs = load_ini_data_hands()
    agents: dict[str, DobotAgent] = {}
    buttons: dict[str, ButtonAState] = {}

    try:
        for section in selected_sections(args.hand):
            label = section.removeprefix("HAND_").lower()
            agents[label] = DobotAgent(
                which_hand=label.upper(), dobot_config=hand_configs[section]
            )
            buttons[label] = ButtonAState()
            agents[label].set_torque(True)

        print("Leader hand(s) are LOCKED. Short-press each hand's A button to toggle it.")
        print("Only the X-Trainer leaders are connected; no follower or simulator is controlled.")
        deadline = time.monotonic() + args.seconds if args.seconds else None
        last_log = 0.0

        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            positions: dict[str, list[float]] = {}
            for label, agent in agents.items():
                positions[label] = agent.act({})[:6].tolist()
                keys = agent.get_keys()
                if len(keys) < 1:
                    continue
                event = buttons[label].update(keys[0], now, args.short_press_max)
                if event is not None:
                    locked, duration = event
                    agent.set_torque(locked)
                    status = "LOCKED" if locked else "UNLOCKED"
                    print(f"[A {label} short {duration:.2f}s] leader {status}")

            if now - last_log >= args.log_every:
                for label, current_rad in positions.items():
                    section = HAND_SECTIONS[label]
                    target_rad = list(hand_configs[section].start_joints)
                    current_deg = [math.degrees(value) for value in current_rad]
                    error_deg = [
                        (math.degrees(current - target) + 180.0) % 360.0 - 180.0
                        for current, target in zip(current_rad, target_rad, strict=True)
                    ]
                    max_error = max(abs(value) for value in error_deg)
                    pose_status = "HOME" if max_error <= args.home_tolerance_deg else "ADJUST"
                    lock_status = "LOCKED" if buttons[label].locked else "UNLOCKED"
                    print(
                        f"{label:>5} current_deg={format_six(current_deg)} "
                        f"error_deg={format_six(error_deg)} max_error={max_error:.1f} "
                        f"pose={pose_status} {lock_status}"
                    )
                last_log = now
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for label, agent in agents.items():
            try:
                agent.set_torque(True)
                print(f"{label} leader locked")
            finally:
                agent.close(release_torque=False)


if __name__ == "__main__":
    main()
