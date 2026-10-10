from __future__ import annotations

import argparse
import configparser
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PACKAGE_ROOT = PROJECT_ROOT / "source" / "leisaac"
sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from leisaac.xtrainer_utils.dobot_control.dynamixel.driver import DynamixelDriver  # noqa: E402
from leisaac.xtrainer_utils.utils.manipulate_utils import load_ini_data_hands  # noqa: E402

SETTINGS_PATH = (
    PROJECT_PACKAGE_ROOT
    / "leisaac"
    / "xtrainer_utils"
    / "utils"
    / "dobot_config"
    / "dobot_settings.ini"
)
HAND_SECTIONS = {"left": "HAND_LEFT", "right": "HAND_RIGHT"}
GRIPPER_IDS = {"HAND_LEFT": 8, "HAND_RIGHT": 18}


def selected_sections(hand: str) -> list[str]:
    if hand == "both":
        return ["HAND_LEFT", "HAND_RIGHT"]
    return [HAND_SECTIONS[hand]]


def format_values(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: inspect or calibrate X-Trainer hand offsets."
    )
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Write calculated offsets. Without this flag the command is read-only.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Require typing CALIBRATE before changing the settings file.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Calculate calibration values without saving."
    )
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if (args.confirm or args.no_save) and not args.calibrate:
        parser.error("--confirm and --no-save require --calibrate")

    ini_file, hands = load_ini_data_hands()
    candidates: dict[str, tuple[list[float], list[int]]] = {}

    for section in selected_sections(args.hand):
        config = hands[section]
        driver = DynamixelDriver(
            ids=[*config.joint_ids, GRIPPER_IDS[section]],
            append_id=config.append_id,
            port=config.port,
            baudrate=config.baud_rate,
        )
        try:
            rows: list[list[float]] = []
            for _ in range(args.samples):
                rows.append(driver.get_joints().tolist())
                time.sleep(0.02)
        finally:
            driver.close()

        averaged_raw = [
            statistics.fmean(row[index] for row in rows) for index in range(7)
        ]
        target = list(config.start_joints)
        offsets = [
            raw - target_value * sign
            for raw, target_value, sign in zip(
                averaged_raw[:6], target, config.joint_signs, strict=True
            )
        ]
        current = [
            (raw - offset) * sign
            for raw, offset, sign in zip(
                averaged_raw[:6], config.joint_offsets, config.joint_signs, strict=True
            )
        ]
        gripper_angle_deg = float(np.rad2deg(averaged_raw[6]))
        gripper_config = [
            GRIPPER_IDS[section],
            int(gripper_angle_deg + 30),
            int(gripper_angle_deg - 0.2),
        ]
        candidates[section] = (offsets, gripper_config)

        print(f"-------------------- {section} --------------------")
        print(f"current calibrated rad: {format_values(current)}")
        print(f"target rad:             {format_values(target)}")
        print(f"loaded offsets:         {format_values(list(config.joint_offsets))}")
        print(f"candidate offsets:      {format_values(offsets)}")
        print(f"candidate gripper:      {gripper_config}")

    if not args.calibrate:
        print("read-only inspection complete; use --calibrate to write these values")
        return

    print("Keep the selected leader hand(s) fixed at the configured start pose.")
    if args.confirm and input("Type CALIBRATE to continue: ").strip() != "CALIBRATE":
        raise SystemExit("Calibration cancelled; no file was changed")

    for section, (offsets, gripper_config) in candidates.items():
        ini_file.set(
            section, "joint_offsets", ", ".join(f"{value:.6f}" for value in offsets)
        )
        ini_file.set(
            section, "gripper_config", ", ".join(str(value) for value in gripper_config)
        )

    if args.no_save:
        print("calibration calculated; configuration was not changed")
        return

    _save_settings(ini_file)
    print(f"saved: {SETTINGS_PATH}")


def _save_settings(config: configparser.ConfigParser) -> None:
    with SETTINGS_PATH.open("w", encoding="utf-8") as settings_file:
        config.write(settings_file)


if __name__ == "__main__":
    main()
