from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PACKAGE_ROOT = PROJECT_ROOT / "source" / "leisaac"
sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from leisaac.xtrainer_utils.dobot_control.dynamixel.driver import DynamixelDriver  # noqa: E402
from leisaac.xtrainer_utils.dobot_control.gripper.dobot_gripper import DobotGripper  # noqa: E402
from leisaac.xtrainer_utils.utils.manipulate_utils import (  # noqa: E402
    load_ini_data_gripper,
    load_ini_data_hands,
)

SETTINGS_PATH = (
    PROJECT_PACKAGE_ROOT
    / "leisaac"
    / "xtrainer_utils"
    / "utils"
    / "dobot_config"
    / "dobot_settings.ini"
)


def serial_candidates(explicit_ports: list[str]) -> list[str]:
    if explicit_ports:
        return list(dict.fromkeys(explicit_ports))

    import serial.tools.list_ports

    return list(
        dict.fromkeys(
            port.device
            for port in serial.tools.list_ports.comports()
            if "ACM" in port.device or "USB" in port.device
        )
    )


def save_settings(config: configparser.ConfigParser) -> None:
    with SETTINGS_PATH.open("w", encoding="utf-8") as settings_file:
        config.write(settings_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: identify the two X-Trainer leader hands."
    )
    parser.add_argument(
        "--port",
        action="append",
        default=[],
        help="Only probe this serial port; repeat for multiple ports.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Probe without changing dobot_settings.ini."
    )
    parser.add_argument(
        "--include-grippers",
        action="store_true",
        help="Also require and identify the two separate follower gripper ports.",
    )
    args = parser.parse_args()

    ports = serial_candidates(args.port)
    if not ports:
        raise SystemExit("No ttyACM or ttyUSB serial device was found")

    ini_file, hands = load_ini_data_hands()
    available = ports.copy()
    failures: list[str] = []
    print(f"serial candidates: {ports}")

    for hand_name, hand_config in hands.items():
        found = False
        for port in available.copy():
            for baud_rate in (2000000, 1000000):
                driver = None
                try:
                    driver = DynamixelDriver(
                        ids=hand_config.joint_ids,
                        append_id=hand_config.append_id,
                        port=port,
                        baudrate=baud_rate,
                    )
                    driver.get_joints()
                    print(f"FOUND {hand_name}: port={port} baud={baud_rate}")
                    ini_file.set(hand_name, "port", port)
                    ini_file.set(hand_name, "baud_rate", str(baud_rate))
                    available.remove(port)
                    found = True
                    break
                except Exception as exc:
                    failures.append(f"{hand_name} {port}@{baud_rate}: {exc}")
                finally:
                    if driver is not None:
                        driver.close()
            if found:
                break
        if not found:
            _print_failures(failures)
            raise SystemExit(f"Could not identify {hand_name}")

    if args.include_grippers:
        _, grippers = load_ini_data_gripper()
        for gripper_name, gripper_config in grippers.items():
            found = False
            for port in available.copy():
                gripper = None
                try:
                    gripper = DobotGripper(
                        port=port,
                        servo_pos=gripper_config.pos,
                        id_name=gripper_config.id_name,
                    )
                    print(f"FOUND {gripper_name}: port={port}")
                    ini_file.set(gripper_name, "port", port)
                    available.remove(port)
                    found = True
                    break
                except Exception as exc:
                    failures.append(f"{gripper_name} {port}: {exc}")
                finally:
                    if gripper is not None:
                        gripper.disconnect()
            if not found:
                _print_failures(failures)
                raise SystemExit(f"Could not identify {gripper_name}")

    if not args.no_save:
        save_settings(ini_file)
        print(f"saved: {SETTINGS_PATH}")
    else:
        print("probe complete; configuration was not changed")

    if available:
        print(f"unused serial ports: {available}")


def _print_failures(failures: list[str]) -> None:
    if not failures:
        return
    print("Probe failures:")
    for failure in failures:
        print(f"  - {failure}")


if __name__ == "__main__":
    main()
