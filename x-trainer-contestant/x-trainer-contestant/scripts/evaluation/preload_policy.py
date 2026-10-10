"""Load a LeRobot policy before Isaac Sim allocates the full scene."""

import argparse
import pickle

import grpc

from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.transport import services_pb2, services_pb2_grpc


XTRAINER_JOINT_NAMES = [
    *(f"J1_{index}" for index in range(1, 9)),
    *(f"J2_{index}" for index in range(1, 9)),
]


def xtrainer_features() -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": [f"{name}.pos" for name in XTRAINER_JOINT_NAMES],
        }
    }
    for camera in ("left_wrist", "right_wrist", "top"):
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--policy-type", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--actions-per-chunk", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    config = RemotePolicyConfig(
        policy_type=args.policy_type,
        pretrained_name_or_path=args.checkpoint,
        lerobot_features=xtrainer_features(),
        actions_per_chunk=args.actions_per_chunk,
        device=args.device,
    )
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    try:
        stub.Ready(services_pb2.Empty(), timeout=args.timeout_seconds)
        stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(config)),
            timeout=args.timeout_seconds,
        )
    finally:
        channel.close()
    print("POLICY_PRELOAD_READY", flush=True)


if __name__ == "__main__":
    main()
