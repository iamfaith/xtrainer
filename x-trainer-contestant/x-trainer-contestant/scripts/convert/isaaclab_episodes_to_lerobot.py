#!/usr/bin/env python3
"""Convert X-trainer HDF5 files/directories to one local LeRobot 0.6.x dataset.

Source HDF5 files are read-only. Every accepted source demo becomes a separate
LeRobot episode. No frame deduplication, resampling, angle correction, or upload.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CAMERAS = ("top", "left_wrist", "right_wrist")
JOINT_NAMES = [f"J{arm}_{joint}.pos" for arm in (1, 2) for joint in range(1, 9)]
TASK3 = "Insert the test tubes on the desktop into the rack."


@dataclass
class SourceEpisode:
    file: str
    source: str
    demo: str
    raw_frames: int
    frames: int
    size_bytes: int
    mtime_ns: int


def natural_key(value):
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", str(value))
    )


def require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required in this Python environment. "
            "Install it with: python -m pip install h5py"
        ) from exc
    return h5py


def collect_episodes(input_path, skip_first, min_frames, excludes=()):
    """Preflight ALL eligible demos before creating an output dataset."""
    h5py = require_h5py()
    root = Path(input_path).expanduser().resolve(strict=True)
    if root.is_file():
        files, base = [root], root.parent
    elif root.is_dir():
        files = sorted(root.rglob("*.hdf5"), key=lambda p: natural_key(p.relative_to(root)))
        base = root
    else:
        raise ValueError(f"Not a regular file/directory: {root}")
    if not files:
        raise ValueError(f"No HDF5 files found in {root}")
    exclusions = {Path(p).as_posix() for p in excludes}
    candidates = {p.relative_to(base).as_posix() for p in files}
    if exclusions - candidates:
        raise ValueError(f"Excluded paths not found: {sorted(exclusions - candidates)}")
    episodes, skipped = [], []
    numeric_shapes = {
        "actions": (16,),
        "obs/left_joint_pos_rel": (8,),
        "obs/right_joint_pos_rel": (8,),
    }
    for path in files:
        relative = path.relative_to(base).as_posix()
        if relative in exclusions:
            skipped.append({"source": relative, "reason": "explicitly excluded"})
            continue
        stat = path.stat()
        with h5py.File(path, "r") as file:
            if "data" not in file or not isinstance(file["data"], h5py.Group):
                raise ValueError(f"{path}: missing /data group")
            for demo_name in sorted(file["data"], key=natural_key):
                group = file["data"][demo_name]
                label = f"{relative}/{demo_name}"
                if not isinstance(group, h5py.Group):
                    raise ValueError(f"{label}: expected an episode group")
                success = group.attrs.get("success")
                if success is None or not isinstance(success, (bool, int, np.bool_, np.integer)) or success != 1:
                    skipped.append({"source": label, "reason": "failed or unfinished (success != True)"})
                    continue
                if "actions" not in group:
                    raise ValueError(f"{label}: missing actions")
                n = len(group["actions"])
                if n < min_frames or n <= skip_first:
                    skipped.append({"source": label, "reason": f"too short ({n} frames)"})
                    continue
                if group.attrs.get("num_samples") != n:
                    raise ValueError(f"{label}: num_samples does not match actions")
                for key, tail in numeric_shapes.items():
                    if key not in group:
                        raise ValueError(f"{label}: missing {key}")
                    ds = group[key]
                    if ds.shape != (n, *tail) or ds.dtype != np.dtype("float32"):
                        raise ValueError(f"{label}/{key}: expected float32 {(n, *tail)}, got {ds.dtype} {ds.shape}")
                    if not np.isfinite(ds[()]).all():
                        raise ValueError(f"{label}/{key}: NaN/Inf")
                for camera in CAMERAS:
                    key = f"obs/{camera}"
                    if key not in group:
                        raise ValueError(f"{label}: missing {key}")
                    ds = group[key]
                    if ds.shape != (n, 480, 640, 3) or ds.dtype != np.dtype("uint8"):
                        raise ValueError(f"{label}/{key}: expected uint8 {(n, 480, 640, 3)}, got {ds.dtype} {ds.shape}")
                episodes.append(SourceEpisode(
                    file=str(path), source=relative, demo=demo_name,
                    raw_frames=n, frames=n - skip_first,
                    size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ))
    if not episodes:
        raise ValueError("No eligible successful episodes found.")
    return episodes, skipped


def features():
    result = {
        "action": {"dtype": "float32", "shape": (16,), "names": JOINT_NAMES.copy()},
        "observation.state": {"dtype": "float32", "shape": (16,), "names": JOINT_NAMES.copy()},
    }
    for camera in CAMERAS:
        result[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def create_writer(output, repo_id, fps, encoder_threads=2):
    """Use the installed 0.6.x streaming writer, without intermediate PNGs."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    required = {"root", "rgb_encoder", "streaming_encoding", "encoder_threads"}
    if not required.issubset(inspect.signature(LeRobotDataset.create).parameters):
        raise RuntimeError("This script requires the LeRobot 0.6.x dataset writer API.")
    if not callable(getattr(LeRobotDataset, "finalize", None)):
        raise RuntimeError("LeRobotDataset.finalize() is required.")
    print(f"LeRobot {importlib.metadata.version('lerobot')}: {inspect.getfile(LeRobotDataset)}", flush=True)
    return LeRobotDataset.create(
        repo_id=repo_id, root=Path(output), fps=fps,
        robot_type="xtrainer_follower", features=features(), use_videos=True,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="fast"),
        streaming_encoding=True, encoder_queue_maxsize=8,
        encoder_threads=encoder_threads, metadata_buffer_size=1,
        batch_encoding_size=1,
    )


def stream_episode(dataset, item, skip_first, batch_size, task):
    h5py = require_h5py()
    path = Path(item.file)
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (item.size_bytes, item.mtime_ns):
        raise RuntimeError(f"Source changed after preflight: {path}")
    written = 0
    with h5py.File(path, "r") as file:
        group = file[f"data/{item.demo}"]
        # Read blocks aligned with the original 100-frame HDF5 chunks.
        for start in range(0, item.raw_frames, batch_size):
            end = min(start + batch_size, item.raw_frames)
            if end <= skip_first:
                continue
            action = group["actions"][start:end]
            state = np.concatenate([
                group["obs/left_joint_pos_rel"][start:end],
                group["obs/right_joint_pos_rel"][start:end],
            ], axis=1)
            images = {camera: group[f"obs/{camera}"][start:end] for camera in CAMERAS}
            for offset in range(max(0, skip_first - start), end - start):
                frame = {
                    "action": action[offset], "observation.state": state[offset],
                    "task": task,
                }
                frame.update({
                    f"observation.images.{camera}": images[camera][offset]
                    for camera in CAMERAS
                })
                dataset.add_frame(frame)
                written += 1
    if written != item.frames:
        raise RuntimeError(f"{item.source}: wrote {written}, expected {item.frames}")
    dataset.save_episode(parallel_encoding=False)


def save_manifest(root, report):
    target = Path(root) / "conversion_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One HDF5 or directory (recursive).")
    parser.add_argument("--output", type=Path, help="New local LeRobot output directory; must not exist.")
    parser.add_argument("--repo-id", default="local/xtrainer_task3")
    parser.add_argument("--task", default=TASK3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip-first-frames", type=int, default=5)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--read-batch-size", type=int, default=100)
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--exclude", action="append", default=[], help="Exact path relative to input directory.")
    parser.add_argument("--dry-run", action="store_true", help="Read-only preflight; no LeRobot environment needed.")
    parser.add_argument("--max-episodes", type=int, help="Optional cap for a user-run conversion test.")
    args = parser.parse_args()
    if args.fps <= 0 or args.read_batch_size <= 0 or args.encoder_threads <= 0 or args.min_frames <= 0:
        parser.error("fps, read-batch-size, encoder-threads and min-frames must be positive.")
    if args.skip_first_frames < 0 or (args.max_episodes is not None and args.max_episodes <= 0):
        parser.error("skip-first-frames must be nonnegative; max-episodes must be positive.")
    if not args.dry_run and args.output is None:
        parser.error("--output is required for conversion.")
    return args


def main():
    args = parse_args()
    items, skipped = collect_episodes(args.input, args.skip_first_frames, args.min_frames, args.exclude)
    if args.max_episodes is not None:
        items = items[:args.max_episodes]
    expected_frames = sum(item.frames for item in items)
    print(f"Selected {len(items)} episodes; {expected_frames} frames after dropping "
          f"the first {args.skip_first_frames} frames of each episode.", flush=True)
    print(f"Skipped {len(skipped)} entries; export FPS={args.fps}; 3 cameras; action/state=16/16.")
    for entry in skipped:
        print("SKIP", entry["source"], entry["reason"])
    if args.dry_run:
        print("DRY_RUN_OK: no images decoded, no files written, no conversion started.")
        return
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    input_root = args.input.expanduser().resolve()
    if output == input_root or (input_root.is_dir() and input_root in output.parents):
        raise ValueError("Keep output outside the source HDF5 directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    print(f"Output free space: {free / 2**30:.1f} GiB. Streaming H.264 encoding, CPU only.", flush=True)
    if free < 2 * 2**30:
        raise RuntimeError("Less than 2 GiB free. Choose another output disk.")
    report = {
        "status": "in_progress", "input": str(input_root), "output": str(output),
        "repo_id": args.repo_id, "task": args.task, "fps": args.fps,
        "skip_first_frames": args.skip_first_frames, "codec": "h264", "crf": 18,
        "expected_episodes": len(items), "expected_frames": expected_frames,
        "completed_episodes": 0, "skipped": skipped,
        "sources": [dict(episode_index=i, **asdict(item)) for i, item in enumerate(items)],
    }
    dataset = create_writer(output, args.repo_id, args.fps, args.encoder_threads)
    save_manifest(output, report)
    try:
        for i, item in enumerate(items):
            if shutil.disk_usage(output).free < 1024**3:
                raise RuntimeError("Less than 1 GiB free; stopping before starting another episode.")
            print(f"[{i+1}/{len(items)}] {item.source}/{item.demo}: {item.frames} frames", flush=True)
            stream_episode(dataset, item, args.skip_first_frames, args.read_batch_size, args.task)
            report["completed_episodes"] = i + 1
            save_manifest(output, report)
        # Required for the 0.6.x parquet and metadata writers.
        dataset.finalize()
        info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
        if (info["total_episodes"], info["total_frames"]) != (len(items), expected_frames):
            raise RuntimeError(f"Output counts mismatch: {info['total_episodes']} / {info['total_frames']}")
        report["status"] = "complete"
        report["lerobot_version"] = importlib.metadata.version("lerobot")
        report["codebase_version"] = info.get("codebase_version")
        save_manifest(output, report)
        print(f"DONE: {len(items)} episodes, {expected_frames} frames -> {output}", flush=True)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        try:
            dataset.finalize()
        except BaseException as close_error:
            report["finalize_error"] = repr(close_error)
        save_manifest(output, report)
        print(f"CONVERSION INCOMPLETE. Sources unchanged. See {output / 'conversion_manifest.json'}", flush=True)
        raise


if __name__ == "__main__":
    main()
