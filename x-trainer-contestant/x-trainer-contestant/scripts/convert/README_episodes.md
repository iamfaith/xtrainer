# X-trainer per-episode HDF5 conversion

`isaaclab_episodes_to_lerobot.py` accepts one HDF5 file or a directory of HDF5
files (recursive, natural numeric ordering). Each successful source demo becomes
one LeRobot episode. The old converters are unchanged.

## Environment

Checked on 2026-09-03: the remote `lerobot` and `dobot-lerobot` Conda environments
both contain LeRobot 0.6.1 from the same editable source. Synthetic two-episode,
three-camera H.264 write/finalize/read tests passed in both environments using
the PyAV decoder. Both environments were missing `h5py`; no packages were
installed or upgraded during these checks.

```bash
conda activate dobot-lerobot
python -m pip install h5py
cd ~/golden_brick_competition/x-trainer-contestant
```

## Read-only preflight

```bash
python scripts/convert/isaaclab_episodes_to_lerobot.py \
  --input ./datasets/task3 \
  --dry-run
```

The reorganized Task3 data passed preflight in the base environment, which
already has h5py: 119 episodes, 56,104 raw frames, 55,509 exported frames after
dropping the first five frames of each episode. Preflight checks all numeric
values and image shapes/dtypes, but does not decode every image.

## Full conversion (run manually)

```bash
python scripts/convert/isaaclab_episodes_to_lerobot.py \
  --input ./datasets/task3 \
  --output ./datasets/lerobot_task3_119 \
  --repo-id local/xtrainer_task3_119 \
  --fps 30 \
  --skip-first-frames 5
```

The output directory must not already exist, and must be outside the source
HDF5 directory. The repo ID is a local dataset identifier; there is no upload.
To test a few episodes yourself, use `--max-episodes 2` and a DIFFERENT output
directory. For a single source file, pass its path to `--input`.

Defaults preserve the old converter's first-five-frame trimming. Change
`--skip-first-frames 0` to keep every recorded frame. State, action and images
are trimmed together, with no deduplication, resampling, or angle correction.
Images are encoded as lossy H.264 videos using CPU streaming encoding, without
intermediate PNGs. The output is LeRobot dataset format v3.0, with 16-dimensional
state/action and `top`, `left_wrist`, `right_wrist` camera features.

The converter writes `conversion_manifest.json` with source mappings and
completion status. If interrupted, the partial output is not a completed
dataset; automatic resume is not supported. Choose a new output directory to
retry. Raw HDF5 files are opened read-only and never removed by the converter.

## Source file organization

The batches were processed in order `17`, `33`, `70`, with ascending original
file numbers within each batch. The known anomalous original file
`33/task3_000034.hdf5` was moved to the desktop trash at the user's request.
The remaining 119 files are directly in `datasets/task3`, named
`task3_000000.hdf5` through `task3_000118.hdf5`.

The original-to-new mapping and deletion record are saved in
`datasets/task3/reindex_manifest_20260903.json`. Renumbering moved existing
files on the same filesystem without rewriting their contents. The three
now-empty batch directories were removed. The anomalous file can be restored
from the remote machine's trash if needed; restoring it does not automatically
put it into the 119-file sequence.

At inspection time the output filesystem had approximately 32 GiB free. Final
encoded size is not known in advance. The converter checks free space before
starting and before each episode; use an output directory on a larger disk
if needed.
