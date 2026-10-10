#!/bin/bash

xhost +
docker run --name isaac-lab --entrypoint bash -it --gpus all \
   -e "ACCEPT_EULA=Y" \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY=$DISPLAY \
   --network=host \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /mnt/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v /mnt/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v /mnt/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v /mnt/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v /mnt/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v /mnt/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v /mnt/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v /mnt/docker/isaac-sim/documents:/root/Documents:rw \
   -v /mnt/x-trainer_old:/workspace/xtrainer_leisaac:rw \
   gdpi_isaac-lab
