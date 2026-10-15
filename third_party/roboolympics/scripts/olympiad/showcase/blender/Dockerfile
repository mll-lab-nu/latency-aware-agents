# Viewer-only image for the offline path-traced showcase render (Blender 4.2 LTS, Cycles with OptiX/CUDA).
# Never used to run, verify or observe a match. Build:
#   docker build -t rcb-showcase-blender:4.2.23 scripts/olympiad/showcase/blender
# Run with: docker run --rm --gpus device=0 -e NVIDIA_DRIVER_CAPABILITIES=all ...
FROM docker.io/library/ubuntu:22.04
ARG BLENDER=blender-4.2.23-linux-x64
ARG BLENDER_SHA256=bea0eb3146be13eae6225409a117b215184f41b7f79e799f97cb3abb8f6dc404
RUN apt-get update -qq && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libgl1 libegl1 \
      libglu1-mesa libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/b.tar.xz https://download.blender.org/release/Blender4.2/${BLENDER}.tar.xz \
    && echo "${BLENDER_SHA256}  /tmp/b.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/b.tar.xz -C /opt && mv /opt/${BLENDER} /opt/blender && rm /tmp/b.tar.xz
ENV PATH=/opt/blender:$PATH
WORKDIR /work
