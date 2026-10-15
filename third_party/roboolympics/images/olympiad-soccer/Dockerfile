# eai-olympiad-soccer: the Embodied Agent Olympiad's soccer simulator stack. The eai-olympiad stack (MuJoCo 3.3.7 on CPU,
# OSMesa rendering, same pinned Python base) plus
#   * torch 2.8.0 (CPU build) for the Booster T1 walking policy (TorchScript, one thread per process), and
#   * the Booster T1 robot model (MJCF + meshes) of sai-mujoco 0.1.30 (MIT, ArenaX Labs), copied to /opt/sai_t1; the
#     package itself is removed again (the soccer scene is built from these files, no gym environment is used).
# The task mounts its code at /srv (docker-compose: ./sim:/srv:ro). Build locally:
#   docker build -t ghcr.io/mll-lab-nu/eai-olympiad-soccer:$(cat VERSION) images/olympiad-soccer
FROM docker.io/library/python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RCB_T1_ASSETS=/opt/sai_t1
COPY requirements.txt /opt/eai-olympiad-soccer/requirements.txt
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libosmesa6 libgl1 \
    && pip install -r /opt/eai-olympiad-soccer/requirements.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu \
    && pip install --no-deps sai-mujoco==0.1.30 \
    && SAI=$(python -c "import importlib.util as u; print(u.find_spec('sai_mujoco').submodule_search_locations[0])") \
    && mkdir -p /opt/sai_t1 && cp -r $SAI/assets/robots/t1/v1 $SAI/assets/robots/t1/meshes /opt/sai_t1/ \
    && cp $(dirname $SAI)/sai_mujoco-0.1.30.dist-info/licenses/LICENSE /opt/sai_t1/LICENSE \
    && pip uninstall -y sai-mujoco && pip check \
    && rm -rf /var/lib/apt/lists/*
RUN python -c "import mujoco, numpy, torch, yaml, imageio, OpenGL, PIL; print('mujoco', mujoco.__version__, 'torch', torch.__version__)" \
    && python -c "import mujoco; m = mujoco.MjModel.from_xml_path('/opt/sai_t1/v1/robot_lower.xml'); print('T1 lower', m.nq)"
WORKDIR /srv
