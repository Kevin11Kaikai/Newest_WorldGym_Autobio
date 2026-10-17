FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive MUJOCO_GL=egl PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git ffmpeg build-essential libegl1 libgl1 libgles2 libglfw3 libosmesa6 libxext6 libxrender1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
ENV PATH=/root/.local/bin:/opt/worldgym/bin:$PATH
RUN uv python install 3.11.13 && uv venv --python 3.11.13 /opt/worldgym
COPY Scripts/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN uv pip install --python /opt/worldgym/bin/python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 && uv pip install --python /opt/worldgym/bin/python -r /tmp/requirements-gpu.txt
WORKDIR /repo
CMD ["bash"]
