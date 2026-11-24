FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Системные библиотеки для open3d / vedo(VTK) / headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl unzip \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libx11-6 libxcb1 libxi6 libxkbcommon0 libglu1-mesa libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Чистые Python-зависимости (кроме PyG-стека и torch)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) PyG-стек под torch 2.3.1 + CUDA 12.1
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-2.3.1+cu121.html \
 && pip install torch-geometric

# 3) Код проекта (включая шим torch_points3d/ и все сети/даталоадеры)
COPY . /app

# Точка входа по умолчанию (переопределим в hf jobs run)
CMD ["python", "Train_Hybrid.py"]