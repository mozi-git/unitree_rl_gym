# 基于NVIDIA CUDA的基础镜像
FROM nvidia/cuda:12.1-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/bin:$PATH

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# 安装MiniConda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# 初始化conda
RUN /opt/conda/bin/conda init bash

# 创建虚拟环境
RUN conda create -n unitree python=3.8 -y

# 激活虚拟环境并安装依赖
SHELL ["/bin/bash", "-c"]
RUN echo "source activate unitree" > ~/.bashrc
ENV PATH /opt/conda/envs/unitree/bin:$PATH

# 安装PyTorch
RUN conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装其他Python依赖
RUN pip install \
    numpy \
    matplotlib \
    tensorboard \
    gym \
    mujoco-py \
    opencv-python \
    scipy \
    pyyaml

# 安装Isaac Gym（需要手动下载并复制到容器中）
# 注意：Isaac Gym需要从NVIDIA官网下载，这里假设已经下载并放在当前目录的isaacgym文件夹中
# 复制当前项目文件
COPY . /workspace/unitree_rl_gym
RUN mv /workspace/unitree_rl_gym/isaacgym /workspace/isaacgym
WORKDIR /workspace/isaacgym/python
RUN pip install -e .
WORKDIR /workspace

# 安装rsl_rl
RUN git clone https://github.com/leggedrobotics/rsl_rl.git \
    && cd rsl_rl \
    && git checkout v1.0.2 \
    && pip install -e . \
    && cd .. 


# 安装unitree_rl_gym
WORKDIR /workspace/unitree_rl_gym
RUN pip install -e .

# 设置LD_LIBRARY_PATH（用于Isaac Gym）
ENV LD_LIBRARY_PATH="/opt/conda/envs/unitree/lib/:$LD_LIBRARY_PATH"

# 设置默认命令
CMD ["/bin/bash"]