FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace"
ENV HF_HOME="/workspace/.cache/huggingface"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


RUN nvcc --version && \
    echo "CUDA_HOME: $CUDA_HOME" && \
    python3.10 --version

COPY . /workspace/

RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

RUN python3.10 -m pip install -e .

RUN python3.10 -m pip install flash-attn==2.5.7 --no-build-isolation
RUN python3.10 -m pip install torch-geometric==2.3.1
RUN python3.10 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

EXPOSE 8888 6006 22

CMD ["/bin/bash"]
