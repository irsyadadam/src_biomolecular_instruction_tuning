# Use CUDA development image that includes nvcc and development tools
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/workspace"
ENV HF_HOME="/workspace/.cache/huggingface"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /workspace

# Install system dependencies including Python 3.10
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

# Use python3.10 and pip3 directly (no symbolic links needed)
# Verify CUDA and nvcc installation
RUN nvcc --version && \
    echo "CUDA_HOME: $CUDA_HOME" && \
    python3.10 --version

# Copy your entire project
COPY . /workspace/

# Install PyTorch first with CUDA 11.8 support (use python3.10 directly)
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install your project
RUN python3.10 -m pip install -e .

# Install flash-attention (should work now with CUDA development tools)
RUN python3.10 -m pip install flash-attn==2.5.7 --no-build-isolation

# Expose ports
EXPOSE 8888 6006 22

CMD ["/bin/bash"]
