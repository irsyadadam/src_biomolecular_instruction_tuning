FROM python:3.10.18

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your entire project (pyproject.toml included)
COPY . /workspace/

# Install PyTorch with CUDA 11.7 support (matching your current setup)
RUN pip install --upgrade pip && \
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Install your project using the existing pyproject.toml
RUN pip install -e .

# Install flash-attention separately
RUN pip install flash-attn==2.5.7 --no-build-isolation

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/workspace"
ENV HF_HOME="/workspace/.cache/huggingface"

EXPOSE 8888 6006 22

CMD ["/bin/bash"]