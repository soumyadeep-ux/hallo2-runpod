# Hallo2 RunPod Serverless Worker
# Audio-driven portrait animation
#
# Build: docker build -t yourusername/hallo2-runpod:v1 .
# Push:  docker push yourusername/hallo2-runpod:v1

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Clone Hallo2 repository
RUN git clone https://github.com/fudan-generative-vision/hallo2.git /app/hallo2

WORKDIR /app/hallo2

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# Install requirements (excluding torch as it's already installed)
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt || true

# Install Hallo2 requirements
RUN pip install -r requirements.txt || true

# Install additional dependencies for RunPod
RUN pip install \
    runpod \
    boto3 \
    requests \
    huggingface_hub

# Install xformers for memory efficiency
RUN pip install xformers==0.0.25.post1 || true

# Create directories for models and outputs
RUN mkdir -p /app/hallo2/pretrained_models \
    && mkdir -p /app/output \
    && mkdir -p /app/input

# Download models (this is done at build time for faster cold starts)
# Note: This significantly increases image size but reduces startup time
# Alternative: Download models at runtime using start.sh

# Copy handler and startup scripts
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
COPY download_models.py /app/download_models.py

RUN chmod +x /app/start.sh

# Set environment variables for HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV TRANSFORMERS_CACHE=/app/.cache/transformers

# Expose port (not needed for serverless but useful for testing)
EXPOSE 8000

# Start the handler
CMD ["/app/start.sh"]
