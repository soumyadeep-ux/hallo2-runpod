#!/bin/bash
# Hallo2 RunPod Serverless Worker Startup Script

echo "=============================================="
echo "Starting Hallo2 RunPod Serverless Worker"
echo "=============================================="

# Navigate to app directory
cd /app

# Check if models exist, download if not
MODELS_DIR="/app/hallo2/pretrained_models"
if [ ! -d "$MODELS_DIR/hallo2" ] || [ -z "$(ls -A $MODELS_DIR/hallo2 2>/dev/null)" ]; then
    echo "Models not found, downloading..."
    python /app/download_models.py
else
    echo "Models already present, skipping download"
fi

# Set environment variables
export PYTHONPATH="/app/hallo2:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# Log GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Start the RunPod serverless handler
echo "Starting RunPod handler..."
python /app/handler.py
