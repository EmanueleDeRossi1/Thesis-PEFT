#!/bin/bash

# Detect if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" | grep "True"; then
    echo "🔹 CUDA detected! Installing GPU dependencies..."
    pip install -r requirements.txt
    pip install -r requirements-cuda.txt
else
    echo "🔹 No GPU found. Installing CPU dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
fi