#!/bin/bash

# CUDA Fix script for Linux

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
    echo "This script is intended to run on Linux only."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup_linux.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Show current PyTorch status
echo "Current PyTorch status:"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Force uninstall any existing PyTorch
echo "Removing existing PyTorch installation..."
pip uninstall -y torch torchvision torchaudio

# Install CUDA version of PyTorch
echo "Installing CUDA-enabled PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python3 -c "
import torch
import platform
print(f'Platform: {platform.system()} {platform.machine()}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Fix complete! If CUDA is still not available, please check:"
echo "1. You have an NVIDIA GPU"
echo "2. You have installed the latest NVIDIA drivers"
echo "3. Your GPU supports CUDA" 