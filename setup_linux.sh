#!/bin/bash

# Setup script for Linux

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
    echo "This script is intended to run on Linux only."
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    NVIDIA_GPU=true
else
    echo "No NVIDIA GPU detected or nvidia-smi not found."
    echo "The application will run on CPU, which may be slow."
    echo "If you have an NVIDIA GPU, please install the latest NVIDIA drivers."
    NVIDIA_GPU=false
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Force uninstall any existing PyTorch
echo "Removing any existing PyTorch installations..."
pip uninstall -y torch torchvision torchaudio

# Install CUDA version of PyTorch
echo "Installing CUDA-enabled PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other requirements..."
pip install librosa numpy moviepy openai-whisper transformers tortoise-tts opencv-python pydub soundfile demucs matplotlib scikit-learn streamlit sentencepiece psutil

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
echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run the application, use: bash run_streamlit.sh" 