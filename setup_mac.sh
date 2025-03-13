#!/bin/bash

# Setup script for macOS with Apple Silicon

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script is intended to run on macOS only."
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Generate platform-specific requirements.txt
echo "Generating platform-specific requirements.txt..."
python3 generate_requirements.py

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

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python3 -c "
import torch
import platform
print(f'Platform: {platform.system()} {platform.machine()}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')
    print(f'MPS built: {torch.backends.mps.is_built()}')
else:
    print('MPS not available in this PyTorch build')
"

echo ""
echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To test MPS acceleration, run: python test_mps.py" 