@echo off
echo AutoDub CUDA Fix for Windows
echo ===========================

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Please run setup_windows.bat first.
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Show current PyTorch status
echo Current PyTorch status:
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

:: Force uninstall any existing PyTorch
echo Removing existing PyTorch installation...
pip uninstall -y torch torchvision torchaudio

:: Install CUDA version of PyTorch
echo Installing CUDA-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Verify PyTorch installation
echo Verifying PyTorch installation...
python -c "import torch; import platform; print(f'Platform: {platform.system()} {platform.machine()}'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); if torch.cuda.is_available(): print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"

echo.
echo Fix complete! If CUDA is still not available, please check:
echo 1. You have an NVIDIA GPU
echo 2. You have installed the latest NVIDIA drivers
echo 3. Your GPU supports CUDA

pause 