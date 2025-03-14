@echo off
echo AutoDub Setup for Windows
echo ========================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is required but not found. Please install Python 3.8 or newer.
    exit /b 1
)

:: Check for NVIDIA GPU
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected!
) else (
    echo No NVIDIA GPU detected or nvidia-smi not found.
    echo The application will run on CPU, which may be slow.
    echo If you have an NVIDIA GPU, please install the latest NVIDIA drivers.
)

:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Force uninstall any existing PyTorch
echo Removing any existing PyTorch installations...
pip uninstall -y torch torchvision torchaudio

:: Install CUDA version of PyTorch
echo Installing CUDA-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install other requirements
echo Installing other requirements...
pip install librosa numpy moviepy openai-whisper transformers tortoise-tts opencv-python pydub soundfile demucs matplotlib scikit-learn streamlit sentencepiece psutil

:: Verify PyTorch installation
echo Verifying PyTorch installation...
python -c "import torch; import platform; print(f'Platform: {platform.system()} {platform.machine()}'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); if torch.cuda.is_available(): print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"

echo.
echo Setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat
echo To run the application, use: run_streamlit.bat

pause 