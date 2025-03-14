#!/usr/bin/env python3
import platform
import os
import sys
import subprocess

def generate_requirements():
    """
    Generate a platform-specific requirements.txt file.
    - For macOS with Apple Silicon: Use native PyTorch without CUDA
    - For Windows/Linux with NVIDIA GPU: Use PyTorch with CUDA support
    - For other platforms: Use CPU version of PyTorch
    """
    system = platform.system()
    is_mac = system == "Darwin"
    is_windows = system == "Windows"
    is_linux = system == "Linux"
    is_arm = platform.machine() == "arm64" or platform.machine() == "aarch64"
    is_apple_silicon = is_mac and is_arm
    
    # Check for NVIDIA GPU on Windows/Linux
    has_nvidia_gpu = False
    if is_windows:
        # Check for NVIDIA GPU on Windows using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            has_nvidia_gpu = result.returncode == 0
        except FileNotFoundError:
            has_nvidia_gpu = False
    elif is_linux:
        # Check for NVIDIA GPU on Linux using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            has_nvidia_gpu = result.returncode == 0
        except FileNotFoundError:
            has_nvidia_gpu = False
    
    print(f"Platform: {system} {platform.release()}")
    print(f"Machine architecture: {platform.machine()}")
    print(f"Apple Silicon: {is_apple_silicon}")
    print(f"NVIDIA GPU detected: {has_nvidia_gpu}")
    
    # Common requirements for all platforms
    common_requirements = [
        "librosa",
        "numpy",
        "moviepy",
        "openai-whisper",
        "transformers",
        "tortoise-tts",
        "opencv-python",
        "pydub",
        "soundfile",
        "demucs",
        "matplotlib",
        "scikit-learn",
        "streamlit",
        "sentencepiece",
        "psutil"
    ]
    
    # Create the requirements file content
    lines = []
    
    if is_apple_silicon:
        print("Generating requirements for macOS with Apple Silicon")
        lines.append("# Requirements for macOS with Apple Silicon (M1/M2/M3)")
        lines.append("# Using native PyTorch with MPS support")
        lines.append("torch>=1.12.0")
        lines.append("torchaudio>=0.12.0")
    elif (is_windows or is_linux) and has_nvidia_gpu:
        print("Generating requirements for Windows/Linux with NVIDIA GPU")
        lines.append("# Requirements for Windows/Linux with NVIDIA GPU")
        lines.append("--extra-index-url https://download.pytorch.org/whl/cu118")
        lines.append("torch")
        lines.append("torchaudio")
    else:
        print("Generating requirements for CPU-only platform")
        lines.append("# Requirements for CPU-only platform")
        lines.append("torch")
        lines.append("torchaudio")
    
    # Add common requirements
    lines.extend(common_requirements)
    
    # Write to requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("\n".join(lines))
    
    if is_apple_silicon:
        print("Generated requirements.txt for macOS with Apple Silicon")
    elif (is_windows or is_linux) and has_nvidia_gpu:
        print("Generated requirements.txt for Windows/Linux with NVIDIA GPU")
    else:
        print("Generated requirements.txt for CPU-only platform")

if __name__ == "__main__":
    generate_requirements() 