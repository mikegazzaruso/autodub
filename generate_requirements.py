#!/usr/bin/env python3
import platform
import os
import sys

def generate_requirements():
    """
    Generate a platform-specific requirements.txt file.
    - For macOS with Apple Silicon: Use native PyTorch without CUDA
    - For other platforms: Use PyTorch with CUDA support
    """
    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    is_apple_silicon = is_mac and is_arm
    
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
        print("Detected macOS with Apple Silicon")
        lines.append("# Requirements for macOS with Apple Silicon (M1/M2/M3)")
        lines.append("# Using native PyTorch with MPS support")
        lines.append("torch>=1.12.0")
        lines.append("torchaudio>=0.12.0")
    else:
        print("Detected non-Apple Silicon platform")
        lines.append("# Requirements for platforms with NVIDIA GPU")
        lines.append("--extra-index-url https://download.pytorch.org/whl/cu118 # Comment if you don't have Nvidia GPU / CUDA Support")
        lines.append("torch")
        lines.append("torchaudio")
    
    # Add common requirements
    lines.extend(common_requirements)
    
    # Write to requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("\n".join(lines))
    
    print(f"Generated requirements.txt for {'macOS with Apple Silicon' if is_apple_silicon else 'standard platform'}")

if __name__ == "__main__":
    generate_requirements() 