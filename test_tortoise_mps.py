#!/usr/bin/env python3
import torch
import platform
import os
from device_utils import get_optimal_device
from tortoise.api import TextToSpeech
from tortoise_patch import patch_tortoise_for_mps

def test_tortoise_mps():
    """
    Test if Tortoise-TTS can use MPS acceleration.
    """
    print("Testing Tortoise-TTS with MPS acceleration")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    is_mac = platform.system() == "Darwin"
    has_mps = False
    if is_mac:
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            print(f"MPS available: {has_mps}")
    
    if not has_mps:
        print("MPS is not available. Exiting.")
        return
    
    # Apply MPS patches
    patch_tortoise_for_mps()
    
    # Create MPS device
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Initialize Tortoise-TTS with MPS
    print("Initializing Tortoise-TTS with MPS...")
    tts = TextToSpeech(device=device)
    
    # Create a simple tensor on MPS
    print("Creating test tensor on MPS...")
    x = torch.ones(5, device=device)
    print(f"Test tensor: {x}")
    
    print("Tortoise-TTS is configured to use MPS acceleration!")
    print("Note: Full inference tests may take a long time to run.")

if __name__ == "__main__":
    test_tortoise_mps() 