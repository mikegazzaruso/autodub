#!/usr/bin/env python3
import platform
import sys

print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Machine: {platform.machine()}")

try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check MPS availability
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        
        # Try to create MPS device
        if torch.backends.mps.is_available():
            try:
                device = torch.device("mps")
                print(f"Successfully created MPS device: {device}")
                
                # Try a simple operation
                x = torch.ones(5, device=device)
                y = x + 1
                print(f"Test tensor on MPS: {y}")
                print("MPS is working correctly!")
            except Exception as e:
                print(f"Error using MPS device: {e}")
    else:
        print("PyTorch was not built with MPS support")
        print("You need PyTorch 1.12 or newer for MPS support")
except ImportError:
    print("PyTorch is not installed")

# Check for Apple Silicon
is_mac = platform.system() == "Darwin"
is_arm = platform.machine() == "arm64"
if is_mac and is_arm:
    print("\nYou are running on Apple Silicon (M1/M2/M3)")
    print("MPS acceleration should be available with the right PyTorch version")
else:
    print("\nYou are NOT running on Apple Silicon")
    print("MPS acceleration is only available on Apple Silicon Macs") 