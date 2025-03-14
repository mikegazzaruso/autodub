#!/usr/bin/env python3
import torch
import platform
import os
from device_utils import get_optimal_device

def test_cuda():
    """
    Test if CUDA is properly detected and can be used.
    """
    system = platform.system()
    print(f"Platform: {system} {platform.release()}")
    print(f"Machine architecture: {platform.machine()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    
    if has_cuda:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")
        
        # Test CUDA with a simple tensor operation
        try:
            device = torch.device("cuda")
            print(f"Using device: {device}")
            
            # Create a test tensor on CUDA
            x = torch.ones(5, device=device)
            y = x + 1
            print(f"Test tensor on CUDA: {y}")
            print("CUDA is working correctly!")
        except Exception as e:
            print(f"Error using CUDA device: {e}")
    else:
        print("CUDA is not available on this system.")
    
    # Test get_optimal_device function
    print("\nTesting get_optimal_device function:")
    device = get_optimal_device()
    print(f"Optimal device selected: {device}")
    
    # Test tensor operation on the optimal device
    try:
        x = torch.ones(5, device=device)
        y = x + 1
        print(f"Test tensor on {device}: {y}")
        print(f"Device {device} is working correctly!")
    except Exception as e:
        print(f"Error using {device}: {e}")

if __name__ == "__main__":
    test_cuda() 