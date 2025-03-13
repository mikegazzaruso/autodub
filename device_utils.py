import torch
import platform
import os

def get_optimal_device():
    """
    Get the optimal device for PyTorch based on the platform and available hardware.
    
    Returns:
        torch.device: The optimal device (cuda, mps, or cpu)
    """
    # Check if we're on macOS (Darwin)
    is_mac = platform.system() == "Darwin"
    print(f"Running on macOS: {is_mac}")
    print(f"Machine architecture: {platform.machine()}")
    
    # Check for CUDA (NVIDIA GPU)
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    
    # Check for MPS (Apple Silicon)
    has_mps = False
    if is_mac:
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            print(f"MPS available: {has_mps}")
            print(f"MPS built: {torch.backends.mps.is_built()}")
            
            # Print PyTorch version for debugging
            print(f"PyTorch version: {torch.__version__}")
        else:
            print("PyTorch was not built with MPS support")
    
    # Determine the best device
    if has_cuda:
        print("CUDA (NVIDIA GPU) is available. Using GPU acceleration.")
        return torch.device('cuda')
    elif has_mps and is_mac:
        print("MPS (Apple Silicon) is available. Using Metal acceleration.")
        return torch.device('mps')
    else:
        print("No GPU acceleration available. Using CPU.")
        return torch.device('cpu')

def safe_to_device(tensor, device):
    """
    Safely move a tensor to the specified device with fallback to CPU if operation not supported.
    
    Args:
        tensor: PyTorch tensor to move
        device: Target device
        
    Returns:
        The tensor on the target device or CPU if the operation failed
    """
    try:
        return tensor.to(device)
    except Exception as e:
        if device.type == 'mps':
            print(f"Warning: Could not move tensor to MPS, falling back to CPU: {str(e)}")
            return tensor.to('cpu')
        else:
            # Re-raise the exception for non-MPS devices
            raise 