import torch
import platform
import importlib
import sys
from device_utils import safe_to_device

def patch_tortoise_for_mps():
    """
    Apply patches to Tortoise-TTS for better MPS compatibility.
    This function should be called before initializing Tortoise-TTS if running on macOS.
    """
    # Only apply patches on macOS with MPS
    is_mac = platform.system() == "Darwin"
    has_mps = False
    if is_mac:
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    
    if not (is_mac and has_mps):
        print("Not on macOS with MPS, skipping patches")
        return
    
    print("Applying MPS compatibility patches to Tortoise-TTS...")
    
    # Patch 1: Fix for autocast issues on MPS
    # MPS doesn't support autocast yet, so we need to disable it
    try:
        from tortoise.api import TextToSpeech
        original_tts = TextToSpeech.tts
        
        def patched_tts(self, text, voice_samples=None, conditioning_latents=None, **kwargs):
            # Disable autocast on MPS
            if self.device.type == 'mps':
                # Remove autocast parameter instead of setting it to False
                if 'use_autocast' in kwargs:
                    del kwargs['use_autocast']
            return original_tts(self, text, voice_samples, conditioning_latents, **kwargs)
        
        TextToSpeech.tts = patched_tts
        print("Patched TextToSpeech.tts for MPS compatibility")
    except Exception as e:
        print(f"Failed to patch TextToSpeech.tts: {e}")
    
    # Patch 2: Fix for unsupported operations on MPS
    # Some operations might not be supported on MPS, so we need to move them to CPU
    try:
        from tortoise.models.vocoder import UnivNetGenerator
        original_forward = UnivNetGenerator.forward
        
        def patched_forward(self, x, *args, **kwargs):
            device = x.device
            if device.type == 'mps':
                # Move to CPU for operations that might not be supported on MPS
                x_cpu = x.to('cpu')
                result = original_forward(self, x_cpu, *args, **kwargs)
                # Move back to MPS
                return safe_to_device(result, device)
            else:
                return original_forward(self, x, *args, **kwargs)
        
        UnivNetGenerator.forward = patched_forward
        print("Patched UnivNetGenerator.forward for MPS compatibility")
    except Exception as e:
        print(f"Failed to patch UnivNetGenerator.forward: {e}")
    
    print("MPS compatibility patches applied")

if __name__ == "__main__":
    # Apply patches when run directly
    patch_tortoise_for_mps() 