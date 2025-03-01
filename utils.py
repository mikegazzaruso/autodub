import os
import hashlib
import torch
import warnings
import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

def create_work_directory(input_video_path=None):
    """
    Create a working directory for the translation process.
    
    Args:
        input_video_path: Path to the input video to create a dedicated directory
        
    Returns:
        Path to the created directory
    """
    if input_video_path:
        video_filename = os.path.basename(input_video_path).split('.')[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir_name = f"{video_filename}_{timestamp}"
        
        # Create the directory in the same folder as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, "conversions", work_dir_name)
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Working directory created: {temp_dir}")
    else:
        import tempfile
        temp_dir = tempfile.mkdtemp()
    
    return temp_dir

def setup_cache_directory():
    """
    Set up the cache directory.
    
    Returns:
        Path to the cache directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_voice_cache_path(cache_dir, voice_samples_dir):
    """
    Generate a unique cache path based on the voice samples.
    
    Args:
        cache_dir: Path to the cache directory
        voice_samples_dir: Directory containing voice samples
        
    Returns:
        Path to the voice cache file or None if no voice samples directory
    """
    if not voice_samples_dir:
        return None
    
    # Generate a hash of the voice samples directory content
    hash_obj = hashlib.md5()
    wav_files = sorted([f for f in os.listdir(voice_samples_dir) if f.endswith('.wav')])
    
    for wav_file in wav_files:
        file_path = os.path.join(voice_samples_dir, wav_file)
        file_stat = os.stat(file_path)
        hash_obj.update(f"{wav_file}_{file_stat.st_size}_{file_stat.st_mtime}".encode())
    
    dir_hash = hash_obj.hexdigest()
    return os.path.join(cache_dir, f"voice_latents_{dir_hash}.pt")

def load_cached_voice(cache_path):
    """
    Load cached voice conditioning latents if available.
    
    Args:
        cache_path: Path to the cached voice file
        
    Returns:
        Tuple of (voice_samples, conditioning_latents) or (None, None) if not found
    """
    if not cache_path or not os.path.exists(cache_path):
        return None, None
    
    try:
        print(f"Loading cached voice from {cache_path}")
        cached_data = torch.load(cache_path)
        voice_samples = cached_data.get('voice_samples')
        conditioning_latents = cached_data.get('conditioning_latents')
        return voice_samples, conditioning_latents
    except Exception as e:
        print(f"Error loading cached voice: {e}")
        return None, None

def save_voice_to_cache(cache_path, voice_samples, conditioning_latents):
    """
    Save voice conditioning latents to cache.
    
    Args:
        cache_path: Path to save the cache file
        voice_samples: Voice samples to cache
        conditioning_latents: Conditioning latents to cache
    """
    if not cache_path:
        return
    
    try:
        print(f"Saving voice to cache: {cache_path}")
        cache_data = {
            'voice_samples': voice_samples,
            'conditioning_latents': conditioning_latents
        }
        torch.save(cache_data, cache_path)
    except Exception as e:
        print(f"Error saving voice to cache: {e}")

def clear_cache(cache_dir, voice_only=False):
    """
    Clear the cache directory.
    
    Args:
        cache_dir: Path to the cache directory
        voice_only: Whether to clear only voice cache files
    """
    if voice_only:
        print("Clearing voice cache...")
        voice_cache_files = [f for f in os.listdir(cache_dir) if f.startswith("voice_latents_")]
        for file in voice_cache_files:
            try:
                os.remove(os.path.join(cache_dir, file))
            except Exception as e:
                print(f"Error removing {file}: {e}")
    else:
        print("Clearing all cache...")
        for file in os.listdir(cache_dir):
            try:
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file}: {e}")

def get_language_code_map():
    """
    Get the mapping of language codes for MBart.
    
    Returns:
        Dictionary mapping language codes
    """
    return {
        "it": "it_IT",  # Italian
        "en": "en_XX",  # English
        "fr": "fr_XX",  # French
        "es": "es_XX",  # Spanish
        "de": "de_DE",  # German
        "zh": "zh_CN",  # Chinese
        "ru": "ru_RU",  # Russian
        "ja": "ja_XX",  # Japanese
        "pt": "pt_XX",  # Portuguese
        "ar": "ar_AR",  # Arabic
        "hi": "hi_IN",  # Hindi
        # Add more languages as needed
    } 