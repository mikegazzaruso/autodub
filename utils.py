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
    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist. Creating it...")
        os.makedirs(cache_dir, exist_ok=True)
        return
    
    if voice_only:
        print("Clearing voice cache...")
        voice_cache_files = [f for f in os.listdir(cache_dir) if f.startswith("voice_latents_")]
        if not voice_cache_files:
            print("No voice cache files found.")
        for file in voice_cache_files:
            try:
                file_path = os.path.join(cache_dir, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file}: {e}")
    else:
        print("Clearing all cache...")
        files_removed = 0
        for file in os.listdir(cache_dir):
            try:
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_removed += 1
            except Exception as e:
                print(f"Error removing {file}: {e}")
        
        if files_removed > 0:
            print(f"Removed {files_removed} cache files.")
        else:
            print("No cache files found to remove.")

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

def get_language_properties(source_lang=None, target_lang=None):
    """
    Returns a dictionary with language properties useful for synchronization.
    
    Args:
        source_lang (str, optional): Source language code
        target_lang (str, optional): Target language code
        
    Returns:
        dict: Dictionary with language properties
    """
    # Base properties for various languages
    language_properties = {
        'en': {  # English (reference language)
            'mbart_code': 'en_XX',
            'avg_syllable_duration': 200,  # in milliseconds
            'avg_words_per_minute': 150,
            'length_ratio_to_english': 1.0,
            'common_pause_markers': ['.', ',', ';', ':', '!', '?'],
            'speech_rate_factor': 1.0
        },
        'it': {  # Italian
            'mbart_code': 'it_IT',
            'avg_syllable_duration': 210,
            'avg_words_per_minute': 140,
            'length_ratio_to_english': 1.1,  # Italian tends to be longer than English
            'common_pause_markers': ['.', ',', ';', ':', '!', '?'],
            'speech_rate_factor': 0.95
        },
        'fr': {  # French
            'mbart_code': 'fr_XX',
            'avg_syllable_duration': 215,
            'avg_words_per_minute': 135,
            'length_ratio_to_english': 1.15,
            'common_pause_markers': ['.', ',', ';', ':', '!', '?'],
            'speech_rate_factor': 0.9
        },
        'es': {  # Spanish
            'mbart_code': 'es_XX',
            'avg_syllable_duration': 205,
            'avg_words_per_minute': 145,
            'length_ratio_to_english': 1.05,
            'common_pause_markers': ['.', ',', ';', ':', '!', '?', '¡', '¿'],
            'speech_rate_factor': 0.98
        },
        'de': {  # German
            'mbart_code': 'de_DE',
            'avg_syllable_duration': 220,
            'avg_words_per_minute': 130,
            'length_ratio_to_english': 1.2,  # German words tend to be longer
            'common_pause_markers': ['.', ',', ';', ':', '!', '?'],
            'speech_rate_factor': 0.85
        },
        'ja': {  # Japanese
            'mbart_code': 'ja_XX',
            'avg_syllable_duration': 180,
            'avg_words_per_minute': 170,
            'length_ratio_to_english': 0.8,  # Japanese can be more compact
            'common_pause_markers': ['。', '、', '！', '？', '…'],
            'speech_rate_factor': 1.2
        },
        'zh': {  # Chinese
            'mbart_code': 'zh_CN',
            'avg_syllable_duration': 190,
            'avg_words_per_minute': 160,
            'length_ratio_to_english': 0.7,  # Chinese is typically more compact
            'common_pause_markers': ['。', '，', '；', '：', '！', '？'],
            'speech_rate_factor': 1.15
        },
        'ru': {  # Russian
            'mbart_code': 'ru_RU',
            'avg_syllable_duration': 225,
            'avg_words_per_minute': 125,
            'length_ratio_to_english': 1.25,
            'common_pause_markers': ['.', ',', ';', ':', '!', '?'],
            'speech_rate_factor': 0.8
        }
    }
    
    # If no specific languages are provided, return the full dictionary
    if source_lang is None and target_lang is None:
        return language_properties
    
    # If only source language is provided
    if target_lang is None and source_lang in language_properties:
        return language_properties[source_lang]
    
    # If both languages are provided, return a dictionary with comparative properties
    if source_lang in language_properties and target_lang in language_properties:
        source_props = language_properties[source_lang]
        target_props = language_properties[target_lang]
        
        # Calculate relative properties between the two languages
        relative_props = {
            'source_lang': source_lang,
            'target_lang': target_lang,
            'length_ratio': target_props['length_ratio_to_english'] / source_props['length_ratio_to_english'],
            'speech_rate_ratio': target_props['speech_rate_factor'] / source_props['speech_rate_factor'],
            'source': source_props,
            'target': target_props
        }
        
        return relative_props
    
    # Default fallback if languages are not found
    return language_properties.get('en', {})

def get_sync_defaults():
    """
    Get default synchronization parameters.
    
    Returns:
        Dictionary with default sync parameters
    """
    return {
        "max_speed_factor": 1.8,        # Maximum acceleration factor
        "min_speed_factor": 0.7,        # Minimum slowdown factor
        "pause_threshold": -35,         # dB threshold for pause detection
        "min_pause_duration": 100,      # Minimum pause duration in ms
        "sync_tolerance": 300,          # Sync tolerance in ms
        "adaptive_timing": True,        # Use adaptive timing based on language
        "preserve_sentence_breaks": True # Preserve pauses between sentences
    } 