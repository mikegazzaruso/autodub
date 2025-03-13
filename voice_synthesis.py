import os
import torch
import librosa
import numpy as np
import subprocess
import math
from pydub import AudioSegment
from tortoise.api import TextToSpeech
import traceback
from device_utils import get_optimal_device, safe_to_device
import platform

# Import and apply MPS patches if on macOS
is_mac = platform.system() == "Darwin"
if is_mac:
    try:
        from tortoise_patch import patch_tortoise_for_mps
        patch_tortoise_for_mps()
    except Exception as e:
        print(f"Warning: Failed to apply MPS patches: {e}")

def initialize_tts():
    """
    Initialize Tortoise TTS for voice cloning.
    
    Returns:
        Initialized TextToSpeech object
    """
    # Get the optimal device based on platform and hardware
    device = get_optimal_device()
    
    # Initialize TorToise TTS
    tts = TextToSpeech(device=device)
    return tts

def prepare_voice_samples(tts, voice_samples_dir, voice_samples=None, conditioning_latents=None):
    """
    Prepare voice samples for cloning.
    
    Args:
        tts: TextToSpeech object
        voice_samples_dir: Directory containing voice samples
        voice_samples: Cached voice samples (optional)
        conditioning_latents: Cached conditioning latents (optional)
        
    Returns:
        Tuple of (voice_samples, conditioning_latents)
    """
    if conditioning_latents is not None and voice_samples is not None:
        print("Using provided voice samples and conditioning latents")
        return voice_samples, conditioning_latents
    
    if not voice_samples_dir or not os.path.exists(voice_samples_dir):
        print("Voice samples directory not found.")
        return None, None
    
    wav_files = [f for f in os.listdir(voice_samples_dir) if f.endswith('.wav')]
    if not wav_files:
        print("No WAV files found in voice samples directory.")
        return None, None
    
    print(f"Loading {len(wav_files)} voice samples...")
    
    # Load voice samples
    voice_samples = []
    for wav_file in wav_files:
        file_path = os.path.join(voice_samples_dir, wav_file)
        try:
            # Load audio at 24kHz (used by Tortoise)
            audio, sr = librosa.load(file_path, sr=24000, mono=True)
            # Convert to tensor and add batch dimension
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(tts.device)
            voice_samples.append(audio_tensor)
        except Exception as e:
            print(f"Error loading file {wav_file}: {e}")
    
    if not voice_samples:
        print("No valid voice samples loaded.")
        return None, None
    
    # Generate conditioning latents once
    print("Generating voice conditioning latents...")
    try:
        conditioning_latents = tts.get_conditioning_latents(voice_samples)
        return voice_samples, conditioning_latents
    except Exception as e:
        print(f"Error generating voice conditioning latents: {e}")
        print(traceback.format_exc())
        return None, None

def clone_voice(tts, text, voice_samples, conditioning_latents, segment_idx=0, use_cache=True):
    """
    Generate audio with cloned voice.
    
    Args:
        tts: TextToSpeech object
        text: Text to synthesize
        voice_samples: Voice samples for cloning
        conditioning_latents: Conditioning latents for voice cloning
        segment_idx: Index of the current segment
        use_cache: Whether to use cached conditioning latents
        
    Returns:
        Generated audio as numpy array
    """
    print(f"Generating audio for segment {segment_idx + 1}...")
    
    try:
        if use_cache and conditioning_latents is not None:
            print(f"Using pre-computed voice conditioning latents for generation...")
            # Use the conditioning latents
            gen = tts.tts_with_preset(text, 
                                     conditioning_latents=conditioning_latents,
                                     preset="fast")  # Options: ultra_fast, fast, standard, high_quality
        elif voice_samples is not None and len(voice_samples) > 0:
            print(f"Using voice samples directly for generation...")
            # Use the voice samples directly
            gen = tts.tts_with_preset(text, 
                                     voice_samples=voice_samples,
                                     preset="fast")
        else:
            print("Using default voice...")
            gen = tts.tts(text, voice_samples=None)
        
        # Get the first result and safely move to CPU
        audio = safe_to_device(gen[0], torch.device('cpu')).numpy()
        return audio
    except Exception as e:
        print(f"Error during audio generation: {e}")
        print(traceback.format_exc())
        return None

def calculate_adaptive_speed_factor(current_duration, target_duration, sync_options):
    """
    Calculate an adaptive speed factor based on the difference between current and target duration.
    
    Args:
        current_duration: Current audio duration in seconds
        target_duration: Target duration in seconds
        sync_options: Synchronization options dictionary
        
    Returns:
        Adjusted speed factor
    """
    # Get limits from sync options
    max_speed = sync_options.get("max_speed_factor", 1.8)
    min_speed = sync_options.get("min_speed_factor", 0.7)
    
    # Calculate raw speed factor
    if target_duration <= 0:
        return 1.0  # Avoid division by zero
    
    raw_factor = current_duration / target_duration
    
    # Apply logarithmic scaling for more natural adjustment
    if raw_factor > 1.0:
        # Acceleration with logarithmic scale to preserve naturalness
        # log base 2 means that doubling the duration results in a factor of 1.5
        adjusted_factor = 1.0 + math.log(raw_factor, 2) * 0.5
        return min(adjusted_factor, max_speed)
    elif raw_factor < 1.0:
        # Deceleration with different approach to avoid extreme slowdown
        # We use a square root function to make the slowdown more gradual
        adjusted_factor = max(min_speed, math.sqrt(raw_factor))
        return adjusted_factor
    else:
        return 1.0  # No adjustment needed

def generate_audio_segments(tts, aligned_segments, temp_dir, voice_samples, conditioning_latents, sync_options=None, use_cache=True):
    """
    Generate audio segments for each translated segment.
    
    Args:
        tts: TextToSpeech object
        aligned_segments: List of aligned segments with translations
        temp_dir: Directory to store temporary files
        voice_samples: Voice samples for cloning
        conditioning_latents: Conditioning latents for voice cloning
        sync_options: Dictionary with synchronization options
        use_cache: Whether to use cached conditioning latents
        
    Returns:
        List of audio segment information
    """
    print("Generating audio segments...")
    audio_segments = []
    
    # Default sync options if none provided
    if sync_options is None:
        sync_options = {
            "max_speed_factor": 1.8,
            "min_speed_factor": 0.7,
            "adaptive_timing": True
        }
    
    for i, segment in enumerate(aligned_segments):
        text = segment["text"]
        start_time = segment["start"]
        end_time = segment["end"]
        duration = end_time - start_time
        is_split = segment.get("is_split", False)
        
        # Generate audio with cloned voice
        audio = clone_voice(tts, text, voice_samples, conditioning_latents, i, use_cache)
        
        # Save audio temporarily using numpy.save which doesn't rely on audio format
        temp_audio_path = os.path.join(temp_dir, f"segment_{i}.npy")
        np.save(temp_audio_path, audio)
        
        # Convert numpy array to wav using ffmpeg which is more reliable
        wav_path = os.path.join(temp_dir, f"segment_{i}.wav")
        temp_raw_path = os.path.join(temp_dir, f"segment_{i}.raw")
        
        try:
            # First save as raw PCM data
            with open(temp_raw_path, 'wb') as f:
                # Convert to 16-bit PCM
                audio_16bit = (audio * 32767).astype(np.int16)
                audio_16bit.tofile(f)
            
            # Then use ffmpeg to convert to proper WAV
            result = subprocess.call([
                "ffmpeg", "-y",
                "-f", "s16le",  # 16-bit signed little endian
                "-ar", "24000",  # 24kHz sample rate
                "-ac", "1",      # 1 channel (mono)
                "-i", temp_raw_path,
                wav_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if result != 0 or not os.path.exists(wav_path):
                print(f"ERROR: Failed to create WAV file for segment {i}")
                continue
                
            # Load audio to adjust speed
            audio_segment = AudioSegment.from_file(wav_path)
            
            # Calculate current duration in seconds
            current_duration = len(audio_segment) / 1000.0  # duration in seconds
            
            # Calculate adaptive speed factor for better synchronization
            speed_factor = calculate_adaptive_speed_factor(current_duration, duration, sync_options)
            
            # For split segments, we might want to be more aggressive with timing
            if is_split:
                # Adjust speed factor more aggressively for split segments
                if speed_factor > 1.0:
                    speed_factor = min(speed_factor * 1.1, sync_options.get("max_speed_factor", 1.8))
            
            # Log the adjustment being made
            print(f"Segment {i}: Duration {current_duration:.2f}s -> Target {duration:.2f}s, Speed factor: {speed_factor:.2f}")
            
            # Adjust audio speed if necessary
            if abs(speed_factor - 1.0) > 0.05:  # If difference is significant (5%)
                # Use ffmpeg to adjust speed while maintaining pitch
                adjusted_path = os.path.join(temp_dir, f"adjusted_{i}.wav")
                
                # Use different filters based on the speed factor
                if speed_factor >= 0.5 and speed_factor <= 2.0:
                    # For moderate adjustments, use atempo
                    filter_complex = f"atempo={speed_factor}"
                    
                    # For larger adjustments, chain multiple atempo filters (atempo only works in range 0.5-2.0)
                    if speed_factor > 2.0:
                        factor1 = min(2.0, speed_factor)
                        factor2 = speed_factor / factor1
                        filter_complex = f"atempo={factor1},atempo={factor2}"
                    elif speed_factor < 0.5:
                        factor1 = max(0.5, speed_factor)
                        factor2 = speed_factor / factor1
                        filter_complex = f"atempo={factor1},atempo={factor2}"
                else:
                    # For extreme adjustments, use rubberband which handles extreme cases better
                    filter_complex = f"asetrate=24000*{speed_factor},aresample=24000"
                
                result = subprocess.call([
                    "ffmpeg", "-y", "-i", wav_path, 
                    "-filter:a", filter_complex, 
                    adjusted_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if result != 0 or not os.path.exists(adjusted_path):
                    print(f"ERROR: Failed to adjust audio speed for segment {i}")
                    # Use original file as fallback
                    adjusted_path = wav_path
                else:
                    # Load adjusted audio
                    audio_segment = AudioSegment.from_file(adjusted_path)
            else:
                # No adjustment needed
                adjusted_path = wav_path
            
            # Clean up temporary raw file
            try:
                if os.path.exists(temp_raw_path):
                    os.remove(temp_raw_path)
            except:
                pass
                
            # Add segment to list
            audio_segments.append({
                "path": adjusted_path,
                "start": start_time,
                "end": end_time,
                "duration": len(audio_segment) / 1000.0,
                "original_duration": duration,
                "speed_factor": speed_factor,
                "is_split": is_split
            })
        except Exception as e:
            print(f"Error generating audio segment {i}: {e}")
            print(traceback.format_exc())
        
    return audio_segments 