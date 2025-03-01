import os
import torch
import librosa
import numpy as np
import subprocess
from pydub import AudioSegment
from tortoise.api import TextToSpeech
import traceback

def initialize_tts():
    """
    Initialize Tortoise TTS for voice cloning.
    
    Returns:
        Initialized TextToSpeech object
    """
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: No GPU detected. Tortoise-TTS may be very slow on CPU.")
    
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
        print("Using cached voice conditioning latents")
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

def clone_voice(tts, text, voice_samples, conditioning_latents, segment_idx=0):
    """
    Generate audio with cloned voice.
    
    Args:
        tts: TextToSpeech object
        text: Text to synthesize
        voice_samples: Voice samples for cloning
        conditioning_latents: Conditioning latents for voice cloning
        segment_idx: Index of the current segment
        
    Returns:
        Generated audio as numpy array
    """
    print(f"Generating audio for segment {segment_idx + 1}...")
    
    try:
        if conditioning_latents is not None:
            print(f"Using cached voice conditioning for generation...")
            # Use the cached conditioning latents
            gen = tts.tts_with_preset(text, 
                                     conditioning_latents=conditioning_latents,
                                     preset="fast")  # Options: ultra_fast, fast, standard, high_quality
        else:
            print("Using default voice...")
            gen = tts.tts(text, voice_samples=None)
        
        # Get the first result
        audio = gen[0].cpu().numpy()
        return audio
    except Exception as e:
        print(f"Error during audio generation: {e}")
        print(traceback.format_exc())
        
        # Create silent audio as fallback
        print("Creating silent audio as fallback...")
        return np.zeros(int(24000 * 3))  # 3 seconds of silence at 24kHz

def generate_audio_segments(tts, aligned_segments, temp_dir, voice_samples, conditioning_latents):
    """
    Generate audio segments for each translated segment.
    
    Args:
        tts: TextToSpeech object
        aligned_segments: List of aligned segments with translations
        temp_dir: Directory to store temporary files
        voice_samples: Voice samples for cloning
        conditioning_latents: Conditioning latents for voice cloning
        
    Returns:
        List of audio segment information
    """
    print("Generating audio segments...")
    audio_segments = []
    
    for i, segment in enumerate(aligned_segments):
        text = segment["text"]
        start_time = segment["start"]
        end_time = segment["end"]
        duration = end_time - start_time
        
        # Generate audio with cloned voice
        audio = clone_voice(tts, text, voice_samples, conditioning_latents, i)
        
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
            
            # Calculate speed factor for lip sync
            current_duration = len(audio_segment) / 1000.0  # duration in seconds
            speed_factor = current_duration / duration if duration > 0 else 1.0
            
            # Adjust audio speed if necessary
            if abs(speed_factor - 1.0) > 0.1:  # If difference is significant
                if speed_factor > 1.5:  # Limit maximum acceleration
                    speed_factor = 1.5
                    
                # Use ffmpeg to adjust speed while maintaining pitch
                adjusted_path = os.path.join(temp_dir, f"adjusted_{i}.wav")
                result = subprocess.call([
                    "ffmpeg", "-y", "-i", wav_path, 
                    "-filter:a", f"atempo={speed_factor}", 
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
                "duration": len(audio_segment) / 1000.0
            })
        except Exception as e:
            print(f"Error generating audio segment {i}: {e}")
            print(traceback.format_exc())
        
    return audio_segments 