import os
import subprocess
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import demucs.separate
import traceback
import shutil

def extract_audio(video_path, temp_dir):
    """
    Extract audio from the video.
    
    Args:
        video_path: Path to the video file
        temp_dir: Directory to store temporary files
        
    Returns:
        Path to the extracted audio file
    """
    print("Extracting audio from video...")
    audio_path = os.path.join(temp_dir, "extracted_audio.wav")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
    return audio_path

def combine_audio_segments(audio_segments, original_audio_path, temp_dir):
    """
    Combine audio segments using AI voice separation.
    
    Args:
        audio_segments: List of audio segment information
        original_audio_path: Path to the original audio file
        temp_dir: Directory to store temporary files
        
    Returns:
        Path to the final combined audio file
    """
    print("Combining audio segments with AI voice separation...")
    
    if not audio_segments:
        print("WARNING: No audio segments to combine.")
        return original_audio_path
    
    try:
        # Step 1: Use Demucs to separate voice from music/background in the original audio
        print("Separating voice and background from original audio...")
        
        # Create a directory for Demucs output
        demucs_output_dir = os.path.join(temp_dir, "demucs_output")
        os.makedirs(demucs_output_dir, exist_ok=True)
        
        # Run Demucs separation on original audio
        # First, ensure the original audio is in WAV format (Demucs requires it)
        wav_original = os.path.join(temp_dir, "original_for_separation.wav")
        
        # Convert to wav if it's not already
        if not original_audio_path.lower().endswith('.wav'):
            subprocess.call([
                "ffmpeg", "-y", "-i", original_audio_path, wav_original
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.copy(original_audio_path, wav_original)
        
        # Separate using Demucs (the most advanced model is "htdemucs")
        print("Running Demucs AI separation...")
        
        # Call demucs.separate directly
        demucs.separate.main(["--out", demucs_output_dir, "--two-stems", "vocals", wav_original])
        
        # Find the output directory created by Demucs (it creates a subdirectory based on the model)
        htdemucs_dir = os.path.join(demucs_output_dir, "htdemucs")
        if not os.path.exists(htdemucs_dir):
            # Try other model names
            possible_dirs = [d for d in os.listdir(demucs_output_dir) if os.path.isdir(os.path.join(demucs_output_dir, d))]
            if possible_dirs:
                htdemucs_dir = os.path.join(demucs_output_dir, possible_dirs[0])
            else:
                raise FileNotFoundError("Could not find Demucs output directory")
        
        # Find the audio files created by Demucs
        base_name = os.path.basename(wav_original).split('.')[0]
        no_vocals_path = os.path.join(htdemucs_dir, base_name, "no_vocals.wav")
        vocals_path = os.path.join(htdemucs_dir, base_name, "vocals.wav")
        
        if not (os.path.exists(no_vocals_path) and os.path.exists(vocals_path)):
            raise FileNotFoundError(f"Demucs output files not found. Looking for: {no_vocals_path} and {vocals_path}")
        
        print("Successfully separated voice and background!")
        
        # Step 2: Create the final audio: background + translated speech
        # Load the background (no vocals) track
        background_audio = AudioSegment.from_file(no_vocals_path)
        
        # Create a silent track with the same duration
        final_audio = AudioSegment.silent(duration=len(background_audio))
        
        # Add each translated audio segment at the appropriate time
        for segment in audio_segments:
            try:
                path = segment["path"]
                if os.path.exists(path):
                    start_ms = int(segment["start"] * 1000)
                    audio = AudioSegment.from_file(path)
                    
                    # Overlay translated audio on silence
                    final_audio = final_audio.overlay(audio, position=start_ms)
                else:
                    print(f"WARNING: Missing audio file: {path}")
            except Exception as e:
                print(f"Error during audio segment combination: {e}")
        
        # Combine translated speech with original background
        final_mixed_audio = background_audio.overlay(final_audio)
        
        # Save final audio
        final_audio_path = os.path.join(temp_dir, "final_audio.wav")
        final_mixed_audio.export(final_audio_path, format="wav")
        
        return final_audio_path
    
    except Exception as e:
        print(f"Error in AI voice separation: {e}")
        print(traceback.format_exc())
        
        # Fallback to the previous method
        print("Falling back to basic audio combination...")
        
        # Load original audio
        original_audio = AudioSegment.from_file(original_audio_path)
        
        # Create a silent track with the same duration
        final_audio = AudioSegment.silent(duration=len(original_audio))
        
        # Add translated audio segments
        for segment in audio_segments:
            try:
                path = segment["path"]
                if os.path.exists(path):
                    start_ms = int(segment["start"] * 1000)
                    audio = AudioSegment.from_file(path)
                    final_audio = final_audio.overlay(audio, position=start_ms)
            except Exception as e:
                print(f"Error: {e}")
        
        # Save final audio
        final_audio_path = os.path.join(temp_dir, "final_audio.wav")
        final_audio.export(final_audio_path, format="wav")
        
        return final_audio_path

def combine_video_and_audio(video_path, audio_path, output_path):
    """
    Combine original video with translated audio.
    
    Args:
        video_path: Path to the original video file
        audio_path: Path to the translated audio file
        output_path: Path to save the output video
        
    Returns:
        Path to the output video
    """
    print("Combining video and audio...")
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ]
    subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path 