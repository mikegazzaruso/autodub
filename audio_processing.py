import os
import subprocess
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import demucs.separate
import traceback
import shutil
from sync_evaluation import detect_speech_pauses

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
    
    try:
        # First try using MoviePy
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
    except Exception as e:
        print(f"MoviePy extraction failed: {str(e)}")
        print("Trying direct FFmpeg extraction...")
        
        # Use FFmpeg directly with increased analyzeduration and probesize
        # This should handle most common formats including iPhone videos
        ffmpeg_cmd = [
            "ffmpeg", 
            "-i", video_path,
            "-analyzeduration", "100000000",  # Increased analyzeduration (100M)
            "-probesize", "100000000",        # Increased probesize (100M)
            "-ac", "1",                       # Convert to mono
            "-ar", "16000",                   # Set sample rate to 16kHz
            "-vn",                            # No video
            "-acodec", "pcm_s16le",           # PCM 16-bit output
            audio_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print("FFmpeg extraction successful")
        except subprocess.CalledProcessError as ffmpeg_error:
            print(f"FFmpeg extraction failed: {ffmpeg_error}")
            
            # Try one more time with map_channel to extract only the first audio stream
            print("Trying to extract only the first audio stream...")
            ffmpeg_cmd = [
                "ffmpeg", 
                "-i", video_path,
                "-map", "0:a:0",              # Map only the first audio stream
                "-analyzeduration", "100000000",
                "-probesize", "100000000",
                "-ac", "1",
                "-ar", "16000",
                "-vn",
                "-acodec", "pcm_s16le",
                audio_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True)
            print("FFmpeg extraction with first audio stream mapping successful")
    
    # Verify the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Failed to extract audio to {audio_path}")
        
    return audio_path

def find_best_segment_position(segment, pauses, original_audio_length, sync_options):
    """
    Find the best position to place an audio segment based on natural pauses.
    
    Args:
        segment: Audio segment information
        pauses: List of detected pauses (start_time, end_time)
        original_audio_length: Length of the original audio in milliseconds
        sync_options: Synchronization options
        
    Returns:
        Best position in milliseconds
    """
    start_ms = int(segment["start"] * 1000)
    end_ms = int(segment["end"] * 1000)
    duration_ms = end_ms - start_ms
    
    # Get sync tolerance from options
    sync_tolerance = sync_options.get("sync_tolerance", 300)  # Default 300ms
    
    # Check if we should preserve sentence breaks
    preserve_breaks = sync_options.get("preserve_sentence_breaks", True)
    
    # If segment is marked as split, be more precise with timing
    if segment.get("is_split", False):
        sync_tolerance = sync_tolerance // 2
    
    # Find the closest pause to the desired start time
    best_position = start_ms
    min_distance = float('inf')
    
    for pause_start, pause_end in pauses:
        pause_start_ms = int(pause_start * 1000)
        pause_end_ms = int(pause_end * 1000)
        
        # Check if pause is within tolerance
        if abs(pause_start_ms - start_ms) <= sync_tolerance:
            # Calculate distance to start time
            distance = abs(pause_start_ms - start_ms)
            
            # If this is a better match than what we've found so far
            if distance < min_distance:
                min_distance = distance
                
                # If we're preserving sentence breaks, use the pause start
                # Otherwise, use the original start time
                if preserve_breaks:
                    best_position = pause_start_ms
                else:
                    best_position = start_ms
    
    # Ensure we don't place audio beyond the original audio length
    if best_position + duration_ms > original_audio_length:
        best_position = max(0, original_audio_length - duration_ms)
    
    return best_position

def combine_audio_segments(audio_segments, original_audio_path, temp_dir, sync_options=None):
    """
    Combine audio segments using AI voice separation.
    
    Args:
        audio_segments: List of audio segment information
        original_audio_path: Path to the original audio file
        temp_dir: Directory to store temporary files
        sync_options: Dictionary with synchronization options
        
    Returns:
        Path to the final combined audio file
    """
    print("Combining audio segments with AI voice separation...")
    
    if not audio_segments:
        print("WARNING: No audio segments to combine.")
        return original_audio_path
    
    # Default sync options if none provided
    if sync_options is None:
        sync_options = {
            "pause_threshold": -35,
            "min_pause_duration": 100,
            "sync_tolerance": 300,
            "preserve_sentence_breaks": True
        }
    
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
        
        # Detect natural pauses in the original audio for better synchronization
        pause_threshold = sync_options.get("pause_threshold", -35)
        min_pause_ms = sync_options.get("min_pause_duration", 100)
        
        print(f"Detecting natural pauses in original audio (threshold: {pause_threshold}dB, min duration: {min_pause_ms}ms)...")
        pauses = detect_speech_pauses(wav_original, threshold_db=pause_threshold, min_pause_ms=min_pause_ms)
        print(f"Detected {len(pauses)} natural pauses in the original audio")
        
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
        
        # Get original audio length in milliseconds
        original_audio_length = len(background_audio)
        
        # Sort segments by start time to ensure proper ordering
        sorted_segments = sorted(audio_segments, key=lambda x: x["start"])
        
        # Add each translated audio segment at the appropriate time
        for i, segment in enumerate(sorted_segments):
            try:
                path = segment["path"]
                if os.path.exists(path):
                    # Find the best position for this segment based on natural pauses
                    best_position = find_best_segment_position(
                        segment, pauses, original_audio_length, sync_options
                    )
                    
                    # Load the audio segment
                    audio = AudioSegment.from_file(path)
                    
                    # Check for overlaps with previous segments
                    if i > 0:
                        prev_segment = sorted_segments[i-1]
                        prev_end = int(prev_segment["end"] * 1000)
                        
                        # If this segment would overlap with the previous one
                        if best_position < prev_end:
                            # If the overlap is small, just place it after the previous segment
                            if prev_end - best_position < 500:  # Less than 500ms overlap
                                best_position = prev_end
                            # Otherwise, try to find a better position
                            else:
                                # Look for a pause after the previous segment
                                for pause_start, pause_end in pauses:
                                    pause_start_ms = int(pause_start * 1000)
                                    if pause_start_ms >= prev_end and pause_start_ms - prev_end < 300:
                                        best_position = pause_start_ms
                                        break
                                else:
                                    # If no suitable pause found, just place after previous
                                    best_position = prev_end
                    
                    # Log the placement
                    original_pos = int(segment["start"] * 1000)
                    shift = best_position - original_pos
                    print(f"Placing segment {i} at {best_position}ms (shifted by {shift}ms from original)")
                    
                    # Overlay translated audio on silence
                    final_audio = final_audio.overlay(audio, position=best_position)
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
                else:
                    print(f"WARNING: Missing audio file: {path}")
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