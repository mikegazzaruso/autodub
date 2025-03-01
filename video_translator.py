import os
import time
import json
import traceback
import warnings
import torch

# Import modules
from speech_recognition import load_whisper_model, transcribe_audio
from translation import load_translation_model, align_segments
from voice_synthesis import initialize_tts, prepare_voice_samples, generate_audio_segments
from audio_processing import extract_audio, combine_audio_segments, combine_video_and_audio
from utils import (
    create_work_directory, setup_cache_directory, get_voice_cache_path,
    load_cached_voice, save_voice_to_cache, clear_cache, get_language_code_map
)

# Suppress warnings
warnings.filterwarnings('ignore')

class VideoTranslator:
    def __init__(self, source_lang="it", target_lang="en", voice_samples_dir=None, input_video_path=None, use_cache=True):
        """
        Initializes the video translator.
        
        Args:
            source_lang: Source language (default: Italian)
            target_lang: Target language (default: English)
            voice_samples_dir: Directory containing voice samples for voice cloning
            input_video_path: Path to the input video to create a dedicated directory
            use_cache: Whether to use cached voice and models (default: True)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.voice_samples_dir = voice_samples_dir
        self.use_cache = use_cache
        
        # Create working directory
        self.temp_dir = create_work_directory(input_video_path)
        
        # Setup cache directory
        self.cache_dir = setup_cache_directory()
        
        # Debug for voice samples path
        if voice_samples_dir:
            print(f"Voice samples directory specified: {voice_samples_dir}")
            if os.path.exists(voice_samples_dir):
                print(f"Directory exists!")
                wav_files = [f for f in os.listdir(voice_samples_dir) if f.endswith('.wav')]
                print(f"WAV files found: {len(wav_files)}")
                if wav_files:
                    print(f"Examples: {wav_files[:3]}")
            else:
                print(f"Directory does NOT exist!")
        
        print("Initializing models...")
        
        # Initialize Whisper speech recognition model
        self.transcriber = load_whisper_model(self.cache_dir, self.use_cache)
        
        # Initialize MBart translation model
        self.translator_model, self.translator_tokenizer = load_translation_model()
        
        # Initialize TorToise TTS for voice cloning
        self.tts = initialize_tts()
        
        # Language codes map for MBart
        self.lang_map = get_language_code_map()
        
        # Initialize voice conditioning latents
        self.voice_samples = None
        self.conditioning_latents = None
        
        # Try to load cached voice if available
        if self.use_cache and self.voice_samples_dir:
            self._load_cached_voice()
        
        print("Models loaded successfully.")

    def _load_cached_voice(self):
        """Load cached voice conditioning latents if available."""
        cache_path = get_voice_cache_path(self.cache_dir, self.voice_samples_dir)
        voice_samples, conditioning_latents = load_cached_voice(cache_path)
        
        if voice_samples is not None and conditioning_latents is not None:
            self.voice_samples = voice_samples
            self.conditioning_latents = conditioning_latents
            return True
        
        return False

    def _save_voice_to_cache(self, voice_samples, conditioning_latents):
        """Save voice conditioning latents to cache."""
        if not self.use_cache:
            return
        
        cache_path = get_voice_cache_path(self.cache_dir, self.voice_samples_dir)
        save_voice_to_cache(cache_path, voice_samples, conditioning_latents)
        
        # Store in instance for reuse
        self.voice_samples = voice_samples
        self.conditioning_latents = conditioning_latents

    def clear_cache(self, voice_only=False):
        """Clear the cache directory."""
        clear_cache(self.cache_dir, voice_only)

    def prepare_voice_samples(self):
        """Prepare voice samples for cloning."""
        if self.conditioning_latents is not None and self.voice_samples is not None:
            print("Using cached voice conditioning latents")
            return self.voice_samples, self.conditioning_latents
        
        voice_samples, conditioning_latents = prepare_voice_samples(
            self.tts, self.voice_samples_dir, self.voice_samples, self.conditioning_latents
        )
        
        if voice_samples is not None and conditioning_latents is not None:
            # Save to cache for future use
            self._save_voice_to_cache(voice_samples, conditioning_latents)
        
        return voice_samples, conditioning_latents

    def cleanup(self):
        """Clean up temporary files."""
        print("Cleaning up temporary files...")
        # In this version we keep files in the dedicated directory
        # We don't delete self.temp_dir
        pass

    def process_video(self, video_path, output_path):
        """Process video from start to finish."""
        try:
            start_time = time.time()
            print(f"Starting video processing: {video_path}")
            
            # Prepare voice conditioning latents before processing (only once)
            if self.voice_samples_dir:
                self.prepare_voice_samples()
            
            # Extract audio from video
            audio_path = extract_audio(video_path, self.temp_dir)
            
            # Transcribe audio
            transcription = transcribe_audio(self.transcriber, audio_path, self.source_lang)
            
            # Save transcription to file for reference
            transcription_path = os.path.join(self.temp_dir, "transcription.json")
            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            
            # Align translated segments with original timings
            aligned_segments = align_segments(
                transcription, 
                self.translator_model, 
                self.translator_tokenizer,
                self.lang_map[self.source_lang],
                self.lang_map[self.target_lang]
            )
            
            # Save aligned segments for reference
            aligned_path = os.path.join(self.temp_dir, "aligned_segments.json")
            with open(aligned_path, 'w', encoding='utf-8') as f:
                json.dump(aligned_segments, f, ensure_ascii=False, indent=2)
            
            # Generate audio segments with cloned voice
            audio_segments = generate_audio_segments(
                self.tts, 
                aligned_segments, 
                self.temp_dir,
                self.voice_samples,
                self.conditioning_latents
            )
            
            # Combine audio segments
            final_audio_path = combine_audio_segments(audio_segments, audio_path, self.temp_dir)
            
            # Combine original video with translated audio
            result_path = combine_video_and_audio(video_path, final_audio_path, output_path)
            
            elapsed_time = time.time() - start_time
            print(f"Processing completed in {elapsed_time:.2f} seconds")
            print(f"Translated video saved to: {output_path}")
            
            return result_path
        except Exception as e:
            print(f"Error during video processing: {e}")
            print(traceback.format_exc())
            return None
        finally:
            self.cleanup() 