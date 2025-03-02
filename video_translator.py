import os
import time
import json
import traceback
import warnings
import torch
import librosa

# Import modules
from speech_recognition import load_whisper_model, transcribe_audio
from translation import load_translation_model, align_segments
from voice_synthesis import initialize_tts, prepare_voice_samples, generate_audio_segments
from audio_processing import extract_audio, combine_audio_segments, combine_video_and_audio
from utils import (
    create_work_directory, setup_cache_directory, get_voice_cache_path,
    load_cached_voice, save_voice_to_cache, clear_cache, get_language_code_map,
    get_language_properties, get_sync_defaults
)
from sync_evaluation import evaluate_sync_quality, visualize_sync

# Suppress warnings
warnings.filterwarnings('ignore')

class VideoTranslator:
    def __init__(self, source_lang="it", target_lang="en", voice_samples_dir=None, 
                 input_video_path=None, use_cache=True, sync_options=None, keep_temp=False):
        """
        Initializes the video translator.
        
        Args:
            source_lang: Source language (default: Italian)
            target_lang: Target language (default: English)
            voice_samples_dir: Directory containing voice samples for voice cloning
            input_video_path: Path to the input video to create a dedicated directory
            use_cache: Whether to use cached voice and models (default: True)
            sync_options: Dictionary with synchronization options (optional)
            keep_temp: Whether to keep temporary files after processing (default: False)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.voice_samples_dir = voice_samples_dir
        self.use_cache = use_cache
        self.keep_temp = keep_temp
        
        # Create working directory
        self.temp_dir = create_work_directory(input_video_path)
        
        # Setup cache directory
        self.cache_dir = setup_cache_directory()
        
        # Set synchronization options
        self.sync_options = get_sync_defaults()
        if sync_options:
            self.sync_options.update(sync_options)
        
        # Get language properties
        self.language_properties = get_language_properties()
        
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
        # Only load cached voice if use_cache is True
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
        # If use_cache is False, we should always regenerate the voice samples
        # even if they are already loaded in memory
        if not self.use_cache:
            print("Cache disabled, generating new voice conditioning latents...")
            self.voice_samples = None
            self.conditioning_latents = None
        
        # If we already have the conditioning latents in memory and use_cache is True
        if self.use_cache and self.conditioning_latents is not None and self.voice_samples is not None:
            print("Using voice conditioning latents from memory")
            return self.voice_samples, self.conditioning_latents
        
        voice_samples, conditioning_latents = prepare_voice_samples(
            self.tts, self.voice_samples_dir, self.voice_samples, self.conditioning_latents
        )
        
        if voice_samples is not None and conditioning_latents is not None:
            # Save to cache for future use only if use_cache is True
            if self.use_cache:
                self._save_voice_to_cache(voice_samples, conditioning_latents)
            else:
                # Just store in instance for this run
                print("Storing voice conditioning latents in memory (not in cache)")
                self.voice_samples = voice_samples
                self.conditioning_latents = conditioning_latents
        
        return voice_samples, conditioning_latents

    def cleanup(self):
        """Clean up temporary files."""
        print("Cleaning up temporary files...")
        # In this version we keep files in the dedicated directory
        # We don't delete self.temp_dir
        pass

    def evaluate_sync(self, original_audio_path=None, translated_audio_path=None, aligned_segments=None):
        """
        Valuta la qualità della sincronizzazione tra l'audio originale e quello tradotto.
        
        Args:
            original_audio_path: Percorso dell'audio originale (opzionale)
            translated_audio_path: Percorso dell'audio tradotto (opzionale)
            aligned_segments: Segmenti allineati (opzionale)
            
        Returns:
            dict: Metriche di qualità della sincronizzazione
        """
        try:
            # Usa i percorsi forniti o quelli predefiniti
            if original_audio_path is None:
                original_audio_path = os.path.join(self.temp_dir, "extracted_audio.wav")
            
            if translated_audio_path is None:
                translated_audio_path = os.path.join(self.temp_dir, "final_audio.wav")
            
            if aligned_segments is None:
                # Se non sono forniti segmenti, usa quelli predefiniti o crea un segmento fittizio
                if hasattr(self, 'aligned_segments') and self.aligned_segments:
                    aligned_segments = self.aligned_segments
                else:
                    # Crea un segmento fittizio che copre l'intero audio
                    y, sr = librosa.load(original_audio_path, sr=None)
                    duration = len(y) / sr
                    aligned_segments = [{"start": 0, "end": duration, "text": "Full audio"}]
            
            # Verifica che i file esistano
            if not os.path.exists(original_audio_path):
                print(f"Warning: Original audio file not found: {original_audio_path}")
                return {"error": "Original audio file not found"}
            
            if not os.path.exists(translated_audio_path):
                print(f"Warning: Translated audio file not found: {translated_audio_path}")
                return {"error": "Translated audio file not found"}
            
            # Crea una visualizzazione per il debug
            visualization_path = os.path.join(self.temp_dir, "sync_debug.wav")
            try:
                visualize_sync(original_audio_path, translated_audio_path, visualization_path)
            except Exception as e:
                print(f"Visualization error: {e}")
                visualization_path = None
            
            # Valuta la qualità della sincronizzazione
            sync_metrics = evaluate_sync_quality(aligned_segments, translated_audio_path, original_audio_path)
            
            # Salva le metriche in un file JSON
            metrics_path = os.path.join(self.temp_dir, "sync_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(sync_metrics, f, indent=2)
            
            return sync_metrics
        
        except Exception as e:
            print(f"Error evaluating synchronization: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def process_video(self, video_path, output_path):
        """
        Process a video by extracting audio, transcribing, translating, and generating new audio.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video
        """
        try:
            start_time = time.time()
            print(f"Starting video processing: {video_path}")
            
            # Prepare voice conditioning latents before processing (only once)
            if self.voice_samples_dir:
                self.prepare_voice_samples()
            
            # Extract audio from video
            audio_path = extract_audio(video_path, self.temp_dir)
            
            # Transcribe audio
            print("Transcribing audio...")
            transcription = transcribe_audio(self.transcriber, audio_path, self.source_lang)
            
            # Save transcription to file for reference
            transcription_path = os.path.join(self.temp_dir, "transcription.json")
            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            
            # Get language-specific properties for better synchronization
            lang_props = get_language_properties(self.source_lang, self.target_lang)
            if isinstance(lang_props, dict):
                length_ratio = lang_props.get('length_ratio', 1.0)
                speech_rate_ratio = lang_props.get('speech_rate_ratio', 1.0)
            else:
                # Fallback to default values
                length_ratio = 1.0
                speech_rate_ratio = 1.0
            
            # Align translated segments with original timings
            print("Translating and aligning segments...")
            self.aligned_segments = align_segments(
                transcription, 
                self.translator_model, 
                self.translator_tokenizer,
                self.lang_map[self.source_lang],
                self.lang_map[self.target_lang],
                length_ratio=length_ratio,
                speech_rate_ratio=speech_rate_ratio
            )
            
            # Save aligned segments for reference
            aligned_path = os.path.join(self.temp_dir, "aligned_segments.json")
            with open(aligned_path, 'w', encoding='utf-8') as f:
                json.dump(self.aligned_segments, f, ensure_ascii=False, indent=2)
            
            # Generate audio segments with cloned voice
            print("Generating audio with cloned voice...")
            audio_segments = generate_audio_segments(
                self.tts, 
                self.aligned_segments, 
                self.temp_dir,
                self.voice_samples,
                self.conditioning_latents,
                sync_options=self.sync_options,
                use_cache=self.use_cache
            )
            
            # Combine audio segments
            print("Combining audio segments...")
            final_audio_path = combine_audio_segments(
                audio_segments, 
                audio_path,
                self.temp_dir,
                sync_options=self.sync_options
            )
            
            # Evaluate synchronization quality
            print("Evaluating synchronization quality...")
            try:
                sync_metrics = self.evaluate_sync(
                    original_audio_path=audio_path, 
                    translated_audio_path=final_audio_path, 
                    aligned_segments=self.aligned_segments
                )
                print(f"Synchronization quality: {sync_metrics.get('overall_alignment_score', 0):.2f}/100")
            except Exception as e:
                print(f"Error during synchronization evaluation: {e}")
            
            # Combine original video with translated audio
            print("Creating final video...")
            result_path = combine_video_and_audio(video_path, final_audio_path, output_path)
            
            elapsed_time = time.time() - start_time
            print(f"Processing completed in {elapsed_time:.2f} seconds")
            print(f"Translated video saved to: {output_path}")
            
            # Clean up temporary files unless keep_temp is True
            if not self.keep_temp:
                print("Cleaning up temporary files...")
                for file in os.listdir(self.temp_dir):
                    if file.endswith(('.wav', '.json', '.txt')) and not file.startswith('sync_'):
                        try:
                            os.remove(os.path.join(self.temp_dir, file))
                        except:
                            pass
            
            return result_path
            
        except Exception as e:
            print(f"Error during video processing: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.cleanup() 