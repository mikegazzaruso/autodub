import os
import subprocess
import torch
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice
import cv2
from pydub import AudioSegment
import tempfile
import shutil
import time
import argparse
import sys
import datetime
import soundfile as sf
import warnings
import traceback
import json
import hashlib
import pickle
import demucs.separate
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
        
        # Create a dedicated directory in the project root instead of using temp
        if input_video_path:
            video_filename = os.path.basename(input_video_path).split('.')[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir_name = f"{video_filename}_{timestamp}"
            
            # Create the directory in the same folder as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.temp_dir = os.path.join(script_dir, "conversions", work_dir_name)
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"Working directory created: {self.temp_dir}")
        else:
            self.temp_dir = tempfile.mkdtemp()
        
        # Cache directory setup
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(script_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
        self.transcriber = self._load_whisper_model()
        
        # Initialize MBart translation model
        self.translator_model, self.translator_tokenizer = self._load_translation_model()
        
        # Check GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if device.type == 'cpu':
            print("WARNING: No GPU detected. Tortoise-TTS may be very slow on CPU.")
        
        # Initialize TorToise TTS for voice cloning
        self.tts = TextToSpeech(device=device)
        
        # Language codes map for MBart
        self.lang_map = {
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
        
        # Initialize voice conditioning latents
        self.voice_samples = None
        self.conditioning_latents = None
        
        # Try to load cached voice if available
        if self.use_cache and self.voice_samples_dir:
            self._load_cached_voice()
        
        print("Models loaded successfully.")

    def _load_whisper_model(self):
        """Load Whisper model, using cache if available."""
        model_cache_path = os.path.join(self.cache_dir, "whisper_model.pkl")
        
        if self.use_cache and os.path.exists(model_cache_path):
            try:
                print("Loading Whisper model from cache...")
                with open(model_cache_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            except Exception as e:
                print(f"Error loading cached Whisper model: {e}")
                print("Loading fresh model...")
        
        model = whisper.load_model("medium")
        
        # Cache the model
        if self.use_cache:
            try:
                with open(model_cache_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"Failed to cache Whisper model: {e}")
        
        return model

    def _load_translation_model(self):
        """Load MBart translation model, using cache if available."""
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        
        return model, tokenizer

    def _get_voice_cache_path(self):
        """Generate a unique cache path based on the voice samples."""
        if not self.voice_samples_dir:
            return None
        
        # Generate a hash of the voice samples directory content
        hash_obj = hashlib.md5()
        wav_files = sorted([f for f in os.listdir(self.voice_samples_dir) if f.endswith('.wav')])
        
        for wav_file in wav_files:
            file_path = os.path.join(self.voice_samples_dir, wav_file)
            file_stat = os.stat(file_path)
            hash_obj.update(f"{wav_file}_{file_stat.st_size}_{file_stat.st_mtime}".encode())
        
        dir_hash = hash_obj.hexdigest()
        return os.path.join(self.cache_dir, f"voice_latents_{dir_hash}.pt")

    def _load_cached_voice(self):
        """Load cached voice conditioning latents if available."""
        cache_path = self._get_voice_cache_path()
        
        if not cache_path:
            return False
        
        if os.path.exists(cache_path):
            try:
                print(f"Loading cached voice from {cache_path}")
                cached_data = torch.load(cache_path)
                self.voice_samples = cached_data.get('voice_samples')
                self.conditioning_latents = cached_data.get('conditioning_latents')
                return True
            except Exception as e:
                print(f"Error loading cached voice: {e}")
        
        return False

    def _save_voice_to_cache(self, voice_samples, conditioning_latents):
        """Save voice conditioning latents to cache."""
        if not self.use_cache:
            return
        
        cache_path = self._get_voice_cache_path()
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

    def clear_cache(self, voice_only=False):
        """Clear the cache directory."""
        if voice_only:
            print("Clearing voice cache...")
            voice_cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith("voice_latents_")]
            for file in voice_cache_files:
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                except Exception as e:
                    print(f"Error removing {file}: {e}")
        else:
            print("Clearing all cache...")
            for file in os.listdir(self.cache_dir):
                try:
                    file_path = os.path.join(self.cache_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file}: {e}")

    def extract_audio(self, video_path):
        """Extract audio from the video."""
        print("Extracting audio from video...")
        audio_path = os.path.join(self.temp_dir, "extracted_audio.wav")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
        return audio_path

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper."""
        print("Transcribing audio...")
        result = self.transcriber.transcribe(audio_path, language=self.source_lang)
        return result

    def translate_text(self, text):
        """Translate text to the target language using MBart."""
        print("Translating text...")
        self.translator_tokenizer.src_lang = self.lang_map[self.source_lang]
        
        # Tokenize the text
        encoded = self.translator_tokenizer(text, return_tensors="pt")
        
        # Generate translation
        generated_tokens = self.translator_model.generate(
            **encoded,
            forced_bos_token_id=self.translator_tokenizer.lang_code_to_id[self.lang_map[self.target_lang]]
        )
        
        # Decode translation
        translation = self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation

    def align_segments(self, transcription, translation):
        """Align translated segments with original timings."""
        print("Aligning segments...")
        aligned_segments = []
        segments = transcription["segments"]
        
        # Split translation into segments proportional to the originals
        for segment in segments:
            original_text = segment["text"]
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Get segment translation
            translation_segment = self.translate_text(original_text)
            
            aligned_segments.append({
                "start": start_time,
                "end": end_time,
                "text": translation_segment
            })
            
        return aligned_segments

    def prepare_voice_samples(self):
        """Prepare voice samples for cloning."""
        if self.conditioning_latents is not None and self.voice_samples is not None:
            print("Using cached voice conditioning latents")
            return self.voice_samples, self.conditioning_latents
        
        if not self.voice_samples_dir or not os.path.exists(self.voice_samples_dir):
            print("Voice samples directory not found.")
            return None, None
        
        wav_files = [f for f in os.listdir(self.voice_samples_dir) if f.endswith('.wav')]
        if not wav_files:
            print("No WAV files found in voice samples directory.")
            return None, None
        
        print(f"Loading {len(wav_files)} voice samples...")
        
        # Load voice samples
        voice_samples = []
        for wav_file in wav_files:
            file_path = os.path.join(self.voice_samples_dir, wav_file)
            try:
                # Load audio at 24kHz (used by Tortoise)
                audio, sr = librosa.load(file_path, sr=24000, mono=True)
                # Convert to tensor and add batch dimension
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.tts.device)
                voice_samples.append(audio_tensor)
            except Exception as e:
                print(f"Error loading file {wav_file}: {e}")
        
        if not voice_samples:
            print("No valid voice samples loaded.")
            return None, None
        
        # Generate conditioning latents once
        print("Generating voice conditioning latents...")
        try:
            conditioning_latents = self.tts.get_conditioning_latents(voice_samples)
            
            # Save to cache for future use
            self._save_voice_to_cache(voice_samples, conditioning_latents)
            
            # Store in instance for reuse
            self.voice_samples = voice_samples
            self.conditioning_latents = conditioning_latents
            
            return voice_samples, conditioning_latents
        except Exception as e:
            print(f"Error generating voice conditioning latents: {e}")
            print(traceback.format_exc())
            return None, None

    def clone_voice(self, text, segment_idx=0):
        """Generate audio with cloned voice."""
        print(f"Generating audio for segment {segment_idx + 1}...")
        
        try:
            # Prepare voice conditioning if not already done
            if self.conditioning_latents is None:
                voice_samples, conditioning_latents = self.prepare_voice_samples()
            else:
                voice_samples, conditioning_latents = self.voice_samples, self.conditioning_latents
            
            if conditioning_latents is not None:
                print(f"Using cached voice conditioning for generation...")
                # Use the cached conditioning latents
                gen = self.tts.tts_with_preset(text, 
                                              conditioning_latents=conditioning_latents,
                                              preset="fast")  # Options: ultra_fast, fast, standard, high_quality
            else:
                print("Using default voice...")
                gen = self.tts.tts(text, voice_samples=None)
            
            # Get the first result
            audio = gen[0].cpu().numpy()
            return audio
        except Exception as e:
            print(f"Error during audio generation: {e}")
            print(traceback.format_exc())
            
            # Create silent audio as fallback
            print("Creating silent audio as fallback...")
            return np.zeros(int(24000 * 3))  # 3 seconds of silence at 24kHz

    def generate_audio_segments(self, aligned_segments):
        """Generate audio segments for each translated segment."""
        print("Generating audio segments...")
        audio_segments = []
        
        for i, segment in enumerate(aligned_segments):
            text = segment["text"]
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time
            
            # Generate audio with cloned voice
            audio = self.clone_voice(text, i)
            
            # Save audio temporarily using numpy.save which doesn't rely on audio format
            temp_audio_path = os.path.join(self.temp_dir, f"segment_{i}.npy")
            np.save(temp_audio_path, audio)
            
            # Convert numpy array to wav using ffmpeg which is more reliable
            wav_path = os.path.join(self.temp_dir, f"segment_{i}.wav")
            temp_raw_path = os.path.join(self.temp_dir, f"segment_{i}.raw")
            
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
                    adjusted_path = os.path.join(self.temp_dir, f"adjusted_{i}.wav")
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

    def combine_audio_segments(self, audio_segments, original_audio_path):
        """Combine audio segments using AI voice separation."""
        print("Combining audio segments with AI voice separation...")
        
        if not audio_segments:
            print("WARNING: No audio segments to combine.")
            return original_audio_path
        
        # Step 1: Use Demucs to separate voice from music/background in the original audio
        print("Separating voice and background from original audio...")
        
        try:
            # Create a directory for Demucs output
            demucs_output_dir = os.path.join(self.temp_dir, "demucs_output")
            os.makedirs(demucs_output_dir, exist_ok=True)
            
            # Run Demucs separation on original audio
            # First, ensure the original audio is in WAV format (Demucs requires it)
            wav_original = os.path.join(self.temp_dir, "original_for_separation.wav")
            
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
            final_audio_path = os.path.join(self.temp_dir, "final_audio.wav")
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
            final_audio_path = os.path.join(self.temp_dir, "final_audio.wav")
            final_audio.export(final_audio_path, format="wav")
            
            return final_audio_path

    def combine_video_and_audio(self, video_path, audio_path, output_path):
        """Combine original video with translated audio."""
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
            audio_path = self.extract_audio(video_path)
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            # Save transcription to file for reference
            transcription_path = os.path.join(self.temp_dir, "transcription.json")
            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            
            # Align translated segments with original timings
            aligned_segments = self.align_segments(transcription, None)
            
            # Save aligned segments for reference
            aligned_path = os.path.join(self.temp_dir, "aligned_segments.json")
            with open(aligned_path, 'w', encoding='utf-8') as f:
                json.dump(aligned_segments, f, ensure_ascii=False, indent=2)
            
            # Generate audio segments with cloned voice
            audio_segments = self.generate_audio_segments(aligned_segments)
            
            # Combine audio segments
            final_audio_path = self.combine_audio_segments(audio_segments, audio_path)
            
            # Combine original video with translated audio
            result_path = self.combine_video_and_audio(video_path, final_audio_path, output_path)
            
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

def main():
    parser = argparse.ArgumentParser(description='Video Translator with voice cloning and lip sync')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to output video')
    parser.add_argument('--source-lang', default='it', help='Source language (default: it)')
    parser.add_argument('--target-lang', default='en', help='Target language (default: en)')
    parser.add_argument('--voice-samples', help='Directory containing voice samples for voice cloning')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of voice samples and models')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached data before processing')
    parser.add_argument('--clear-voice-cache', action='store_true', help='Clear only voice cache before processing')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files after processing')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
    
    # Create conversions directory in script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversions_dir = os.path.join(script_dir, "conversions")
    os.makedirs(conversions_dir, exist_ok=True)
    
    # Create the video translator
    translator = VideoTranslator(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        voice_samples_dir=args.voice_samples,
        input_video_path=args.input,
        use_cache=not args.no_cache
    )
    
    # Handle cache clearing if requested
    if args.clear_cache:
        translator.clear_cache(voice_only=False)
    elif args.clear_voice_cache:
        translator.clear_cache(voice_only=True)
    
    # Process the video
    translator.process_video(args.input, args.output)

if __name__ == "__main__":
    main()