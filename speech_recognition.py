import os
import pickle
import whisper

def load_whisper_model(cache_dir, use_cache=True):
    """
    Load Whisper model, using cache if available.
    
    Args:
        cache_dir: Directory to store cached models
        use_cache: Whether to use cached models
        
    Returns:
        Loaded Whisper model
    """
    model_cache_path = os.path.join(cache_dir, "whisper_model.pkl")
    
    if use_cache and os.path.exists(model_cache_path):
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
    if use_cache:
        try:
            with open(model_cache_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Failed to cache Whisper model: {e}")
    
    return model

def transcribe_audio(model, audio_path, language):
    """
    Transcribe audio using Whisper.
    
    Args:
        model: Whisper model
        audio_path: Path to the audio file
        language: Source language code
        
    Returns:
        Transcription result
    """
    print("Transcribing audio...")
    result = model.transcribe(audio_path, language=language)
    return result 