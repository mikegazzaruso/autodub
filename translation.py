import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_translation_model():
    """
    Load MBart translation model.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    return model, tokenizer

def translate_text(model, tokenizer, text, source_lang_code, target_lang_code):
    """
    Translate text to the target language using MBart.
    
    Args:
        model: MBart model
        tokenizer: MBart tokenizer
        text: Text to translate
        source_lang_code: Source language code in MBart format
        target_lang_code: Target language code in MBart format
        
    Returns:
        Translated text
    """
    print("Translating text...")
    tokenizer.src_lang = source_lang_code
    
    # Tokenize the text
    encoded = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code]
    )
    
    # Decode translation
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

def align_segments(transcription, model, tokenizer, source_lang_code, target_lang_code):
    """
    Align translated segments with original timings.
    
    Args:
        transcription: Transcription result from Whisper
        model: MBart model
        tokenizer: MBart tokenizer
        source_lang_code: Source language code in MBart format
        target_lang_code: Target language code in MBart format
        
    Returns:
        List of aligned segments with translations
    """
    print("Aligning segments...")
    aligned_segments = []
    segments = transcription["segments"]
    
    # Split translation into segments proportional to the originals
    for segment in segments:
        original_text = segment["text"]
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Get segment translation
        translation_segment = translate_text(
            model, tokenizer, original_text, source_lang_code, target_lang_code
        )
        
        aligned_segments.append({
            "start": start_time,
            "end": end_time,
            "text": translation_segment
        })
        
    return aligned_segments 