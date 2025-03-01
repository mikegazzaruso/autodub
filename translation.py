import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import re

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

def estimate_segment_duration(text, language_code, length_ratio=1.0, speech_rate_ratio=1.0):
    """
    Estimate the duration of a text segment based on language properties.
    
    Args:
        text: Text to estimate duration for
        language_code: Language code in MBart format
        length_ratio: Ratio of text length compared to English
        speech_rate_ratio: Ratio of speech rate compared to reference language
        
    Returns:
        Estimated duration in seconds
    """
    # Count words and characters
    words = len(re.findall(r'\b\w+\b', text))
    chars = len(text)
    
    # Base duration calculation (very approximate)
    # Assuming average speaking rate of ~150 words per minute
    word_duration = words * 0.4  # 0.4 seconds per word
    
    # Adjust for language-specific factors
    adjusted_duration = word_duration * length_ratio / speech_rate_ratio
    
    # Ensure minimum duration
    return max(adjusted_duration, 0.5)

def should_split_segment(text, max_duration=5.0, language_code=None, length_ratio=1.0, speech_rate_ratio=1.0):
    """
    Determine if a segment should be split based on estimated duration.
    
    Args:
        text: Text to check
        max_duration: Maximum desired duration in seconds
        language_code: Language code
        length_ratio: Ratio of text length compared to English
        speech_rate_ratio: Ratio of speech rate compared to reference language
        
    Returns:
        Boolean indicating if segment should be split
    """
    estimated_duration = estimate_segment_duration(text, language_code, length_ratio, speech_rate_ratio)
    return estimated_duration > max_duration

def find_split_point(text):
    """
    Find a good point to split a text segment.
    
    Args:
        text: Text to split
        
    Returns:
        Index where text should be split
    """
    # Try to split at sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        # Find the middle sentence boundary
        middle_idx = len(sentences) // 2
        split_point = 0
        for i in range(middle_idx):
            split_point += len(sentences[i]) + 1  # +1 for the space
        return split_point
    
    # If no sentence boundaries, try to split at commas or other punctuation
    punctuation_matches = list(re.finditer(r'[,;:](?=\s)', text))
    if punctuation_matches:
        # Find the punctuation closest to the middle
        middle_idx = len(text) // 2
        closest_match = min(punctuation_matches, key=lambda m: abs(m.start() - middle_idx))
        return closest_match.start() + 1  # +1 to include the punctuation
    
    # If no good punctuation, split at a word boundary near the middle
    words = list(re.finditer(r'\b\w+\b', text))
    if words:
        middle_idx = len(text) // 2
        closest_word = min(words, key=lambda m: abs(m.start() - middle_idx))
        # Split after the word
        return closest_word.end()
    
    # Last resort: split in the middle
    return len(text) // 2

def align_segments(transcription, model, tokenizer, source_lang_code, target_lang_code, 
                  length_ratio=1.0, speech_rate_ratio=1.0):
    """
    Align translated segments with original timings, considering language properties.
    
    Args:
        transcription: Transcription result from Whisper
        model: MBart model
        tokenizer: MBart tokenizer
        source_lang_code: Source language code in MBart format
        target_lang_code: Target language code in MBart format
        length_ratio: Ratio of target language text length to source language
        speech_rate_ratio: Ratio of target language speech rate to source language
        
    Returns:
        List of aligned segments with translations
    """
    print("Aligning segments...")
    aligned_segments = []
    segments = transcription["segments"]
    
    for segment in segments:
        original_text = segment["text"]
        start_time = segment["start"]
        end_time = segment["end"]
        duration = end_time - start_time
        
        # Get segment translation
        translation_segment = translate_text(
            model, tokenizer, original_text, source_lang_code, target_lang_code
        )
        
        # Estimate if the translated segment will be too long
        estimated_duration = estimate_segment_duration(
            translation_segment, 
            target_lang_code, 
            length_ratio, 
            speech_rate_ratio
        )
        
        # If the estimated duration is significantly longer than the original segment
        # and the original segment is long enough, consider splitting
        if estimated_duration > duration * 1.5 and duration > 2.0:
            # Find a good split point
            split_point = find_split_point(translation_segment)
            
            # Split the translation into two parts
            part1 = translation_segment[:split_point].strip()
            part2 = translation_segment[split_point:].strip()
            
            # Calculate the split time proportionally
            split_ratio = len(part1) / (len(part1) + len(part2))
            split_time = start_time + duration * split_ratio
            
            # Add both segments
            aligned_segments.append({
                "start": start_time,
                "end": split_time,
                "text": part1,
                "original_text": original_text,
                "is_split": True
            })
            
            aligned_segments.append({
                "start": split_time,
                "end": end_time,
                "text": part2,
                "original_text": original_text,
                "is_split": True
            })
        else:
            # Add the segment as is
            aligned_segments.append({
                "start": start_time,
                "end": end_time,
                "text": translation_segment,
                "original_text": original_text,
                "is_split": False
            })
    
    return aligned_segments 