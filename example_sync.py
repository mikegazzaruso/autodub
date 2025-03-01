#!/usr/bin/env python3
"""
Example script demonstrating the synchronization features of the Video Translator.
This script shows how to use the advanced synchronization options to improve
the alignment between original and translated audio.
"""

import os
import json
from video_translator import VideoTranslator
from utils import get_sync_defaults, get_language_properties
from sync_evaluation import evaluate_sync_quality, visualize_sync

def main():
    # Define input and output paths
    input_video = "syncdir\input_video.mov"
    output_video = "syncdir\output_video_synced.mov"
    
    # Define language pair
    source_lang = "it"
    target_lang = "en"
    
    # Get default synchronization options
    sync_options = get_sync_defaults()
    
    # Customize synchronization options
    sync_options.update({
        "max_speed_factor": 1.3,        # Maximum speed up factor
        "min_speed_factor": 0.8,        # Minimum slow down factor
        "pause_threshold": -35,         # dB threshold for pause detection
        "min_pause_duration": 150,      # Minimum pause duration in ms
        "adaptive_timing": True,        # Use adaptive timing based on language
        "preserve_sentence_breaks": True # Preserve pauses between sentences
    })
    
    # Get language-specific properties
    lang_props = get_language_properties(source_lang, target_lang)
    print(f"Language properties for {source_lang}-{target_lang}:")
    print(f"  Length ratio: {lang_props['length_ratio']}")
    print(f"  Speech rate ratio: {lang_props['speech_rate_ratio']}")
    print(f"  Source language avg words per minute: {lang_props['source']['avg_words_per_minute']}")
    print(f"  Target language avg words per minute: {lang_props['target']['avg_words_per_minute']}")
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found.")
        print("Please place a video file named 'input_video.mp4' in the current directory")
        print("or modify the script to point to your video file.")
        return
    
    # Create the video translator with custom sync options
    translator = VideoTranslator(
        source_lang=source_lang,
        target_lang=target_lang,
        input_video_path=input_video,
        sync_options=sync_options,
        keep_temp=True  # Keep temporary files for debugging
    )
    
    # Process the video
    print(f"\nProcessing video: {input_video}")
    print(f"This may take some time depending on the video length and your hardware...")
    translator.process_video(input_video, output_video)
    
    # Evaluate synchronization quality
    print("\nEvaluating synchronization quality...")
    metrics = translator.evaluate_sync()
    
    # Print synchronization metrics
    print("\nSynchronization Quality Metrics:")
    print(f"  Overall alignment score: {metrics['overall_alignment_score']:.2f}")
    print(f"  Average timing error: {metrics['avg_timing_error']:.2f} ms")
    print(f"  Max timing error: {metrics['max_timing_error']:.2f} ms")
    print(f"  Percentage of well-aligned segments: {metrics['percent_well_aligned']:.1f}%")
    
    # Save metrics to file
    metrics_file = "sync_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nOutput video saved to: {output_video}")
    print(f"Synchronization metrics saved to: {metrics_file}")
    print(f"Debug visualization saved to: {metrics.get('visualization_file', 'No visualization file created')}")

if __name__ == "__main__":
    main() 