import argparse
import os
import sys
import json
from video_translator import VideoTranslator
from utils import get_sync_defaults, setup_cache_directory, clear_cache

def main():
    parser = argparse.ArgumentParser(description='AutoDub v0.5.1 - Video Translator with voice cloning and advanced synchronization')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to output video')
    parser.add_argument('--source-lang', default='it', help='Source language (default: it)')
    parser.add_argument('--target-lang', default='en', help='Target language (default: en)')
    parser.add_argument('--voice-samples', help='Directory containing voice samples for voice cloning')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of voice samples and models')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached data before processing')
    parser.add_argument('--clear-voice-cache', action='store_true', help='Clear only voice cache before processing')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files after processing')
    
    # Add synchronization options
    parser.add_argument('--sync-config', help='Path to JSON file with synchronization configuration')
    parser.add_argument('--max-speed', type=float, help='Maximum speed factor for audio adjustment')
    parser.add_argument('--min-speed', type=float, help='Minimum speed factor for audio adjustment')
    parser.add_argument('--pause-threshold', type=float, help='dB threshold for pause detection')
    parser.add_argument('--min-pause-duration', type=int, help='Minimum pause duration in milliseconds')
    parser.add_argument('--no-adaptive-timing', action='store_true', help='Disable adaptive timing based on language')
    parser.add_argument('--no-preserve-breaks', action='store_true', help='Do not preserve sentence breaks')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)
    
    # Create conversions directory in script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversions_dir = os.path.join(script_dir, "conversions")
    os.makedirs(conversions_dir, exist_ok=True)
    
    # Set up synchronization options
    sync_options = get_sync_defaults()
    
    # Load sync config from file if provided
    if args.sync_config and os.path.exists(args.sync_config):
        try:
            with open(args.sync_config, 'r') as f:
                file_options = json.load(f)
                sync_options.update(file_options)
        except Exception as e:
            print(f"Error loading sync config file: {e}")
    
    # Override with command line arguments
    if args.max_speed is not None:
        sync_options["max_speed_factor"] = args.max_speed
    if args.min_speed is not None:
        sync_options["min_speed_factor"] = args.min_speed
    if args.pause_threshold is not None:
        sync_options["pause_threshold"] = args.pause_threshold
    if args.min_pause_duration is not None:
        sync_options["min_pause_duration"] = args.min_pause_duration
    if args.no_adaptive_timing:
        sync_options["adaptive_timing"] = False
    if args.no_preserve_breaks:
        sync_options["preserve_sentence_breaks"] = False
    
    # Handle cache clearing before creating the VideoTranslator
    cache_dir = setup_cache_directory()
    if args.clear_cache:
        print("Clearing all cache before initialization...")
        clear_cache(cache_dir, voice_only=False)
    elif args.clear_voice_cache:
        print("Clearing voice cache before initialization...")
        clear_cache(cache_dir, voice_only=True)
    
    # Create the video translator
    translator = VideoTranslator(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        voice_samples_dir=args.voice_samples,
        input_video_path=args.input,
        use_cache=not args.no_cache,
        sync_options=sync_options,
        keep_temp=args.keep_temp
    )
    
    # Process the video
    translator.process_video(args.input, args.output)

if __name__ == "__main__":
    main() 