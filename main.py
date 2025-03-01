import argparse
import os
import sys
from video_translator import VideoTranslator

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