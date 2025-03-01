#!/usr/bin/env python3
"""
Script to run the original monolithic version for comparison.
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run original monolithic video translator')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path to output video')
    parser.add_argument('--source-lang', default='it', help='Source language (default: it)')
    parser.add_argument('--target-lang', default='en', help='Target language (default: en)')
    parser.add_argument('--voice-samples', help='Directory containing voice samples for voice cloning')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of voice samples and models')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached data before processing')
    parser.add_argument('--clear-voice-cache', action='store_true', help='Clear only voice cache before processing')
    
    args = parser.parse_args()
    
    # Import the original monolithic version
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from translate_video import VideoTranslator, main as original_main
        
        # Run the original main function with the provided arguments
        sys.argv = [
            'translate_video.py',
            '--input', args.input,
            '--output', args.output,
            '--source-lang', args.source_lang,
            '--target-lang', args.target_lang
        ]
        
        if args.voice_samples:
            sys.argv.extend(['--voice-samples', args.voice_samples])
        
        if args.no_cache:
            sys.argv.append('--no-cache')
        
        if args.clear_cache:
            sys.argv.append('--clear-cache')
        
        if args.clear_voice_cache:
            sys.argv.append('--clear-voice-cache')
        
        # Run the original main function
        original_main()
        
    except ImportError:
        print("Error: Could not import the original translate_video.py module.")
        print("Make sure the file exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running the original version: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 