#!/usr/bin/env python3
"""
Simple test script to verify that the modular implementation works correctly.
This script imports all modules and creates a VideoTranslator instance.
"""

import os
import sys

# Test imports
try:
    print("Testing imports...")
    from video_translator import VideoTranslator
    import speech_recognition
    import translation
    import voice_synthesis
    import audio_processing
    import utils
    print("All modules imported successfully!")
    
    # Test VideoTranslator initialization
    print("\nTesting VideoTranslator initialization...")
    translator = VideoTranslator(
        source_lang="en",
        target_lang="it",
        use_cache=True
    )
    print("VideoTranslator initialized successfully!")
    
    print("\nAll tests passed!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1) 