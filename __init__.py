"""
AutoDub - Video Translator with voice cloning and AI voice separation

This package provides tools for translating videos with voice cloning.
"""

__version__ = "0.6.0"

from .video_translator import VideoTranslator
from .utils import get_sync_defaults, setup_cache_directory, clear_cache

__all__ = ['VideoTranslator', 'get_sync_defaults', 'setup_cache_directory', 'clear_cache'] 