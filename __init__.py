"""
Video Translator - A tool for translating videos with voice cloning
"""

from .video_translator import VideoTranslator

__version__ = "0.3.0"

from .utils import get_sync_defaults, get_language_properties
from .sync_evaluation import evaluate_sync_quality, visualize_sync

__all__ = [
    'VideoTranslator',
    'get_sync_defaults',
    'get_language_properties',
    'evaluate_sync_quality',
    'visualize_sync'
] 