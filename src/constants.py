"""
Constants for audio analysis and mixing
"""

import numpy as np

# Krumhansl-Schmuckler key profiles (cognitive weights for each pitch class)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Key names for display
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}

# Cache version for compatibility
CACHE_VERSION = '1.0'

# Default parameters
DEFAULT_CROSSFADE_DURATION = 8.0  # seconds
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_TARGET_LUFS = -14.0  # EBU R128 standard
DEFAULT_MAX_WORKERS = 4
