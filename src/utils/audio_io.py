"""
Audio I/O and normalization utilities
"""

import librosa
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
from pathlib import Path
from typing import Tuple


def load_audio(file_path: Path, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    """Load audio file at specified sample rate, preserving original channels"""
    y, sr = librosa.load(str(file_path), sr=sample_rate, mono=False)
    # librosa returns (n_channels, n_samples) if multi-channel, else (n_samples,)
    if y.ndim == 1:
        channel_count = 1
    else:
        channel_count = y.shape[0]
        y = y.T  # (n_channels, n_samples) -> (n_samples, n_channels)
    return y, sr, channel_count


def save_audio(file_path: Path, audio: np.ndarray, sample_rate: int = 44100):
    """Save audio to file, preserving original channel count"""
    # If audio is (n_samples,), save as mono
    # If audio is (n_samples, n_channels), save as is
    sf.write(str(file_path), audio, sample_rate)


def normalize_audio(audio: np.ndarray, sample_rate: int, target_lufs: float = -14.0) -> np.ndarray:
    """
    Normalize audio using EBU R128 loudness standard
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        target_lufs: Target integrated loudness in LUFS
        
    Returns:
        Loudness-normalized audio with proper peak limiting
    """
    if len(audio) == 0:
        return audio
    
    # Create EBU R128 meter
    meter = pyln.Meter(sample_rate)
    
    # Measure current loudness
    try:
        current_loudness = meter.integrated_loudness(audio)
    except:
        # Fallback if audio is too quiet
        return audio
    
    # Calculate required gain
    gain_db = target_lufs - current_loudness
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Check peak after gain
    peak_after_gain = np.max(np.abs(audio)) * gain_linear
    
    # If it would clip, reduce target to prevent clipping
    if peak_after_gain > 0.891:  # -1 dBTP = 0.891 linear
        # Adjust target to keep peak at -1 dBTP
        max_gain = 0.891 / np.max(np.abs(audio))
        adjusted_target = current_loudness + (20 * np.log10(max_gain))
        normalized = pyln.normalize.loudness(audio, current_loudness, adjusted_target)
    else:
        # Safe to normalize to target
        normalized = pyln.normalize.loudness(audio, current_loudness, target_lufs)
    
    return normalized
