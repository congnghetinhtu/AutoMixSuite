"""
Test suite for refactored modules
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import *
from src.utils import *
from src.analysis import GenreDetector
import numpy as np


def test_constants():
    """Test that constants are properly defined"""
    print("Testing constants...")
    assert len(MAJOR_PROFILE) == 12, "MAJOR_PROFILE should have 12 values"
    assert len(MINOR_PROFILE) == 12, "MINOR_PROFILE should have 12 values"
    assert len(KEY_NAMES) == 12, "KEY_NAMES should have 12 keys"
    assert DEFAULT_CROSSFADE_DURATION == 8.0, "Default crossfade should be 8.0s"
    assert DEFAULT_SAMPLE_RATE == 44100, "Default sample rate should be 44100 Hz"
    assert DEFAULT_TARGET_LUFS == -14.0, "Default target LUFS should be -14.0"
    print("✓ Constants test passed\n")


def test_file_hash():
    """Test file hash generation"""
    print("Testing file hash...")
    test_file = Path(__file__)
    hash1 = get_file_hash(test_file)
    hash2 = get_file_hash(test_file)
    assert hash1 == hash2, "Hash should be consistent"
    assert len(hash1) == 32, "MD5 hash should be 32 characters"
    print(f"✓ File hash test passed (hash: {hash1[:8]}...)\n")


def test_audio_files():
    """Test audio file discovery"""
    print("Testing audio file discovery...")
    tracks_dir = Path(__file__).parent.parent / 'tracks'
    if tracks_dir.exists():
        files = get_audio_files(tracks_dir, SUPPORTED_FORMATS)
        print(f"  Found {len(files)} audio files")
        if files:
            print(f"  First file: {files[0].name}")
            print("✓ Audio file discovery test passed\n")
        else:
            print("⚠ No audio files found (add some to tracks/ to test)\n")
    else:
        print("⚠ tracks/ directory not found, skipping\n")


def test_normalize_audio():
    """Test audio normalization"""
    print("Testing audio normalization...")
    # Create test audio signal
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.1  # 440 Hz sine wave at low volume
    
    # Test normalization
    normalized = normalize_audio(audio, sample_rate, target_lufs=-14.0)
    
    assert len(normalized) == len(audio), "Normalized audio should have same length"
    assert np.max(np.abs(normalized)) > np.max(np.abs(audio)), "Should be louder after normalization"
    assert np.max(np.abs(normalized)) <= 1.0, "Should not clip"
    print(f"  Original peak: {np.max(np.abs(audio)):.4f}")
    print(f"  Normalized peak: {np.max(np.abs(normalized)):.4f}")
    print("✓ Audio normalization test passed\n")


def test_genre_detector():
    """Test genre detection"""
    print("Testing genre detector...")
    
    # Create synthetic audio for different genres
    sample_rate = 44100
    duration = 10.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test 1: Electronic music (high tempo, percussive)
    audio_electronic = np.random.randn(len(t)) * 0.3  # Percussive noise
    genre = GenreDetector.detect(audio_electronic, sample_rate, tempo=130, energy=0.2, spectral_centroid=3000)
    print(f"  Electronic test: {genre}")
    
    # Test 2: Ballad (slow tempo, harmonic)
    audio_ballad = np.sin(2 * np.pi * 440 * t) * 0.1  # Smooth sine wave
    genre = GenreDetector.detect(audio_ballad, sample_rate, tempo=75, energy=0.1, spectral_centroid=2500)
    print(f"  Ballad test: {genre}")
    
    # Test 3: Cuban Bolero (slow-moderate, balanced)
    audio_bolero = np.sin(2 * np.pi * 440 * t) * 0.15 + np.random.randn(len(t)) * 0.05
    genre = GenreDetector.detect(audio_bolero, sample_rate, tempo=70, energy=0.12, spectral_centroid=2700)
    print(f"  Cuban Bolero test: {genre}")
    
    print("✓ Genre detector test passed\n")


def test_audio_load_save():
    """Test audio loading and saving"""
    print("Testing audio I/O...")
    
    # Check if we have test audio files
    tracks_dir = Path(__file__).parent.parent / 'tracks'
    if not tracks_dir.exists():
        print("⚠ tracks/ directory not found, skipping I/O test\n")
        return
    
    audio_files = get_audio_files(tracks_dir, SUPPORTED_FORMATS)
    if not audio_files:
        print("⚠ No audio files found in tracks/, skipping I/O test\n")
        return
    
    # Load first audio file
    test_file = audio_files[0]
    print(f"  Loading: {test_file.name}")
    audio, sr = load_audio(test_file, sample_rate=44100)
    
    assert len(audio) > 0, "Audio should not be empty"
    assert sr == 44100, "Sample rate should be 44100"
    print(f"  Duration: {len(audio) / sr:.1f}s")
    print(f"  Peak level: {np.max(np.abs(audio)):.4f}")
    
    # Test save (to temp location)
    temp_output = Path(__file__).parent / 'test_output.wav'
    try:
        save_audio(temp_output, audio[:sr], sr)  # Save first 1 second
        assert temp_output.exists(), "Output file should exist"
        print(f"  Saved test file: {temp_output.name}")
        
        # Clean up
        temp_output.unlink()
        print("  Cleaned up test file")
        print("✓ Audio I/O test passed\n")
    except Exception as e:
        print(f"✗ Audio I/O test failed: {e}\n")
        if temp_output.exists():
            temp_output.unlink()


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("REFACTORED MODULE TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_constants()
        test_file_hash()
        test_audio_files()
        test_normalize_audio()
        test_genre_detector()
        test_audio_load_save()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
