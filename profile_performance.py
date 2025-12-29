#!/usr/bin/env python3
"""Quick performance analysis of AutoMix operations"""

import time
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("AutoMix Performance Analysis")
print("="*60)

# Generate test audio
sr = 44100
duration = 60  # 1 minute of audio
y = np.random.randn(sr * duration).astype(np.float32) * 0.1

print(f"\nTest audio: {duration}s at {sr}Hz")
print()

# Test 1: STFT (CPU vs GPU)
print("1. STFT Performance")
print("-" * 40)

try:
    import librosa
    start = time.perf_counter()
    S_cpu = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    cpu_time = time.perf_counter() - start
    print(f"CPU STFT:  {cpu_time:.3f}s")
    
    try:
        from src.utils.apple_silicon_gpu import AppleSiliconGPU
        from src.analysis.gpu_features import GPUFeatureExtractor
        
        gpu = AppleSiliconGPU()
        if gpu.use_mps:
            extractor = GPUFeatureExtractor(gpu, sample_rate=sr)
            
            # Warmup
            _ = extractor.compute_stft(y, n_fft=2048, hop_length=512)
            
            start = time.perf_counter()
            S_gpu = extractor.compute_stft(y, n_fft=2048, hop_length=512)
            gpu_time = time.perf_counter() - start
            
            print(f"GPU STFT:  {gpu_time:.3f}s")
            print(f"Speedup:   {cpu_time/gpu_time:.1f}x ⚡")
        else:
            print("GPU STFT:  Not available")
    except Exception as e:
        print(f"GPU STFT:  Error - {e}")
        
except Exception as e:
    print(f"STFT test failed: {e}")

print()

# Test 2: Beat tracking
print("2. Beat Tracking Performance")
print("-" * 40)

try:
    import librosa
    start = time.perf_counter()
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    beat_time = time.perf_counter() - start
    print(f"Beat tracking: {beat_time:.3f}s")
    print(f"Found {len(beats)} beats, tempo: {tempo:.1f} BPM")
except Exception as e:
    print(f"Beat tracking failed: {e}")

print()

# Test 3: Genre detection  
print("3. Genre Detection Performance")
print("-" * 40)

try:
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    
    start = time.perf_counter()
    # Simplified genre features
    mfcc = librosa.feature.mfcc(S=S, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    genre_time = time.perf_counter() - start
    print(f"Genre features: {genre_time:.3f}s")
except Exception as e:
    print(f"Genre detection failed: {e}")

print()

# Summary
print("="*60)
print("Estimated Per-Track Analysis Time")
print("="*60)

try:
    total_cpu = cpu_time + beat_time + genre_time
    print(f"Total CPU operations: {total_cpu:.2f}s")
    
    if 'gpu_time' in locals():
        total_with_gpu = gpu_time + beat_time + genre_time
        overall_speedup = total_cpu / total_with_gpu
        print(f"Total with GPU:       {total_with_gpu:.2f}s")
        print(f"Overall speedup:      {overall_speedup:.1f}x")
        print()
        print(f"For 7 tracks: {total_cpu * 7:.1f}s CPU vs {total_with_gpu * 7:.1f}s GPU")
except Exception as e:
    print(f"Summary calculation failed: {e}")

print("="*60)
