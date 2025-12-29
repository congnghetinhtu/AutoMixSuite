#!/usr/bin/env python3
"""
Quick verification script to check GPU acceleration setup.
Runs without requiring audio files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def check_pytorch():
    """Check if PyTorch is installed."""
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch")
        return False

def check_mps():
    """Check if MPS (Metal) is available."""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✓ MPS (Metal Performance Shaders) available")
            return True
        else:
            print("✗ MPS not available (Intel Mac or old macOS)")
            return False
    except Exception as e:
        print(f"✗ MPS check failed: {e}")
        return False

def check_gpu_modules():
    """Check if GPU modules can be imported."""
    try:
        from src.utils.apple_silicon_gpu import AppleSiliconGPU
        print("✓ GPU detection module loaded")
        
        from src.analysis.gpu_features import GPUFeatureExtractor
        print("✓ GPU feature extraction module loaded")
        
        from src.analysis.gpu_correlation import GPUCorrelation
        print("✓ GPU correlation module loaded")
        
        return True
    except ImportError as e:
        print(f"✗ GPU modules import failed: {e}")
        return False

def test_gpu_detection():
    """Test GPU hardware detection."""
    try:
        from src.utils.apple_silicon_gpu import AppleSiliconGPU
        
        gpu = AppleSiliconGPU()
        
        if gpu.use_mps:
            print(f"✓ GPU detected: {gpu.chip_model}")
            print(f"  - GPU cores: {gpu.gpu_cores}")
            print(f"  - Memory: {gpu.memory_gb:.1f} GB")
            print(f"  - Optimal batch size: {gpu.batch_size}")
            return True
        else:
            print("✗ GPU not available (using CPU)")
            return False
            
    except Exception as e:
        print(f"✗ GPU detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stft():
    """Test GPU STFT computation."""
    try:
        import numpy as np
        from src.utils.apple_silicon_gpu import AppleSiliconGPU
        from src.analysis.gpu_features import GPUFeatureExtractor
        
        gpu = AppleSiliconGPU()
        if not gpu.use_mps:
            print("⊘ Skipping STFT test (no GPU)")
            return False
        
        extractor = GPUFeatureExtractor(gpu, sample_rate=44100)
        
        # Create test audio (1 second sine wave)
        t = np.linspace(0, 1, 44100)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Compute STFT
        S = extractor.compute_stft(audio, n_fft=2048, hop_length=512)
        
        print(f"✓ GPU STFT computation successful")
        print(f"  - Output shape: {S.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU STFT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_correlation():
    """Test GPU correlation computation."""
    try:
        import numpy as np
        from src.utils.apple_silicon_gpu import AppleSiliconGPU
        from src.analysis.gpu_correlation import GPUCorrelation
        
        gpu = AppleSiliconGPU()
        if not gpu.use_mps:
            print("⊘ Skipping correlation test (no GPU)")
            return False
        
        correlator = GPUCorrelation(gpu)
        
        # Create test audio
        t = np.linspace(0, 1, 44100)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Split in half
        mid = len(audio) // 2
        audio1 = audio[:mid]
        audio2 = audio[mid:]
        
        # Compute correlation
        correlation, offset = correlator.phase_correlation(
            audio1, audio2, window_samples=22050, sr=44100
        )
        
        print(f"✓ GPU correlation computation successful")
        print(f"  - Detected offset: {offset} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU correlation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("AutoMix GPU Acceleration Verification")
    print("="*60)
    print()
    
    results = []
    
    print("1. Checking PyTorch installation...")
    results.append(check_pytorch())
    print()
    
    print("2. Checking MPS availability...")
    results.append(check_mps())
    print()
    
    print("3. Checking GPU modules...")
    results.append(check_gpu_modules())
    print()
    
    print("4. Testing GPU detection...")
    results.append(test_gpu_detection())
    print()
    
    print("5. Testing GPU STFT...")
    results.append(test_stft())
    print()
    
    print("6. Testing GPU correlation...")
    results.append(test_correlation())
    print()
    
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    print()
    
    if passed == total:
        print("🎉 All tests passed! GPU acceleration ready to use.")
        print()
        print("Try it out:")
        print("  python automix.py tracks/ --benchmark")
        return 0
    elif passed >= 3:
        print("⚠️  GPU acceleration partially working")
        print("   AutoMix will fall back to CPU for failed operations")
        return 0
    else:
        print("❌ GPU acceleration not available")
        print("   AutoMix will use CPU only (still works, just slower)")
        print()
        print("To enable GPU acceleration on Apple Silicon:")
        print("  pip install torch")
        return 1

if __name__ == "__main__":
    sys.exit(main())
