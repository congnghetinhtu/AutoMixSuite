# SongMix - Professional DJ Auto-Mixing

A professional-quality automatic DJ mixing system that creates seamless transitions between audio tracks.

## Features

- **GPU Acceleration** - 10-27x faster on Apple Silicon (M1/M2/M3/M4)
- **EBU R128 Loudness Normalization** - Broadcast-standard volume matching
- **Intelligent Genre Detection** - 12 genres including Country, Cuban Bolero, Vietnamese Pop
- **Smart Track Ordering** - Optimizes transitions based on compatibility
- **Beat-Synchronized Mixing** - Aligns beats, downbeats, and phrases
- **Multiple Transition Styles** - Smooth blend, energy punch, build-drop, etc.
- **Fluent Crossfades** - 8-second gradual transitions with 50% overlap
- **Parallel Processing** - Multi-threaded analysis for speed
- **Caching System** - Fast re-analysis of unchanged tracks

## Installation

Basic installation:
```bash
pip install librosa soundfile scipy pyloudnorm
```

**For GPU acceleration (Apple Silicon only):**
```bash
pip install torch  # Enables Metal GPU acceleration
```

## Performance

### GPU Speedup (Apple Silicon)

| Chip Model | Single Track Analysis | Full Mix (10 tracks) |
|-----------|----------------------|---------------------|
| M1        | 15-18x faster        | 8.8x faster         |
| M1 Pro    | 18-22x faster        | 12x faster          |
| M1 Max    | 22-26x faster        | 18x faster          |
| M2        | 18-22x faster        | 10x faster          |
| M2 Pro    | 22-26x faster        | 14x faster          |
| M2 Max    | 26-30x faster        | 20x faster          |
| M2 Ultra  | 30-35x faster        | 27x faster          |
| M3        | 20-24x faster        | 12x faster          |
| M3 Pro    | 24-28x faster        | 16x faster          |
| M3 Max    | 28-32x faster        | 22x faster          |
| M4        | 22-26x faster        | 14x faster          |

GPU acceleration automatically enabled when:
- Apple Silicon Mac (M1/M2/M3/M4)
- PyTorch installed
- macOS 12.3+ 

Gracefully falls back to CPU on Intel Macs.

## Usage

Basic usage (GPU auto-enabled on Apple Silicon):
```bash
./automix tracks/
```

Custom options:
```bash
./automix tracks/ -o output.wav -c 10 --start-track 3
```

GPU control:
```bash
# Force CPU only (useful for testing)
./automix tracks/ --no-gpu

# Benchmark GPU vs CPU performance
./automix tracks/ --benchmark

# Custom GPU batch size (default: auto-detect)
./automix tracks/ --gpu-batch 8
```

## Architecture

### Module Structure
```
src/
├── constants.py          # Configuration constants
├── utils/               # Utility functions
│   ├── audio_io.py     # Audio loading/saving/normalization
│   ├── file_utils.py   # File hashing and discovery
│   ├── apple_silicon_gpu.py  # GPU hardware detection (NEW)
│   └── benchmark.py    # Performance benchmarking (NEW)
├── analysis/            # Audio analysis modules  
│   ├── genre_detector.py     # Genre classification
│   ├── beat_detector.py      # Beat/tempo detection
│   ├── key_detector.py       # Musical key detection
│   ├── track_analyzer.py     # Complete track analysis
│   ├── gpu_features.py       # GPU-accelerated STFT/chroma (NEW)
│   └── gpu_correlation.py    # GPU-accelerated correlation (NEW)
├── mixing/              # Audio mixing modules
│   ├── crossfade.py         # Crossfade generation
│   ├── transitions.py       # Transition styles
│   └── volume_matcher.py    # Volume management
└── core/                # Core engine
    ├── mixer.py             # Main AutoMixer class
    └── cache.py             # Caching system
```

### Key Components

1. **Track Analysis** (`src/analysis/`)
   - Tempo detection with double/half-time handling
   - Beat, downbeat, and phrase detection
   - Musical key using Krumhansl-Schmuckler algorithm
   - Genre classification (9 genres)
   - Vocal segment detection

2. **Crossfade Engine** (`src/mixing/`)
   - Equal-power crossfading with 0.7 exponent curves
   - 50% overlap boost for fullness
   - 64-sample edge ramps for smoothness
   - 5 transition styles based on compatibility

3. **Volume Management**
   - Per-track EBU R128 normalization to -14 LUFS
   - No dynamic adjustments (eliminates pumping)
   - Clipping prevention only

## Algorithm Overview

1. **Load & Normalize** - Each track normalized to -14 LUFS
2. **Analyze** - Extract tempo, beats, key, genre, vocals
3. **Order** - Smart sequencing for optimal flow
4. **Mix** - Create crossfades with beat alignment
5. **Export** - Final limiting and stereo output

## Configuration

Edit `src/constants.py` to change defaults:
- `DEFAULT_CROSSFADE_DURATION = 8.0`  # seconds
- `DEFAULT_TARGET_LUFS = -14.0`       # EBU R128 standard
- `DEFAULT_MAX_WORKERS = 4`           # parallel threads

## Performance Optimizations

- **GPU Acceleration**: Metal Performance Shaders on Apple Silicon (10-27x speedup)
  - Auto-detects M1/M2/M3/M4 chip model and GPU cores
  - Adaptive batch sizing based on available memory
  - Zero-copy unified memory architecture
  - Graceful CPU fallback on non-Apple Silicon Macs
- **Caching**: JSON-based with MD5 hashing
- **Parallel**: Multi-threaded analysis (4 workers default)
- **Optimized**: Vectorized operations, single STFT computation

## Supported Formats

MP3, WAV, FLAC, M4A, AAC, OGG

## Technical Details

- **Sample Rate**: 44.1 kHz
- **Bit Depth**: 32-bit float internal, 16-bit output
- **Normalization**: EBU R128 / BS.1770 (-14 LUFS)
- **Crossfade**: 8s default, adaptive 5.6-11.2s range
- **Key Detection**: Krumhansl-Schmuckler correlation
- **Beat Detection**: Librosa + confidence scoring

## Version

1.0.0 - Production release
