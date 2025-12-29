#!/usr/bin/env python3
"""
Apple Music-style Automix Script
Creates seamless transitions between audio tracks in a folder
"""

import os
import sys
import argparse
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from scipy import signal
from scipy.interpolate import interp1d
import json
import hashlib
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pyloudnorm as pyln

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoMixer:
    # Krumhansl-Schmuckler key profiles (cognitive weights for each pitch class)
    # Based on empirical studies of perceived key strength
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Key names for display
    KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self, input_folder: str, output_file: str = "automix_output.wav", 
                 crossfade_duration: float = 6.0, sample_rate: int = 44100,
                 use_cache: bool = True, cache_dir: Optional[str] = None,
                 max_workers: int = 4, target_lufs: float = -14.0,
                 start_track_index: Optional[int] = None,
                 use_gpu: bool = True, use_beat_grid: bool = True):
        """
        Initialize the AutoMixer
        
        Args:
            input_folder: Path to folder containing audio tracks
            output_file: Output file name for the mixed result
            crossfade_duration: Duration of crossfade between tracks (seconds)
            sample_rate: Target sample rate for processing
            use_cache: Enable caching to speed up repeated analysis
            cache_dir: Custom cache directory (default: .automix_cache in input folder)
            max_workers: Number of parallel threads for analysis (default: 4, use 1 to disable)
            target_lufs: Target loudness in LUFS for EBU R128 normalization (default: -14.0)
            start_track_index: 0-based index of track to start with (None = auto-select)
        """
        self.input_folder = Path(input_folder)
        self.output_file = output_file
        self.crossfade_duration = crossfade_duration
        self.sample_rate = sample_rate
        self.supported_formats = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
        self.max_workers = max(1, max_workers)  # At least 1 worker
        self.target_lufs = target_lufs  # EBU R128 target loudness
        self.start_track_index = start_track_index  # User-selected starting track
        
        # GPU acceleration for Apple Silicon
        self.use_gpu = use_gpu
        self.gpu = None
        self.gpu_features = None
        self.gpu_correlation = None
        self.benchmark_mode = False
        
        if self.use_gpu:
            try:
                from src.utils.apple_silicon_gpu import AppleSiliconGPU
                from src.analysis.gpu_features import GPUFeatureExtractor
                from src.analysis.gpu_correlation import GPUCorrelation
                
                self.gpu = AppleSiliconGPU()
                if self.gpu.use_mps:
                    self.gpu_features = GPUFeatureExtractor(self.gpu, self.sample_rate)
                    self.gpu_correlation = GPUCorrelation(self.gpu)
                    logger.info(f"✓ GPU acceleration enabled: {self.gpu.chip_model}")
                    logger.info(f"✓ Optimal batch size: {self.gpu.batch_size}")
                else:
                    logger.info("GPU acceleration unavailable, using CPU")
                    self.use_gpu = False
            except ImportError as e:
                logger.warning(f"GPU modules not found: {e}")
                logger.info("Install PyTorch for GPU acceleration: pip install torch")
                self.use_gpu = False
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}, using CPU")
                self.use_gpu = False
        
        # Beat grid warping setup
        self.use_beat_grid = use_beat_grid
        if self.use_beat_grid:
            logger.info("✓ Beat grid warping enabled (preserves swing/pitch)")
        
        # Caching setup
        self.CACHE_VERSION = '1.0'
        self.use_cache = use_cache
        self.cache_lock = threading.Lock()  # Thread-safe cache access
        self.use_cache = use_cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.input_folder / '.automix_cache'
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache enabled: {self.cache_dir}")
            # Clean up stale cache files on startup
            self._cleanup_stale_cache()
        
    def _cleanup_stale_cache(self) -> None:
        """Remove cache files for songs that no longer exist"""
        if not self.cache_dir.exists():
            return
        
        # Get all current audio files
        current_files = self.get_audio_files()
        current_hashes = {self._get_file_hash(f) for f in current_files}
        
        # Remove cache files that don't match any current file
        removed = 0
        for cache_file in self.cache_dir.glob('*.json'):
            # Extract hash from filename (last part before .json)
            cache_hash = cache_file.stem.split('_')[-1]
            if cache_hash not in current_hashes:
                cache_file.unlink()
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} stale cache file(s)")
    
    def get_audio_files(self) -> List[Path]:
        """Get all supported audio files from the input folder"""
        audio_files = []
        for file_path in self.input_folder.iterdir():
            if file_path.suffix.lower() in self.supported_formats:
                audio_files.append(file_path)
        
        # Sort files alphabetically
        audio_files.sort(key=lambda x: x.name.lower())
        logger.info(f"Found {len(audio_files)} audio files")
        return audio_files
    
    def _get_file_hash(self, file_path: Path) -> str:
        """
        Generate a hash of the file to detect changes
        Uses file size, modification time, and first 8KB for speed
        Does NOT include filename so cache survives rename/move
        """
        stat = file_path.stat()
        # Don't include filename - only content-based hash
        hash_input = f"{stat.st_size}_{stat.st_mtime_ns}".encode()
        
        # Add first 8KB of file content for better detection
        try:
            with open(file_path, 'rb') as f:
                hash_input += f.read(8192)
        except:
            pass
        
        return hashlib.md5(hash_input).hexdigest()
    
    def _get_cache_path(self, file_path: Path) -> Path:
        """Get cache file path for a given audio file"""
        file_hash = self._get_file_hash(file_path)
        safe_name = "".join(c for c in file_path.stem if c.isalnum() or c in (' ', '-', '_'))[:50]
        cache_name = f"{safe_name}_{file_hash}.json"
        return self.cache_dir / cache_name
    
    def _load_from_cache(self, file_path: Path) -> Optional[Dict]:
        """Load analysis results from cache if available and valid (thread-safe)"""
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(file_path)
        
        if not cache_path.exists():
            return None
        
        try:
            with self.cache_lock:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
            
            # Verify cache version and file hash
            if cached_data.get('cache_version') != '1.0':
                logger.debug(f"Cache version mismatch for {file_path.name}")
                return None
            
            if cached_data.get('file_hash') != self._get_file_hash(file_path):
                logger.debug(f"File changed, cache invalid for {file_path.name}")
                return None
            
            # Convert lists back to numpy arrays
            analysis = cached_data['analysis']
            for key in ['beats', 'downbeats', 'beat_strengths', 'mfcc_mean']:
                if key in analysis and analysis[key] is not None:
                    analysis[key] = np.array(analysis[key])
            
            # Reconstruct beat_frames
            if 'beat_frames' in analysis and analysis['beat_frames'] is not None:
                analysis['beat_frames'] = np.array(analysis['beat_frames'])
            
            # Restore file_path as Path object (was stored as string in JSON)
            if 'file_path' in analysis and isinstance(analysis['file_path'], str):
                analysis['file_path'] = Path(analysis['file_path'])
            
            # Load audio data fresh (not cached - too large)
            audio, sr = librosa.load(str(file_path), sr=44100)
            analysis['audio_data'] = audio
            analysis['sample_rate'] = sr
            
            logger.info(f"  Loaded from cache: {file_path.name}")
            return analysis
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {file_path.name}: {e}")
            return None
    
    def _save_to_cache(self, file_path: Path, analysis: Dict) -> None:
        """Save analysis results to cache (thread-safe)"""
        if not self.use_cache:
            return
        
        try:
            with self.cache_lock:
                cache_path = self._get_cache_path(file_path)
            
            # Prepare cache data (convert numpy arrays to lists for JSON)
            cache_data = {
                'cache_version': '1.0',
                'file_hash': self._get_file_hash(file_path),
                'cached_at': datetime.now().isoformat(),
                'analysis': {}
            }
            
            # Copy analysis data, converting numpy arrays to lists
            for key, value in analysis.items():
                if key in ['audio_data', 'sample_rate']:
                    continue  # Don't cache audio - too large
                elif key == 'file_path':
                    cache_data['analysis'][key] = str(value)
                elif isinstance(value, np.ndarray):
                    cache_data['analysis'][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    cache_data['analysis'][key] = float(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], tuple):
                    # Handle phrases and vocal_segments (list of tuples)
                    cache_data['analysis'][key] = [[float(x) if isinstance(x, (int, float, np.number)) else x for x in item] for item in value]
                else:
                    cache_data['analysis'][key] = value
            
            # Save JSON cache (no pickle - audio not cached)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"  ✓ Cached: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to save cache for {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_cache(self) -> int:
        """Clear all cache files. Returns number of files deleted."""
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.iterdir():
            if cache_file.suffix == '.json':  # Only JSON files now (no pickle)
                cache_file.unlink()
                count += 1
        
        logger.info(f"Cleared {count} cache files")
        return count
    
    def analyze_audio(self, file_path: Path) -> Dict:
        """
        Analyze audio file for mixing parameters with enhanced volume normalization
        
        Returns:
            Dictionary containing tempo, key, energy, beats, and other features
        """
        # Try to load from cache first
        cached_analysis = self._load_from_cache(file_path)
        if cached_analysis is not None:
            return cached_analysis
        
        try:
            logger.info(f"Analyzing: {file_path.name}")
            
            # Load audio file preserving original channels (stereo/mono)
            # librosa returns stereo as (2, samples), we need (samples, 2) for processing
            y, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=False)
            
            # Transpose stereo to (samples, channels) format if needed
            if y.ndim > 1:
                y = y.T  # Convert from (2, samples) to (samples, 2)
            
            # Store original channel format
            is_stereo = y.ndim > 1
            
            # Normalize each track to consistent level for uniform volume across all songs
            # This ensures all tracks have the same perceived loudness before mixing
            y_normalized = self._normalize_audio(y, target_lufs=self.target_lufs)
            
            # CRITICAL: Ensure output is ALWAYS stereo to preserve channel information
            # Check if normalization accidentally converted to mono
            if y_normalized.ndim == 1:
                # Was stereo but became mono - restore stereo from original
                if is_stereo:
                    logger.warning("  Audio became mono after normalization - this shouldn't happen!")
                    logger.warning(f"  Original shape: {y.shape}, After normalize: {y_normalized.shape}")
                    # Re-normalize properly preserving stereo
                    y_normalized = self._normalize_audio(y, target_lufs=self.target_lufs)
                    if y_normalized.ndim == 1:
                        logger.error("  Normalization still returns mono! Converting mono to stereo...")
                        y_normalized = np.column_stack([y_normalized, y_normalized])
                else:
                    # Was mono, convert to stereo
                    logger.info("  Converting mono track to stereo")
                    y_normalized = np.column_stack([y_normalized, y_normalized])
                is_stereo = True
            elif y_normalized.ndim == 2 and not is_stereo:
                # Was mono but normalize returned stereo (shouldn't happen)
                logger.info("  Track was mono but normalized as stereo")
                is_stereo = True
            
            # For analysis, convert to mono (librosa analysis functions expect mono)
            # But keep the stereo version for actual mixing
            y_mono = librosa.to_mono(y_normalized.T)  # to_mono expects (channels, samples)
            
            # PERFORMANCE OPTIMIZATION: Compute STFT once and reuse for multiple features
            hop_length = 512
            n_fft = 2048
            
            # Use GPU STFT if available (15-20x faster on Apple Silicon)
            if self.use_gpu and self.gpu_features:
                S = self.gpu_features.compute_stft(y_mono, n_fft=n_fft, hop_length=hop_length)
            else:
                S = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=hop_length))
            
            # Enhanced tempo and beat detection with downbeat analysis
            # OPTIMIZATION: Compute beat tracking once and convert units
            tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr, units='frames')
            beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
            
            # Detect if this is actually double-time or half-time
            actual_tempo, tempo_multiplier = self._detect_tempo_variations(y_mono, sr, tempo, beats)
            
            # Detect swing/groove characteristics
            swing_ratio, groove_type = self._detect_swing_groove(y_mono, sr, beats, beat_frames)
            
            # Detect downbeats (measure boundaries) for better transitions
            downbeats = self._detect_downbeats(y_mono, sr, beats, beat_frames)
            
            # Detect time signature for accurate phrase detection
            time_signature = self._detect_time_signature(beats, actual_tempo)
            
            # Detect musical phrases (8/16/32 bar sections) with time signature awareness
            phrases = self._detect_phrases(beats, downbeats, actual_tempo, time_signature)
            
            # Calculate beat strength for better transition point selection
            beat_strengths = self._calculate_beat_strengths(y_mono, sr, beat_frames)
            
            # Calculate beat confidence (reliability/quality of each beat)
            beat_confidence = self._calculate_beat_confidence(beats, beat_strengths, actual_tempo)
            
            # Enhanced key detection using Krumhansl-Schmuckler algorithm
            # Use GPU chroma if available (10-15x faster on Apple Silicon)
            if self.use_gpu and self.gpu_features:
                chroma = self.gpu_features.compute_chroma(S, sr)
            else:
                chroma = librosa.feature.chroma_stft(S=S, sr=sr)
            key, key_mode, key_confidence = self._detect_key_krumhansl(chroma)
            key_name = self.KEY_NAMES[key]
            logger.info(f"  Key: {key_name} {key_mode} (confidence: {key_confidence:.2f})")
            
            # Enhanced energy analysis with RMS (reuse precomputed STFT for speed)
            # OPTIMIZATION: Use precomputed STFT magnitude, 2x faster than recomputing
            rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)[0]
            energy = np.mean(rms)
            energy_variation = np.std(rms)
            
            # Spectral features for mixing compatibility (reuse STFT)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
            
            # Detect genre for tempo matching strategy (after energy/spectral features calculated)
            genre_hint = self._detect_genre_hint(y_mono, sr, actual_tempo, energy, spectral_centroid)
            
            # Zero crossing rate for rhythm analysis
            zcr = np.mean(librosa.feature.zero_crossing_rate(y_mono))
            
            # MFCC for timbral analysis
            mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Vocal detection using multiple techniques
            vocal_segments = self._detect_vocals(y_mono, sr)
            
            # Song structure detection (intro/verse/chorus/bridge/outro)
            # Skip for very short tracks (< 60 seconds)
            structure_boundaries = []
            structure_sections = []
            main_section = None
            
            if len(y_mono) / sr >= 60.0 and len(beats) >= 16:
                logger.info(f"  Analyzing song structure...")
                
                # Compute self-similarity matrix for repetition detection
                sim_matrix = self._compute_self_similarity_matrix(chroma, mfccs, beats, sr)
                
                # Detect structural boundaries (verse/chorus transitions)
                structure_boundaries = self._detect_structure_boundaries(
                    y_mono, sr, beats, downbeats, sim_matrix
                )
                
                # Label each section (intro/verse/chorus/etc)
                structure_sections = self._label_structure_sections(
                    structure_boundaries, len(y_mono) / sr, beats, vocal_segments, 
                    rms, sr, sim_matrix
                )
                
                # Identify the "main" section (primary chorus/verse)
                main_section = self._identify_main_section(
                    structure_sections, beats, vocal_segments, rms, sr, sim_matrix
                )
            else:
                logger.info(f"  Skipping structure analysis (track too short or insufficient beats)")
            
            # Onset detection for transition points
            onset_frames = librosa.onset.onset_detect(y=y_mono, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Beat grid analysis for timing correction (if enabled)
            beat_grid = None
            if self.use_beat_grid:
                try:
                    from src.analysis.beat_grid import BeatGridAnalyzer
                    
                    grid_analyzer = BeatGridAnalyzer(sample_rate=sr)
                    beat_grid = grid_analyzer.detect_beat_grid(
                        y_mono, beats, actual_tempo, swing_ratio
                    )
                    
                    # Check if warping recommended
                    if beat_grid and grid_analyzer.should_apply_warping(beat_grid, genre_hint):
                        logger.info(f"    ✓ Beat grid warping recommended")
                    elif beat_grid:
                        logger.info(f"    ✓ Beat grid detected but warping not needed")
                    
                except Exception as e:
                    logger.warning(f"    Beat grid analysis failed: {e}")
                    beat_grid = None
            
            # Find potential intro/outro sections for smooth flow
            intro_end, outro_start = self._detect_intro_outro(y_mono, sr, beats)
            
            # Calculate peak and RMS levels for volume matching (use stereo original)
            peak_level = np.max(np.abs(y_normalized))
            rms_level = np.sqrt(np.mean(y_normalized**2))
            
            # EBU R128 loudness measurement (integrated LUFS)
            meter = pyln.Meter(sr)  # Create BS.1770 meter
            loudness_lufs = meter.integrated_loudness(y_normalized)
            
            logger.info(f"  Building result dictionary...")
            
            result = {
                'file_path': file_path,
                'duration': len(y_normalized) / sr,
                'tempo': tempo.item() if hasattr(tempo, 'item') else float(tempo),
                'actual_tempo': actual_tempo.item() if hasattr(actual_tempo, 'item') else float(actual_tempo),
                'tempo_multiplier': tempo_multiplier,
                'swing_ratio': float(swing_ratio),
                'groove_type': groove_type,
                'genre_hint': genre_hint,
                'beats': beats,
                'beat_frames': beat_frames,
                'downbeats': downbeats,
                'phrases': phrases,
                'beat_strengths': beat_strengths,
                'beat_confidence': beat_confidence,
                'key': int(key),
                'key_name': key_name,
                'key_mode': key_mode,
                'key_confidence': float(key_confidence),
                'energy': float(energy),
                'energy_variation': float(energy_variation),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'spectral_bandwidth': float(spectral_bandwidth),
                'zcr': float(zcr),
                'mfcc_mean': mfcc_mean,
                'vocal_segments': vocal_segments,
                'structure_boundaries': structure_boundaries,  # NEW: Structural boundary times
                'structure_sections': structure_sections,      # NEW: Labeled sections
                'main_section': main_section,                   # NEW: Primary chorus/verse
                'onset_times': onset_times,
                'beat_grid': beat_grid,                         # NEW: Beat grid for timing correction
                'intro_end': intro_end,
                'outro_start': outro_start,
                'peak_level': float(peak_level),
                'rms_level': float(rms_level),
                'loudness_lufs': float(loudness_lufs),  # EBU R128 loudness
                'audio_data': y_normalized,  # Use normalized audio
                'sample_rate': sr
            }
            
            # Save to cache for future use
            logger.info(f"  Saving to cache: {file_path.name}")
            self._save_to_cache(file_path, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_tracks_parallel(self, audio_files: List[Path], max_workers: int = 4) -> List[Dict]:
        """
        Analyze multiple audio files in parallel using ThreadPoolExecutor
        
        Args:
            audio_files: List of audio file paths to analyze
            max_workers: Maximum number of parallel threads (default: 4)
        
        Returns:
            List of analysis dictionaries for each track
        """
        logger.info(f"Analyzing {len(audio_files)} tracks with {max_workers} workers...")
        analyzed_tracks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_file = {
                executor.submit(self.analyze_audio, file_path): file_path 
                for file_path in audio_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        analyzed_tracks.append(result)
                except Exception as e:
                    logger.error(f"Failed to analyze {file_path.name}: {e}")
        
        logger.info(f"Successfully analyzed {len(analyzed_tracks)}/{len(audio_files)} tracks")
        return analyzed_tracks
    
    def _detect_vocals(self, y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Detect vocal segments using spectral analysis and harmonic-percussive separation
        """
        try:
            # PERFORMANCE: Analyze only first 2 minutes for speed (vocals usually appear early)
            max_samples = min(len(y), sr * 120)  # 2 minutes max
            y_short = y[:max_samples]
            
            # Harmonic-percussive separation to isolate vocals (COMPUTE ONCE)
            y_harmonic, y_percussive = librosa.effects.hpss(y_short, margin=2.0)
            
            # Compute spectral features with larger hop for speed
            hop_length = 1024  # Larger hop = 2x faster
            
            # 1. Spectral centroid (vocals typically have higher centroids)
            spec_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr, hop_length=hop_length)[0]
            
            # 2. Spectral rolloff (vocals have characteristic rolloff patterns)
            spec_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr, roll_percent=0.85, hop_length=hop_length)[0]
            
            # 3. Chroma features (vocals often follow harmonic progressions)
            chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, hop_length=hop_length)
            chroma_strength = np.sum(chroma, axis=0)
            
            # 4. Zero crossing rate (speech-like patterns)
            zcr = librosa.feature.zero_crossing_rate(y_harmonic, hop_length=hop_length)[0]
            
            # 5. MFCCs (vocal timbre characteristics) - reduced from 13 to 5 for speed
            mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=5, hop_length=hop_length)
            mfcc_var = np.var(mfccs, axis=0)
            
            # Create vocal probability score
            # Normalize features
            spec_centroid_norm = (spec_centroid - np.mean(spec_centroid)) / (np.std(spec_centroid) + 1e-8)
            chroma_strength_norm = (chroma_strength - np.mean(chroma_strength)) / (np.std(chroma_strength) + 1e-8)
            mfcc_var_norm = (mfcc_var - np.mean(mfcc_var)) / (np.std(mfcc_var) + 1e-8)
            
            # Vocal probability based on multiple features
            vocal_prob = (
                np.clip(spec_centroid_norm * 0.3, -1, 1) +  # Higher centroid suggests vocals
                np.clip(chroma_strength_norm * 0.3, -1, 1) + # Strong harmonic content
                np.clip(mfcc_var_norm * 0.4, -1, 1)          # Vocal timbre variation
            ) / 3.0
            
            # Apply smoothing to reduce noise
            if len(vocal_prob) > 10:
                from scipy import signal
                window_size = min(21, len(vocal_prob) // 5)
                if window_size >= 5:
                    vocal_prob = signal.savgol_filter(vocal_prob, window_size | 1, 2)
            
            # Convert frame indices to time (scaled to analyzed length)
            frame_times = librosa.frames_to_time(np.arange(len(vocal_prob)), sr=sr, hop_length=hop_length)
            
            # Find vocal segments (threshold for vocal detection)
            vocal_threshold = 0.1  # Lower threshold for better detection
            vocal_frames = vocal_prob > vocal_threshold
            
            # Find continuous vocal segments
            vocal_segments = []
            in_vocal = False
            start_time = 0
            
            for i, is_vocal in enumerate(vocal_frames):
                current_time = frame_times[i] if i < len(frame_times) else frame_times[-1]
                
                if is_vocal and not in_vocal:
                    # Start of vocal segment
                    start_time = current_time
                    in_vocal = True
                elif not is_vocal and in_vocal:
                    # End of vocal segment
                    if current_time - start_time > 1.0:  # Keep segments longer than 1 second
                        vocal_segments.append((start_time, current_time))
                    in_vocal = False
            
            # Handle case where track ends during vocal
            if in_vocal and len(frame_times) > 0:
                if frame_times[-1] - start_time > 1.0:
                    vocal_segments.append((start_time, frame_times[-1]))
            
            logger.info(f"    Detected {len(vocal_segments)} vocal segments")
            return vocal_segments
            
        except Exception as e:
            logger.warning(f"    Vocal detection failed: {e}, using fallback")
            # Fallback: assume vocals in middle sections of track
            duration = len(y) / sr
            return [(duration * 0.2, duration * 0.4), (duration * 0.6, duration * 0.8)]
    
    def _compute_self_similarity_matrix(self, chroma: np.ndarray, mfccs: np.ndarray, 
                                       beats: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute self-similarity matrix for structure detection
        Finds repeated sections (chorus, verse patterns) using harmonic and timbral features
        
        Args:
            chroma: Chroma feature matrix (12 x time_frames)
            mfccs: MFCC feature matrix (13 x time_frames) 
            beats: Beat times for beat-synchronization
            sr: Sample rate
            
        Returns:
            NxN similarity matrix where high values indicate similar sections
            N = number of beats (beat-synchronized for efficiency)
        """
        try:
            # Beat-synchronize features for efficiency (one feature vector per beat)
            # This dramatically reduces computation while preserving musical structure
            chroma_sync = librosa.util.sync(chroma, np.searchsorted(
                librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr), beats
            ), aggregate=np.median)
            
            mfcc_sync = librosa.util.sync(mfccs, np.searchsorted(
                librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr), beats
            ), aggregate=np.median)
            
            # Normalize features to [0, 1] range for equal weighting
            chroma_norm = (chroma_sync - np.min(chroma_sync, axis=1, keepdims=True)) / (
                np.max(chroma_sync, axis=1, keepdims=True) - np.min(chroma_sync, axis=1, keepdims=True) + 1e-8
            )
            mfcc_norm = (mfcc_sync - np.min(mfcc_sync, axis=1, keepdims=True)) / (
                np.max(mfcc_sync, axis=1, keepdims=True) - np.min(mfcc_sync, axis=1, keepdims=True) + 1e-8
            )
            
            # Combine features: 60% chroma (harmonic/melodic) + 40% MFCC (timbral)
            combined_features = np.vstack([
                chroma_norm * 0.6,
                mfcc_norm[:5, :] * 0.4  # Use only first 5 MFCCs for speed
            ])
            
            # Compute cosine similarity matrix (efficient for normalized features)
            # Each element [i,j] = similarity between beat i and beat j
            n_beats = combined_features.shape[1]
            sim_matrix = np.dot(combined_features.T, combined_features)
            
            # Apply Gaussian smoothing to reduce noise and emphasize patterns
            from scipy.ndimage import gaussian_filter
            sim_matrix = gaussian_filter(sim_matrix, sigma=1.5)
            
            # Ensure diagonal is maximum (each beat is most similar to itself)
            np.fill_diagonal(sim_matrix, 1.0)
            
            # Clip to [0, 1] range
            sim_matrix = np.clip(sim_matrix, 0, 1)
            
            logger.info(f"    Self-similarity matrix: {n_beats}x{n_beats} (beat-synchronized)")
            return sim_matrix
            
        except Exception as e:
            logger.warning(f"    Self-similarity computation failed: {e}")
            # Fallback: return identity matrix (no repetition detected)
            n_beats = len(beats) if len(beats) > 0 else 100
            return np.eye(n_beats)
    
    def _detect_structure_boundaries(self, y_mono: np.ndarray, sr: int, 
                                    beats: np.ndarray, downbeats: np.ndarray,
                                    sim_matrix: np.ndarray) -> List[float]:
        """
        Detect major structural change points (verse→chorus, chorus→bridge, etc.)
        Uses self-similarity matrix patterns and novelty detection
        
        Args:
            y_mono: Mono audio signal
            sr: Sample rate
            beats: Beat times
            downbeats: Downbeat (measure start) times
            sim_matrix: Self-similarity matrix from _compute_self_similarity_matrix()
            
        Returns:
            List of boundary times in seconds (aligned to downbeats)
        """
        try:
            if len(beats) < 16 or sim_matrix.shape[0] < 16:
                # Too short for structure detection
                return []
            
            # Method 1: Novelty-based segmentation using librosa
            # Compute novelty curve from self-similarity matrix
            # High novelty = structural boundary
            novelty_curve = librosa.segment.recurrence_to_lag(sim_matrix, pad=False, axis=0)
            novelty_curve = np.sum(novelty_curve, axis=0)
            
            # Smooth novelty curve
            from scipy.ndimage import gaussian_filter1d
            novelty_smooth = gaussian_filter1d(novelty_curve, sigma=3)
            
            # Find peaks in novelty curve (potential boundaries)
            from scipy.signal import find_peaks
            peak_threshold = np.mean(novelty_smooth) + 0.5 * np.std(novelty_smooth)
            peaks, _ = find_peaks(novelty_smooth, height=peak_threshold, distance=8)
            
            # Convert beat indices to time
            boundary_times = []
            for peak_idx in peaks:
                if peak_idx < len(beats):
                    boundary_time = beats[peak_idx]
                    
                    # Align to nearest downbeat for musical correctness
                    if len(downbeats) > 0:
                        nearest_downbeat_idx = np.argmin(np.abs(downbeats - boundary_time))
                        aligned_time = downbeats[nearest_downbeat_idx]
                        
                        # Only use if within 2 beats of the detected boundary
                        if abs(aligned_time - boundary_time) < (60.0 / (len(beats) / (beats[-1] - beats[0]) * 60.0)) * 2:
                            boundary_time = aligned_time
                    
                    boundary_times.append(boundary_time)
            
            # Filter out boundaries that are too close together (< 8 seconds)
            filtered_boundaries = []
            last_boundary = -10.0
            for boundary in sorted(boundary_times):
                if boundary - last_boundary >= 8.0:
                    filtered_boundaries.append(boundary)
                    last_boundary = boundary
            
            # Always include start (0.0) if not present
            if len(filtered_boundaries) == 0 or filtered_boundaries[0] > 1.0:
                filtered_boundaries.insert(0, 0.0)
            
            logger.info(f"    Detected {len(filtered_boundaries)} structural boundaries")
            return filtered_boundaries
            
        except Exception as e:
            logger.warning(f"    Structure boundary detection failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: divide track into 4 equal sections
            duration = len(y_mono) / sr
            return [0.0, duration * 0.25, duration * 0.5, duration * 0.75]
    
    def _analyze_intro_mood(self, y: np.ndarray, sr: int, intro_duration: float,
                           vocal_segments: List[Tuple[float, float]],
                           rms: np.ndarray) -> Dict[str, float]:
        """
        Analyze intro mood to detect boring/sparse intros that should be skipped
        
        Boring intro indicators:
        1. Low energy (quiet, minimal dynamics)
        2. No vocals or very sparse vocals
        3. Low spectral complexity (simple/repetitive instrumentation)
        4. Minimal rhythm/percussion
        5. Long duration with little change
        
        Returns:
            Dict with 'boring_score' (0-1, higher = more boring) and component scores
        """
        try:
            intro_samples = int(intro_duration * sr)
            intro_audio = y[:intro_samples] if len(y) > intro_samples else y
            
            if len(intro_audio) < sr * 2:  # Less than 2 seconds
                return {'boring_score': 0.0, 'reason': 'too_short'}
            
            # Factor 1: Energy level (low energy = boring)
            hop_length = 512
            intro_frames = int(intro_duration * sr / hop_length)
            intro_rms = rms[:intro_frames] if len(rms) > intro_frames else rms
            avg_energy = np.mean(intro_rms)
            energy_variation = np.std(intro_rms)
            
            # Normalize energy (typical range 0.01-0.15)
            energy_score = 1.0 - min(avg_energy / 0.08, 1.0)  # Low energy = high boring score
            variation_score = 1.0 - min(energy_variation / 0.03, 1.0)  # Low variation = boring
            
            # Factor 2: Vocal presence (no vocals = more boring)
            intro_vocal_duration = sum(min(vs_end, intro_duration) - max(vs_start, 0.0)
                                      for vs_start, vs_end in vocal_segments
                                      if vs_start < intro_duration)
            vocal_ratio = intro_vocal_duration / intro_duration if intro_duration > 0 else 0
            vocal_score = 1.0 - vocal_ratio  # No vocals = high boring score
            
            # Factor 3: Spectral complexity (simple = boring)
            import librosa
            spectral_contrast = librosa.feature.spectral_contrast(y=intro_audio, sr=sr, hop_length=hop_length)
            avg_contrast = np.mean(spectral_contrast)
            # Typical range 20-40 dB, low contrast = simple/boring
            complexity_score = 1.0 - min(avg_contrast / 35.0, 1.0)
            
            # Factor 4: Rhythmic content (minimal percussion = boring)
            onset_env = librosa.onset.onset_strength(y=intro_audio, sr=sr, hop_length=hop_length)
            onset_density = np.sum(onset_env > np.mean(onset_env)) / len(onset_env)
            rhythm_score = 1.0 - min(onset_density / 0.3, 1.0)  # Low onset density = boring
            
            # Factor 5: Duration penalty (long boring intro = skip it)
            duration_penalty = min(intro_duration / 30.0, 1.0)  # Longer intros penalized more
            
            # Weighted boring score
            boring_score = (
                energy_score * 0.25 +
                variation_score * 0.15 +
                vocal_score * 0.25 +
                complexity_score * 0.20 +
                rhythm_score * 0.15
            ) * (1.0 + duration_penalty * 0.3)  # Apply duration penalty
            
            boring_score = min(boring_score, 1.0)  # Cap at 1.0
            
            # Classify boring level
            if boring_score > 0.7:
                mood = 'very_boring'
            elif boring_score > 0.5:
                mood = 'boring'
            elif boring_score > 0.3:
                mood = 'mild'
            else:
                mood = 'engaging'
            
            logger.info(f"    Intro mood: {mood} (score: {boring_score:.2f}, energy: {avg_energy:.3f}, vocals: {vocal_ratio:.1%}, duration: {intro_duration:.1f}s)")
            
            return {
                'boring_score': float(boring_score),
                'mood': mood,
                'energy': float(avg_energy),
                'energy_variation': float(energy_variation),
                'vocal_ratio': float(vocal_ratio),
                'spectral_complexity': float(avg_contrast),
                'rhythm_density': float(onset_density),
                'duration': float(intro_duration)
            }
            
        except Exception as e:
            logger.warning(f"    Intro mood analysis failed: {e}")
            return {'boring_score': 0.0, 'mood': 'unknown', 'reason': str(e)}
    
    def _label_structure_sections(self, boundaries: List[float], duration: float,
                                  beats: np.ndarray, vocal_segments: List[Tuple[float, float]], 
                                  rms: np.ndarray, sr: int,
                                  sim_matrix: np.ndarray) -> List[Tuple[float, float, str]]:
        """
        Label each section as intro/verse/chorus/bridge/outro
        
        Classification rules:
        - Intro: First section, lower energy, often instrumental or sparse vocals
        - Verse: Lower-medium energy, vocals present, less repetitive
        - Chorus: Highest energy, most repeated section, prominent vocals
        - Bridge: Mid-track, contrasting pattern, different from surrounding sections
        - Outro: Last section, decreasing energy, may mirror intro
        
        Args:
            boundaries: List of section boundary times
            duration: Track duration in seconds
            beats: Beat times
            vocal_segments: List of (start, end) tuples for vocal regions
            rms: RMS energy per frame
            sr: Sample rate
            sim_matrix: Self-similarity matrix (beat-synchronized)
            
        Returns:
            List of (start_time, end_time, label) tuples
        """
        try:
            if len(boundaries) == 0:
                return [(0.0, duration, 'unknown')]
            
            # Add end boundary if not present
            all_boundaries = sorted(list(set(boundaries + [duration])))
            
            # Create sections from boundaries
            sections = []
            for i in range(len(all_boundaries) - 1):
                start = all_boundaries[i]
                end = all_boundaries[i + 1]
                sections.append((start, end, 'unknown'))
            
            if len(sections) == 0:
                return [(0.0, duration, 'unknown')]
            
            # Calculate features for each section
            section_features = []
            hop_length = 512
            
            for start, end, _ in sections:
                # Energy
                start_frame = int(start * sr / hop_length)
                end_frame = int(end * sr / hop_length)
                section_rms = rms[start_frame:min(end_frame, len(rms))]
                avg_energy = np.mean(section_rms) if len(section_rms) > 0 else 0
                
                # Vocal presence
                vocal_duration = sum(min(vs_end, end) - max(vs_start, start) 
                                   for vs_start, vs_end in vocal_segments 
                                   if vs_start < end and vs_end > start)
                vocal_ratio = vocal_duration / (end - start) if end > start else 0
                
                # Repetition score (how similar is this section to others?)
                # Map section time to beat indices in sim_matrix
                section_beats = beats[(beats >= start) & (beats < end)]
                if len(section_beats) > 0 and len(beats) > 0:
                    start_beat_idx = np.searchsorted(beats, start)
                    end_beat_idx = np.searchsorted(beats, end)
                    
                    # Compare this section to all other sections
                    repetition_scores = []
                    for j, (other_start, other_end, _) in enumerate(sections):
                        if i != j:  # Don't compare to self
                            other_start_idx = np.searchsorted(beats, other_start)
                            other_end_idx = np.searchsorted(beats, other_end)
                            
                            # Extract similarity values between these two sections
                            if (start_beat_idx < sim_matrix.shape[0] and end_beat_idx <= sim_matrix.shape[0] and
                                other_start_idx < sim_matrix.shape[1] and other_end_idx <= sim_matrix.shape[1]):
                                section_sim = sim_matrix[start_beat_idx:end_beat_idx, 
                                                        other_start_idx:other_end_idx]
                                if section_sim.size > 0:
                                    repetition_scores.append(np.mean(section_sim))
                    
                    repetition_score = np.max(repetition_scores) if repetition_scores else 0
                else:
                    repetition_score = 0
                
                section_features.append({
                    'energy': avg_energy,
                    'vocal_ratio': vocal_ratio,
                    'repetition': repetition_score,
                    'duration': end - start
                })
            
            # Normalize features across all sections
            if len(section_features) > 1:
                max_energy = max(f['energy'] for f in section_features) + 1e-8
                for f in section_features:
                    f['energy_norm'] = f['energy'] / max_energy
            else:
                section_features[0]['energy_norm'] = 1.0
            
            # Label sections using rules
            labeled_sections = []
            
            for i, ((start, end, _), features) in enumerate(zip(sections, section_features)):
                label = 'unknown'
                metadata = {}  # Store additional section metadata
                
                # Rule 1: First section with low energy or no vocals → Intro
                # Enhanced: Analyze intro mood for boring detection
                if i == 0 and (features['energy_norm'] < 0.5 or features['vocal_ratio'] < 0.3):
                    label = 'intro'
                    # Analyze intro mood
                    intro_mood = self._analyze_intro_mood(
                        rms * sr / 512,  # Approximate audio reconstruction for analysis
                        sr, end - start, vocal_segments, rms
                    )
                    metadata['mood'] = intro_mood
                    # Flag boring intros for aggressive skipping
                    if intro_mood.get('boring_score', 0) > 0.5:
                        metadata['skip_recommended'] = True
                        metadata['skip_confidence'] = intro_mood['boring_score']
                        logger.info(f"    ⚠️  Boring intro detected (score: {intro_mood['boring_score']:.2f}) - recommend skipping")
                
                # Rule 2: Last section → Outro
                elif i == len(sections) - 1:
                    label = 'outro'
                
                # Rule 3: High repetition + high energy + vocals → Chorus
                elif (features['repetition'] > 0.6 and 
                      features['energy_norm'] > 0.65 and 
                      features['vocal_ratio'] > 0.4):
                    label = 'chorus'
                
                # Rule 4: Medium repetition + medium energy + vocals → Verse
                elif (features['repetition'] > 0.4 and 
                      features['energy_norm'] > 0.4 and 
                      features['vocal_ratio'] > 0.5):
                    label = 'verse'
                
                # Rule 5: Middle section with low repetition → Bridge
                elif (i > 0 and i < len(sections) - 1 and 
                      features['repetition'] < 0.4 and
                      features['vocal_ratio'] > 0.3):
                    label = 'bridge'
                
                # Rule 6: Default with vocals → Verse, without → Instrumental
                elif features['vocal_ratio'] > 0.3:
                    label = 'verse'
                else:
                    label = 'instrumental'
                
                # Store section with metadata if available
                if metadata:
                    labeled_sections.append((start, end, label, metadata))
                else:
                    labeled_sections.append((start, end, label))
            
            # Post-processing: Ensure we have at least one verse and one chorus if possible
            has_chorus = any(s[2] == 'chorus' for s in labeled_sections)
            has_verse = any(s[2] == 'verse' for s in labeled_sections)
            
            # If no chorus detected but we have high-energy sections, label the highest as chorus
            if not has_chorus and len(labeled_sections) > 1:
                max_energy_idx = max(range(len(section_features)), 
                                    key=lambda i: section_features[i]['energy_norm'])
                if labeled_sections[max_energy_idx][2] not in ['intro', 'outro']:
                    section = labeled_sections[max_energy_idx]
                    start, end = section[0], section[1]
                    metadata = section[3] if len(section) > 3 else {}
                    if metadata:
                        labeled_sections[max_energy_idx] = (start, end, 'chorus', metadata)
                    else:
                        labeled_sections[max_energy_idx] = (start, end, 'chorus')
            
            logger.info(f"    Labeled {len(labeled_sections)} sections: " + 
                       ", ".join(f"{s[2]}" for s in labeled_sections))
            
            return labeled_sections
            
        except Exception as e:
            logger.warning(f"    Section labeling failed: {e}")
            import traceback
            traceback.print_exc()
            return [(0.0, duration, 'unknown')]
    
    def _identify_main_section(self, sections: List[Tuple[float, float, str]], 
                              beats: np.ndarray, vocal_segments: List[Tuple[float, float]],
                              rms: np.ndarray, sr: int,
                              sim_matrix: np.ndarray) -> Optional[Tuple[float, float, str]]:
        """
        Identify the "main song" section - typically the primary/strongest chorus
        
        Scoring criteria:
        - Repetition count (40%): How many times does this pattern repeat?
        - Peak energy (30%): Is this the highest energy section?
        - Vocal presence (20%): Strong, clear vocals?
        - Duration (10%): Reasonable length (not too short)
        
        Args:
            sections: List of labeled sections
            beats: Beat times
            vocal_segments: Vocal regions
            rms: RMS energy
            sr: Sample rate
            sim_matrix: Self-similarity matrix
            
        Returns:
            (start_time, end_time, 'main_chorus') or None if no clear main section
        """
        try:
            if len(sections) == 0:
                return None
            
            # Focus on chorus sections as candidates for "main"
            chorus_sections = [s for s in sections if s[2] == 'chorus']
            
            # If no chorus, try verse sections
            if len(chorus_sections) == 0:
                chorus_sections = [s for s in sections if s[2] == 'verse']
            
            # If still nothing, use any section with vocals
            if len(chorus_sections) == 0:
                chorus_sections = [s for s in sections if s[2] not in ['intro', 'outro', 'instrumental']]
            
            if len(chorus_sections) == 0:
                return None
            
            hop_length = 512
            best_score = -1
            best_section = None
            
            for section in chorus_sections:
                start, end, label = section[0], section[1], section[2]
                score = 0.0
                
                # Factor 1: Repetition count (40%)
                # Count how many other sections are similar to this one
                section_beats_mask = (beats >= start) & (beats < end)
                if np.any(section_beats_mask):
                    start_beat_idx = np.searchsorted(beats, start)
                    end_beat_idx = np.searchsorted(beats, end)
                    
                    repetition_count = 0
                    for other_section in sections:
                        other_start, other_end = other_section[0], other_section[1]
                        if other_start == start and other_end == end:
                            continue  # Skip self
                        
                        other_start_idx = np.searchsorted(beats, other_start)
                        other_end_idx = np.searchsorted(beats, other_end)
                        
                        if (start_beat_idx < sim_matrix.shape[0] and end_beat_idx <= sim_matrix.shape[0] and
                            other_start_idx < sim_matrix.shape[1] and other_end_idx <= sim_matrix.shape[1]):
                            section_sim = sim_matrix[start_beat_idx:end_beat_idx, 
                                                    other_start_idx:other_end_idx]
                            if section_sim.size > 0 and np.mean(section_sim) > 0.7:
                                repetition_count += 1
                    
                    # Normalize: 0 repeats = 0, 3+ repeats = 1.0
                    repetition_score = min(repetition_count / 3.0, 1.0)
                    score += repetition_score * 0.4
                
                # Factor 2: Peak energy (30%)
                start_frame = int(start * sr / hop_length)
                end_frame = int(end * sr / hop_length)
                section_rms = rms[start_frame:min(end_frame, len(rms))]
                if len(section_rms) > 0:
                    avg_energy = np.mean(section_rms)
                    max_possible_energy = np.max(rms) + 1e-8
                    energy_score = avg_energy / max_possible_energy
                    score += energy_score * 0.3
                
                # Factor 3: Vocal presence (20%)
                vocal_duration = sum(min(vs_end, end) - max(vs_start, start) 
                                   for vs_start, vs_end in vocal_segments 
                                   if vs_start < end and vs_end > start)
                vocal_ratio = vocal_duration / (end - start) if end > start else 0
                score += vocal_ratio * 0.2
                
                # Factor 4: Duration (10%)
                # Prefer sections between 15-45 seconds
                duration = end - start
                if 15 <= duration <= 45:
                    duration_score = 1.0
                elif duration < 15:
                    duration_score = duration / 15.0
                else:
                    duration_score = max(0.5, 1.0 - (duration - 45) / 60.0)
                score += duration_score * 0.1
                
                if score > best_score:
                    best_score = score
                    best_section = section
            
            if best_section and best_score > 0.4:  # Minimum confidence threshold
                start, end, label = best_section
                logger.info(f"    Main section: {label} at {start:.1f}-{end:.1f}s (score: {best_score:.2f})")
                return (start, end, f'main_{label}')
            
            return None
            
        except Exception as e:
            logger.warning(f"    Main section identification failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_tempo_variations(self, y: np.ndarray, sr: int, detected_tempo: float, 
                                beats: np.ndarray) -> Tuple[float, str]:
        """
        Detect if the track is actually double-time or half-time
        Returns (actual_tempo, multiplier_type)
        """
        try:
            if len(beats) < 8:
                return detected_tempo, "normal"
            
            # Calculate inter-beat intervals
            beat_intervals = np.diff(beats)
            
            if len(beat_intervals) < 4:
                return detected_tempo, "normal"
            
            # Calculate median interval and consistency
            median_interval = np.median(beat_intervals)
            interval_std = np.std(beat_intervals)
            
            # Method 1: Check for alternating strong/weak beats (half-time detection)
            hop_length = 512
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            
            # Get onset strength at each beat (SAMPLE: first 30 beats for speed)
            beat_strengths = []
            for beat_time in beats[:min(len(beats), 30)]:
                beat_frame = int(beat_time * sr / hop_length)
                if beat_frame < len(onset_env) - 2:
                    strength = np.max(onset_env[beat_frame-1:beat_frame+2])
                    beat_strengths.append(strength)
            
            if len(beat_strengths) >= 8:
                # Check if every other beat is consistently stronger (OPTIMIZED)
                beat_arr = np.array(beat_strengths)
                even_beats = beat_arr[::2]
                odd_beats = beat_arr[1::2]
                
                if len(even_beats) > 3 and len(odd_beats) > 3:
                    even_avg = np.mean(even_beats)
                    odd_avg = np.mean(odd_beats)
                    
                    # Strong 2:1 pattern indicates half-time
                    ratio = max(even_avg, odd_avg) / (min(even_avg, odd_avg) + 1e-8)
                    if ratio > 1.4:  # 40% stronger alternating pattern
                        actual_tempo = detected_tempo / 2
                        logger.info(f"    Detected half-time pattern: {detected_tempo.item() if hasattr(detected_tempo, 'item') else float(detected_tempo):.1f} → {actual_tempo.item() if hasattr(actual_tempo, 'item') else float(actual_tempo):.1f} BPM (ratio: {ratio.item() if hasattr(ratio, 'item') else float(ratio):.2f})")
                        return actual_tempo, "half-time"
            
            # Method 2: Fast tempo double-time check (threshold lowered to 140)
            if detected_tempo > 140:
                # Calculate what half-tempo would be
                half_tempo = detected_tempo / 2
                
                # If half-tempo falls in common range (85-110 BPM), it's likely double-time
                if 85 <= half_tempo <= 110:
                    # Verify with beat strength analysis (OPTIMIZED)
                    if len(beat_strengths) >= 8:
                        beat_arr = np.array(beat_strengths)
                        even_beats = beat_arr[::2]
                        odd_beats = beat_arr[1::2]
                        
                        if len(even_beats) > 3 and len(odd_beats) > 3:
                            even_avg = np.mean(even_beats)
                            odd_avg = np.mean(odd_beats)
                            ratio = max(even_avg, odd_avg) / (min(even_avg, odd_avg) + 1e-8)
                            
                            if ratio > 1.25:  # Even weak alternating pattern confirms
                                logger.info(f"    Detected double-time: {detected_tempo.item() if hasattr(detected_tempo, 'item') else float(detected_tempo):.1f} → {half_tempo.item() if hasattr(half_tempo, 'item') else float(half_tempo):.1f} BPM")
                                return half_tempo, "double-time"
            
            # Method 3: Slow tempo that should be doubled
            elif detected_tempo < 70:
                doubled_tempo = detected_tempo * 2
                # If doubling gives reasonable tempo, do it
                if 90 <= doubled_tempo <= 150:
                    logger.info(f"    Adjusted slow tempo: {detected_tempo.item() if hasattr(detected_tempo, 'item') else float(detected_tempo):.1f} → {doubled_tempo.item() if hasattr(doubled_tempo, 'item') else float(doubled_tempo):.1f} BPM")
                    return doubled_tempo, "adjusted-up"
            
            # Method 4: Check beat interval consistency
            # If beats are very inconsistent, librosa might have locked onto subdivisions
            if interval_std > median_interval * 0.4:
                # Try grouping every 2nd beat
                if len(beat_intervals) >= 4:
                    grouped_intervals = []
                    for i in range(0, len(beat_intervals)-1, 2):
                        grouped_intervals.append(beat_intervals[i] + beat_intervals[i+1])
                    
                    if len(grouped_intervals) > 2:
                        grouped_std = np.std(grouped_intervals)
                        grouped_median = np.median(grouped_intervals)
                        
                        # If grouped beats are much more consistent
                        if grouped_std < interval_std * 0.6 and grouped_std < grouped_median * 0.2:
                            actual_tempo = detected_tempo / 2
                            logger.info(f"    Beat grouping correction: {detected_tempo.item() if hasattr(detected_tempo, 'item') else float(detected_tempo):.1f} → {actual_tempo.item() if hasattr(actual_tempo, 'item') else float(actual_tempo):.1f} BPM")
                            return actual_tempo, "half-time"
            
            return detected_tempo, "normal"
            
        except Exception as e:
            logger.warning(f"    Tempo variation detection failed: {e}")
            return detected_tempo, "normal"
    
    def _detect_swing_groove(self, y: np.ndarray, sr: int, beats: np.ndarray, 
                           beat_frames: np.ndarray) -> Tuple[float, str]:
        """
        Detect swing ratio and groove type (straight, swing, shuffle)
        Returns (swing_ratio, groove_type)
        """
        try:
            if len(beats) < 8:
                return 0.5, "straight"
            
            # Analyze micro-timing between beats
            beat_intervals = np.diff(beats)
            
            if len(beat_intervals) < 4:
                return 0.5, "straight"
            
            # Detect onset times with high precision
            hop_length = 128  # Smaller hop for better timing precision
            # Compute onset strength envelope first
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            # Then detect onsets from the envelope
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, 
                                                     hop_length=hop_length)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
            
            # For each beat, find subdivisions (8th notes, triplets)
            subdivision_ratios = []
            
            for i in range(len(beats) - 1):
                beat_start = beats[i]
                beat_end = beats[i + 1]
                beat_duration = beat_end - beat_start
                
                # Find onsets within this beat
                beat_onsets = onset_times[(onset_times >= beat_start) & (onset_times < beat_end)]
                
                if len(beat_onsets) >= 2:
                    # Analyze timing of subdivisions
                    first_onset = beat_onsets[0] - beat_start
                    second_onset = beat_onsets[1] - beat_start if len(beat_onsets) > 1 else beat_duration / 2
                    
                    # Calculate ratio (0.5 = straight, 0.67 = triplet swing, 0.75 = hard shuffle)
                    if beat_duration > 0:
                        ratio = second_onset / beat_duration
                        if 0.4 < ratio < 0.8:  # Valid subdivision range
                            subdivision_ratios.append(ratio)
            
            if len(subdivision_ratios) > 4:
                avg_ratio = np.median(subdivision_ratios)  # Use median for robustness
                ratio_std = np.std(subdivision_ratios)
                
                # Classify groove type
                if ratio_std < 0.08:  # Consistent timing
                    if 0.48 <= avg_ratio <= 0.52:
                        groove_type = "straight"
                    elif 0.58 <= avg_ratio <= 0.70:
                        groove_type = "swing"
                        logger.info(f"    Detected swing groove: ratio {avg_ratio:.2f}")
                    elif avg_ratio > 0.70:
                        groove_type = "shuffle"
                        logger.info(f"    Detected shuffle groove: ratio {avg_ratio:.2f}")
                    else:
                        groove_type = "straight"
                    
                    return float(avg_ratio), groove_type
            
            # Default to straight
            return 0.5, "straight"
            
        except Exception as e:
            logger.warning(f"    Swing/groove detection failed: {e}")
            return 0.5, "straight"
    
    def _detect_genre_hint(self, y: np.ndarray, sr: int, tempo: float, 
                         energy: float, spectral_centroid: float) -> str:
        """
        Detect genre hint for tempo matching strategy
        Returns genre category: 'electronic', 'pop', 'rock', 'jazz', 'classical', 'hiphop', 
                               'vietnamese_ballad', 'vietnamese_pop', 'cuba_bolero', 'country', 'unknown'
        """
        try:
            # PERFORMANCE: Analyze only first 90 seconds for genre detection
            max_samples = min(len(y), sr * 90)
            y_short = y[:max_samples]
            
            # Percussive vs harmonic balance with proper normalization (COMPUTE ONCE)
            y_harmonic, y_percussive = librosa.effects.hpss(y_short, margin=2.0)
            total_energy = np.sqrt(np.mean(y_short**2)) + 1e-10
            harmonic_ratio = np.sqrt(np.mean(y_harmonic**2)) / total_energy
            percussive_ratio = np.sqrt(np.mean(y_percussive**2)) / total_energy
            
            # Normalize to sum to ~1.0
            total_ratio = harmonic_ratio + percussive_ratio
            if total_ratio > 0:
                harmonic_ratio /= total_ratio
                percussive_ratio /= total_ratio
            
            # PERFORMANCE: Use larger hop for faster computation
            hop_length = 1024
            
            # Spectral contrast for genre-specific timbral characteristics
            contrast = librosa.feature.spectral_contrast(y=y_short, sr=sr, hop_length=hop_length)
            avg_contrast = np.mean(contrast)
            contrast_var = np.std(np.mean(contrast, axis=0))  # Variation over time
            
            # Zero crossing rate variation
            zcr = librosa.feature.zero_crossing_rate(y_short, hop_length=hop_length)[0]
            zcr_var = np.std(zcr)
            zcr_mean = np.mean(zcr)
            
            # Tempo stability
            tempo_category = "slow" if tempo < 100 else "moderate" if tempo < 130 else "fast"
            
            # Genre classification with weighted scoring (0-1 scale)
            genre_scores = {}
            
            # Electronic/EDM: High percussive, tight tempo range, consistent
            electronic_score = 0.0
            if 115 <= tempo <= 145:
                electronic_score += 0.25
                if 125 <= tempo <= 135:  # House/techno sweet spot
                    electronic_score += 0.15
            if percussive_ratio > 0.52:  # Strong percussion
                electronic_score += 0.25
            if contrast_var < 3.5:  # Consistent timbre
                electronic_score += 0.15
            if spectral_centroid > 2500:  # Bright electronic sound
                electronic_score += 0.15
            if energy > 0.15:  # High energy typical
                electronic_score += 0.10
            if electronic_score > 0.5:
                genre_scores['electronic'] = electronic_score
            
            # Hip-Hop: Moderate tempo, strong percussion, lower spectral content
            hiphop_score = 0.0
            if 80 <= tempo <= 110:
                hiphop_score += 0.3
            if percussive_ratio > 0.5:
                hiphop_score += 0.3
            if spectral_centroid < 2500:
                hiphop_score += 0.2
            if zcr_mean < 0.08:  # Lower ZCR typical
                hiphop_score += 0.2
            if hiphop_score > 0.4:
                genre_scores['hiphop'] = hiphop_score
            
            # Rock: Moderate-high tempo, balanced, high energy variation
            rock_score = 0.0
            if 110 <= tempo <= 170:
                rock_score += 0.25
            if 0.35 <= harmonic_ratio <= 0.65:  # Balanced
                rock_score += 0.25
            if avg_contrast > 20:
                rock_score += 0.25
            if energy > 0.15:  # Decent energy
                rock_score += 0.25
            if rock_score > 0.5:
                genre_scores['rock'] = rock_score
            
            # Pop: Moderate tempo, balanced, accessible
            pop_score = 0.0
            if 95 <= tempo <= 135:
                pop_score += 0.25
            if 0.4 <= harmonic_ratio <= 0.6:
                pop_score += 0.25
            if energy > 0.1:
                pop_score += 0.25
            if 2000 <= spectral_centroid <= 3500:
                pop_score += 0.25
            if pop_score > 0.5:
                genre_scores['pop'] = pop_score
            
            # Jazz: High harmonic, variable, complex
            jazz_score = 0.0
            if harmonic_ratio > 0.58:  # More harmonic (raised threshold)
                jazz_score += 0.25
            if contrast_var > 4.5:  # Variable timbre (raised threshold)
                jazz_score += 0.25
            if zcr_var > 0.03:  # Rhythmic complexity (raised threshold)
                jazz_score += 0.25
            if 80 <= tempo <= 200:  # Wide tempo range
                jazz_score += 0.15
            # Penalty if too percussive (jazz has more subtlety)
            if percussive_ratio > 0.6:
                jazz_score -= 0.2
            if jazz_score > 0.6:  # Raised confidence threshold
                genre_scores['jazz'] = jazz_score
            
            # Classical: Very high harmonic, wide dynamic range, orchestral
            classical_score = 0.0
            if harmonic_ratio > 0.7:  # Strongly harmonic
                classical_score += 0.4
            if avg_contrast > 23:  # Wide dynamic range
                classical_score += 0.3
            if contrast_var > 3.5:  # Dynamic variation
                classical_score += 0.2
            if spectral_centroid > 2000:  # Orchestral brightness
                classical_score += 0.1
            if classical_score > 0.6:
                genre_scores['classical'] = classical_score
            
            # Vietnamese Ballad: Slow, emotional, melodic, vocal-centric
            viet_ballad_score = 0.0
            if 70 <= tempo <= 95:  # Slower, emotional tempo
                viet_ballad_score += 0.3
            if harmonic_ratio > 0.6:  # Very melodic/harmonic
                viet_ballad_score += 0.3
            if spectral_centroid < 2800:  # Softer, warmer sound
                viet_ballad_score += 0.2
            if 0.08 <= energy <= 0.15:  # Gentle dynamics
                viet_ballad_score += 0.2
            if viet_ballad_score > 0.5:
                genre_scores['vietnamese_ballad'] = viet_ballad_score
            
            # Vietnamese Pop (V-pop): Moderate tempo, balanced, often syncopated
            viet_pop_score = 0.0
            if 95 <= tempo <= 130:  # Moderate, danceable tempo
                viet_pop_score += 0.25
            if 0.45 <= harmonic_ratio <= 0.55:  # Balanced harmonic/percussive
                viet_pop_score += 0.25
            if 2200 <= spectral_centroid <= 3200:  # Modern production sound
                viet_pop_score += 0.25
            if energy > 0.12:  # Moderate to high energy
                viet_pop_score += 0.25
            if viet_pop_score > 0.5:
                genre_scores['vietnamese_pop'] = viet_pop_score
            
            # Cuban Bolero: Slow-moderate tempo, romantic, rhythmic with clave pattern, guitar-based
            cuba_bolero_score = 0.0
            if 60 <= tempo <= 90:  # Slow to moderate romantic tempo
                cuba_bolero_score += 0.3
            if harmonic_ratio > 0.55:  # Melodic and harmonic (guitar, vocals)
                cuba_bolero_score += 0.3
            if 0.45 <= percussive_ratio <= 0.55:  # Balanced with rhythmic clave
                cuba_bolero_score += 0.2
            if spectral_centroid < 3000:  # Warm, acoustic sound
                cuba_bolero_score += 0.1
            if 0.08 <= energy <= 0.18:  # Moderate, romantic energy
                cuba_bolero_score += 0.1
            if cuba_bolero_score > 0.5:
                genre_scores['cuba_bolero'] = cuba_bolero_score
            
            # Future Funk (Young Franco style): Groovy, funky basslines, disco samples, moderate-high tempo
            future_funk_score = 0.0
            if 105 <= tempo <= 128:  # Funky groove tempo
                future_funk_score += 0.3
                if 110 <= tempo <= 120:  # Sweet spot for funk
                    future_funk_score += 0.1
            if 0.4 <= harmonic_ratio <= 0.6:  # Balanced funky groove
                future_funk_score += 0.3
            if percussive_ratio > 0.4:  # Strong rhythm section
                future_funk_score += 0.2
            if 2500 <= spectral_centroid <= 3800:  # Bright, funky sound
                future_funk_score += 0.1
            if energy > 0.13:  # Energetic and danceable
                future_funk_score += 0.1
            if future_funk_score > 0.5:
                genre_scores['future_funk'] = future_funk_score
            
            # House (Future House, Future Bounce): 4/4 beat, 120-130 BPM, energetic, synth-heavy
            house_score = 0.0
            if 118 <= tempo <= 132:  # House tempo range
                house_score += 0.30
                if 124 <= tempo <= 128:  # Classic house sweet spot
                    house_score += 0.10
            if percussive_ratio > 0.52:  # Strong 4/4 kick pattern
                house_score += 0.25
            if 2800 <= spectral_centroid <= 4500:  # Bright synth-heavy sound
                house_score += 0.20
            if energy > 0.16:  # High energy, club-ready
                house_score += 0.15
            if house_score > 0.6:  # Raised threshold to distinguish from electronic
                genre_scores['house'] = house_score
            
            # Country: Moderate tempo, storytelling, acoustic instruments, twangy guitars
            country_score = 0.0
            if 90 <= tempo <= 145:  # Country tempo range (ballads to uptempo)
                country_score += 0.25
                if 110 <= tempo <= 130:  # Sweet spot for modern country
                    country_score += 0.10
            if 0.48 <= harmonic_ratio <= 0.68:  # Melodic with strong vocals and guitars (wider range)
                country_score += 0.25
            if 1600 <= spectral_centroid <= 3200:  # Warm acoustic sound (wider range for variety)
                country_score += 0.20
            if 0.08 <= energy <= 0.20:  # Moderate energy, storytelling pace (wider range)
                country_score += 0.15
            if zcr_var > 0.02:  # Variable but not extreme (storytelling dynamics)
                country_score += 0.10
            # Bonus for swing groove (common in country)
            if 0.55 <= zcr_mean <= 0.12:
                country_score += 0.10
            if country_score > 0.55:
                genre_scores['country'] = country_score
            
            # Select genre with highest score, with minimum threshold
            if genre_scores:
                # Sort by score descending
                sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Get top genre
                top_genre, top_score = sorted_genres[0]
                
                # Only accept if confidence is reasonable
                if top_score >= 0.5:
                    # Check if there's ambiguity (multiple high scores)
                    if len(sorted_genres) > 1:
                        second_score = sorted_genres[1][1]
                        if top_score - second_score < 0.15:  # Very close
                            logger.info(f"    Genre hint: {top_genre} (confidence: {top_score:.2f}, ambiguous with {sorted_genres[1][0]})")
                        else:
                            logger.info(f"    Genre hint: {top_genre} (confidence: {top_score:.2f})")
                    else:
                        logger.info(f"    Genre hint: {top_genre} (confidence: {top_score:.2f})")
                    return top_genre
                else:
                    logger.info(f"    Genre uncertain: best guess {top_genre} (confidence: {top_score:.2f})")
            
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"    Genre detection failed: {e}")
            return 'unknown'
    
    def _detect_time_signature(self, beats: np.ndarray, tempo: float) -> int:
        """
        Detect time signature by analyzing beat interval patterns
        Returns beats per measure (2, 3, 4, or 6)
        """
        try:
            if len(beats) < 8:
                return 4  # Default to 4/4
            
            # Calculate inter-beat intervals
            intervals = np.diff(beats)
            
            if len(intervals) < 4:
                return 4
            
            # Calculate median interval (quarter note duration)
            median_interval = np.median(intervals)
            beat_period = 60.0 / tempo
            
            # Try different meter hypotheses by grouping beats
            meter_scores = {}
            
            # Test 4/4 (most common)
            score_4 = self._score_meter_hypothesis(intervals, 4, beat_period)
            meter_scores[4] = score_4
            
            # Test 3/4 (waltz - common in Vietnamese ballads)
            score_3 = self._score_meter_hypothesis(intervals, 3, beat_period)
            meter_scores[3] = score_3
            
            # Test 2/4 (march)
            score_2 = self._score_meter_hypothesis(intervals, 2, beat_period)
            meter_scores[2] = score_2
            
            # Test 6/8 (compound meter - 2 groups of 3)
            # 6/8 has emphasis on beats 1 and 4
            score_6 = self._score_meter_hypothesis(intervals, 6, beat_period)
            meter_scores[6] = score_6
            
            # Select best meter
            best_meter = max(meter_scores, key=meter_scores.get)
            best_score = meter_scores[best_meter]
            
            # Only accept if confidence is reasonable (score > 0.6)
            if best_score > 0.6:
                logger.info(f"    Detected time signature: {best_meter}/4 (confidence: {best_score:.2f})")
                return best_meter
            else:
                logger.info(f"    Time signature uncertain, defaulting to 4/4 (best: {best_meter}/4, score: {best_score:.2f})")
                return 4
            
        except Exception as e:
            logger.warning(f"    Time signature detection failed: {e}, defaulting to 4/4")
            return 4
    
    def _score_meter_hypothesis(self, intervals: np.ndarray, beats_per_measure: int, beat_period: float) -> float:
        """
        Score how well the beat intervals fit a given meter hypothesis
        Returns confidence score 0-1
        """
        try:
            if len(intervals) < beats_per_measure:
                return 0.0
            
            # Group intervals into measures
            num_measures = len(intervals) // beats_per_measure
            if num_measures < 2:
                return 0.0
            
            # Calculate expected measure duration
            expected_measure_duration = beat_period * beats_per_measure
            
            # Check consistency of measure durations
            measure_durations = []
            for i in range(num_measures):
                start_idx = i * beats_per_measure
                end_idx = min(start_idx + beats_per_measure, len(intervals))
                measure_sum = np.sum(intervals[start_idx:end_idx])
                measure_durations.append(measure_sum)
            
            if len(measure_durations) < 2:
                return 0.0
            
            # Calculate coefficient of variation (lower is better)
            measure_mean = np.mean(measure_durations)
            measure_std = np.std(measure_durations)
            
            if measure_mean == 0:
                return 0.0
            
            cv = measure_std / measure_mean
            
            # Score based on consistency (low CV = high score)
            if cv < 0.05:
                consistency_score = 1.0
            elif cv < 0.10:
                consistency_score = 0.9
            elif cv < 0.15:
                consistency_score = 0.7
            elif cv < 0.25:
                consistency_score = 0.5
            else:
                consistency_score = 0.3
            
            # Bonus for matching expected duration
            duration_error = abs(measure_mean - expected_measure_duration) / expected_measure_duration
            if duration_error < 0.05:
                duration_score = 1.0
            elif duration_error < 0.10:
                duration_score = 0.8
            elif duration_error < 0.20:
                duration_score = 0.6
            else:
                duration_score = 0.3
            
            # Combined score (70% consistency, 30% duration match)
            total_score = consistency_score * 0.7 + duration_score * 0.3
            
            return total_score
            
        except Exception as e:
            return 0.0
    
    def _detect_downbeats(self, y: np.ndarray, sr: int, beats: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
        """
        Detect downbeats (first beat of each measure) with time signature detection
        Uses multi-feature analysis: spectral flux, harmonic change, low-frequency energy, onset strength
        """
        try:
            # Validate inputs
            if beats is None or len(beats) == 0:
                return np.array([])
            if len(beats) < 4:
                return beats  # Not enough beats to detect measures
            
            # Step 1: Detect time signature
            tempo = len(beats) / (beats[-1] - beats[0]) * 60.0 if len(beats) > 1 else 120.0
            beats_per_measure = self._detect_time_signature(beats, tempo)
            
            # PERFORMANCE: Use larger hop for faster computation
            hop_length = 1024  # 2x faster than 512
            n_fft = 2048
            S = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=n_fft))
            
            # Feature 1: Spectral flux (frequency changes) - 25%
            spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
            spectral_flux = np.pad(spectral_flux, (1, 0), mode='edge')
            
            # Feature 2: Low-frequency energy (bass/kick drum) - 20%
            # Focus on 20-250 Hz range (bass instruments)
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            low_freq_mask = (freq_bins >= 20) & (freq_bins <= 250)
            low_freq_energy = np.sum(S[low_freq_mask, :], axis=0)
            
            # Feature 3: Harmonic change (chord changes) - 20%
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
            harmonic_change = np.zeros(chroma.shape[1])
            for i in range(1, chroma.shape[1]):
                harmonic_change[i] = np.linalg.norm(chroma[:, i] - chroma[:, i-1])
            
            # Feature 4: Onset strength peaks - 20%
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            
            # Feature 5: Beat periodicity - 15% (how well beat fits expected pattern)
            beat_period = 60.0 / tempo
            expected_downbeat_times = np.arange(beats[0], beats[-1], beat_period * beats_per_measure)
            
            # Normalize all features
            def normalize_feature(feature):
                if len(feature) > 0 and np.max(feature) > 0:
                    return feature / np.max(feature)
                return feature
            
            spectral_flux_norm = normalize_feature(spectral_flux)
            low_freq_norm = normalize_feature(low_freq_energy)
            harmonic_change_norm = normalize_feature(harmonic_change)
            onset_env_norm = normalize_feature(onset_env)
            
            # Calculate importance score for each beat
            beat_importance = []
            for i, beat_frame in enumerate(beat_frames):
                adjusted_frame = int(beat_frame * 512 / hop_length)
                
                # Get features at beat location
                flux = spectral_flux_norm[adjusted_frame] if adjusted_frame < len(spectral_flux_norm) else 0
                low_freq = low_freq_norm[adjusted_frame] if adjusted_frame < len(low_freq_norm) else 0
                harm = harmonic_change_norm[adjusted_frame] if adjusted_frame < len(harmonic_change_norm) else 0
                onset = onset_env_norm[adjusted_frame] if adjusted_frame < len(onset_env_norm) else 0
                
                # Periodicity score: how close to expected downbeat position?
                beat_time = beats[i]
                min_distance = np.min(np.abs(expected_downbeat_times - beat_time))
                periodicity = 1.0 - min(min_distance / beat_period, 1.0)
                
                # Weighted combination
                importance = (
                    flux * 0.25 +
                    low_freq * 0.20 +
                    harm * 0.20 +
                    onset * 0.20 +
                    periodicity * 0.15
                )
                beat_importance.append(importance)
            
            beat_importance = np.array(beat_importance)
            
            # Find downbeats using dynamic programming for optimal selection
            downbeat_indices = [0]  # First beat is always a downbeat
            
            for i in range(beats_per_measure, len(beats), beats_per_measure):
                # Look for strongest beat in measure window
                window_start = max(0, i - 1)
                window_end = min(len(beat_importance), i + 2)
                window = beat_importance[window_start:window_end]
                
                if len(window) > 0:
                    local_max_idx = window_start + np.argmax(window)
                    downbeat_indices.append(local_max_idx)
            
            downbeats = beats[downbeat_indices]
            logger.info(f"    Detected {len(downbeats)} downbeats in {beats_per_measure}/4 time")
            
            return downbeats
            
        except Exception as e:
            logger.warning(f"    Downbeat detection failed: {e}, using every 4th beat")
            import traceback
            traceback.print_exc()
            # Fallback: assume every 4th beat is a downbeat
            return beats[::4]
    
    def _detect_phrases(self, beats: np.ndarray, downbeats: np.ndarray, tempo: float, time_signature: int = 4) -> List[Tuple[float, float, int]]:
        """
        Detect musical phrases (8, 16, or 32 bar sections)
        Returns list of (start_time, end_time, bar_count) tuples
        Now time-signature aware: handles 2/4, 3/4, 4/4, 6/8
        """
        try:
            # Validate inputs
            if downbeats is None or len(downbeats) == 0:
                return []
            if len(downbeats) < 8:
                return []  # Not enough measures for phrase detection
            
            phrases = []
            beats_per_phrase = [32, 16, 8]  # Common phrase lengths in order of preference
            
            # Calculate average measure duration using detected time signature
            if len(downbeats) > 1:
                measure_duration = np.mean(np.diff(downbeats))
            else:
                measure_duration = 60.0 / tempo * time_signature  # Use actual time signature
            
            # Detect phrases of different lengths
            for phrase_bars in beats_per_phrase:
                if len(downbeats) >= phrase_bars:
                    for i in range(0, len(downbeats) - phrase_bars + 1, phrase_bars):
                        phrase_start = downbeats[i]
                        phrase_end_idx = min(i + phrase_bars, len(downbeats) - 1)
                        phrase_end = downbeats[phrase_end_idx]
                        
                        phrases.append((phrase_start, phrase_end, phrase_bars))
            
            # Sort by start time
            phrases.sort(key=lambda x: x[0])
            
            logger.info(f"    Detected {len(phrases)} musical phrases")
            return phrases
            
        except Exception as e:
            logger.warning(f"    Phrase detection failed: {e}")
            return []
    
    def _calculate_beat_strengths(self, y: np.ndarray, sr: int, beat_frames: np.ndarray) -> np.ndarray:
        """
        Calculate the strength/importance of each beat using multiple features
        Combines onset strength, spectral flux, transient detection, and harmonic changes
        """
        try:
            hop_length = 512
            n_fft = 2048
            
            # Feature 1: Onset strength (rhythmic energy) - Weight: 40%
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            
            # Feature 2: Spectral flux (frequency domain energy changes) - Weight: 25%
            S = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=n_fft))
            spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
            # Pad to match onset_env length
            spectral_flux = np.pad(spectral_flux, (1, 0), mode='edge')
            
            # Feature 3: Transient sharpness (attack detection) - Weight: 20%
            # High-frequency energy changes indicate sharp transients (drum hits, etc.)
            S_high_freq = S[S.shape[0]//2:, :]  # Upper half of spectrum
            transient_env = np.sum(np.abs(np.diff(S_high_freq, axis=1)), axis=0)
            transient_env = np.pad(transient_env, (1, 0), mode='edge')
            
            # Feature 4: Harmonic change (chord/note changes) - Weight: 15%
            chroma = librosa.feature.chroma_stft(S=S, sr=sr, hop_length=hop_length)
            harmonic_change = np.zeros(chroma.shape[1])
            for i in range(1, chroma.shape[1]):
                harmonic_change[i] = np.linalg.norm(chroma[:, i] - chroma[:, i-1])
            
            # Normalize all features to 0-1 range
            def normalize_feature(feature):
                if len(feature) > 0 and np.max(feature) > 0:
                    return feature / np.max(feature)
                return feature
            
            onset_env_norm = normalize_feature(onset_env)
            spectral_flux_norm = normalize_feature(spectral_flux)
            transient_env_norm = normalize_feature(transient_env)
            harmonic_change_norm = normalize_feature(harmonic_change)
            
            # Extract strength at each beat frame with weighted combination (VECTORIZED)
            beat_strengths = np.zeros(len(beat_frames))
            valid_frames = beat_frames < len(onset_env_norm)
            
            # Vectorized window extraction
            for i, beat_frame in enumerate(beat_frames[valid_frames]):
                window_start = max(0, beat_frame - 2)
                window_end = min(len(onset_env_norm), beat_frame + 3)
                
                # Weighted combination (pre-slice arrays once)
                beat_strengths[i] = (
                    np.mean(onset_env_norm[window_start:window_end]) * 0.40 +
                    np.mean(spectral_flux_norm[window_start:min(len(spectral_flux_norm), beat_frame + 3)]) * 0.25 +
                    np.mean(transient_env_norm[window_start:min(len(transient_env_norm), beat_frame + 3)]) * 0.20 +
                    np.mean(harmonic_change_norm[window_start:min(len(harmonic_change_norm), beat_frame + 3)]) * 0.15
                )
            
            # Final normalization to 0-1 range
            if len(beat_strengths) > 0 and np.max(beat_strengths) > 0:
                beat_strengths = beat_strengths / np.max(beat_strengths)
            
            # Apply light smoothing to reduce noise while preserving peaks
            if len(beat_strengths) > 5:
                from scipy.ndimage import gaussian_filter1d
                beat_strengths = gaussian_filter1d(beat_strengths, sigma=0.8)
                # Re-normalize after smoothing
                if np.max(beat_strengths) > 0:
                    beat_strengths = beat_strengths / np.max(beat_strengths)
            
            logger.info(f"    Beat strengths: mean={np.mean(beat_strengths):.3f}, std={np.std(beat_strengths):.3f}, max={np.max(beat_strengths):.3f}")
            
            return beat_strengths
            
        except Exception as e:
            logger.warning(f"    Beat strength calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return np.ones(len(beat_frames))
    
    def _calculate_beat_confidence(self, beats: np.ndarray, beat_strengths: np.ndarray, tempo: float) -> np.ndarray:
        """
        Calculate confidence score (0-1) for each beat based on:
        - Relative strength compared to neighbors
        - Temporal consistency with expected beat grid
        - Regularity of surrounding beats
        """
        try:
            # Validate inputs
            if beats is None or beat_strengths is None:
                return np.ones(0) if beats is None else np.ones(len(beats))
            if len(beats) < 3 or len(beat_strengths) != len(beats):
                return np.ones(len(beats))
            
            # Pre-allocate for speed
            beat_confidence = np.zeros(len(beats))
            beat_period = 60.0 / tempo
            
            # Vectorized calculations where possible
            for i in range(len(beats)):
                confidence = 0.0
                
                # Factor 1: Relative strength (30%)
                # Compare to local neighborhood (5 beats)
                window_start = max(0, i - 2)
                window_end = min(len(beat_strengths), i + 3)
                local_strengths = beat_strengths[window_start:window_end]
                
                if len(local_strengths) > 0 and np.max(local_strengths) > 0:
                    relative_strength = beat_strengths[i] / np.max(local_strengths)
                    confidence += relative_strength * 0.30
                else:
                    confidence += 0.15  # Medium confidence if no neighbors
                
                # Factor 2: Temporal consistency (40%)
                # How well does this beat align with the expected grid?
                if i > 0:
                    actual_interval = beats[i] - beats[i-1]
                    expected_interval = beat_period
                    
                    # Calculate deviation from expected interval
                    interval_error = abs(actual_interval - expected_interval) / expected_interval
                    
                    # High confidence if error < 10%, low if > 30%
                    if interval_error < 0.10:
                        temporal_score = 1.0
                    elif interval_error < 0.20:
                        temporal_score = 0.7
                    elif interval_error < 0.30:
                        temporal_score = 0.4
                    else:
                        temporal_score = 0.2
                    
                    confidence += temporal_score * 0.40
                else:
                    confidence += 0.30  # First beat gets decent confidence
                
                # Factor 3: Regularity of surrounding beats (30%)
                # Check consistency of intervals in local window
                if i >= 2 and i < len(beats) - 2:
                    # Get intervals in neighborhood
                    intervals = []
                    for j in range(max(0, i-2), min(len(beats)-1, i+3)):
                        intervals.append(beats[j+1] - beats[j])
                    
                    if len(intervals) > 1:
                        # Calculate coefficient of variation (std/mean)
                        interval_mean = np.mean(intervals)
                        interval_std = np.std(intervals)
                        
                        if interval_mean > 0:
                            cv = interval_std / interval_mean
                            
                            # Low CV means regular beats = high confidence
                            if cv < 0.10:
                                regularity_score = 1.0
                            elif cv < 0.20:
                                regularity_score = 0.7
                            elif cv < 0.35:
                                regularity_score = 0.4
                            else:
                                regularity_score = 0.2
                            
                            confidence += regularity_score * 0.30
                        else:
                            confidence += 0.15
                    else:
                        confidence += 0.15
                else:
                    # Edge beats get medium regularity score
                    confidence += 0.20
                
                # Ensure confidence stays in [0, 1] range
                beat_confidence[i] = np.clip(confidence, 0.0, 1.0)
            
            # Log statistics
            high_conf_count = np.sum(beat_confidence > 0.8)
            low_conf_count = np.sum(beat_confidence < 0.5)
            logger.info(f"    Beat confidence: mean={np.mean(beat_confidence):.3f}, high={high_conf_count}/{len(beats)}, low={low_conf_count}/{len(beats)}")
            
            return beat_confidence
            
        except Exception as e:
            logger.warning(f"    Beat confidence calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return np.ones(len(beats))  # Default to full confidence
    
    def _analyze_energy_progression(self, audio: np.ndarray, sr: int, beats: np.ndarray, bars: int = 8) -> Tuple[str, float]:
        """
        Analyze energy progression in the last N bars to detect building/dropping/stable patterns
        
        Args:
            audio: Audio signal
            sr: Sample rate
            beats: Beat times array
            bars: Number of bars to analyze (default: 8)
        
        Returns:
            Tuple of (progression_type, slope_magnitude)
            - progression_type: 'building', 'dropping', or 'stable'
            - slope_magnitude: Rate of energy change
        """
        try:
            if len(beats) < bars * 2:  # Need at least 2 beats per bar
                return 'stable', 0.0
            
            # Get last N bars worth of beats (assume 4 beats per bar)
            beats_per_bar = 4
            num_beats = min(bars * beats_per_bar, len(beats))
            last_beats = beats[-num_beats:]
            
            if len(last_beats) < 4:
                return 'stable', 0.0
            
            # Get audio for this section
            start_sample = int(last_beats[0] * sr)
            end_sample = min(len(audio), int(last_beats[-1] * sr) + sr)  # Add 1s buffer
            
            if end_sample <= start_sample:
                return 'stable', 0.0
            
            section_audio = audio[start_sample:end_sample]
            
            # Split into 4 segments
            segment_length = len(section_audio) // 4
            if segment_length < sr // 10:  # Too short
                return 'stable', 0.0
            
            energies = []
            for i in range(4):
                seg_start = i * segment_length
                seg_end = min((i + 1) * segment_length, len(section_audio))
                segment = section_audio[seg_start:seg_end]
                
                # Calculate RMS energy
                rms = np.sqrt(np.mean(segment**2))
                energies.append(rms)
            
            # Linear regression to find slope
            x = np.arange(len(energies))
            slope = np.polyfit(x, energies, 1)[0]
            
            # Normalize slope by mean energy
            mean_energy = np.mean(energies)
            if mean_energy > 0:
                normalized_slope = slope / mean_energy
            else:
                normalized_slope = 0.0
            
            # Classify based on slope
            if normalized_slope > 0.15:  # Building
                return 'building', float(normalized_slope)
            elif normalized_slope < -0.15:  # Dropping
                return 'dropping', float(normalized_slope)
            else:  # Stable
                return 'stable', float(normalized_slope)
            
        except Exception as e:
            logger.warning(f"    Energy progression analysis failed: {e}")
            return 'stable', 0.0
    
    def _detect_intro_outro(self, y: np.ndarray, sr: int, beats: np.ndarray) -> Tuple[float, float]:
        """
        Detect intro and outro sections for better transition points
        """
        duration = len(y) / sr
        
        # Simple heuristic: assume intro is first 16-32 beats, outro is last 16-32 beats
        if len(beats) > 32:
            intro_end = beats[min(16, len(beats)//4)]
            outro_start = beats[max(-16, -len(beats)//4)]
        else:
            intro_end = duration * 0.15  # 15% of track
            outro_start = duration * 0.85  # 85% of track
            
        return intro_end, outro_start
    
    def _detect_key_krumhansl(self, chroma: np.ndarray) -> Tuple[int, str, float]:
        """
        Detect musical key using Krumhansl-Schmuckler algorithm
        
        Args:
            chroma: Chroma feature matrix (12 x time_frames)
        
        Returns:
            Tuple of (key_index, mode, confidence)
            - key_index: 0-11 (C, C#, D, ..., B)
            - mode: 'major' or 'minor'
            - confidence: 0-1 correlation strength
        """
        try:
            # Average chroma across time to get overall pitch class distribution
            chroma_mean = np.mean(chroma, axis=1)
            
            # Normalize to sum to 1
            if np.sum(chroma_mean) > 0:
                chroma_mean = chroma_mean / np.sum(chroma_mean)
            else:
                return 0, 'major', 0.0
            
            best_correlation = -1
            best_key = 0
            best_mode = 'major'
            
            # Test all 24 possible keys (12 major + 12 minor)
            for key_idx in range(12):
                # Test major key
                # Rotate the major profile to match this key
                rotated_major = np.roll(self.MAJOR_PROFILE, key_idx)
                # Normalize profile
                rotated_major = rotated_major / np.sum(rotated_major)
                # Calculate Pearson correlation
                correlation_major = np.corrcoef(chroma_mean, rotated_major)[0, 1]
                
                if correlation_major > best_correlation:
                    best_correlation = correlation_major
                    best_key = key_idx
                    best_mode = 'major'
                
                # Test minor key
                rotated_minor = np.roll(self.MINOR_PROFILE, key_idx)
                rotated_minor = rotated_minor / np.sum(rotated_minor)
                correlation_minor = np.corrcoef(chroma_mean, rotated_minor)[0, 1]
                
                if correlation_minor > best_correlation:
                    best_correlation = correlation_minor
                    best_key = key_idx
                    best_mode = 'minor'
            
            # Convert correlation (-1 to 1) to confidence (0 to 1)
            # Correlation typically ranges from 0.3 to 0.9 for real keys
            confidence = np.clip((best_correlation - 0.3) / 0.6, 0.0, 1.0)
            
            return best_key, best_mode, confidence
            
        except Exception as e:
            logger.warning(f"    Key detection failed: {e}, using default")
            return 0, 'major', 0.0
    
    def _normalize_audio(self, audio: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
        """
        Normalize audio using EBU R128 loudness standard (broadcast quality)
        Handles both mono and stereo audio properly
        
        Args:
            audio: Input audio signal (mono or stereo)
            target_lufs: Target integrated loudness in LUFS (default: -14.0 for streaming)
        
        Returns:
            Loudness-normalized audio with proper peak limiting
        """
        if len(audio) == 0:
            return audio
        
        # Ensure audio is in correct format for pyloudnorm (samples, channels)
        # If stereo, should be shape (samples, 2), if mono should be (samples,)
        if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
            # If shape is (channels, samples), transpose to (samples, channels)
            audio = audio.T
        
        # Create EBU R128 meter
        meter = pyln.Meter(self.sample_rate)
        
        # Measure current loudness
        try:
            current_loudness = meter.integrated_loudness(audio)
        except Exception as e:
            # Fallback if audio format is incompatible - just return original
            logger.warning(f"    Loudness measurement failed: {e}, skipping normalization")
            return audio
        
        # Check if normalization would cause clipping
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
    
    def _ensure_smooth_flow(self, track1: Dict, track2: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Preserve full song duration while optimizing vocal-to-vocal transitions
        Returns: (audio1, audio2, intro_skip_samples) - skip amount for beat alignment adjustment
        """
        audio1 = track1['audio_data']
        audio2 = track2['audio_data']
        sr = track1['sample_rate']
        
        # Get vocal segments for both tracks
        vocal_segments1 = track1.get('vocal_segments', [])
        vocal_segments2 = track2.get('vocal_segments', [])
        
        # Get structure information for intelligent transitions
        structure_sections1 = track1.get('structure_sections', [])
        structure_sections2 = track2.get('structure_sections', [])
        beats1 = track1.get('beats', np.array([]))
        beats2 = track2.get('beats', np.array([]))
        
        # Find optimal structure-aware transition points
        crossfade_start_time, intro_skip_time, transition_meta = self._find_vocal_transition_points(
            vocal_segments1, vocal_segments2, 
            len(audio1) / sr, len(audio2) / sr,
            structure_sections1, structure_sections2,
            beats1, beats2
        )
        
        # Log transition strategy
        if transition_meta.get('confidence', 0) > 0.7:
            logger.info(f"    🎯 High-quality {transition_meta['strategy']} transition (confidence: {transition_meta['confidence']:.2f})")
        
        # Convert to samples with safety limits
        intro_skip_samples = int(intro_skip_time * sr)
        
        # Ensure intro skip doesn't remove too much or cause issues
        max_skip = int(len(audio2) * 0.2)  # Don't skip more than 20% of track
        intro_skip_samples = min(intro_skip_samples, max_skip)
        
        # Keep at least 1 second of intro for smooth transitions
        if intro_skip_samples > 0 and intro_skip_samples < sr:  # Less than 1 second
            intro_skip_samples = 0  # Keep full intro if skipping very little
        
        # Validate that we have enough audio left after skipping
        if len(audio2) - intro_skip_samples < sr * 2:  # Less than 2 seconds left
            logger.warning(f"    Intro skip too aggressive ({intro_skip_time:.1f}s), keeping full track")
            intro_skip_samples = 0
        
        # Keep full songs with minimal intro skip for vocal alignment
        full_audio1 = audio1  # Keep complete first song
        vocal_aligned_audio2 = audio2[intro_skip_samples:] if intro_skip_samples > 0 else audio2
        
        if intro_skip_samples > 0:
            logger.info(f"    Vocal-to-vocal transition: Crossfade at {crossfade_start_time:.1f}s, vocal intro skip {intro_skip_time:.1f}s")
        else:
            logger.info(f"    Using full tracks for transition (no intro skip)")
        
        return full_audio1, vocal_aligned_audio2, intro_skip_samples
    
    def _find_vocal_transition_points(self, vocal_segments1: List[Tuple[float, float]], 
                                    vocal_segments2: List[Tuple[float, float]], 
                                    duration1: float, duration2: float,
                                    structure_sections1: List[Tuple[float, float, str]] = None,
                                    structure_sections2: List[Tuple[float, float, str]] = None,
                                    beats1: np.ndarray = None,
                                    beats2: np.ndarray = None) -> Tuple[float, float, Dict]:
        """
        Find optimal transition points using song structure analysis
        
        Strategy Priority:
        1. Outro → Intro (most natural)
        2. Outro → Verse (good energy match)
        3. Bridge → Intro (transitional)
        4. Chorus end → Verse (dramatic shift)
        5. Instrumental → Intro (clean)
        6. Fallback to vocal detection only
        
        Returns:
            Tuple[float, float, Dict]: (crossfade_start_time, intro_skip_time, transition_metadata)
        """
        # Default fallback positions
        default_crossfade_start = duration1 - self.crossfade_duration
        default_intro_skip = 0.0
        transition_meta = {'strategy': 'fallback', 'section1': None, 'section2': None, 'confidence': 0.0}
        
        # STRUCTURE-AWARE TRANSITION (Priority 1)
        if structure_sections1 and structure_sections2:
            result = self._find_structure_transition(
                structure_sections1, structure_sections2, 
                duration1, duration2, beats1, beats2
            )
            if result:
                return result
        
        # VOCAL-AWARE FALLBACK (Priority 2)
        if not vocal_segments1 or not vocal_segments2:
            # If no vocals detected in one track, use fallback but still log info
            if vocal_segments1:
                logger.info(f"    Track1 has {len(vocal_segments1)} vocal segments, Track2 instrumental")
            elif vocal_segments2:
                logger.info(f"    Track1 instrumental, Track2 has {len(vocal_segments2)} vocal segments")
            else:
                logger.info(f"    Both tracks appear instrumental, using standard transition")
            return default_crossfade_start, default_intro_skip, transition_meta
        
        # Find vocal segments in the outro section of track1
        outro_start_time = duration1 * 0.7  # Look for vocals in last 30% of track
        track1_outro_vocals = [
            (start, end) for start, end in vocal_segments1 
            if start >= outro_start_time and end <= duration1
        ]
        
        # Find vocal segments in the intro section of track2
        intro_end_time = duration2 * 0.3  # Look for vocals in first 30% of track
        track2_intro_vocals = [
            (start, end) for start, end in vocal_segments2 
            if start >= 0 and end <= intro_end_time
        ]
        
        if not track1_outro_vocals and not track2_intro_vocals:
            logger.info(f"    No outro/intro vocals found, using standard transition")
            return default_crossfade_start, default_intro_skip, transition_meta
        
        # Strategy 1: Vocal outro to vocal intro
        if track1_outro_vocals and track2_intro_vocals:
            # Find the last vocal segment in track1 outro
            last_outro_vocal = max(track1_outro_vocals, key=lambda x: x[1])
            # Find the first vocal segment in track2 intro
            first_intro_vocal = min(track2_intro_vocals, key=lambda x: x[0])
            
            # Time crossfade to blend vocals naturally
            crossfade_start = last_outro_vocal[0] + (last_outro_vocal[1] - last_outro_vocal[0]) * 0.7
            # Be more conservative with intro skip to avoid cutting into music or exposing silence
            intro_skip = max(0, first_intro_vocal[0] - 2.0)  # Start 2 seconds before vocals (safer)
            intro_skip = min(intro_skip, duration2 * 0.15)  # Never skip more than 15% of intro
            
            transition_meta = {
                'strategy': 'vocal_to_vocal',
                'section1': f'outro_vocal_{last_outro_vocal[0]:.1f}s',
                'section2': f'intro_vocal_{first_intro_vocal[0]:.1f}s',
                'confidence': 0.7
            }
            logger.info(f"    Vocal-to-vocal: outro vocal at {last_outro_vocal[0]:.1f}-{last_outro_vocal[1]:.1f}s, intro vocal at {first_intro_vocal[0]:.1f}-{first_intro_vocal[1]:.1f}s")
            return crossfade_start, intro_skip, transition_meta
        
        # Strategy 2: Vocal outro to instrumental intro (let vocals finish)
        elif track1_outro_vocals and not track2_intro_vocals:
            last_outro_vocal = max(track1_outro_vocals, key=lambda x: x[1])
            # Start crossfade near end of last vocal
            crossfade_start = last_outro_vocal[1] - self.crossfade_duration * 0.3
            
            transition_meta = {
                'strategy': 'vocal_to_instrumental',
                'section1': f'outro_vocal_{last_outro_vocal[1]:.1f}s',
                'section2': 'instrumental_intro',
                'confidence': 0.6
            }
            logger.info(f"    Vocal outro to instrumental: outro vocal ends at {last_outro_vocal[1]:.1f}s")
            return crossfade_start, 0.0, transition_meta
        
        # Strategy 3: Instrumental outro to vocal intro (prepare for vocals)
        elif not track1_outro_vocals and track2_intro_vocals:
            first_intro_vocal = min(track2_intro_vocals, key=lambda x: x[0])
            # Start crossfade earlier to build up to vocals
            crossfade_start = duration1 - self.crossfade_duration * 1.2
            # Be very conservative - keep full intro for instrumental to vocal
            intro_skip = max(0, first_intro_vocal[0] - 3.0)  # Start 3 seconds before vocals
            intro_skip = min(intro_skip, duration2 * 0.1)  # Never skip more than 10% of intro
            
            transition_meta = {
                'strategy': 'instrumental_to_vocal',
                'section1': 'instrumental_outro',
                'section2': f'intro_vocal_{first_intro_vocal[0]:.1f}s',
                'confidence': 0.6
            }
            logger.info(f"    Instrumental to vocal intro: intro vocal starts at {first_intro_vocal[0]:.1f}s")
            return crossfade_start, intro_skip, transition_meta
        
        # Fallback to standard transition
        return default_crossfade_start, default_intro_skip, transition_meta
    
    def _find_structure_transition(self, sections1: List[Tuple],
                                   sections2: List[Tuple],
                                   duration1: float, duration2: float,
                                   beats1: np.ndarray = None,
                                   beats2: np.ndarray = None) -> Optional[Tuple[float, float, Dict]]:
        """
        Find optimal transition using song structure analysis with boring intro detection
        
        Priority mapping:
        0. Very Boring Intro → Skip to Verse (NEW - highest priority)
        1. Outro → Intro (natural flow)
        2. Outro → Verse (smooth energy)
        3. Bridge → Intro (transitional)
        4. Instrumental → Intro (clean)
        5. Chorus end → Verse (dramatic)
        """
        try:
            # Extract section types for easier lookup
            sections1_by_type = {}
            for section in sections1:
                start, end, label = section[0], section[1], section[2]
                if label not in sections1_by_type:
                    sections1_by_type[label] = []
                sections1_by_type[label].append((start, end))
            
            sections2_by_type = {}
            sections2_metadata = {}  # Track metadata for track2 sections
            for section in sections2:
                start, end, label = section[0], section[1], section[2]
                metadata = section[3] if len(section) > 3 else {}
                if label not in sections2_by_type:
                    sections2_by_type[label] = []
                sections2_by_type[label].append((start, end, metadata))
                sections2_metadata[(start, end)] = metadata
            
            # Find last section in track1 (for outro analysis)
            last_section1 = sections1[-1] if sections1 else None
            # Find first section in track2 (for intro analysis)
            first_section2 = sections2[0] if sections2 else None
            first_section2_metadata = first_section2[3] if first_section2 and len(first_section2) > 3 else {}
            
            # STRATEGY 0: Very Boring Intro → Skip directly to verse (HIGHEST PRIORITY)
            if (first_section2 and first_section2[2] == 'intro' and 
                first_section2_metadata.get('mood', {}).get('boring_score', 0) > 0.7 and
                'verse' in sections2_by_type):
                
                first_verse = min(sections2_by_type['verse'], key=lambda x: x[0])
                verse_start, verse_end = first_verse[0], first_verse[1]
                
                # Skip boring intro entirely, go to verse
                if last_section1:
                    outro_start, outro_end = last_section1[0], last_section1[1]
                    crossfade_start = outro_start + (outro_end - outro_start) * 0.65
                else:
                    crossfade_start = duration1 - self.crossfade_duration
                
                intro_skip = max(0, verse_start - 0.8)  # Minimal lead-in to verse
                
                if beats1 is not None and len(beats1) > 0:
                    crossfade_start = self._snap_to_beat(crossfade_start, beats1, max_offset=1.0)
                if beats2 is not None and len(beats2) > 0:
                    intro_skip = self._snap_to_beat(intro_skip, beats2, max_offset=0.5)
                
                logger.info(f"    🎯🎯 VERY BORING intro - direct skip to verse at {intro_skip:.1f}s (boring score: {first_section2_metadata['mood']['boring_score']:.2f})")
                return crossfade_start, intro_skip, {
                    'strategy': 'boring_intro_skip_to_verse',
                    'section1': last_section1[2] if last_section1 else 'unknown',
                    'section2': 'verse',
                    'confidence': 0.92,
                    'boring_score': first_section2_metadata['mood']['boring_score']
                }
            
            # STRATEGY 1: Outro → Intro (BEST) - Enhanced with boring intro detection
            if last_section1 and last_section1[2] == 'outro' and first_section2 and first_section2[2] == 'intro':
                outro_start, outro_end = last_section1[0], last_section1[1]
                intro_start, intro_end = first_section2[0], first_section2[1]
                
                # Start crossfade 70% into the outro
                crossfade_start = outro_start + (outro_end - outro_start) * 0.7
                
                # Enhanced intro skip logic based on mood analysis
                intro_mood = first_section2_metadata.get('mood', {})
                boring_score = intro_mood.get('boring_score', 0.0)
                
                if boring_score > 0.7:  # Very boring intro
                    # Skip almost entire intro, go straight to verse
                    intro_skip = intro_end * 0.85  # Skip 85% of boring intro
                    logger.info(f"    🎯 Very boring intro detected - aggressive skip to {intro_skip:.1f}s")
                elif boring_score > 0.5:  # Boring intro
                    # Skip most of intro
                    intro_skip = intro_end * 0.6  # Skip 60% of boring intro
                    logger.info(f"    ⚡ Boring intro detected - skip to {intro_skip:.1f}s")
                elif intro_end > 10.0:  # Long intro even if not boring
                    # Skip significant portion of long intros
                    intro_skip = min(intro_end * 0.4, intro_end - 3.0)  # Keep last 3s
                else:
                    # Normal intro - minimal skip
                    intro_skip = min(intro_start + 1.0, intro_end * 0.3) if intro_end > 3.0 else 0.0
                
                # Align to nearest beat for musical precision
                if beats1 is not None and len(beats1) > 0:
                    crossfade_start = self._snap_to_beat(crossfade_start, beats1, max_offset=1.0)
                if beats2 is not None and len(beats2) > 0 and intro_skip > 0:
                    intro_skip = self._snap_to_beat(intro_skip, beats2, max_offset=0.5)
                
                logger.info(f"    ✨ Structure: outro→intro transition at {crossfade_start:.1f}s, skip {intro_skip:.1f}s")
                return crossfade_start, intro_skip, {
                    'strategy': 'structure_outro_intro',
                    'section1': 'outro',
                    'section2': 'intro' if boring_score < 0.5 else 'intro_skipped',
                    'confidence': 0.95,
                    'boring_score': boring_score
                }
            
            # STRATEGY 2: Outro → Verse (GOOD) - Prioritized for boring intros
            if last_section1 and last_section1[2] == 'outro' and 'verse' in sections2_by_type:
                outro_start, outro_end = last_section1[0], last_section1[1]
                # Find first verse in track2
                first_verse = min(sections2_by_type['verse'], key=lambda x: x[0])
                verse_start, verse_end = first_verse[0], first_verse[1]
                
                # Check if intro is boring - if yes, skip directly to verse
                intro_mood = first_section2_metadata.get('mood', {})
                boring_score = intro_mood.get('boring_score', 0.0)
                
                # Start crossfade 60% into outro
                crossfade_start = outro_start + (outro_end - outro_start) * 0.6
                
                if boring_score > 0.5:  # Boring intro detected
                    # Skip directly to verse start with minimal lead-in
                    intro_skip = max(0, verse_start - 0.5)  # Just 0.5s before verse
                    logger.info(f"    🎯 Skipping boring intro, jumping to verse at {intro_skip:.1f}s")
                else:
                    # Normal transition - keep some buildup
                    intro_skip = max(0, verse_start - 1.5)
                
                if beats1 is not None and len(beats1) > 0:
                    crossfade_start = self._snap_to_beat(crossfade_start, beats1, max_offset=1.0)
                if beats2 is not None and len(beats2) > 0:
                    intro_skip = self._snap_to_beat(intro_skip, beats2, max_offset=0.5)
                
                logger.info(f"    ✨ Structure: outro→verse transition at {crossfade_start:.1f}s, skip to {intro_skip:.1f}s")
                return crossfade_start, intro_skip, {
                    'strategy': 'structure_outro_verse',
                    'section1': 'outro',
                    'section2': 'verse',
                    'confidence': 0.9,
                    'boring_score': boring_score
                }
            
            # No good structure-based transition found
            return None
            
        except Exception as e:
            logger.warning(f"    Structure transition analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _snap_to_beat(self, time: float, beats: np.ndarray, max_offset: float = 1.0) -> float:
        """
        Snap a time position to the nearest beat within max_offset seconds
        """
        try:
            if beats is None or len(beats) == 0:
                return time
            
            # Find nearest beat
            beat_diffs = np.abs(beats - time)
            nearest_idx = np.argmin(beat_diffs)
            nearest_beat = beats[nearest_idx]
            
            # Only snap if within max_offset
            if abs(nearest_beat - time) <= max_offset:
                return float(nearest_beat)
            
            return time
        except:
            return time
    
    def calculate_compatibility(self, track1: Dict, track2: Dict) -> float:
        """
        Calculate compatibility score between two tracks (0-1) with enhanced metrics
        Higher score means better transition
        """
        if not track1 or not track2:
            return 0.0
        
        # Use actual tempo (after doubling/halving detection)
        tempo1 = track1.get('actual_tempo', track1['tempo'])
        tempo2 = track2.get('actual_tempo', track2['tempo'])
        tempo_ratio = max(tempo1, tempo2) / min(tempo1, tempo2)
        
        # Genre-aware tempo matching
        genre1 = track1.get('genre_hint', 'unknown')
        genre2 = track2.get('genre_hint', 'unknown')
        
        # Check for harmonic ratios (2:1, 3:2, 4:3)
        if abs(tempo_ratio - 2.0) < 0.1 or abs(tempo_ratio - 1.5) < 0.1 or abs(tempo_ratio - 1.33) < 0.1:
            tempo_score = 0.9  # High score for harmonic ratios
        else:
            tempo_diff = abs(tempo1 - tempo2)
            
            # Genre-specific tolerance
            if genre1 == genre2:
                # Same genre - tighter tolerance
                if genre1 in ['electronic', 'hiphop']:
                    tolerance = 8  # Electronic music needs tight tempo matching
                elif genre1 in ['jazz', 'classical']:
                    tolerance = 25  # More flexible for jazz/classical
                elif genre1 in ['vietnamese_ballad']:
                    tolerance = 12  # Ballads can vary slightly in tempo
                elif genre1 in ['vietnamese_pop']:
                    tolerance = 10  # V-pop similar to pop
                elif genre1 == 'country':
                    tolerance = 12  # Country can vary in tempo but maintains flow
                else:
                    tolerance = 15  # Default
                tempo_score = max(0, 1 - (tempo_diff / tolerance))
            else:
                # Check for compatible genre pairs
                viet_genres = {'vietnamese_ballad', 'vietnamese_pop'}
                western_pop_genres = {'pop', 'rock', 'country'}
                
                # Vietnamese genres mix well with each other
                if genre1 in viet_genres and genre2 in viet_genres:
                    tolerance = 15
                # Vietnamese pop can mix with western pop
                elif (genre1 == 'vietnamese_pop' and genre2 in western_pop_genres) or \
                     (genre2 == 'vietnamese_pop' and genre1 in western_pop_genres):
                    tolerance = 18
                # Country mixes well with pop and rock
                elif (genre1 == 'country' and genre2 in {'pop', 'rock'}) or \
                     (genre2 == 'country' and genre1 in {'pop', 'rock'}):
                    tolerance = 16
                else:
                    tolerance = 20  # Default cross-genre
                
                tempo_score = max(0, 1 - (tempo_diff / tolerance))
        
        # Bonus for matching groove types
        groove1 = track1.get('groove_type', 'straight')
        groove2 = track2.get('groove_type', 'straight')
        groove_bonus = 0.1 if groove1 == groove2 and groove1 != 'straight' else 0
        
        tempo_score = min(1.0, tempo_score + groove_bonus)
        
        # Enhanced key compatibility with mode awareness (major/minor)
        key1, key2 = track1['key'], track2['key']
        mode1 = track1.get('key_mode', 'major')
        mode2 = track2.get('key_mode', 'major')
        key_conf1 = track1.get('key_confidence', 0.5)
        key_conf2 = track2.get('key_confidence', 0.5)
        
        # Calculate semitone distance on circle
        key_distance = min(abs(key1 - key2), 12 - abs(key1 - key2))
        
        # Determine key relationship
        if key1 == key2 and mode1 == mode2:
            # Perfect match (same key and mode)
            key_score = 1.0
        elif key1 == key2 and mode1 != mode2:
            # Parallel keys (C major / C minor) - acceptable but not perfect
            key_score = 0.70
        elif key_distance == 3:
            # Relative keys (C major / A minor, or E minor / G major)
            if mode1 != mode2:
                key_score = 0.85  # Very compatible
            else:
                key_score = 0.65  # Same mode, minor third apart
        elif key_distance == 7 or key_distance == 5:
            # Perfect fifth relationship (C / G)
            if mode1 == mode2:
                key_score = 0.80  # Strong harmonic relationship
            else:
                key_score = 0.65  # Fifth but different modes
        elif key_distance == 2 or key_distance == 10:
            # Whole step (C / D) - neighboring keys
            key_score = 0.60
        elif key_distance == 1 or key_distance == 11:
            # Half step (C / C#) - very dissonant
            key_score = 0.20
        else:
            # Other relationships
            key_score = max(0, 1 - (key_distance / 6))
        
        # Reduce key score if either detection has low confidence
        avg_confidence = (key_conf1 + key_conf2) / 2
        if avg_confidence < 0.5:
            key_score *= 0.7  # Reduce impact of uncertain keys
        
        # Energy compatibility with variation consideration
        energy_diff = abs(track1['energy'] - track2['energy'])
        max_energy = max(track1['energy'], track2['energy'], 0.1)
        energy_score = max(0, 1 - (energy_diff / max_energy))
        
        # Energy variation compatibility (smoother transitions)
        energy_var_diff = abs(track1['energy_variation'] - track2['energy_variation'])
        energy_var_score = max(0, 1 - energy_var_diff)
        
        # Spectral compatibility (multiple features)
        spectral_centroid_diff = abs(track1['spectral_centroid'] - track2['spectral_centroid'])
        spectral_score = max(0, 1 - (spectral_centroid_diff / 2000))
        
        # Timbral compatibility using MFCC
        mfcc_distance = np.linalg.norm(track1['mfcc_mean'] - track2['mfcc_mean'])
        timbral_score = max(0, 1 - (mfcc_distance / 50))
        
        # Rhythm compatibility using ZCR
        zcr_diff = abs(track1['zcr'] - track2['zcr'])
        rhythm_score = max(0, 1 - (zcr_diff / 0.1))
        
        # Weighted average with Apple Music-style priorities
        compatibility = (
            tempo_score * 0.30 +      # Tempo is crucial
            key_score * 0.25 +        # Key harmony important
            energy_score * 0.20 +     # Energy flow
            spectral_score * 0.10 +   # Timbre matching
            timbral_score * 0.10 +    # MFCC timbral
            rhythm_score * 0.05       # Rhythm consistency
        )
        
        return min(1.0, compatibility)
    
    def _determine_transition_style(self, track1: Dict, track2: Dict) -> Tuple[str, Dict]:
        """
        Determine optimal transition style based on musical context
        
        Returns:
            Tuple of (style_name, parameters_dict)
            
        Styles:
            - smooth_blend: Default, similar energy/genre
            - energy_punch: Low to high energy jump
            - build_drop: High to high energy with build-up
            - harmonic_layer: Compatible keys, long melodic blend
            - palate_cleanser: Incompatible keys/genres, needs separation
        """
        try:
            # Extract features
            energy1 = track1.get('energy', 0.1)
            energy2 = track2.get('energy', 0.1)
            key1 = track1.get('key', 0)
            key2 = track2.get('key', 0)
            key_conf1 = track1.get('key_confidence', 0.5)
            key_conf2 = track2.get('key_confidence', 0.5)
            mode1 = track1.get('key_mode', 'major')
            mode2 = track2.get('key_mode', 'major')
            genre1 = track1.get('genre_hint', 'unknown')
            genre2 = track2.get('genre_hint', 'unknown')
            tempo1 = track1.get('actual_tempo', track1['tempo'])
            tempo2 = track2.get('actual_tempo', track2['tempo'])
            
            # Get structure information for enhanced decisions
            sections1 = track1.get('structure_sections', [])
            sections2 = track2.get('structure_sections', [])
            main_section1 = track1.get('main_section')
            main_section2 = track2.get('main_section')
            
            # Analyze structural compatibility
            has_outro1 = any(s[2] == 'outro' for s in sections1) if sections1 else False
            has_intro2 = any(s[2] == 'intro' for s in sections2) if sections2 else False
            both_structured = bool(sections1) and bool(sections2)
            
            # Calculate key distance
            key_distance = min(abs(key1 - key2), 12 - abs(key1 - key2))
            avg_key_conf = (key_conf1 + key_conf2) / 2
            
            # Analyze energy progression in track1
            audio1 = track1.get('audio_data')
            beats1 = track1.get('beats', np.array([]))
            sr1 = track1.get('sample_rate', self.sample_rate)
            
            progression = 'stable'
            if audio1 is not None and len(beats1) > 0:
                progression, _ = self._analyze_energy_progression(audio1, sr1, beats1)
            
            # Decision tree for style selection
            
            # 1. PALATE CLEANSER: Incompatible keys or clashing genres
            incompatible_keys = key_distance in [1, 6, 8, 11]  # Dissonant intervals
            clashing_genres = (
                (genre1 == 'classical' and genre2 in ['electronic', 'hiphop']) or
                (genre2 == 'classical' and genre1 in ['electronic', 'hiphop']) or
                (genre1 == 'vietnamese_ballad' and genre2 == 'electronic')
            )
            
            # NOTE: Reduced palate_cleanser usage - with adaptive alignment, we can handle more transitions smoothly
            # Only use for SEVERE incompatibility (tritone interval or drastically clashing genres)
            severe_incompatibility = key_distance in [6]  # Tritone only (most dissonant)
            very_clashing_genres = (
                (genre1 == 'classical' and genre2 in ['hiphop']) or
                (genre2 == 'classical' and genre1 in ['hiphop'])
            )
            
            if (severe_incompatibility and avg_key_conf > 0.8) or very_clashing_genres:
                return 'palate_cleanser', {
                    'duration': 5.5,
                    'gap_duration': 1.5,
                    'fade_curve_power': 0.9,
                    'reason': 'severe_incompatibility' if severe_incompatibility else 'clashing_genres'
                }
            
            # 2. HARMONIC LAYER: Compatible keys with high confidence
            compatible_keys = key_distance in [0, 3, 5, 7]  # Unison, minor 3rd, 4th, 5th
            same_mode = mode1 == mode2
            
            if compatible_keys and avg_key_conf > 0.7 and same_mode:
                return 'harmonic_layer', {
                    'duration': 13.0,
                    'overlap_intensity': 0.75,
                    'fade_curve_power': 0.5,
                    'reason': f'compatible_keys_distance_{key_distance}'
                }
            
            # 3. ENERGY PUNCH: Low to high energy transition
            energy_jump = energy2 - energy1
            if energy1 < 0.12 and energy2 > 0.18 and energy_jump > 0.08:
                return 'energy_punch', {
                    'duration': 3.8,
                    'gap_duration': 0.3,
                    'fade_curve_power': 1.2,
                    'reason': f'energy_jump_{energy_jump:.2f}'
                }
            
            # 4. BUILD DROP: High energy to high energy with build-up
            both_high_energy = energy1 > 0.16 and energy2 > 0.16
            similar_genre = genre1 == genre2 or (genre1 in ['pop', 'electronic'] and genre2 in ['pop', 'electronic'])
            
            if both_high_energy and similar_genre and progression == 'building':
                return 'build_drop', {
                    'duration': 9.0,
                    'gap_duration': 0.5,
                    'fade_curve_power': 1.0,
                    'extend_build': 2.0,  # Extend track1 by 2 seconds to complete build
                    'reason': 'high_energy_build'
                }
            
            # 5. SMOOTH BLEND: Default with genre-adaptive parameters
            # Determine optimal duration and curve based on genre characteristics
            base_duration = 8.0
            base_curve = 0.7
            
            # Genre-specific adaptations
            if genre1 == 'jazz' or genre2 == 'jazz':
                base_duration = 10.0  # Jazz needs longer, musical transitions
                base_curve = 0.6  # Gentler for improvisational feel
            elif genre1 == 'classical' or genre2 == 'classical':
                base_duration = 12.0  # Classical needs very long, orchestral transitions
                base_curve = 0.5  # Very gentle for dynamic range
            elif genre1 == 'vietnamese_ballad' or genre2 == 'vietnamese_ballad':
                base_duration = 10.0  # Ballads need emotional, long transitions
                base_curve = 0.65  # Smooth and romantic
            elif genre1 == 'cuba_bolero' or genre2 == 'cuba_bolero':
                base_duration = 9.0  # Bolero needs romantic, flowing transitions
                base_curve = 0.7  # Balanced smoothness
            elif genre1 == 'future_funk' or genre2 == 'future_funk':
                base_duration = 7.0  # Funk needs groovier, tighter transitions
                base_curve = 0.75  # Slightly more punchy
            elif genre1 == 'house' or genre2 == 'house':
                base_duration = 8.0  # House needs tight, energetic transitions
                base_curve = 0.8  # Smooth but punchy for club feel
            elif genre1 == 'electronic' and genre2 == 'electronic':
                base_duration = 8.0  # Electronic can sustain long mixes
                base_curve = 0.8  # Smoother for beat-matched content
            elif genre1 == 'hiphop' or genre2 == 'hiphop':
                base_duration = 5.0  # Hip-hop typically shorter
                base_curve = 0.9  # More abrupt for rhythmic impact
            
            # Tempo-based fine-tuning
            avg_tempo = (tempo1 + tempo2) / 2
            if avg_tempo < 80:  # Slow songs need longer transitions
                base_duration *= 1.2
            elif avg_tempo > 140:  # Fast songs can handle tighter transitions
                base_duration *= 0.85
            
            # Energy-based adjustments
            avg_energy = (energy1 + energy2) / 2
            if avg_energy < 0.1:  # Very quiet/ambient songs
                base_duration *= 1.15
                base_curve *= 0.9  # Even gentler
            elif avg_energy > 0.2:  # High energy songs
                base_duration *= 0.9
            
            # STRUCTURE-AWARE ENHANCEMENT: Adjust duration based on section types
            if both_structured and has_outro1 and has_intro2:
                # Perfect outro→intro gets longer, more musical blend
                base_duration *= 1.3
                base_curve *= 0.85  # Gentler for structural transitions
                reason_suffix = '_perfect_structure'
            elif both_structured:
                # Good structure info, optimize duration
                base_duration *= 1.1
                reason_suffix = '_structure_aware'
            else:
                reason_suffix = '_default'
            
            return 'smooth_blend', {
                'duration': base_duration,
                'fade_curve_power': base_curve,
                'overlap_boost': 0.5,  # Standard overlap
                'reason': f'adaptive{reason_suffix}_genre_{genre1}_to_{genre2}'
            }
            
        except Exception as e:
            logger.warning(f"    Transition style determination failed: {e}, using smooth_blend")
            return 'smooth_blend', {'duration': 8.0, 'fade_curve_power': 0.7, 'reason': 'fallback'}
    
    def _calculate_optimal_crossfade_duration(self, track1: Dict, track2: Dict) -> Tuple[float, Dict]:
        """
        Calculate optimal crossfade duration and style parameters
        
        Returns:
            Tuple of (duration, style_params)
        """
        # Determine transition style first
        style, style_params = self._determine_transition_style(track1, track2)
        
        # Get base duration from style
        duration = style_params.get('duration', self.crossfade_duration)
        
        # Still apply genre/tempo adjustments for fine-tuning
        tempo1 = track1.get('actual_tempo', track1['tempo'])
        tempo2 = track2.get('actual_tempo', track2['tempo'])
        genre1 = track1.get('genre_hint', 'unknown')
        genre2 = track2.get('genre_hint', 'unknown')
        groove1 = track1.get('groove_type', 'straight')
        groove2 = track2.get('groove_type', 'straight')
        
        tempo_diff = abs(tempo1 - tempo2)
        
        # Base duration from settings
        base_duration = self.crossfade_duration
        
        # Adjust based on tempo difference
        if tempo_diff < 5:
            duration = base_duration * 1.4  # Longer for very similar tempos (more fluent)
        elif tempo_diff < 10:
            duration = base_duration * 1.2  # Slightly longer for smooth blend
        elif tempo_diff < 20:
            duration = base_duration
        else:
            duration = base_duration * 0.85  # Still fairly long even for big differences
        
        # Genre-specific adjustments with phrase awareness
        phrases1 = track1.get('phrases', [])
        phrases2 = track2.get('phrases', [])
        
        if genre1 == genre2:
            if genre1 == 'electronic':
                # EDM typically uses longer, beat-matched transitions
                beats_per_bar = 4
                bar_duration = (60.0 / tempo1) * beats_per_bar
                # Align to 4, 8, or 16 bars based on tempo similarity
                target_bars = 8 if tempo_diff < 5 else 4
                duration = bar_duration * target_bars
                logger.info(f"    Electronic genre: {target_bars}-bar crossfade ({duration:.1f}s)")
            elif genre1 == 'hiphop':
                # Hip-hop often uses shorter, punch transitions
                # Align to 2 or 4-bar phrases if possible
                beats_per_bar = 4
                bar_duration = (60.0 / tempo1) * beats_per_bar
                target_bars = 2 if tempo_diff > 10 else 4
                duration = bar_duration * target_bars
            elif genre1 in ['jazz', 'classical']:
                # Jazz/classical use longer, musical phrase-based transitions
                # Try to align with natural phrase boundaries (typically 8-16 bars)
                if phrases1 and len(phrases1) > 0:
                    # Calculate average phrase length from phrase tuples (start, end, bars)
                    # Each phrase is (start_time, end_time, bar_count)
                    phrase_durations = [phrase[1] - phrase[0] for phrase in phrases1]
                    if phrase_durations:
                        avg_phrase_duration = np.mean(phrase_durations)
                        duration = min(avg_phrase_duration, base_duration * 1.8)
                    else:
                        duration = base_duration * 1.5
                else:
                    duration = base_duration * 1.5
            elif genre1 == 'vietnamese_ballad':
                # Ballads use longer, emotional transitions
                # Align with melodic phrases (often 4 or 8 bars)
                duration = base_duration * 1.3
            elif genre1 == 'cuba_bolero':
                # Bolero uses romantic, flowing 4-8 bar transitions
                beats_per_bar = 4
                bar_duration = (60.0 / tempo1) * beats_per_bar
                target_bars = 6  # Sweet spot for bolero feel
                duration = bar_duration * target_bars
            elif genre1 == 'future_funk':
                # Funk uses tight, groovy 2-4 bar transitions
                beats_per_bar = 4
                bar_duration = (60.0 / tempo1) * beats_per_bar
                target_bars = 4 if tempo_diff < 5 else 2
                duration = bar_duration * target_bars
            elif genre1 == 'house':
                # House uses tight 4/8/16 bar phrase transitions
                beats_per_bar = 4
                bar_duration = (60.0 / tempo1) * beats_per_bar
                # House is very strict about phrase lengths
                if tempo_diff < 3:
                    target_bars = 16  # Long blend for same tempo
                elif tempo_diff < 6:
                    target_bars = 8  # Medium blend
                else:
                    target_bars = 4  # Quick transition for tempo change
                duration = bar_duration * target_bars
            elif genre1 == 'vietnamese_pop':
                # V-pop similar to pop, moderate duration
                duration = base_duration * 1.0
            elif genre1 == 'country':
                # Country storytelling pacing
                if tempo1 < 110:  # Ballad pacing
                    duration = base_duration * 1.2
                else:  # Uptempo country
                    duration = base_duration * 1.0
        
        # Groove matching bonus
        if groove1 == groove2 and groove1 != 'straight':
            # Matching grooves can sustain longer crossfades
            duration *= 1.1
        elif groove1 != groove2:
            # Different grooves need shorter transitions
            duration *= 0.9
        
        # Ensure reasonable bounds (but respect style-specific durations)
        if style == 'energy_punch':
            duration = max(3.0, min(5.0, duration))
        elif style == 'harmonic_layer':
            duration = max(10.0, min(18.0, duration))
        else:
            duration = max(4.0, min(16.0, duration))
        
        # Update style_params with final duration
        style_params['duration'] = duration
        style_params['style'] = style
        
        return duration, style_params
    
    def create_crossfade(self, track1_audio: np.ndarray, track2_audio: np.ndarray, 
                        crossfade_samples: int, track1_tempo: float = None, track2_tempo: float = None,
                        style_params: Dict = None) -> np.ndarray:
        """
        Create contextual crossfade - dispatcher to style-specific methods
        
        Args:
            track1_audio: Audio from first track
            track2_audio: Audio from second track
            crossfade_samples: Number of samples for crossfade
            track1_tempo: Tempo of first track
            track2_tempo: Tempo of second track
            style_params: Style parameters dict from _determine_transition_style
        
        Returns:
            Combined audio with crossfade applied
        """
        # Default to smooth_blend if no style specified
        if style_params is None:
            style_params = {'style': 'smooth_blend', 'duration': 6.5, 'fade_curve_power': 0.8}
        
        style = style_params.get('style', 'smooth_blend')
        
        logger.info(f"  Applying '{style}' transition style (reason: {style_params.get('reason', 'unknown')})")
        
        # Route to appropriate style handler
        if style == 'energy_punch':
            return self._crossfade_energy_punch(track1_audio, track2_audio, crossfade_samples,
                                               track1_tempo, track2_tempo, style_params)
        elif style == 'build_drop':
            return self._crossfade_build_drop(track1_audio, track2_audio, crossfade_samples,
                                             track1_tempo, track2_tempo, style_params)
        elif style == 'harmonic_layer':
            return self._crossfade_harmonic_layer(track1_audio, track2_audio, crossfade_samples,
                                                 track1_tempo, track2_tempo, style_params)
        elif style == 'palate_cleanser':
            return self._crossfade_palate_cleanser(track1_audio, track2_audio, crossfade_samples,
                                                  track1_tempo, track2_tempo, style_params)
        else:  # smooth_blend or fallback
            return self._crossfade_smooth_blend(track1_audio, track2_audio, crossfade_samples,
                                               track1_tempo, track2_tempo, style_params)
    
    def _match_audio_dimensions(self, *arrays) -> tuple:
        """
        Ensure all audio arrays have matching dimensions
        ALWAYS converts to STEREO to preserve original channel information
        
        Returns:
            Tuple of matched stereo arrays
        """
        if not arrays or len(arrays) == 0:
            return arrays
        
        # Check if any array is stereo - if so, convert all to stereo
        has_stereo = any(arr.ndim > 1 for arr in arrays if arr is not None and len(arr) > 0)
        
        matched = []
        for arr in arrays:
            if arr is None or len(arr) == 0:
                matched.append(arr)
            elif arr.ndim == 1:
                # Convert mono to stereo by duplicating channels
                matched.append(np.column_stack([arr, arr]))
            else:
                # Already stereo
                matched.append(arr)
        
        return tuple(matched)
    
    def _crossfade_smooth_blend(self, track1_audio: np.ndarray, track2_audio: np.ndarray,
                               crossfade_samples: int, track1_tempo: float = None,
                               track2_tempo: float = None, style_params: Dict = None) -> np.ndarray:
        """
        Standard smooth blend crossfade - extracted from original create_crossfade
        Default Apple Music-style crossfade with tempo adjustment during transition
        """
        # Ensure we don't exceed track lengths
        min_length = min(len(track1_audio), len(track2_audio))
        
        # Minimum track length check (need at least 2 seconds)
        if min_length < self.sample_rate * 2:
            logger.warning(f"  Track too short for crossfade ({min_length/self.sample_rate:.1f}s), using simple concatenation")
            return np.concatenate([track1_audio, track2_audio])
        
        # Adjust crossfade if tracks are short
        if crossfade_samples > min_length * 0.5:  # Don't use more than 50% of shortest track
            original_duration = crossfade_samples / self.sample_rate
            crossfade_samples = int(min_length * 0.5)
            new_duration = crossfade_samples / self.sample_rate
            logger.warning(f"  Reducing crossfade from {original_duration:.1f}s to {new_duration:.1f}s (track too short)")
        
        crossfade_samples = min(crossfade_samples, len(track1_audio), len(track2_audio))
        
        # Get crossfade sections
        track1_end = track1_audio[-crossfade_samples:].copy()
        track2_start = track2_audio[:crossfade_samples].copy()
        
        # DISABLED: All volume adjustments - use original track levels
        volume_adjustment = 1.0  # No adjustment
        
        # Apply tempo ramping for interesting transitions
        # Slight acceleration on Track 1, slight deceleration on Track 2
        # Creates excitement and anticipation while preserving pitch
        adjusted_track1_end = self._apply_tempo_ramp(
            track1_end, track1_tempo, track2_tempo, 
            ramp_type='accelerate', style_params=style_params
        )
        adjusted_track2_start = self._apply_tempo_ramp(
            track2_start, track2_tempo, track1_tempo,
            ramp_type='decelerate', style_params=style_params
        )
        
        # Create ultra-smooth invisible fade curves with advanced smoothing
        fade_curve = np.linspace(0, 1, crossfade_samples)
        
        # Use advanced equal-power crossfading for fuller sound
        # Apply gentle S-curve for smooth, natural perception with more overlap
        def clarity_s_curve(x):
            # Gentle S-curve for smoother, fuller transitions
            # Less steep = more overlap = fuller sound
            return x  # Linear for maximum overlap and fullness
        
        # Apply clarity-enhanced S-curve
        smooth_fade = np.array([clarity_s_curve(f) for f in fade_curve])
        
        # Create fuller equal-power crossfade with adaptive overlap boost
        # Use adaptive exponential curves based on style parameters
        curve_power = style_params.get('fade_curve_power', 0.7) if style_params else 0.7
        fade_out = np.power(np.cos(smooth_fade * np.pi / 2), curve_power)
        fade_in = np.power(np.sin(smooth_fade * np.pi / 2), curve_power)
        
        # Add adaptive overlap boost based on style parameters
        # This creates more intentional overlap for seamless, imperceptible transitions
        overlap_boost = style_params.get('overlap_boost', 0.5) if style_params else 0.5
        overlap_curve = np.sin(smooth_fade * np.pi) * overlap_boost  # Peak boost in the middle
        fade_out = fade_out + overlap_curve
        fade_in = fade_in + overlap_curve
        
        # Normalize but keep the boost for fuller sound
        fade_sum = fade_out + fade_in
        fade_out = fade_out / (fade_sum + 1e-10)
        fade_in = fade_in / (fade_sum + 1e-10)
        
        # Apply additional smoothing passes for invisible transitions
        if crossfade_samples > 256:
            # Multi-pass smoothing with decreasing intensity
            for pass_num in range(3):
                window_size = max(5, min(41, crossfade_samples // (20 + pass_num * 10)))
                if window_size >= 5:
                    fade_out = signal.savgol_filter(fade_out, window_size | 1, 2)
                    fade_in = signal.savgol_filter(fade_in, window_size | 1, 2)
            
            # Ensure fade curves maintain proper bounds
            fade_out = np.clip(fade_out, 0, 1)
            fade_in = np.clip(fade_in, 0, 1)
            
            # Force exact start and end points
            fade_out[0] = 1.0
            fade_out[-1] = 0.0
            fade_in[0] = 0.0
            fade_in[-1] = 1.0
        
        # Apply adaptive micro-ramping at edges based on genre and tempo
        # Longer ramps for slower/smoother genres, shorter for energetic/punchy ones
        base_ramp_samples = 64  # Default
        if style_params:
            # Genre-based adaptive ramp sizing
            fade_curve_power = style_params.get('fade_curve_power', 0.7)
            if fade_curve_power < 0.6:  # Classical, jazz - very gentle
                base_ramp_samples = 96
            elif fade_curve_power > 0.85:  # Hip-hop, punk - punchy
                base_ramp_samples = 48
        
        # Ensure fade curves match the (possibly changed) audio lengths after tempo ramping
        actual_crossfade_samples = min(len(adjusted_track1_end), len(adjusted_track2_start))
        if actual_crossfade_samples != crossfade_samples:
            # Resample fade curves to match new length
            old_indices = np.linspace(0, crossfade_samples - 1, crossfade_samples)
            new_indices = np.linspace(0, crossfade_samples - 1, actual_crossfade_samples)
            fade_out = np.interp(new_indices, old_indices, fade_out)
            fade_in = np.interp(new_indices, old_indices, fade_in)
            crossfade_samples = actual_crossfade_samples
            
            # Trim audio to match
            adjusted_track1_end = adjusted_track1_end[:crossfade_samples]
            adjusted_track2_start = adjusted_track2_start[:crossfade_samples]
        
        ramp_samples = min(base_ramp_samples, crossfade_samples // 16)
        if ramp_samples > 0:
            # Ultra-gentle ramps using raised cosine
            ramp_curve = (1 - np.cos(np.linspace(0, np.pi, ramp_samples))) / 2
            
            # Apply to start
            fade_out[:ramp_samples] *= ramp_curve
            fade_in[:ramp_samples] *= ramp_curve
            
            # Apply to end (reverse curve)
            fade_out[-ramp_samples:] *= ramp_curve[::-1]
            fade_in[-ramp_samples:] *= ramp_curve[::-1]
        
        # Apply vocal-aware frequency crossfading
        crossfade_section = self._vocal_aware_crossfade(
            adjusted_track1_end, adjusted_track2_start, fade_out, fade_in
        )
        
        # Apply gentle final limiting to prevent clipping only
        peak = np.max(np.abs(crossfade_section))
        if peak > 0.95:
            crossfade_section = crossfade_section * (0.95 / peak)
        
        # DISABLED: All volume adjustments - use original levels
        # FIXED: Don't cut track1 - keep it full and fade during overlap
        # This prevents cutting songs short at the end
        track1_prefix = track1_audio[:-crossfade_samples] if len(track1_audio) > crossfade_samples else np.array([])
        track2_remainder = track2_audio[crossfade_samples:]
        
        # Ensure all parts have matching dimensions before concatenation
        track1_prefix, crossfade_section, track2_remainder = self._match_audio_dimensions(
            track1_prefix, crossfade_section, track2_remainder
        )
        
        # Combine: track1 (full, faded at end) + track2 (remainder after crossfade)
        # Note: crossfade_section already contains the overlapped portion of both tracks
        if len(track1_prefix) > 0:
            result = np.concatenate([
                track1_prefix,      # Beginning of track1 (before crossfade)
                crossfade_section,  # Overlapped crossfade section
                track2_remainder    # Remainder of track2 (after crossfade)
            ])
        else:
            # Track1 is shorter than crossfade - just blend what we have
            result = np.concatenate([crossfade_section, track2_remainder])
        
        return result
    
    def _crossfade_energy_punch(self, track1_audio: np.ndarray, track2_audio: np.ndarray,
                               crossfade_samples: int, track1_tempo: float = None,
                               track2_tempo: float = None, style_params: Dict = None) -> np.ndarray:
        """
        Energy punch transition - short, impactful, with optional gap
        For low-energy to high-energy transitions
        """
        gap_duration = style_params.get('gap_duration', 0.3)
        fade_curve_power = style_params.get('fade_curve_power', 1.2)
        
        # Match dimensions first
        track1_audio, track2_audio = self._match_audio_dimensions(track1_audio, track2_audio)
        
        gap_samples = int(gap_duration * self.sample_rate)
        actual_crossfade = crossfade_samples - gap_samples
        
        if actual_crossfade < self.sample_rate:  # Minimum 1 second
            actual_crossfade = crossfade_samples
            gap_samples = 0
        
        # Split into fadeout and fadein sections
        fadeout_samples = actual_crossfade // 2
        fadein_samples = actual_crossfade - fadeout_samples
        
        track1_end = track1_audio[-fadeout_samples:].copy()
        track2_start = track2_audio[fadein_samples:fadein_samples * 2].copy()
        
        # Create steeper fade curves for punch
        fadeout_curve = np.linspace(1, 0, fadeout_samples) ** fade_curve_power
        fadein_curve = np.linspace(0, 1, fadein_samples) ** (1.0 / fade_curve_power)
        
        # Apply fades with proper dimension handling
        if track1_end.ndim > 1:
            fadeout_curve = fadeout_curve.reshape(-1, 1)
            fadein_curve = fadein_curve.reshape(-1, 1)
        
        track1_faded = track1_end * fadeout_curve
        track2_faded = track2_start * fadein_curve
        
        # Create gap with decay (not pure silence)
        if gap_samples > 0:
            gap_section = np.zeros((gap_samples, track1_end.shape[1])) if track1_end.ndim > 1 else np.zeros(gap_samples)
            # Add subtle reverb tail from track1
            if len(track1_end) > 0:
                tail_length = min(gap_samples, len(track1_end) // 4)
                decay_curve = np.exp(-5 * np.linspace(0, 1, tail_length))
                if track1_end.ndim > 1:
                    decay_curve = decay_curve.reshape(-1, 1)
                gap_section[:tail_length] = track1_end[-tail_length:] * decay_curve * 0.3
            
            logger.info(f"    Energy punch with {gap_duration:.1f}s dramatic gap")
        
        # Combine - keep full track1
        track1_prefix = track1_audio[:-fadeout_samples] if len(track1_audio) > fadeout_samples else np.array([])
        track2_remainder = track2_audio[fadein_samples * 2:]
        
        parts = [p for p in [
            track1_prefix,
            track1_faded,
            gap_section if gap_samples > 0 else None,
            track2_faded,
            track2_remainder
        ] if p is not None and len(p) > 0]
        
        result = np.concatenate(parts)
        
        return result
    
    def _crossfade_build_drop(self, track1_audio: np.ndarray, track2_audio: np.ndarray,
                             crossfade_samples: int, track1_tempo: float = None,
                             track2_tempo: float = None, style_params: Dict = None) -> np.ndarray:
        """
        Build and drop transition - let track1 build to peak, brief gap, drop into track2
        For high-energy to high-energy transitions
        """
        gap_duration = style_params.get('gap_duration', 0.5)
        extend_build = style_params.get('extend_build', 2.0)
        
        # Match dimensions first
        track1_audio, track2_audio = self._match_audio_dimensions(track1_audio, track2_audio)
        
        gap_samples = int(gap_duration * self.sample_rate)
        extend_samples = int(extend_build * self.sample_rate)
        
        # Extend track1 to let build complete
        # Use last bar and fade out
        fadeout_samples = crossfade_samples // 2
        track1_extended = track1_audio.copy()
        
        # Create fadeout from extended endpoint
        track1_end = track1_extended[-fadeout_samples:].copy()
        fadeout_curve = np.power(np.linspace(1, 0, fadeout_samples), 0.8)
        if track1_end.ndim > 1:
            fadeout_curve = fadeout_curve.reshape(-1, 1)
        track1_faded = track1_end * fadeout_curve
        
        # Gap (dramatic pause)
        gap_section = np.zeros((gap_samples, track1_end.shape[1])) if gap_samples > 0 and track1_end.ndim > 1 else np.zeros(gap_samples) if gap_samples > 0 else np.array([])
        
        # Sharp fadein for track2 (the "drop")
        fadein_samples = crossfade_samples // 3  # Shorter fadein for impact
        track2_start = track2_audio[:fadein_samples * 2].copy()
        fadein_curve = np.power(np.linspace(0, 1, fadein_samples), 1.5)  # Sharp attack
        if track2_start.ndim > 1:
            fadein_curve = fadein_curve.reshape(-1, 1)
        track2_faded = track2_start[:fadein_samples] * fadein_curve
        
        logger.info(f"    Build & drop with {gap_duration:.1f}s pause, {extend_build:.1f}s build extension")
        
        # Combine - keep full track1
        track1_prefix = track1_audio[:-fadeout_samples] if len(track1_audio) > fadeout_samples else np.array([])
        track2_remainder = track2_audio[fadein_samples:]
        
        parts = [p for p in [
            track1_prefix,
            track1_faded,
            gap_section if len(gap_section) > 0 else None,
            track2_faded,
            track2_remainder
        ] if p is not None and len(p) > 0]
        
        result = np.concatenate(parts)
        
        return result
    
    def _crossfade_harmonic_layer(self, track1_audio: np.ndarray, track2_audio: np.ndarray,
                                  crossfade_samples: int, track1_tempo: float = None,
                                  track2_tempo: float = None, style_params: Dict = None) -> np.ndarray:
        """
        Harmonic layer transition - long overlap with both tracks audible
        For compatible keys, let harmonies blend
        """
        overlap_intensity = style_params.get('overlap_intensity', 0.75)
        fade_curve_power = style_params.get('fade_curve_power', 0.5)
        
        # Match dimensions first
        track1_audio, track2_audio = self._match_audio_dimensions(track1_audio, track2_audio)
        
        # Ensure we have enough audio
        crossfade_samples = min(crossfade_samples, len(track1_audio), len(track2_audio))
        
        track1_end = track1_audio[-crossfade_samples:].copy()
        track2_start = track2_audio[:crossfade_samples].copy()
        
        # Gentle S-curve for smooth perception
        fade_curve = np.linspace(0, 1, crossfade_samples)
        smooth_curve = fade_curve ** fade_curve_power  # Gentler than smooth_blend
        
        # Both tracks stay relatively loud in the middle
        fade_out = np.power(np.cos(smooth_curve * np.pi / 2), fade_curve_power)
        fade_in = np.power(np.sin(smooth_curve * np.pi / 2), fade_curve_power)
        
        # Scale to allow overlap (both at ~75% in middle instead of 50%)
        fade_out = fade_out * (1 - overlap_intensity * 0.5)  + overlap_intensity * 0.5
        fade_in = fade_in * (1 - overlap_intensity * 0.5) + overlap_intensity * 0.5
        
        # Apply fades with proper dimension handling
        if track1_end.ndim > 1:
            fade_out_shaped = fade_out.reshape(-1, 1)
            fade_in_shaped = fade_in.reshape(-1, 1)
        else:
            fade_out_shaped = fade_out
            fade_in_shaped = fade_in
        
        track1_faded = track1_end * fade_out_shaped
        track2_faded = track2_start * fade_in_shaped
        
        # Mix with emphasis on harmonic content
        crossfade_section = track1_faded + track2_faded
        
        # Normalize to prevent clipping while preserving dynamics
        peak = np.max(np.abs(crossfade_section))
        if peak > 0.95:
            crossfade_section = crossfade_section * (0.95 / peak)
        
        logger.info(f"    Harmonic layer with {overlap_intensity:.0%} overlap intensity")
        
        # Combine - keep full track1
        track1_prefix = track1_audio[:-crossfade_samples] if len(track1_audio) > crossfade_samples else np.array([])
        track2_remainder = track2_audio[crossfade_samples:]
        
        if len(track1_prefix) > 0:
            result = np.concatenate([
                track1_prefix,
                crossfade_section,
                track2_remainder
            ])
        else:
            result = np.concatenate([crossfade_section, track2_remainder])
        
        return result
    
    def _crossfade_palate_cleanser(self, track1_audio: np.ndarray, track2_audio: np.ndarray,
                                   crossfade_samples: int, track1_tempo: float = None,
                                   track2_tempo: float = None, style_params: Dict = None) -> np.ndarray:
        """
        Palate cleanser transition - complete separation with gap
        For incompatible keys/genres that need a "reset"
        """
        gap_duration = style_params.get('gap_duration', 1.5)
        fade_curve_power = style_params.get('fade_curve_power', 0.9)
        
        gap_samples = int(gap_duration * self.sample_rate)
        
        # Split crossfade time into fadeout, gap, fadein
        fade_samples = (crossfade_samples - gap_samples) // 2
        if fade_samples < self.sample_rate // 2:  # Minimum 0.5s fades
            fade_samples = crossfade_samples // 3
            gap_samples = crossfade_samples - (2 * fade_samples)
        
        # Complete fadeout of track1
        track1_end = track1_audio[-fade_samples:].copy()
        fadeout_curve = np.power(np.linspace(1, 0, fade_samples), fade_curve_power)
        if track1_end.ndim > 1:
            fadeout_curve = fadeout_curve.reshape(-1, 1)
        track1_faded = track1_end * fadeout_curve
        
        # Gap with subtle ambient/reverb - match dimensions with track audio
        is_stereo = track1_audio.ndim > 1
        if is_stereo:
            gap_section = np.zeros((gap_samples, 2))
        else:
            gap_section = np.zeros(gap_samples)
            
        if gap_samples > self.sample_rate // 4 and len(track1_end) > 0:
            # Add reverb tail from track1 (not pure silence)
            tail_length = min(gap_samples, self.sample_rate)
            decay_curve = np.exp(-3 * np.linspace(0, 1, tail_length))
            if is_stereo:
                decay_curve_shaped = decay_curve.reshape(-1, 1)
                gap_section[:tail_length] = track1_end[-tail_length:] * decay_curve_shaped * 0.2
            else:
                gap_section[:tail_length] = track1_end[-tail_length:] * decay_curve * 0.2
        
        # Fresh fadein of track2
        track2_start = track2_audio[:fade_samples].copy()
        fadein_curve = np.power(np.linspace(0, 1, fade_samples), 1.0 / fade_curve_power)
        if track2_start.ndim > 1:
            fadein_curve = fadein_curve.reshape(-1, 1)
        track2_faded = track2_start * fadein_curve
        
        logger.info(f"    Palate cleanser with {gap_duration:.1f}s separation (reset for new musical context)")
        
        # Combine - keep full track1
        track1_prefix = track1_audio[:-fade_samples] if len(track1_audio) > fade_samples else np.array([])
        track2_remainder = track2_audio[fade_samples:]
        
        # Ensure all parts have matching dimensions before concatenating
        if len(track1_prefix) > 0:
            parts = self._match_audio_dimensions(
                track1_prefix,
                track1_faded,
                gap_section,
                track2_faded,
                track2_remainder
            )
        else:
            parts = self._match_audio_dimensions(
                track1_faded,
                gap_section,
                track2_faded,
                track2_remainder
            )
        
        # Combine
        result = np.concatenate(parts)
        
        return result
    
    def _apply_tempo_ramp(self, audio: np.ndarray, current_tempo: float, 
                          target_tempo: float, ramp_type: str = 'accelerate',
                          style_params: Dict = None) -> np.ndarray:
        """
        Apply subtle tempo ramping for interesting transitions
        Creates excitement (accelerate) or anticipation (decelerate)
        Uses pitch-preserving time stretching with minimal artifacts
        
        Args:
            audio: Audio segment to process
            current_tempo: Current BPM
            target_tempo: Target BPM to ramp toward
            ramp_type: 'accelerate' or 'decelerate'
            style_params: Style parameters for adaptive ramping
            
        Returns:
            Tempo-ramped audio segment
        """
        try:
            # Determine ramp intensity based on genre and tempo difference
            tempo_diff = abs(target_tempo - current_tempo)
            
            # Skip ramping if tempos are too similar (less than 3 BPM difference)
            if tempo_diff < 3.0:
                logger.info(f"  Skipping tempo ramp (tempos too similar: {tempo_diff:.1f} BPM)")
                return audio
            
            # Adaptive ramp intensity based on style
            if style_params:
                genre_hint = style_params.get('reason', '')
                
                # More aggressive ramping for electronic/energetic genres
                if 'electronic' in genre_hint or 'future_funk' in genre_hint:
                    max_ramp_percent = 2.5  # Up to 2.5% tempo change
                elif 'jazz' in genre_hint or 'classical' in genre_hint:
                    max_ramp_percent = 1.0  # Subtle 1% for musical genres
                elif 'ballad' in genre_hint or 'bolero' in genre_hint:
                    max_ramp_percent = 0.8  # Very subtle for emotional genres
                else:
                    max_ramp_percent = 1.5  # Default 1.5%
            else:
                max_ramp_percent = 1.5
            
            # Calculate actual ramp amount (cap at max_ramp_percent)
            ramp_direction = 1 if ramp_type == 'accelerate' else -1
            ramp_percent = min(max_ramp_percent, tempo_diff * 0.3) * ramp_direction
            
            # Create gradual tempo curve (exponential for natural feel)
            num_samples = len(audio)
            time_curve = np.linspace(0, 1, num_samples)
            
            if ramp_type == 'accelerate':
                # Exponential acceleration (slow start, faster end)
                tempo_curve = 1.0 + (ramp_percent / 100.0) * (time_curve ** 2)
                effect_description = f"accelerating +{abs(ramp_percent):.1f}%"
            else:  # decelerate
                # Exponential deceleration (fast start, slower end)
                tempo_curve = 1.0 + (ramp_percent / 100.0) * ((1 - time_curve) ** 2)
                effect_description = f"decelerating -{abs(ramp_percent):.1f}%"
            
            # Apply pitch-preserving time stretch with tempo curve
            # Use librosa's phase vocoder for high-quality pitch preservation
            logger.info(f"  Applying tempo ramp: {effect_description}")
            
            # Calculate cumulative time mapping
            cumulative_stretch = np.cumsum(tempo_curve)
            cumulative_stretch = cumulative_stretch / cumulative_stretch[-1] * num_samples
            
            # Resample to create tempo variation
            # Use high-quality interpolation
            if audio.ndim > 1:  # Stereo
                ramped_audio = np.zeros_like(audio)
                for channel in range(audio.shape[1]):
                    original_indices = np.arange(num_samples)
                    interpolator = interp1d(original_indices, audio[:, channel], 
                                           kind='cubic', bounds_error=False, 
                                           fill_value='extrapolate')
                    ramped_audio[:, channel] = interpolator(cumulative_stretch)
            else:  # Mono
                original_indices = np.arange(num_samples)
                interpolator = interp1d(original_indices, audio, 
                                       kind='cubic', bounds_error=False, 
                                       fill_value='extrapolate')
                ramped_audio = interpolator(cumulative_stretch)
            
            # Ensure no clipping from interpolation
            peak = np.max(np.abs(ramped_audio))
            if peak > 1.0:
                ramped_audio = ramped_audio / peak
            
            return ramped_audio
            
        except Exception as e:
            logger.warning(f"  Tempo ramping failed: {e}, using original audio")
            return audio
    
    def _vocal_aware_crossfade(self, track1_end: np.ndarray, track2_start: np.ndarray,
                              fade_out: np.ndarray, fade_in: np.ndarray) -> np.ndarray:
        """
        Apply smooth crossfading with consistent volume - NO dynamic adjustments
        """
        # Simple equal-power crossfade without volume adjustments that cause pumping
        # Just apply the fade curves directly to avoid any beat-synchronized artifacts
        
        # Ensure both tracks have matching dimensions first
        track1_end, track2_start = self._match_audio_dimensions(track1_end, track2_start)
        
        # Ensure arrays are properly shaped for multiplication
        if track1_end.ndim > 1:
            fade_out_shaped = fade_out.reshape(-1, 1)
            fade_in_shaped = fade_in.reshape(-1, 1)
        else:
            fade_out_shaped = fade_out
            fade_in_shaped = fade_in
        
        # Simple crossfade - no normalization, no filtering, no dynamic processing
        crossfade_section = track1_end * fade_out_shaped + track2_start * fade_in_shaped
        
        # Final gentle limiting to prevent clipping (very conservative)
        peak = np.max(np.abs(crossfade_section))
        if peak > 0.9:
            # Very gentle soft limiting with smooth curve
            limit_ratio = 0.9 / peak
            crossfade_section = crossfade_section * limit_ratio
        
        return crossfade_section
    
    def _intelligent_vocal_crossfade(self, vocal1: np.ndarray, vocal2: np.ndarray,
                                   fade_out: np.ndarray, fade_in: np.ndarray) -> np.ndarray:
        """
        Intelligent crossfade specifically for vocal frequency ranges
        """
        # Detect vocal energy in both tracks
        vocal1_energy = np.sqrt(np.mean(vocal1**2))
        vocal2_energy = np.sqrt(np.mean(vocal2**2))
        
        # If one track has significantly more vocal energy, adjust crossfade
        energy_ratio = vocal2_energy / (vocal1_energy + 1e-8)
        
        if energy_ratio > 2.0:
            # Track 2 has much stronger vocals - fade in faster
            adjusted_fade_in = fade_in ** 0.7
            adjusted_fade_out = fade_out ** 1.3
        elif energy_ratio < 0.5:
            # Track 1 has much stronger vocals - fade out slower
            adjusted_fade_in = fade_in ** 1.3
            adjusted_fade_out = fade_out ** 0.7
        else:
            # Similar vocal energy - use equal-power crossfade
            adjusted_fade_in = fade_in
            adjusted_fade_out = fade_out
        
        # Apply crossfade with minimal ducking for fuller sound
        crossfaded = vocal1 * adjusted_fade_out + vocal2 * adjusted_fade_in
        
        # Apply very gentle ducking only in the middle to avoid harsh vocal conflicts
        # Reduced ducking for fuller transitions
        duck_start = len(crossfaded) // 3
        duck_end = 2 * len(crossfaded) // 3
        duck_curve = np.ones(len(crossfaded))
        
        # Create very gentle ducking curve
        duck_amount = 0.95  # Only reduce to 95% (was 85%) for fuller sound
        for i in range(duck_start, duck_end):
            position = (i - duck_start) / (duck_end - duck_start)
            # Wider bell curve for subtler ducking
            duck_factor = 1.0 - (1.0 - duck_amount) * np.exp(-((position - 0.5) * 4)**2)  # Wider (4 instead of 6)
            duck_curve[i] = duck_factor
        
        crossfaded *= duck_curve
        
        return crossfaded
    
    def _gradual_tempo_sync(self, track1_end: np.ndarray, track2_start: np.ndarray,
                           track1_tempo: float, track2_tempo: float, crossfade_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ultra-smooth tempo curves with exponential easing for natural transitions
        """
        sr = self.sample_rate
        
        # Calculate the tempo change direction and amount
        tempo_change = track2_tempo - track1_tempo
        
        if abs(tempo_change) < 0.5:  # Only skip very minimal changes
            return track1_end, track2_start
        
        logger.info(f"    Creating progressive tempo curve: {track1_tempo:.1f} → {track2_tempo:.1f} BPM")
        
        crossfade_duration = crossfade_samples / sr
        
        # Calculate beat periods for both tracks
        beat_period_1 = 60.0 / track1_tempo
        beat_period_2 = 60.0 / track2_tempo
        
        logger.info(f"    Beat periods: {beat_period_1:.3f}s → {beat_period_2:.3f}s")
        
        # Create time array for the crossfade duration with fine granularity
        time_points = np.linspace(0, crossfade_duration, crossfade_samples)
        progress = time_points / crossfade_duration
        
        # Use smoothstep interpolation (cubic Hermite) for ultra-smooth BPM transitions
        def smoothstep(t):
            # S-curve with zero derivatives at endpoints (no sudden changes)
            return t * t * (3.0 - 2.0 * t)
        
        # Apply double smoothstep for even smoother acceleration
        smooth_curve = np.array([smoothstep(smoothstep(p)) for p in progress])
        
        # Add very subtle musical timing modulation aligned with beats
        beats_in_crossfade = crossfade_duration / beat_period_1
        
        if beats_in_crossfade > 1:
            # Minimal musical modulation synchronized with beat phase (1.5% intensity)
            beat_phase = (time_points / beat_period_1) * 2 * np.pi
            musical_modulation = np.sin(beat_phase) * 0.015
            
            # Apply only in middle 70% with very gentle windowing
            modulation_window = np.where(
                (progress >= 0.15) & (progress <= 0.85),
                np.sin((progress - 0.15) / 0.7 * np.pi) ** 2,
                0
            )
            smooth_curve += musical_modulation * modulation_window
        
        # Ensure curve stays within bounds
        smooth_curve = np.clip(smooth_curve, 0, 1)
        
        # Apply multiple passes of smoothing for ultra-smooth BPM changes
        if len(smooth_curve) > 32:
            # First pass: light smoothing
            window_size = min(21, len(smooth_curve) // 8)
            if window_size >= 5:
                smooth_curve = signal.savgol_filter(smooth_curve, window_size | 1, 3)
                smooth_curve = np.clip(smooth_curve, 0, 1)
            
            # Second pass: very light smoothing to eliminate micro-bumps
            window_size = min(11, len(smooth_curve) // 16)
            if window_size >= 5:
                smooth_curve = signal.savgol_filter(smooth_curve, window_size | 1, 2)
                smooth_curve = np.clip(smooth_curve, 0, 1)
        
        # Calculate instantaneous tempo with adaptive blending for smooth transitions
        # Use 90% for differences >10 BPM, scale down for smaller differences
        if abs(tempo_change) > 10:
            tempo_blend_factor = 0.90
        elif abs(tempo_change) > 5:
            tempo_blend_factor = 0.70 + (abs(tempo_change) - 5) / 5 * 0.20
        else:
            tempo_blend_factor = 0.50 + abs(tempo_change) / 5 * 0.20
        
        instantaneous_tempo = track1_tempo + tempo_change * smooth_curve * tempo_blend_factor
        
        # For track2: use complementary curve with matching intensity
        final_tempo_1 = track1_tempo + tempo_change * tempo_blend_factor
        track2_tempo_curve = final_tempo_1 + (track2_tempo - final_tempo_1) * smooth_curve * tempo_blend_factor
        
        # Apply progressive tempo stretching using the calculated curves
        logger.info(f"    Track 1 outro: {track1_tempo:.1f} → {instantaneous_tempo[-1]:.1f} BPM")
        logger.info(f"    Track 2 intro: {track2_tempo_curve[0]:.1f} → {track2_tempo:.1f} BPM")
        adjusted_track1_end = self._apply_tempo_curve_gentle(track1_end, track1_tempo, instantaneous_tempo, sr)
        adjusted_track2_start = self._apply_tempo_curve_gentle(track2_start, track2_tempo, track2_tempo_curve, sr)
        
        # Ensure exact length match with high-quality resampling
        if len(adjusted_track1_end) != crossfade_samples:
            adjusted_track1_end = signal.resample(adjusted_track1_end, crossfade_samples)
        
        if len(adjusted_track2_start) != crossfade_samples:
            adjusted_track2_start = signal.resample(adjusted_track2_start, crossfade_samples)
        
        return adjusted_track1_end, adjusted_track2_start
    
    def _apply_tempo_curve_gentle(self, audio: np.ndarray, original_tempo: float, 
                                 tempo_curve: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply progressive tempo curve using simple approach: just use average tempo ratio
        Complex tempo curves cause artifacts - keep it simple for smoothness
        """
        try:
            # Calculate the tempo ratios
            tempo_ratios = tempo_curve / original_tempo
            avg_ratio = np.mean(tempo_ratios)
            
            # Apply changes above 0.5% threshold
            if abs(avg_ratio - 1.0) < 0.005:
                return audio
            
            # Simple approach: use librosa's time_stretch with average ratio
            # This is the smoothest method - no segmentation, no artifacts
            # tempo_ratios = tempo_curve / original_tempo, so avg_ratio is the stretch factor
            # librosa rate: >1 = faster (higher tempo), <1 = slower (lower tempo)
            stretched = librosa.effects.time_stretch(audio, rate=avg_ratio)
            
            # Resample to exact original length to maintain sync
            if len(stretched) != len(audio):
                stretched = signal.resample_poly(stretched, len(audio), len(stretched))
            
            return stretched
                    
        except Exception as e:
            logger.warning(f"    Tempo adjustment failed: {e}, using original audio")
            import traceback
            traceback.print_exc()
            return audio
    
    def _apply_tempo_restoration(self, audio: np.ndarray, start_tempo: float, 
                                 target_tempo: float, restoration_samples: int) -> np.ndarray:
        """
        Smoothly return audio from adjusted tempo back to its natural BPM
        Uses reverse tempo curve to gradually restore original speed
        """
        try:
            if len(audio) == 0 or restoration_samples == 0:
                return audio
            
            # Only apply if there's a meaningful difference
            if abs(start_tempo - target_tempo) < 0.5:
                return audio
            
            logger.info(f"    Restoring tempo: {start_tempo:.1f} → {target_tempo:.1f} BPM over {restoration_samples / self.sample_rate:.1f}s")
            
            # Create smooth restoration curve (reverse of acceleration)
            progress = np.linspace(0, 1, restoration_samples)
            
            # Use smoothstep for gentle deceleration back to natural tempo
            def smoothstep(t):
                return t * t * (3.0 - 2.0 * t)
            
            # Apply single smoothstep (gentler than double)
            restoration_curve = np.array([smoothstep(p) for p in progress])
            
            # Calculate tempo change
            tempo_change = target_tempo - start_tempo
            
            # Determine blend factor based on tempo difference
            if abs(tempo_change) > 10:
                blend_factor = 0.90
            elif abs(tempo_change) > 5:
                blend_factor = 0.70 + (abs(tempo_change) - 5) / 5 * 0.20
            else:
                blend_factor = 0.50 + abs(tempo_change) / 5 * 0.20
            
            # Create tempo curve from adjusted tempo back to natural
            tempo_curve = start_tempo + tempo_change * restoration_curve * blend_factor
            
            # Apply the restoration curve
            restored_audio = self._apply_tempo_curve_gentle(audio, start_tempo, tempo_curve, self.sample_rate)
            
            # Ensure exact length match
            if len(restored_audio) != len(audio):
                restored_audio = signal.resample_poly(restored_audio, len(audio), len(restored_audio))
            
            return restored_audio
            
        except Exception as e:
            logger.warning(f"    Tempo restoration failed: {e}, using original audio")
            import traceback
            traceback.print_exc()
            return audio
    
    def _match_rms(self, audio: np.ndarray, target_rms: float) -> np.ndarray:
        """
        Match audio RMS level to target value to preserve loudness after processing
        """
        if len(audio) == 0 or target_rms == 0:
            return audio
        
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms == 0:
            return audio
        
        # Calculate gain needed
        gain = target_rms / current_rms
        
        # Apply safety limits to prevent extreme volume changes
        gain = np.clip(gain, 0.5, 2.0)
        
        return audio * gain
    
    def _apply_volume_envelope_restoration(self, audio: np.ndarray, 
                                           start_rms: float, end_rms: float) -> np.ndarray:
        """
        Apply smooth volume envelope during restoration to eliminate volume jumps
        Gradually transitions from crossfade volume to restoration section volume
        """
        if len(audio) == 0 or start_rms == 0 or end_rms == 0:
            return audio
        
        # Create smooth envelope from start to end RMS
        progress = np.linspace(0, 1, len(audio))
        
        # Use smoothstep for perceptually smooth volume change
        def smoothstep(t):
            return t * t * (3.0 - 2.0 * t)
        
        envelope = np.array([smoothstep(p) for p in progress])
        
        # Calculate current RMS at each point
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms == 0:
            return audio
        
        # Create RMS curve from start to end
        rms_curve = start_rms + (end_rms - start_rms) * envelope
        
        # Calculate target overall scaling
        target_avg_rms = np.mean(rms_curve)
        current_to_target = target_avg_rms / current_rms
        
        # Apply base scaling
        normalized_audio = audio * current_to_target
        
        # Apply smooth envelope (limit to prevent extreme changes)
        final_envelope = rms_curve / target_avg_rms
        final_envelope = np.clip(final_envelope, 0.7, 1.3)
        
        # Apply envelope to audio
        if audio.ndim == 1:
            result = normalized_audio * final_envelope
        else:
            result = normalized_audio * final_envelope.reshape(-1, 1)
        
        return result
    
    def _apply_tempo_curve(self, audio: np.ndarray, original_tempo: float, 
                          tempo_curve: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply a smooth tempo curve to audio using phase accumulation
        """
        try:
            # Calculate the cumulative tempo ratios
            tempo_ratios = tempo_curve / original_tempo
            
            # Use librosa's phase vocoder for smooth tempo changes
            # This preserves pitch while changing tempo smoothly
            
            # For very smooth transitions, we'll use a windowed approach
            if len(tempo_ratios) > 1024:  # For longer crossfades
                # Apply tempo stretching in overlapping windows for ultra-smooth results
                window_size = len(audio) // 8  # 8 overlapping windows
                hop_size = window_size // 2
                
                result = np.zeros_like(audio)
                window_func = np.hanning(window_size)
                
                for i in range(0, len(audio) - window_size + 1, hop_size):
                    # Get current window
                    window_audio = audio[i:i + window_size] * window_func
                    
                    # Calculate average tempo ratio for this window
                    ratio_start_idx = int((i / len(audio)) * len(tempo_ratios))
                    ratio_end_idx = int(((i + window_size) / len(audio)) * len(tempo_ratios))
                    ratio_end_idx = min(ratio_end_idx, len(tempo_ratios) - 1)
                    
                    if ratio_start_idx < len(tempo_ratios):
                        avg_ratio = np.mean(tempo_ratios[ratio_start_idx:ratio_end_idx + 1])
                        
                        # Apply tempo stretching to window (phase vocoder preserves pitch)
                        if abs(avg_ratio - 1.0) > 0.01:
                            stretched_window = librosa.effects.time_stretch(window_audio, rate=avg_ratio)
                            
                            # Resize back to original window size with quality resampling
                            if len(stretched_window) != window_size:
                                stretched_window = signal.resample(stretched_window, window_size)
                            
                            # Apply window function again after stretching
                            stretched_window *= window_func
                        else:
                            stretched_window = window_audio
                        
                        # Overlap-add into result
                        end_idx = min(i + window_size, len(result))
                        actual_size = end_idx - i
                        result[i:end_idx] += stretched_window[:actual_size]
                    
                return result
                
            else:
                # For shorter crossfades, use simpler approach
                avg_ratio = np.mean(tempo_ratios)
                if abs(avg_ratio - 1.0) > 0.01:
                    return librosa.effects.time_stretch(audio, rate=avg_ratio)
                else:
                    return audio
                    
        except Exception as e:
            logger.warning(f"    Tempo curve application failed: {e}, using original audio")
            return audio
    
    def _apply_invisible_tempo_sync(self, audio: np.ndarray, original_tempo: float, 
                                  target_tempo: float, is_outro: bool = True) -> np.ndarray:
        """
        Apply subtle tempo synchronization (fallback for very small differences)
        """
        tempo_diff = abs(original_tempo - target_tempo)
        
        # Only apply gentle adjustments for small differences
        if tempo_diff < 0.5:
            return audio
        
        # Use moderate tempo adjustments for subtle sync
        max_tempo_change = min(0.04, tempo_diff / original_tempo * 0.25)  # Max 4% change
        
        # Calculate gradual stretch factor
        if is_outro:
            # For outro: move very gradually toward target tempo
            if target_tempo > original_tempo:
                stretch_factor = 1.0 - max_tempo_change * 0.5  # Even gentler
            else:
                stretch_factor = 1.0 + max_tempo_change * 0.5
        else:
            # For intro: start much closer to original tempo
            if original_tempo > target_tempo:
                stretch_factor = 1.0 - max_tempo_change * 0.3
            else:
                stretch_factor = 1.0 + max_tempo_change * 0.3
        
        try:
            # Apply very gentle time stretching with enhanced quality
            # Use phase vocoder for better pitch preservation
            if abs(stretch_factor - 1.0) > 0.005:  # Only if meaningful change
                # Apply stretching in smaller chunks for smoother results
                chunk_size = len(audio) // 4  # Process in quarters
                adjusted_chunks = []
                
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) > 1024:  # Only process significant chunks
                        # Apply gradual stretch factor that varies across the chunk
                        position_factor = i / len(audio)
                        local_stretch = 1.0 + (stretch_factor - 1.0) * position_factor
                        
                        adjusted_chunk = librosa.effects.time_stretch(chunk, rate=local_stretch)
                        
                        # Resample back to exact chunk size to maintain timing
                        if len(adjusted_chunk) != len(chunk):
                            adjusted_chunk = signal.resample(adjusted_chunk, len(chunk))
                        
                        adjusted_chunks.append(adjusted_chunk)
                    else:
                        adjusted_chunks.append(chunk)
                
                adjusted_audio = np.concatenate(adjusted_chunks)
                
                # Ensure exact length match
                if len(adjusted_audio) != len(audio):
                    adjusted_audio = signal.resample(adjusted_audio, len(audio))
                    
                logger.info(f"    Applied subtle tempo sync: max {stretch_factor:.4f}x stretch")
                return adjusted_audio
            else:
                return audio
            
        except Exception as e:
            logger.warning(f"    Subtle tempo sync failed: {e}, using original")
            return audio
    
    def adjust_tempo(self, audio: np.ndarray, source_tempo: float, target_tempo: float) -> np.ndarray:
        """
        Adjust tempo with enhanced BPM synchronization like Apple Music
        Preserves pitch using phase vocoder
        """
        if abs(source_tempo - target_tempo) < 1:  # Very small difference, don't adjust
            return audio
        
        # Calculate stretch factor: rate > 1 = faster, rate < 1 = slower
        # To match target tempo: rate = target / source
        # Example: 120 BPM -> 90 BPM needs rate = 90/120 = 0.75 (slower)
        rate = target_tempo / source_tempo
        
        # Use librosa's high-quality time stretching (phase vocoder preserves pitch)
        adjusted_audio = librosa.effects.time_stretch(audio, rate=rate)
        return adjusted_audio
    
    def _calculate_phase_correlation(self, audio1: np.ndarray, audio2: np.ndarray, 
                                     window_samples: int, sr: int) -> Tuple[np.ndarray, int]:
        """
        Calculate cross-correlation between two audio segments to find optimal alignment
        Uses GPU (Metal) on Apple Silicon if available (~50x faster)
        
        Args:
            audio1: End segment of first track
            audio2: Start segment of second track
            window_samples: Search window size in samples (±offset)
            sr: Sample rate
            
        Returns:
            Tuple of (correlation values, optimal offset in samples)
        """
        try:
            # Use GPU correlation if available (50x faster on Apple Silicon)
            if self.use_gpu and self.gpu_correlation:
                return self.gpu_correlation.phase_correlation(
                    audio1, audio2, window_samples, sr
                )
            
            # CPU fallback (original implementation)
            # Ensure mono for correlation analysis
            if audio1.ndim > 1:
                audio1 = np.mean(audio1, axis=1)
            if audio2.ndim > 1:
                audio2 = np.mean(audio2, axis=1)
            
            # Extract segments for comparison - use 2 seconds for better accuracy
            segment_length = min(len(audio1), len(audio2), int(sr * 2.0))
            seg1 = audio1[-segment_length:] if len(audio1) >= segment_length else audio1
            seg2 = audio2[:segment_length] if len(audio2) >= segment_length else audio2
            
            # Apply onset emphasis for better beat correlation
            onset_env1 = librosa.onset.onset_strength(y=seg1, sr=sr)
            onset_env2 = librosa.onset.onset_strength(y=seg2, sr=sr)
            
            # Resample onset envelopes to match audio length
            from scipy.interpolate import interp1d
            x1 = np.linspace(0, len(seg1), len(onset_env1))
            x2 = np.linspace(0, len(seg2), len(onset_env2))
            onset1 = interp1d(x1, onset_env1, bounds_error=False, fill_value=0)(np.arange(len(seg1)))
            onset2 = interp1d(x2, onset_env2, bounds_error=False, fill_value=0)(np.arange(len(seg2)))
            
            # Combine audio with onset emphasis (70% audio, 30% onset)
            seg1_enhanced = seg1 * 0.7 + onset1 * 0.3
            seg2_enhanced = seg2 * 0.7 + onset2 * 0.3
            
            # Normalize segments to prevent amplitude bias
            seg1_enhanced = seg1_enhanced / (np.max(np.abs(seg1_enhanced)) + 1e-8)
            seg2_enhanced = seg2_enhanced / (np.max(np.abs(seg2_enhanced)) + 1e-8)
            
            # Use FFT-based correlation for efficiency
            correlation = signal.correlate(seg1_enhanced, seg2_enhanced, mode='same', method='fft')
            
            # Find peak in search window (center ± window_samples)
            center = len(correlation) // 2
            search_start = max(0, center - window_samples)
            search_end = min(len(correlation), center + window_samples)
            
            search_region = correlation[search_start:search_end]
            peak_idx = np.argmax(search_region)
            optimal_offset = peak_idx + search_start - center
            
            return correlation, optimal_offset
            
        except Exception as e:
            logger.warning(f"    Phase correlation failed: {e}, using zero offset")
            return np.array([0]), 0
    
    def _find_optimal_beat_alignment(self, track1: Dict, track2: Dict, 
                                     transition_point: float, genre1: str, genre2: str,
                                     beat_idx1: int, beat_idx2: int) -> Tuple[float, float, float]:
        """
        Find optimal beat alignment using adaptive window-based search
        
        Args:
            track1: First track dictionary
            track2: Second track dictionary
            transition_point: Intended transition time in track1 (seconds)
            genre1: Genre of first track
            genre2: Genre of second track
            beat_idx1: Index of beat in track1 to align
            beat_idx2: Index of beat in track2 to align
            
        Returns:
            Tuple of (adjusted_beat1_time, adjusted_beat2_time, confidence_score)
        """
        try:
            audio1 = track1['audio_data']
            audio2 = track2['audio_data']
            beats1 = track1.get('beats', np.array([]))
            beats2 = track2.get('beats', np.array([]))
            sr = track1['sample_rate']
            
            if len(beats1) <= beat_idx1 or len(beats2) <= beat_idx2:
                return beats1[beat_idx1] if len(beats1) > beat_idx1 else transition_point, \
                       beats2[beat_idx2] if len(beats2) > beat_idx2 else 0.0, 0.5
            
            beat1_time = beats1[beat_idx1]
            beat2_time = beats2[beat_idx2]
            
            # Determine search window based on genre (adaptive)
            # Tighter windows for better beat alignment precision
            genre_windows = {
                'electronic': 0.010,  # ±10ms (ultra-tight, beat-matched)
                'hiphop': 0.015,      # ±15ms (very tight, punchy)
                'pop': 0.025,         # ±25ms (tight)
                'rock': 0.025,        # ±25ms (tight)
                'house': 0.010,       # ±10ms (ultra-tight, 4/4 kick)
                'future_funk': 0.012, # ±12ms (ultra-tight funk groove)
                'jazz': 0.050,        # ±50ms (moderate, musical)
                'classical': 0.075,   # ±75ms (moderate-loose, phrase-based)
                'vietnamese_ballad': 0.040,  # ±40ms (moderate)
                'vietnamese_pop': 0.025,     # ±25ms (tight)
                'cuba_bolero': 0.040,        # ±40ms (moderate)
            }
            
            # Use average of both genres
            window1 = genre_windows.get(genre1, 0.050)
            window2 = genre_windows.get(genre2, 0.050)
            window_time = (window1 + window2) / 2.0
            window_samples = int(window_time * sr)
            
            # Extract audio segments around beats
            beat1_sample = int(beat1_time * sr)
            beat2_sample = int(beat2_time * sr)
            
            # Extract segments for correlation (2 seconds around each beat)
            seg_length = int(2.0 * sr)
            start1 = max(0, beat1_sample - seg_length // 2)
            end1 = min(len(audio1), beat1_sample + seg_length // 2)
            start2 = max(0, beat2_sample - seg_length // 2)
            end2 = min(len(audio2), beat2_sample + seg_length // 2)
            
            seg1 = audio1[start1:end1]
            seg2 = audio2[start2:end2]
            
            if len(seg1) < sr // 10 or len(seg2) < sr // 10:  # Need at least 100ms
                return beat1_time, beat2_time, 0.3
            
            # Calculate phase correlation
            correlation, optimal_offset = self._calculate_phase_correlation(
                seg1, seg2, window_samples, sr
            )
            
            # Calculate confidence based on correlation peak sharpness
            if len(correlation) > 0:
                peak_value = np.max(np.abs(correlation))
                mean_value = np.mean(np.abs(correlation))
                confidence = min(1.0, (peak_value - mean_value) / (mean_value + 1e-8))
            else:
                confidence = 0.3
            
            # Apply offset to beat times
            offset_time = optimal_offset / sr
            adjusted_beat1 = beat1_time
            adjusted_beat2 = beat2_time - offset_time  # Shift track2 beat
            
            # Multi-point verification: Check surrounding beats for consistency
            verify_confidence = self._verify_beat_alignment(
                track1, track2, beat_idx1, beat_idx2, offset_time
            )
            
            # Combine correlation confidence with verification
            final_confidence = (confidence * 0.6 + verify_confidence * 0.4)
            
            logger.info(f"    Adaptive alignment: offset={offset_time*1000:.1f}ms, "
                       f"window=±{window_time*1000:.0f}ms, confidence={final_confidence:.2f}")
            
            return adjusted_beat1, adjusted_beat2, final_confidence
            
        except Exception as e:
            logger.warning(f"    Optimal beat alignment failed: {e}, using original beats")
            import traceback
            traceback.print_exc()
            return beats1[beat_idx1] if len(beats1) > beat_idx1 else transition_point, \
                   beats2[beat_idx2] if len(beats2) > beat_idx2 else 0.0, 0.3
    
    def _verify_beat_alignment(self, track1: Dict, track2: Dict, 
                               beat_idx1: int, beat_idx2: int, offset_time: float) -> float:
        """
        Verify beat alignment quality by checking multiple beats for consistency
        
        Args:
            track1: First track dictionary
            track2: Second track dictionary
            beat_idx1: Beat index in track1
            beat_idx2: Beat index in track2
            offset_time: Proposed time offset (seconds)
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            beats1 = track1.get('beats', np.array([]))
            beats2 = track2.get('beats', np.array([]))
            tempo1 = track1.get('actual_tempo', track1.get('tempo', 120))
            tempo2 = track2.get('actual_tempo', track2.get('tempo', 120))
            
            # Check if enough beats available
            if len(beats1) < beat_idx1 + 4 or len(beats2) < beat_idx2 + 4:
                return 0.5  # Moderate confidence if not enough beats
            
            # Expected beat period for each track
            beat_period1 = 60.0 / tempo1
            beat_period2 = 60.0 / tempo2
            
            # Check 4 beats ahead for timing consistency
            errors = []
            for i in range(1, 5):
                if beat_idx1 + i >= len(beats1) or beat_idx2 + i >= len(beats2):
                    break
                
                # Expected position after offset
                expected1 = beats1[beat_idx1] + i * beat_period1
                expected2 = beats2[beat_idx2] - offset_time + i * beat_period2
                
                # Actual positions
                actual1 = beats1[beat_idx1 + i]
                actual2 = beats2[beat_idx2 + i] - offset_time
                
                # Calculate error between expected and actual
                error1 = abs(actual1 - expected1)
                error2 = abs(actual2 - expected2)
                errors.append((error1 + error2) / 2)
            
            if not errors:
                return 0.5
            
            # Convert errors to confidence (lower error = higher confidence)
            avg_error = np.mean(errors)
            max_acceptable_error = 0.025  # 25ms tolerance (tighter)
            confidence = max(0.0, 1.0 - (avg_error / max_acceptable_error))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"    Beat verification failed: {e}")
            return 0.5
    
    def align_beats(self, track1: Dict, track2: Dict, crossfade_samples: int, intro_skip_samples: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align beats between tracks for seamless transitions using downbeats and phrases
        Uses beat confidence to prefer high-quality transition points
        
        Args:
            track1: First track dict with audio_data, beats, etc.
            track2: Second track dict
            crossfade_samples: Number of samples for crossfade
            intro_skip_samples: Number of samples already skipped from track2 intro (for vocal alignment)
        """
        audio1 = track1['audio_data']
        audio2 = track2['audio_data']
        beats1 = track1.get('beats', np.array([]))
        beats2 = track2.get('beats', np.array([]))
        
        # Validate inputs - if no beats detected, skip alignment
        if beats1 is None or len(beats1) == 0:
            logger.warning(f"    No beats detected in track 1, skipping beat alignment")
            return audio1, audio2
        if beats2 is None or len(beats2) == 0:
            logger.warning(f"    No beats detected in track 2, skipping beat alignment")
            return audio1, audio2
        
        # Adjust beat positions for track2 if intro was already skipped
        sr = track1['sample_rate']
        intro_skip_time = intro_skip_samples / sr
        
        if intro_skip_samples > 0:
            # Shift all beat times backward by the skip amount
            beats2_original = beats2.copy()
            beats2 = beats2 - intro_skip_time
            # Remove any beats that are now negative (were in the skipped section)
            valid_beat_mask = beats2 >= 0
            beats2 = beats2[valid_beat_mask]
            
            if len(beats2) == 0:
                logger.warning(f"    All beats were in skipped intro ({intro_skip_time:.1f}s), no beat alignment possible")
                return audio1, audio2
            
            # Update track2 dict with adjusted beats (for adaptive alignment)
            track2['beats'] = beats2
            
            # Also adjust beat_strengths and beat_confidence arrays to match
            if 'beat_strengths' in track2 and len(track2['beat_strengths']) == len(beats2_original):
                track2['beat_strengths'] = track2['beat_strengths'][valid_beat_mask]
            if 'beat_confidence' in track2 and len(track2['beat_confidence']) == len(beats2_original):
                track2['beat_confidence'] = track2['beat_confidence'][valid_beat_mask]
            if 'downbeats' in track2:
                downbeats2_adjusted = track2['downbeats'] - intro_skip_time
                track2['downbeats'] = downbeats2_adjusted[downbeats2_adjusted >= 0]
            
            logger.info(f"    Adjusted {len(beats2)} beats for {intro_skip_time:.1f}s intro skip")
        
        downbeats1 = track1.get('downbeats', beats1[::4] if len(beats1) >= 4 else beats1)
        downbeats2 = track2.get('downbeats', beats2[::4] if len(beats2) >= 4 else beats2)
        phrases1 = track1.get('phrases', [])
        phrases2 = track2.get('phrases', [])
        beat_strengths1 = track1.get('beat_strengths', np.ones(len(beats1)))
        beat_strengths2 = track2.get('beat_strengths', np.ones(len(beats2)))  # FIX: was track1
        beat_confidence1 = track1.get('beat_confidence', np.ones(len(beats1)))
        beat_confidence2 = track2.get('beat_confidence', np.ones(len(beats2)))
        sr = track1['sample_rate']
        
        crossfade_time = crossfade_samples / sr
        track1_duration = len(audio1) / sr
        transition_point = track1_duration - crossfade_time
        
        # Get genre context for adaptive alignment
        genre1 = track1.get('genre_hint', 'unknown')
        genre2 = track2.get('genre_hint', 'unknown')
        
        logger.debug(f"    Genre context: {genre1} -> {genre2}, beats: {len(beats1)}/{len(beats2)}")
        
        # Strategy 0: ADAPTIVE WINDOW-BASED BEAT ALIGNMENT (NEW - HIGHEST PRIORITY)
        # Use phase correlation with genre-specific windows for smoothest transitions
        if len(beats1) > 0 and len(beats2) > 0 and genre1 != 'unknown':
            try:
                logger.debug(f"    Attempting adaptive alignment...")
                # Find beats near transition point
                nearby_beat_indices = np.where(np.abs(beats1 - transition_point) < 2.0)[0]
                logger.debug(f"    Found {len(nearby_beat_indices)} nearby beats at transition point")
                
                if len(nearby_beat_indices) > 0:
                    # Select best beat in track1 (prefer strong, confident beats)
                    valid_indices = nearby_beat_indices[nearby_beat_indices < len(beat_strengths1)]
                    if len(valid_indices) > 0:
                        combined_scores1 = beat_strengths1[valid_indices] * beat_confidence1[valid_indices]
                        best_beat_idx1 = int(valid_indices[np.argmax(combined_scores1)])
                        
                        # Find corresponding beat in track2
                        early_beat_indices = np.where(beats2 < 4.0)[0]
                        if len(early_beat_indices) > 0:
                            beat_strengths2_actual = track2.get('beat_strengths', np.ones(len(beats2)))
                            beat_confidence2_actual = track2.get('beat_confidence', np.ones(len(beats2)))
                            valid_early = early_beat_indices[early_beat_indices < len(beat_strengths2_actual)]
                            
                            if len(valid_early) > 0:
                                combined_scores2 = beat_strengths2_actual[valid_early] * beat_confidence2_actual[valid_early]
                                best_beat_idx2 = int(valid_early[np.argmax(combined_scores2)])
                                
                                # Apply adaptive window-based alignment
                                adjusted_beat1, adjusted_beat2, confidence = self._find_optimal_beat_alignment(
                                    track1, track2, transition_point, genre1, genre2,
                                    best_beat_idx1, best_beat_idx2
                                )
                                
                                # Use adaptive alignment if confidence is high enough
                                if confidence > 0.65:
                                    # Apply the optimized alignment
                                    target_end_time = track1_duration - (transition_point - adjusted_beat1)
                                    target_end_samples = int(target_end_time * sr)
                                    
                                    if 0 < target_end_samples <= len(audio1):
                                        audio1 = audio1[:target_end_samples]
                                    
                                    skip_samples = int(adjusted_beat2 * sr)
                                    if skip_samples >= 0 and skip_samples < len(audio2):
                                        audio2 = audio2[skip_samples:]
                                    
                                    logger.info(f"    ✓ Adaptive window-based alignment applied (confidence: {confidence:.2f})")
                                    return audio1, audio2
                                else:
                                    logger.info(f"    Adaptive alignment confidence low ({confidence:.2f}), falling back to phrase/downbeat")
            except Exception as e:
                logger.debug(f"    Adaptive alignment failed: {e}, falling back to traditional methods")
        
        # Strategy 1: Try to align on phrase boundaries (best for musical flow)
        if phrases1 and phrases2:
            # Find phrase ending near transition point in track1
            phrase1_ends = [p[1] for p in phrases1 if abs(p[1] - transition_point) < 8.0]
            if phrase1_ends:
                best_phrase_end = min(phrase1_ends, key=lambda x: abs(x - transition_point))
                
                # Find phrase beginning near start in track2
                phrase2_starts = [p[0] for p in phrases2 if p[0] < 8.0]
                if phrase2_starts:
                    best_phrase_start = min(phrase2_starts, key=lambda x: x)
                    
                    # Align on phrase boundaries
                    target_end_time = track1_duration - (transition_point - best_phrase_end)
                    target_end_samples = int(target_end_time * sr)
                    
                    if 0 < target_end_samples <= len(audio1):
                        audio1 = audio1[:target_end_samples]
                        logger.info(f"    Aligned on phrase boundary at {best_phrase_end:.1f}s")
                    
                    skip_samples = int(best_phrase_start * sr)
                    if skip_samples < len(audio2):
                        audio2 = audio2[skip_samples:]
                    
                    return audio1, audio2
        
        # Strategy 2: Try to align on downbeats (measures)
        if len(downbeats1) > 0 and len(downbeats2) > 0:
            # Find downbeat near transition point
            downbeat1_idx = np.argmin(np.abs(downbeats1 - transition_point))
            downbeat1_time = downbeats1[downbeat1_idx]
            
            # Find early downbeat in track2
            early_downbeats2 = downbeats2[downbeats2 < 8.0]
            if len(early_downbeats2) > 0:
                downbeat2_time = early_downbeats2[0]
            else:
                downbeat2_time = downbeats2[0] if len(downbeats2) > 0 else 0
            
            # Align on downbeats
            if abs(downbeat1_time - transition_point) < 2.0:
                target_end_time = track1_duration - (transition_point - downbeat1_time)
                target_end_samples = int(target_end_time * sr)
                
                if 0 < target_end_samples <= len(audio1):
                    audio1 = audio1[:target_end_samples]
                    logger.info(f"    Aligned on downbeat at {downbeat1_time:.1f}s")
                
                skip_samples = int(downbeat2_time * sr)
                if skip_samples < len(audio2):
                    audio2 = audio2[skip_samples:]
                
                return audio1, audio2
        
        # Strategy 3: Use high-confidence strong beats (ENHANCED)
        if len(beats1) > 0 and len(beats2) > 0:
            # Find beats near transition point
            nearby_beat_indices = np.where(np.abs(beats1 - transition_point) < 2.0)[0]
            
            if len(nearby_beat_indices) > 0 and len(beat_strengths1) > 0:
                valid_indices = nearby_beat_indices[nearby_beat_indices < len(beat_strengths1)]
                
                if len(valid_indices) > 0:
                    # NEW: Weight strength by confidence for better selection
                    combined_scores1 = beat_strengths1[valid_indices] * beat_confidence1[valid_indices]
                    strongest_idx = valid_indices[np.argmax(combined_scores1)]
                    beat1_time = beats1[strongest_idx]
                    conf1 = beat_confidence1[strongest_idx] if strongest_idx < len(beat_confidence1) else 0.5
                    
                    # Find high-confidence beat at start of track2
                    early_beat_indices = np.where(beats2 < 4.0)[0]
                    if len(early_beat_indices) > 0 and len(beat_strengths2) > 0:
                        # Get beat strengths for track2 (not track1!)
                        beat_strengths2_actual = track2.get('beat_strengths', np.ones(len(beats2)))
                        beat_confidence2_actual = track2.get('beat_confidence', np.ones(len(beats2)))
                        
                        valid_early = early_beat_indices[early_beat_indices < len(beat_strengths2_actual)]
                        if len(valid_early) > 0:
                            # NEW: Weight by confidence using correct track2 data
                            combined_scores2 = beat_strengths2_actual[valid_early] * beat_confidence2_actual[valid_early]
                            strongest_idx2 = valid_early[np.argmax(combined_scores2)]
                            beat2_time = beats2[strongest_idx2]
                            conf2 = beat_confidence2[strongest_idx2] if strongest_idx2 < len(beat_confidence2) else 0.5
                        else:
                            beat2_time = beats2[0]
                            conf2 = 0.5
                    else:
                        beat2_time = beats2[0] if len(beats2) > 0 else 0
                        conf2 = 0.5
                    
                    # Align on strong, confident beats
                    target_end_time = track1_duration - (transition_point - beat1_time)
                    target_end_samples = int(target_end_time * sr)
                    
                    if 0 < target_end_samples <= len(audio1):
                        audio1 = audio1[:target_end_samples]
                        logger.info(f"    Aligned on confident beat at {beat1_time:.1f}s (conf: {conf1:.2f} → {conf2:.2f})")
                    
                    skip_samples = int(beat2_time * sr)
                    if skip_samples < len(audio2):
                        audio2 = audio2[skip_samples:]
                    
                    return audio1, audio2
        
        # Strategy 4: Original simple beat alignment
        if len(beats1) > 0 and len(beats2) > 0:
            beat1_idx = np.argmin(np.abs(beats1 - transition_point))
            beat2_idx = 0
            
            if beat1_idx < len(beats1):
                beat1_time = beats1[beat1_idx]
                beat2_time = beats2[beat2_idx] if beat2_idx < len(beats2) else 0
                
                target_adjustment = min(0.5, abs(beat1_time - transition_point))
                if beat1_time > transition_point:
                    target_end_time = track1_duration - target_adjustment
                else:
                    target_end_time = track1_duration + target_adjustment
                
                target_end_samples = int(target_end_time * sr)
                if 0 < target_end_samples <= len(audio1):
                    audio1 = audio1[:target_end_samples]
                
                beat2_samples = int(beat2_time * sr)
                if beat2_samples < len(audio2):
                    audio2 = audio2[beat2_samples:]
        
        return audio1, audio2
    
    def smart_track_ordering(self, analyzed_tracks: List[Dict], start_index: Optional[int] = None) -> List[Dict]:
        """
        Reorder tracks for optimal transitions using a greedy approach
        
        Args:
            analyzed_tracks: List of analyzed track dictionaries
            start_index: 0-based index of track to start with (None = auto-select)
        """
        if len(analyzed_tracks) <= 1:
            return analyzed_tracks
        
        # Determine starting track
        if start_index is not None and 0 <= start_index < len(analyzed_tracks):
            # User specified start track
            start_track = analyzed_tracks[start_index]
            remaining_tracks = [t for i, t in enumerate(analyzed_tracks) if i != start_index]
            logger.info(f"Starting with user-selected track #{start_index + 1}: {start_track['file_path'].name}")
        else:
            # Auto-select: find the BEST starting track
            # Good starting tracks have: engaging intro, moderate energy, common key/tempo
            best_start_track = None
            best_start_score = -1
            
            for i, track in enumerate(analyzed_tracks):
                score = 0.0
                
                # Factor 1: Intro quality (30%) - prefer tracks WITHOUT boring intros
                intro_sections = [s for s in track.get('structure_sections', []) if s[2] == 'intro']
                if intro_sections and len(intro_sections[0]) > 3:
                    intro_mood = intro_sections[0][3].get('mood', {}) if len(intro_sections[0]) > 3 else {}
                    boring_score = intro_mood.get('boring_score', 0.0)
                    # Invert boring score - lower is better
                    score += (1.0 - boring_score) * 0.30
                else:
                    score += 0.20  # No intro data = assume decent
                
                # Factor 2: Energy level (20%) - prefer moderate energy starts (not too low/high)
                energy = track.get('energy', 0.1)
                if 0.10 <= energy <= 0.18:  # Sweet spot for starting
                    score += 0.20
                elif 0.08 <= energy <= 0.22:  # Acceptable range
                    score += 0.12
                else:
                    score += 0.05
                
                # Factor 3: Tempo (15%) - prefer moderate, danceable tempos
                tempo = track.get('actual_tempo', track.get('tempo', 120))
                if 95 <= tempo <= 130:  # Good starting tempo
                    score += 0.15
                elif 80 <= tempo <= 140:  # Acceptable
                    score += 0.08
                
                # Factor 4: Key commonality (15%) - prefer keys that work with many tracks
                track_key = track.get('key', 0)
                compatible_count = 0
                for other_track in analyzed_tracks:
                    if other_track == track:
                        continue
                    other_key = other_track.get('key', 0)
                    key_distance = min(abs(track_key - other_key), 12 - abs(track_key - other_key))
                    if key_distance in [0, 3, 5, 7]:  # Compatible intervals
                        compatible_count += 1
                
                key_compatibility = compatible_count / max(len(analyzed_tracks) - 1, 1)
                score += key_compatibility * 0.15
                
                # Factor 5: Genre (10%) - prefer popular/versatile genres
                genre = track.get('genre_hint', 'unknown')
                if genre in ['pop', 'electronic', 'house']:  # Versatile starters
                    score += 0.10
                elif genre in ['hiphop', 'future_funk', 'vietnamese_pop']:
                    score += 0.07
                else:
                    score += 0.03
                
                # Factor 6: Overall compatibility (10%) - avg compatibility with all other tracks
                avg_compatibility = 0.0
                for other_track in analyzed_tracks:
                    if other_track == track:
                        continue
                    avg_compatibility += self.calculate_compatibility(track, other_track)
                avg_compatibility /= max(len(analyzed_tracks) - 1, 1)
                score += avg_compatibility * 0.10
                
                logger.debug(f"  Start candidate: {track['file_path'].name} (score: {score:.3f})")
                
                if score > best_start_score:
                    best_start_score = score
                    best_start_track = track
                    best_start_index = i
            
            start_track = best_start_track
            remaining_tracks = [t for i, t in enumerate(analyzed_tracks) if i != best_start_index]
            logger.info(f"Auto-selected best starting track: {start_track['file_path'].name} (score: {best_start_score:.3f})")
        
        ordered_tracks = [start_track]
        
        while remaining_tracks:
            current_track = ordered_tracks[-1]
            best_track = None
            best_score = -1
            
            # Enhanced greedy selection with look-ahead
            # Don't just pick the best immediate match - consider if it leaves us stuck
            for track in remaining_tracks:
                compatibility = self.calculate_compatibility(current_track, track)
                
                # Look-ahead penalty: if this track is incompatible with ALL remaining tracks,
                # it might leave us stuck with bad transitions later
                if len(remaining_tracks) > 1:
                    # Calculate track's average compatibility with other remaining tracks
                    other_remaining = [t for t in remaining_tracks if t != track]
                    if other_remaining:
                        forward_compatibility = sum(
                            self.calculate_compatibility(track, other) 
                            for other in other_remaining
                        ) / len(other_remaining)
                        
                        # Blend immediate and forward compatibility (70% immediate, 30% forward)
                        adjusted_score = compatibility * 0.70 + forward_compatibility * 0.30
                    else:
                        adjusted_score = compatibility
                else:
                    # Last track - only immediate compatibility matters
                    adjusted_score = compatibility
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_track = track
            
            if best_track:
                ordered_tracks.append(best_track)
                remaining_tracks.remove(best_track)
                logger.info(f"Next track: {best_track['file_path'].name} (compatibility: {best_score:.3f})")
            else:
                # If no good match found (shouldn't happen), take first remaining
                fallback_track = remaining_tracks.pop(0)
                ordered_tracks.append(fallback_track)
                logger.warning(f"Fallback: {fallback_track['file_path'].name} (no good match found)")
        
        return ordered_tracks
    
    def create_mix(self) -> bool:
        """
        Create the automix from all tracks in the folder
        """
        try:
            # Get audio files
            audio_files = self.get_audio_files()
            if not audio_files:
                logger.error("No audio files found in the input folder")
                return False
            
            # Interactive track selection if start_track_index is not set
            selected_file_path = None
            custom_order = []
            if self.start_track_index is None and len(audio_files) > 1:
                print("\n" + "="*60)
                print("Available tracks:")
                print("="*60)
                for i, file_path in enumerate(audio_files, 1):
                    print(f"  {i}. {file_path.name}")
                print("="*60)
                
                try:
                    choice = input(f"\nEnter track order (e.g. '3 1 2 4') or just start track (e.g. '2'), or Enter for auto: ").strip()
                    if choice:
                        # Parse the input
                        numbers = [int(x) for x in choice.split()]
                        
                        # Check if it's a full order or just starting track
                        if len(numbers) == len(audio_files) and set(numbers) == set(range(1, len(audio_files) + 1)):
                            # Full custom order
                            custom_order = [audio_files[n - 1] for n in numbers]
                            print(f"✓ Using custom order: {' → '.join([f.name for f in custom_order])}\n")
                        elif len(numbers) == 1 and 1 <= numbers[0] <= len(audio_files):
                            # Just starting track
                            selected_file_path = audio_files[numbers[0] - 1]
                            print(f"✓ Starting with track #{numbers[0]}: {selected_file_path.name}\n")
                        else:
                            print(f"✗ Invalid input. Using auto-selection.\n")
                    else:
                        print(f"✓ Using auto-selection for best flow.\n")
                except (ValueError, KeyboardInterrupt):
                    print(f"\n✓ Using auto-selection for best flow.\n")
            elif self.start_track_index is not None and 0 <= self.start_track_index < len(audio_files):
                # CLI --start-track was used
                selected_file_path = audio_files[self.start_track_index]
            
            # Analyze all tracks (parallel or sequential)
            if self.max_workers > 1:
                logger.info(f"Using {self.max_workers} parallel threads for analysis...")
                analyzed_tracks = self.analyze_tracks_parallel(audio_files, self.max_workers)
            else:
                logger.info("Analyzing tracks sequentially...")
                analyzed_tracks = []
                for file_path in audio_files:
                    analysis = self.analyze_audio(file_path)
                    if analysis:
                        analyzed_tracks.append(analysis)
            
            if len(analyzed_tracks) < 2:
                logger.error("Need at least 2 tracks to create a mix")
                return False
            
            # Apply custom order or find selected track
            if custom_order:
                # User specified full custom order - reorder analyzed tracks to match
                ordered_tracks = []
                for file_path in custom_order:
                    for track in analyzed_tracks:
                        if track['file_path'] == file_path:
                            ordered_tracks.append(track)
                            break
                if len(ordered_tracks) != len(analyzed_tracks):
                    logger.warning("Could not match all tracks in custom order, using auto-selection")
                    logger.info("Optimizing track order...")
                    ordered_tracks = self.smart_track_ordering(analyzed_tracks, None)
                else:
                    logger.info("Using custom track order (no optimization)")
            else:
                # Find the index of selected track in analyzed_tracks
                start_idx = None
                if selected_file_path is not None:
                    for i, track in enumerate(analyzed_tracks):
                        if track['file_path'] == selected_file_path:
                            start_idx = i
                            break
                    if start_idx is None:
                        logger.warning(f"Selected track not found in analyzed tracks, using auto-selection")
                
                # Smart ordering for better transitions
                logger.info("Optimizing track order...")
                ordered_tracks = self.smart_track_ordering(analyzed_tracks, start_idx)
            
            # Create the mix with enhanced Apple Music-style transitions (keeping original BPMs)
            logger.info("Creating Apple Music-style mix with original BPMs and transition-only tempo sync...")
            
            # Start with the first track - ensure it's stereo
            mixed_audio = ordered_tracks[0]['audio_data'].copy()
            # Convert to stereo if mono
            if mixed_audio.ndim == 1:
                mixed_audio = np.column_stack([mixed_audio, mixed_audio])
                logger.info("Converted first track from mono to stereo")
            
            # Log first track info
            first_track = ordered_tracks[0]
            logger.info(f"Starting with: {first_track['file_path'].name}")
            logger.info(f"  Tempo: {first_track.get('actual_tempo', first_track['tempo']):.1f} BPM ({first_track.get('tempo_multiplier', 'normal')})")
            if first_track.get('genre_hint') != 'unknown':
                logger.info(f"  Genre: {first_track.get('genre_hint')}")
            if first_track.get('groove_type') != 'straight':
                logger.info(f"  Groove: {first_track.get('groove_type')} (swing ratio: {first_track.get('swing_ratio', 0.5):.2f})")
            
            # Add each subsequent track with enhanced crossfade
            for i in range(1, len(ordered_tracks)):
                current_track = ordered_tracks[i-1]
                next_track = ordered_tracks[i]
                
                # Validate track has necessary data
                if current_track.get('audio_data') is None or next_track.get('audio_data') is None:
                    logger.error(f"  Skipping transition - missing audio data for {next_track['file_path'].name}")
                    continue
                    
                if len(next_track.get('audio_data', [])) < self.sample_rate:  # Less than 1 second
                    logger.warning(f"  Skipping {next_track['file_path'].name} - track too short (< 1 second)")
                    continue
                
                # Calculate optimal crossfade duration and style for this transition
                optimal_crossfade_duration, style_params = self._calculate_optimal_crossfade_duration(current_track, next_track)
                crossfade_samples = int(optimal_crossfade_duration * self.sample_rate)
                
                logger.info(f"\nMixing: {next_track['file_path'].name}")
                compatibility = self.calculate_compatibility(current_track, next_track)
                logger.info(f"  Compatibility score: {compatibility:.3f}")
                logger.info(f"  Transition style: '{style_params['style']}' ({style_params.get('reason', 'unknown')})")
                logger.info(f"  Crossfade duration: {optimal_crossfade_duration:.1f}s")
                if 'gap_duration' in style_params:
                    logger.info(f"  Gap duration: {style_params['gap_duration']:.1f}s")
                
                # Display tempo info with actual tempo (after doubling/halving detection)
                current_tempo = current_track.get('actual_tempo', current_track['tempo'])
                next_tempo = next_track.get('actual_tempo', next_track['tempo'])
                logger.info(f"  Current tempo: {current_tempo:.1f} BPM ({current_track.get('tempo_multiplier', 'normal')})")
                logger.info(f"  Next tempo: {next_tempo:.1f} BPM ({next_track.get('tempo_multiplier', 'normal')})")
                
                # Display genre and groove info
                if next_track.get('genre_hint') != 'unknown':
                    logger.info(f"  Genre: {next_track.get('genre_hint')}")
                if next_track.get('groove_type') != 'straight':
                    logger.info(f"  Groove: {next_track.get('groove_type')} (swing ratio: {next_track.get('swing_ratio', 0.5):.2f})")
                
                # Ensure smooth flow without cuts
                logger.info(f"  Analyzing flow points for seamless transition...")
                current_audio_for_mix = mixed_audio
                next_audio = next_track['audio_data'].copy()
                
                # Create flow-optimized versions
                if i == 1:  # First transition, use full current track
                    flow_current = current_audio_for_mix
                else:
                    # For subsequent tracks, we already have the flowing mix
                    flow_current = current_audio_for_mix
                
                # Optimize next track for natural flow entry
                flow_current, flow_next, intro_skip_samples = self._ensure_smooth_flow(
                    {'audio_data': flow_current, 'sample_rate': self.sample_rate, 
                     'outro_start': current_track.get('outro_start', len(flow_current) / self.sample_rate * 0.85),
                     'vocal_segments': current_track.get('vocal_segments', [])},
                    {'audio_data': next_audio, 'sample_rate': self.sample_rate,
                     'intro_end': next_track.get('intro_end', 0),
                     'vocal_segments': next_track.get('vocal_segments', [])}
                )
                
                if intro_skip_samples > 0:
                    logger.info(f"  Trimmed {intro_skip_samples/self.sample_rate:.1f}s from intro for vocal alignment")
                
                # Apply beat grid warping if enabled and grids available
                if self.use_beat_grid:
                    grid1 = current_track.get('beat_grid')
                    grid2 = next_track.get('beat_grid')
                    
                    if grid1 and grid2:
                        try:
                            from src.mixing.beat_warping import BeatWarper
                            
                            warper = BeatWarper(sample_rate=self.sample_rate)
                            
                            # Calculate crossfade region for warping
                            crossfade_start = len(flow_current) / self.sample_rate - optimal_crossfade_duration
                            
                            # Apply grid alignment
                            flow_current, flow_next = warper.align_swing_grids(
                                flow_current, flow_next,
                                grid1, grid2,
                                crossfade_start, optimal_crossfade_duration
                            )
                        except Exception as e:
                            logger.warning(f"  Beat grid warping failed: {e}")
                
                # Apply beat alignment for better transitions (without changing overall tempo)
                logger.info(f"  Aligning beats for seamless transition...")
                try:
                    # Pass full track context for adaptive alignment
                    track1_for_alignment = {
                        'audio_data': flow_current, 
                        'beats': current_track['beats'],
                        'downbeats': current_track.get('downbeats', []),
                        'phrases': current_track.get('phrases', []),
                        'beat_strengths': current_track.get('beat_strengths', np.ones(len(current_track['beats']))),
                        'beat_confidence': current_track.get('beat_confidence', np.ones(len(current_track['beats']))),
                        'sample_rate': self.sample_rate,
                        'genre_hint': current_track.get('genre_hint', 'unknown'),
                        'tempo': current_track['tempo'],
                        'actual_tempo': current_track.get('actual_tempo', current_track['tempo'])
                    }
                    track2_for_alignment = {
                        'audio_data': flow_next, 
                        'beats': next_track['beats'],
                        'downbeats': next_track.get('downbeats', []),
                        'phrases': next_track.get('phrases', []),
                        'beat_strengths': next_track.get('beat_strengths', np.ones(len(next_track['beats']))),
                        'beat_confidence': next_track.get('beat_confidence', np.ones(len(next_track['beats']))),
                        'sample_rate': self.sample_rate,
                        'genre_hint': next_track.get('genre_hint', 'unknown'),
                        'tempo': next_track['tempo'],
                        'actual_tempo': next_track.get('actual_tempo', next_track['tempo'])
                    }
                    
                    aligned_current, aligned_next = self.align_beats(
                        track1_for_alignment,
                        track2_for_alignment,
                        crossfade_samples,
                        intro_skip_samples  # Pass the intro skip so align_beats can adjust
                    )
                except Exception as e:
                    logger.warning(f"  Beat alignment failed: {e}, using unaligned audio")
                    aligned_current = flow_current
                    aligned_next = flow_next
                
                # Create enhanced crossfade with contextual style
                try:
                    mixed_audio = self.create_crossfade(
                        aligned_current, aligned_next, crossfade_samples,
                        current_track['tempo'], next_track['tempo'], style_params
                    )
                except Exception as e:
                    logger.error(f"  Crossfade creation failed: {e}")
                    logger.info(f"  Falling back to simple blend")
                    import traceback
                    traceback.print_exc()
                    
                    # Ensure both tracks have matching dimensions
                    aligned_current, aligned_next = self._match_audio_dimensions(aligned_current, aligned_next)
                    
                    # Fallback: simple overlap-add blend to avoid silence gaps
                    overlap_samples = min(crossfade_samples, len(aligned_current), len(aligned_next))
                    if overlap_samples > 0:
                        # Create equal-power crossfade manually
                        fade_out = np.linspace(1, 0, overlap_samples)
                        fade_in = np.linspace(0, 1, overlap_samples)
                        
                        track1_end = aligned_current[-overlap_samples:]
                        track2_start = aligned_next[:overlap_samples]
                        
                        if aligned_current.ndim > 1:
                            fade_out = fade_out.reshape(-1, 1)
                            fade_in = fade_in.reshape(-1, 1)
                        
                        overlap = track1_end * fade_out + track2_start * fade_in
                        mixed_audio = np.concatenate([
                            aligned_current[:-overlap_samples],
                            overlap,
                            aligned_next[overlap_samples:]
                        ])
                    else:
                        mixed_audio = np.concatenate([aligned_current, aligned_next])
            
            # Normalize final mix to consistent level without changing relative volumes
            logger.info("Applying final volume normalization...")
            final_rms = np.sqrt(np.mean(mixed_audio**2))
            target_rms = 0.2  # Conservative target level
            
            if final_rms > 0:
                final_gain = target_rms / final_rms
                # Limit gain to prevent extreme changes
                if final_gain > 2.0:
                    final_gain = 2.0
                elif final_gain < 0.5:
                    final_gain = 0.5
                mixed_audio = mixed_audio * final_gain
            
            # Final gentle limiting to prevent any clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 0.95:
                mixed_audio = mixed_audio * (0.95 / max_val)
                logger.info(f"Applied gentle limiting: {max_val:.3f} → 0.95")
            
            # Save the result with the highest channel count among all input tracks
            from src.utils.audio_io import save_audio
            output_path = self.input_folder / self.output_file
            output_channels = getattr(self, 'max_channel_count', 1)
            # Upmix if needed
            if mixed_audio.ndim == 1 and output_channels > 1:
                mixed_audio = np.tile(mixed_audio[:, None], (1, output_channels))
            elif mixed_audio.ndim == 2 and mixed_audio.shape[1] < output_channels:
                reps = output_channels // mixed_audio.shape[1]
                mixed_audio = np.tile(mixed_audio, (1, reps))
            # Save using save_audio utility (handles WAV and other formats)
            save_audio(output_path, mixed_audio, self.sample_rate)
            
            total_duration = len(mixed_audio) / self.sample_rate
            logger.info(f"Mix created successfully!")
            logger.info(f"Output: {output_path}")
            logger.info(f"Duration: {total_duration:.1f} seconds")
            logger.info(f"Tracks mixed: {len(ordered_tracks)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating mix: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="Create Apple Music-style automix from folder of tracks")
    parser.add_argument("input_folder", help="Path to folder containing audio tracks")
    parser.add_argument("-o", "--output", default="automix_output.wav", 
                       help="Output filename (default: automix_output.wav, use .mp3 for MP3 export)")
    parser.add_argument("-c", "--crossfade", type=float, default=8.0,
                       help="Crossfade duration in seconds (default: 8.0)")
    parser.add_argument("-s", "--sample-rate", type=int, default=44100,
                       help="Sample rate for processing (default: 44100)")
    parser.add_argument("-t", "--threads", type=int, default=4,
                       help="Number of parallel threads for analysis (default: 4, use 1 to disable)")
    parser.add_argument("--start-track", type=int, default=None,
                       help="Index of track to start with (1-based), use 0 for interactive menu, default: auto-select")
    parser.add_argument("--target-lufs", type=float, default=-14.0,
                       help="Target loudness in LUFS (default: -14.0 for streaming services)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching (slower but ensures fresh analysis)")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cache and exit")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Custom cache directory path")
    parser.add_argument("--gpu", action="store_true", default=None,
                       help="Force enable GPU acceleration (Apple Silicon only)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration, use CPU only")
    parser.add_argument("--gpu-batch", type=int, default=None,
                       help="GPU batch size (default: auto-detect based on memory)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Enable benchmark mode to compare GPU vs CPU performance")
    parser.add_argument("--beat-grid", action="store_true", default=None,
                       help="Enable beat grid warping for perfect timing (default: auto)")
    parser.add_argument("--no-beat-grid", action="store_true",
                       help="Disable beat grid warping")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        logger.error(f"Input folder does not exist: {args.input_folder}")
        return 1
    
    # Convert 1-based track index to 0-based (if provided)
    start_index = None
    if args.start_track is not None:
        if args.start_track == 0:
            # Interactive mode
            start_index = -1
        elif args.start_track < 1:
            logger.error("--start-track must be >= 1 (or 0 for interactive)")
            return 1
        else:
            start_index = args.start_track - 1
    
    # Determine GPU usage (default: True unless --no-gpu specified)
    use_gpu = True
    if args.no_gpu:
        use_gpu = False
    elif args.gpu is not None:
        use_gpu = args.gpu
    
    # Determine beat grid usage (default: True unless --no-beat-grid specified)
    use_beat_grid = True
    if args.no_beat_grid:
        use_beat_grid = False
    elif args.beat_grid is not None:
        use_beat_grid = args.beat_grid
    
    # Create automixer
    mixer = AutoMixer(
        input_folder=args.input_folder,
        output_file=args.output,
        crossfade_duration=args.crossfade,
        sample_rate=args.sample_rate,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        max_workers=args.threads,
        target_lufs=args.target_lufs,
        start_track_index=start_index,
        use_gpu=use_gpu,
        use_beat_grid=use_beat_grid
    )
    
    # Override batch size if specified
    if args.gpu_batch and mixer.gpu:
        mixer.gpu.batch_size = args.gpu_batch
        logger.info(f"Using custom GPU batch size: {args.gpu_batch}")
    
    # Enable benchmark mode if requested
    if args.benchmark:
        mixer.benchmark_mode = True
        logger.info("Benchmark mode enabled")
    
    # Handle clear cache command
    if args.clear_cache:
        count = mixer.clear_cache()
        logger.info(f"Cache cleared: {count} files deleted")
        return 0
    
    success = mixer.create_mix()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())