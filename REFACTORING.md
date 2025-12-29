# Project Refactoring Plan

## Current State
- **Single file**: `automix.py` (3,387 lines)
- **Monolithic**: All functionality in one AutoMixer class
- **Working**: Production-ready algorithm with no bugs

## Refactored Structure

### Phase 1: Extract Utilities (✓ Complete)
Created modular utility files:
- `src/constants.py` - Configuration constants
- `src/utils/audio_io.py` - Audio I/O and normalization
- `src/utils/file_utils.py` - File operations
- `src/analysis/genre_detector.py` - Genre classification

### Phase 2: Separate Analysis (In Progress)
Split analysis into focused modules:
- `src/analysis/beat_detector.py` - Beat/tempo/downbeat detection
- `src/analysis/key_detector.py` - Musical key detection  
- `src/analysis/vocal_detector.py` - Vocal segment detection
- `src/analysis/track_analyzer.py` - Orchestrates all analysis

### Phase 3: Extract Mixing Logic
Separate crossfade and transition logic:
- `src/mixing/crossfade.py` - Crossfade curve generation
- `src/mixing/transitions.py` - Transition style implementations
- `src/mixing/beat_aligner.py` - Beat alignment logic
- `src/mixing/flow_optimizer.py` - Intro/outro optimization

### Phase 4: Core Engine
Main mixer orchestration:
- `src/core/mixer.py` - AutoMixer main class
- `src/core/cache.py` - Caching system
- `src/core/compatibility.py` - Track compatibility scoring
- `src/core/ordering.py` - Smart track ordering

### Phase 5: CLI
Command-line interface:
- `src/cli/main.py` - Argument parsing and main entry
- `src/cli/interactive.py` - Interactive track selection

## Migration Strategy

### Option A: Gradual Migration (Recommended)
- Keep `automix.py` working
- Extract modules one at a time
- Import and use new modules in `automix.py`
- Eventually replace `automix.py` with thin wrapper

### Option B: Full Rewrite
- Create complete new structure
- Copy algorithm logic piece by piece
- Test against original for identical output
- Swap when complete

## Benefits

1. **Maintainability**
   - Easier to locate and fix bugs
   - Clear separation of concerns
   - Better code organization

2. **Testability**
   - Unit test individual modules
   - Mock dependencies easily
   - Faster test execution

3. **Extensibility**
   - Add new genres without touching core
   - Plug in new transition styles
   - Replace analysis methods

4. **Readability**
   - Self-documenting module names
   - Focused functions (single responsibility)
   - Better inline documentation

## Implementation Notes

- **No algorithm changes**: Refactoring preserves exact behavior
- **Backward compatible**: Old `automix.py` still works
- **Incremental**: Can adopt modules gradually
- **Tested**: Each module tested against original output

## Next Steps

1. Complete beat detector extraction
2. Extract key detector
3. Create track analyzer orchestrator
4. Test module integration
5. Extract crossfade logic
6. Create new main entry point
7. Deprecate monolithic file
