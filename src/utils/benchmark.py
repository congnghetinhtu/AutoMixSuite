"""
Performance benchmarking utilities for GPU vs CPU comparison.

Measures execution time, memory usage, and speedup factors for AutoMix operations.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    operation: str
    cpu_time: Optional[float] = None
    gpu_time: Optional[float] = None
    speedup: Optional[float] = None
    memory_used: Optional[float] = None  # MB
    batch_size: Optional[int] = None
    device: str = "unknown"
    
    def __str__(self) -> str:
        """Format benchmark results for display."""
        lines = [f"\n{'='*60}", f"Benchmark: {self.operation}", f"{'='*60}"]
        
        if self.cpu_time:
            lines.append(f"CPU Time:     {self.cpu_time:.3f}s")
        if self.gpu_time:
            lines.append(f"GPU Time:     {self.gpu_time:.3f}s")
        if self.speedup:
            lines.append(f"Speedup:      {self.speedup:.2f}x {'⚡' if self.speedup > 5 else ''}")
        if self.memory_used:
            lines.append(f"GPU Memory:   {self.memory_used:.1f} MB")
        if self.batch_size:
            lines.append(f"Batch Size:   {self.batch_size}")
        if self.device != "unknown":
            lines.append(f"Device:       {self.device}")
        
        lines.append('='*60)
        return '\n'.join(lines)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: list[BenchmarkResult] = field(default_factory=list)
    
    def add(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)
        logger.info(str(result))
    
    def summary(self) -> str:
        """Generate summary of all benchmarks."""
        if not self.results:
            return "No benchmarks recorded"
        
        total_cpu = sum(r.cpu_time for r in self.results if r.cpu_time)
        total_gpu = sum(r.gpu_time for r in self.results if r.gpu_time)
        avg_speedup = sum(r.speedup for r in self.results if r.speedup) / len([r for r in self.results if r.speedup])
        
        lines = [
            f"\n{'='*60}",
            "BENCHMARK SUMMARY",
            f"{'='*60}",
            f"Total Operations: {len(self.results)}",
            f"Total CPU Time:   {total_cpu:.3f}s",
            f"Total GPU Time:   {total_gpu:.3f}s",
            f"Overall Speedup:  {total_cpu/total_gpu:.2f}x" if total_gpu > 0 else "Overall Speedup:  N/A",
            f"Average Speedup:  {avg_speedup:.2f}x",
            f"{'='*60}"
        ]
        return '\n'.join(lines)


def benchmark_operation(
    operation_name: str,
    cpu_func: Optional[Callable] = None,
    gpu_func: Optional[Callable] = None,
    device: str = "unknown",
    batch_size: Optional[int] = None,
    warmup: bool = True
) -> BenchmarkResult:
    """
    Benchmark a single operation on CPU and/or GPU.
    
    Args:
        operation_name: Name of the operation being benchmarked
        cpu_func: Function to run on CPU (optional)
        gpu_func: Function to run on GPU (optional)
        device: Device name (e.g., "M1 Pro")
        batch_size: Batch size used (for batch operations)
        warmup: Run a warmup iteration before timing
        
    Returns:
        BenchmarkResult with timing and speedup information
    """
    result = BenchmarkResult(operation=operation_name, device=device, batch_size=batch_size)
    
    # CPU benchmark
    if cpu_func:
        if warmup:
            try:
                cpu_func()  # Warmup run
            except Exception as e:
                logger.warning(f"CPU warmup failed: {e}")
        
        start = time.perf_counter()
        try:
            cpu_func()
            result.cpu_time = time.perf_counter() - start
        except Exception as e:
            logger.error(f"CPU benchmark failed: {e}")
            result.cpu_time = None
    
    # GPU benchmark
    if gpu_func:
        if warmup:
            try:
                gpu_func()  # Warmup run
            except Exception as e:
                logger.warning(f"GPU warmup failed: {e}")
        
        start = time.perf_counter()
        try:
            gpu_func()
            result.gpu_time = time.perf_counter() - start
        except Exception as e:
            logger.error(f"GPU benchmark failed: {e}")
            result.gpu_time = None
    
    # Calculate speedup
    if result.cpu_time and result.gpu_time and result.cpu_time > 0 and result.gpu_time > 0:
        result.speedup = result.cpu_time / result.gpu_time
    
    return result


def estimate_speedup(
    chip_model: str,
    operation: str,
    batch_size: int = 1
) -> float:
    """
    Estimate expected speedup based on chip model and operation.
    
    Args:
        chip_model: Apple Silicon chip model (M1, M2, etc.)
        operation: Operation type (stft, correlation, chroma)
        batch_size: Batch size multiplier
        
    Returns:
        Estimated speedup factor
    """
    # Base speedup factors per chip and operation
    speedup_table = {
        'M1': {'stft': 15, 'correlation': 50, 'chroma': 12},
        'M1 Pro': {'stft': 18, 'correlation': 60, 'chroma': 15},
        'M1 Max': {'stft': 22, 'correlation': 70, 'chroma': 18},
        'M1 Ultra': {'stft': 25, 'correlation': 80, 'chroma': 20},
        'M2': {'stft': 18, 'correlation': 55, 'chroma': 14},
        'M2 Pro': {'stft': 22, 'correlation': 65, 'chroma': 17},
        'M2 Max': {'stft': 26, 'correlation': 75, 'chroma': 20},
        'M2 Ultra': {'stft': 30, 'correlation': 85, 'chroma': 22},
        'M3': {'stft': 20, 'correlation': 60, 'chroma': 16},
        'M3 Pro': {'stft': 24, 'correlation': 70, 'chroma': 19},
        'M3 Max': {'stft': 28, 'correlation': 80, 'chroma': 22},
        'M4': {'stft': 22, 'correlation': 65, 'chroma': 18},
    }
    
    base_speedup = speedup_table.get(chip_model, {}).get(operation, 10)
    
    # Batch operations get additional speedup
    batch_multiplier = min(1.5, 1 + (batch_size - 1) * 0.1)  # Up to 1.5x for large batches
    
    return base_speedup * batch_multiplier
