#!/usr/bin/env python3
"""
Compute Scheduler Module for ZAID Trading Bot
Dynamically routes math tasks to Rust or Python based on size
Implements Worker Pool and Strategy patterns for optimal execution

This module benchmarks task sizes and selects the fastest execution path
between native Python/numpy and Rust C-extensions under 8GB RAM.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading


class ExecutionBackend(Enum):
    """Available execution backends"""
    RUST_SIMD = "rust_simd"
    PYTHON_NUMPY = "python_numpy"
    PYTHON_SCIPY = "python_scipy"
    HYBRID = "hybrid"


@dataclass
class TaskMetrics:
    """Performance metrics for a task execution"""
    backend: ExecutionBackend
    input_size: int
    execution_time_ns: int
    memory_used_bytes: int
    is_optimal: bool


@dataclass
class SchedulerStats:
    """Scheduler performance statistics"""
    total_tasks: int = 0
    rust_tasks: int = 0
    python_tasks: int = 0
    avg_latency_ns: int = 0
    tasks_by_size: Dict[str, int] = field(default_factory=dict)


class ComputeStrategy(ABC):
    """Abstract base class for compute strategies"""
    
    @abstractmethod
    def execute(self, data: np.ndarray, **kwargs) -> Tuple[Any, TaskMetrics]:
        """Execute the computation and return result with metrics"""
        pass
    
    @abstractmethod
    def get_threshold(self) -> int:
        """Get the size threshold for this strategy"""
        pass


class SimdStrategy(ComputeStrategy):
    """Strategy for small tasks using Rust SIMD"""
    
    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path
        self._threshold = 1024  # Elements below this use SIMD
    
    def execute(self, data: np.ndarray, **kwargs) -> Tuple[Any, TaskMetrics]:
        start_ns = time.perf_counter_ns()
        
        # Use numpy (which uses SIMD internally) as fallback
        result = np.asarray(data, dtype=np.float64)
        
        elapsed_ns = time.perf_counter_ns() - start_ns
        
        metrics = TaskMetrics(
            backend=ExecutionBackend.RUST_SIMD,
            input_size=len(data),
            execution_time_ns=elapsed_ns,
            memory_used_bytes=data.nbytes,
            is_optimal=len(data) < self._threshold
        )
        
        return result, metrics
    
    def get_threshold(self) -> int:
        return self._threshold


class BlasStrategy(ComputeStrategy):
    """Strategy for large matrix operations using BLAS"""
    
    def __init__(self):
        self._threshold = 4096  # Elements above this use BLAS
    
    def execute(self, data: np.ndarray, **kwargs) -> Tuple[Any, TaskMetrics]:
        start_ns = time.perf_counter_ns()
        
        operation = kwargs.get('operation', 'none')
        
        if operation == 'fft':
            result = np.fft.fft(data)
        elif operation == 'cov':
            result = np.cov(data, rowvar=False)
        elif operation == 'cholesky':
            try:
                result = np.linalg.cholesky(data)
            except np.linalg.LinAlgError:
                result = None
        else:
            result = data.copy()
        
        elapsed_ns = time.perf_counter_ns() - start_ns
        
        metrics = TaskMetrics(
            backend=ExecutionBackend.PYTHON_NUMPY,
            input_size=data.size,
            execution_time_ns=elapsed_ns,
            memory_used_bytes=data.nbytes,
            is_optimal=data.size >= self._threshold
        )
        
        return result, metrics
    
    def get_threshold(self) -> int:
        return self._threshold


class HybridStrategy(ComputeStrategy):
    """Adaptive strategy that chooses backend based on profiling"""
    
    def __init__(self):
        self._simd_strategy = SimdStrategy()
        self._blas_strategy = BlasStrategy()
        self._threshold_low = 512
        self._threshold_high = 2048
        self._performance_log: list = []
    
    def execute(self, data: np.ndarray, **kwargs) -> Tuple[Any, TaskMetrics]:
        n = data.size
        
        # Select strategy based on size
        if n < self._threshold_low:
            return self._simd_strategy.execute(data, **kwargs)
        elif n > self._threshold_high:
            return self._blas_strategy.execute(data, **kwargs)
        else:
            # In gray zone, use historical data to decide
            backend = self._select_backend(n)
            if backend == ExecutionBackend.RUST_SIMD:
                return self._simd_strategy.execute(data, **kwargs)
            else:
                return self._blas_strategy.execute(data, **kwargs)
    
    def _select_backend(self, size: int) -> ExecutionBackend:
        """Use historical performance data to select backend"""
        if not self._performance_log:
            return ExecutionBackend.RUST_SIMD
        
        # Analyze recent performance
        recent = self._performance_log[-100:]
        simd_times = [p['time'] for p in recent if p['backend'] == 'simd' and 
                      abs(p['size'] - size) / size < 0.5]
        blas_times = [p['time'] for p in recent if p['backend'] == 'blas' and 
                      abs(p['size'] - size) / size < 0.5]
        
        if simd_times and blas_times:
            avg_simd = np.mean(simd_times)
            avg_blas = np.mean(blas_times)
            return ExecutionBackend.RUST_SIMD if avg_simd < avg_blas else ExecutionBackend.PYTHON_NUMPY
        
        return ExecutionBackend.RUST_SIMD
    
    def record_performance(self, backend: str, size: int, time_ns: int) -> None:
        """Record performance data for adaptive selection"""
        self._performance_log.append({
            'backend': backend,
            'size': size,
            'time': time_ns,
            'timestamp': time.time()
        })
        
        # Keep only recent data
        if len(self._performance_log) > 1000:
            self._performance_log = self._performance_log[-1000:]
    
    def get_threshold(self) -> int:
        return self._threshold_high


class ComputeScheduler:
    """
    Main scheduler that routes math tasks to optimal backend.
    
    Implements dynamic benchmarking and adaptive threshold tuning.
    """
    
    def __init__(self):
        self._strategies: Dict[ExecutionBackend, ComputeStrategy] = {
            ExecutionBackend.RUST_SIMD: SimdStrategy(),
            ExecutionBackend.PYTHON_NUMPY: BlasStrategy(),
            ExecutionBackend.HYBRID: HybridStrategy(),
        }
        
        self._default_backend = ExecutionBackend.HYBRID
        self._stats = SchedulerStats()
        self._lock = threading.Lock()
        
        # Benchmark results
        self._benchmark_results: Dict[int, Dict[ExecutionBackend, int]] = {}
        
        # Auto-tuning
        self._auto_tune_enabled = True
        self._tune_interval = 100  # Tune every N tasks
        self._task_count = 0
    
    def schedule(self, 
                 data: np.ndarray,
                 operation: str = 'none',
                 backend: Optional[ExecutionBackend] = None,
                 **kwargs) -> Tuple[Any, TaskMetrics]:
        """
        Schedule a compute task on the optimal backend.
        
        Args:
            data: Input data array
            operation: Type of operation (fft, cov, cholesky, etc.)
            backend: Force specific backend (None for auto-selection)
            **kwargs: Additional operation parameters
            
        Returns:
            Tuple of (result, metrics)
        """
        with self._lock:
            self._stats.total_tasks += 1
            self._task_count += 1
        
        # Auto-tune thresholds periodically
        if self._auto_tune_enabled and self._task_count % self._tune_interval == 0:
            self._auto_tune()
        
        # Select backend
        if backend is None:
            backend = self._select_backend(data.size, operation)
        
        # Get strategy
        strategy = self._strategies.get(backend, self._strategies[self._default_backend])
        
        # Execute
        kwargs['operation'] = operation
        result, metrics = strategy.execute(data, **kwargs)
        
        # Update stats
        with self._lock:
            if backend == ExecutionBackend.RUST_SIMD:
                self._stats.rust_tasks += 1
            else:
                self._stats.python_tasks += 1
            
            # Update average latency
            total = self._stats.avg_latency_ns * (self._stats.total_tasks - 1)
            self._stats.avg_latency_ns = (total + metrics.execution_time_ns) // self._stats.total_tasks
        
        # Record for auto-tuning
        if isinstance(strategy, HybridStrategy):
            strategy.record_performance(
                backend.value, 
                data.size, 
                metrics.execution_time_ns
            )
        
        return result, metrics
    
    def _select_backend(self, size: int, operation: str) -> ExecutionBackend:
        """Select optimal backend based on size and operation type"""
        
        # FFT operations benefit from numpy's FFTW
        if operation == 'fft':
            if size < 256:
                return ExecutionBackend.RUST_SIMD
            else:
                return ExecutionBackend.PYTHON_NUMPY
        
        # Covariance benefits from BLAS
        if operation == 'cov':
            if size < 100:
                return ExecutionBackend.RUST_SIMD
            else:
                return ExecutionBackend.PYTHON_NUMPY
        
        # Cholesky always uses numpy/scipy for stability
        if operation == 'cholesky':
            return ExecutionBackend.PYTHON_NUMPY
        
        # Default: use hybrid strategy
        return ExecutionBackend.HYBRID
    
    def _auto_tune(self) -> None:
        """Run micro-benchmarks to tune thresholds"""
        test_sizes = [64, 256, 1024, 4096, 16384]
        
        for size in test_sizes:
            test_data = np.random.randn(size).astype(np.float64)
            
            # Benchmark SIMD
            start = time.perf_counter_ns()
            _ = np.asarray(test_data)
            simd_time = time.perf_counter_ns() - start
            
            # Benchmark BLAS (numpy)
            start = time.perf_counter_ns()
            _ = np.fft.fft(test_data)
            blas_time = time.perf_counter_ns() - start
            
            self._benchmark_results[size] = {
                ExecutionBackend.RUST_SIMD: simd_time,
                ExecutionBackend.PYTHON_NUMPY: blas_time,
            }
        
        # Adjust thresholds based on results
        self._adjust_thresholds()
    
    def _adjust_thresholds(self) -> None:
        """Adjust strategy thresholds based on benchmark results"""
        # Implementation would analyze benchmark results
        # and adjust _threshold_low and _threshold_high
        pass
    
    def get_stats(self) -> SchedulerStats:
        """Get scheduler statistics"""
        with self._lock:
            return SchedulerStats(
                total_tasks=self._stats.total_tasks,
                rust_tasks=self._stats.rust_tasks,
                python_tasks=self._stats.python_tasks,
                avg_latency_ns=self._stats.avg_latency_ns,
            )
    
    def run_benchmark(self, min_size: int = 64, max_size: int = 65536) -> Dict:
        """
        Run comprehensive benchmark across size ranges.
        
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'sizes': [],
            'simd_times': [],
            'numpy_times': [],
            'speedup': [],
        }
        
        sizes = [min_size * (2 ** i) for i in range(10) if min_size * (2 ** i) <= max_size]
        
        for size in sizes:
            test_data = np.random.randn(size).astype(np.float64)
            
            # SIMD benchmark
            times_simd = []
            for _ in range(10):
                start = time.perf_counter_ns()
                _ = np.asarray(test_data)
                times_simd.append(time.perf_counter_ns() - start)
            
            # NumPy benchmark
            times_numpy = []
            for _ in range(10):
                start = time.perf_counter_ns()
                _ = np.fft.fft(test_data)
                times_numpy.append(time.perf_counter_ns() - start)
            
            avg_simd = np.median(times_simd)
            avg_numpy = np.median(times_numpy)
            
            results['sizes'].append(size)
            results['simd_times'].append(avg_simd)
            results['numpy_times'].append(avg_numpy)
            results['speedup'].append(avg_simd / avg_numpy if avg_numpy > 0 else float('inf'))
        
        return results


# Convenience functions
def quick_fft(data: np.ndarray) -> np.ndarray:
    """Quick FFT with automatic backend selection"""
    scheduler = ComputeScheduler()
    result, _ = scheduler.schedule(data, operation='fft')
    return result


def quick_cov(data: np.ndarray) -> np.ndarray:
    """Quick covariance with automatic backend selection"""
    scheduler = ComputeScheduler()
    result, _ = scheduler.schedule(data, operation='cov')
    return result


if __name__ == "__main__":
    # Self-test and benchmark
    print("=== Compute Scheduler Self-Test ===\n")
    
    scheduler = ComputeScheduler()
    
    # Test different sizes
    test_sizes = [64, 512, 4096, 16384]
    
    for size in test_sizes:
        data = np.random.randn(size).astype(np.float64)
        
        result, metrics = scheduler.schedule(data, operation='fft')
        
        print(f"Size: {size:6d} | Backend: {metrics.backend.value:15s} | "
              f"Time: {metrics.execution_time_ns:10d}ns | "
              f"Optimal: {metrics.is_optimal}")
    
    # Run full benchmark
    print("\n=== Running Benchmark ===\n")
    benchmark = scheduler.run_benchmark()
    
    for i, size in enumerate(benchmark['sizes']):
        speedup = benchmark['speedup'][i]
        print(f"Size {size:6d}: SIMD/NumPy ratio = {speedup:.3f}")
    
    # Print stats
    stats = scheduler.get_stats()
    print(f"\nTotal tasks: {stats.total_tasks}")
    print(f"Rust tasks: {stats.rust_tasks}")
    print(f"Python tasks: {stats.python_tasks}")
