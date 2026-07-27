#!/usr/bin/env python3
"""
S4 Kernel: Structured State-Space Sequence Model Optimization

This module implements the S4 (Structured State-Space) convolution kernel
optimized for continuous-time order flow processing on AMD Ryzen AI 5.

Key Features:
- HiPPO-based state initialization for long-range dependencies
- FFT-based convolution for O(N log N) complexity
- Memory-efficient implementation respecting 8GB RAM limit
- C-extension ready design for maximum performance

Mathematical Foundation:
The S4 model uses a structured state-space representation:
    h'(t) = A*h(t) + B*x(t)
    y(t) = C*h(t)
Where A is initialized using HiPPO-LegS matrix for optimal memory.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

# Type aliases for strict type hinting
Array1D = np.ndarray
Array2D = np.ndarray


@dataclass
class S4Config:
    """Configuration for S4 kernel parameters."""
    hidden_dim: int = 64          # Hidden dimension (optimized for 8GB RAM)
    state_dim: int = 16           # State dimension for HiPPO matrix
    max_seq_len: int = 4096       # Maximum sequence length for FFT buffer
    dtype: np.dtype = np.float64  # Use float64 for numerical stability
    
    def __post_init__(self) -> None:
        if self.hidden_dim * self.state_dim > 1024:
            raise ValueError("State size exceeds memory budget for 8GB system")


class S4Kernel:
    """
    S4 Convolution Kernel for continuous-time order flow modeling.
    
    Implements the diagonal plus low-rank structure for efficient computation.
    Uses FFT-based convolution to achieve O(N log N) time complexity.
    """
    
    def __init__(self, config: Optional[S4Config] = None):
        self.config = config or S4Config()
        self._initialize_hippo_params()
        self._precompute_convolution_kernel()
        
        # Circular buffer for streaming inference
        self.input_buffer: Array1D = np.zeros(
            self.config.max_seq_len, dtype=self.config.dtype
        )
        self.buffer_idx: int = 0
        self.is_initialized: bool = False
    
    def _initialize_hippo_params(self) -> None:
        """
        Initialize HiPPO-LegS matrices for optimal long-range memory.
        
        The HiPPO framework provides theoretically grounded initialization
        for capturing long-term dependencies in continuous-time signals.
        """
        N = self.config.state_dim
        
        # HiPPO-LegS A matrix (diagonal approximation for efficiency)
        # Full HiPPO matrix would be too expensive for 8GB constraint
        # Using diagonal approximation: A_ii = -i
        self.A_diag: Array1D = -np.arange(1, N + 1, dtype=self.config.dtype)
        
        # B vector: projection of input onto Legendre polynomials
        self.B: Array1D = np.sqrt(2 * np.arange(1, N + 1, dtype=self.config.dtype))
        
        # C vector: learned output projection (initialized randomly)
        np.random.seed(42)  # Reproducibility
        self.C: Array1D = np.random.randn(N).astype(self.config.dtype) * 0.1
        
        # Time step parameter (learnable in training, fixed here)
        self.dt: float = 0.01
    
    def _precompute_convolution_kernel(self) -> None:
        """
        Precompute the SSM convolution kernel in frequency domain.
        
        The continuous SSM is discretized and its impulse response is
        computed for FFT-based convolution during inference.
        """
        # Discretize the continuous SSM using zero-order hold
        # A_d = exp(A * dt), B_d = A^{-1}(exp(A*dt) - I) * B
        A_d = np.exp(self.A_diag * self.dt)
        B_d = (A_d - 1.0) / self.A_diag * self.B
        
        # Compute impulse response (kernel) for convolution
        # K[t] = C * A_d^t * B_d for t = 0, 1, ..., max_seq_len-1
        max_len = self.config.max_seq_len
        self.kernel: Array1D = np.zeros(max_len, dtype=self.config.dtype)
        
        A_power = np.ones_like(self.A_diag)  # A_d^0 = I
        for t in range(max_len):
            self.kernel[t] = np.dot(self.C, A_power * B_d)
            A_power *= A_d  # Element-wise power update
        
        # Precompute FFT of kernel for fast convolution
        self.kernel_fft: Array1D = np.fft.rfft(self.kernel)
        
        self.is_initialized = True
    
    def process_sequence(self, inputs: Array1D) -> Array1D:
        """
        Process an entire sequence using FFT-based convolution.
        
        Args:
            inputs: Input sequence of shape [seq_len]
            
        Returns:
            Output sequence of same shape as inputs
        """
        if not self.is_initialized:
            raise RuntimeError("S4Kernel not properly initialized")
        
        seq_len = len(inputs)
        if seq_len > self.config.max_seq_len:
            warnings.warn(
                f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}, "
                "truncating to max length"
            )
            inputs = inputs[:self.config.max_seq_len]
            seq_len = self.config.max_seq_len
        
        # Pad inputs to next power of 2 for efficient FFT
        fft_len = 1
        while fft_len < 2 * seq_len:
            fft_len *= 2
        
        # FFT-based convolution: y = F^{-1}(F(k) * F(x))
        input_padded = np.zeros(fft_len, dtype=self.config.dtype)
        input_padded[:seq_len] = inputs
        
        input_fft = np.fft.rfft(input_padded)
        output_fft = self.kernel_fft[:len(input_fft)] * input_fft
        output_full = np.fft.irfft(output_fft, n=fft_len)
        
        return output_full[:seq_len]
    
    def process_tick_streaming(self, tick_value: float) -> float:
        """
        Process a single tick in streaming mode with O(1) memory per step.
        
        Uses circular buffer and incremental convolution update.
        Suitable for real-time high-frequency trading scenarios.
        
        Args:
            tick_value: Single scalar input value
            
        Returns:
            Single scalar output value
        """
        # Update circular buffer
        self.input_buffer[self.buffer_idx] = tick_value
        self.buffer_idx = (self.buffer_idx + 1) % self.config.max_seq_len
        
        # Compute output via truncated convolution sum
        # y[t] = sum_{tau=0}^{T} K[tau] * x[t-tau]
        output = 0.0
        for tau in range(min(self.config.state_dim * 4, self.config.max_seq_len)):
            idx = (self.buffer_idx - 1 - tau) % self.config.max_seq_len
            output += self.kernel[tau] * self.input_buffer[idx]
        
        return output
    
    def reset_state(self) -> None:
        """Reset internal buffers for new episode/sequence."""
        self.input_buffer.fill(0.0)
        self.buffer_idx = 0
    
    def get_state_dimension(self) -> int:
        """Return the state dimension for monitoring/debugging."""
        return self.config.state_dim
    
    def estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes for capacity planning."""
        base_size = self.config.hidden_dim * self.config.state_dim * 8  # float64
        kernel_size = self.config.max_seq_len * 8
        buffer_size = self.config.max_seq_len * 8
        return base_size + kernel_size + buffer_size


class ContinuousTimeOrderFlow(S4Kernel):
    """
    Specialized S4 kernel for continuous-time order flow modeling.
    
    Extends S4Kernel with domain-specific preprocessing for L2 order book data.
    """
    
    def __init__(self, config: Optional[S4Config] = None):
        super().__init__(config)
        self.order_flow_buffer: List[Tuple[float, float]] = []  # (price_delta, volume)
    
    def add_order_book_event(self, price_delta: float, volume: float) -> None:
        """Add an order book event to the continuous-time stream."""
        self.order_flow_buffer.append((price_delta, volume))
        
        # Keep buffer bounded
        if len(self.order_flow_buffer) > 1000:
            self.order_flow_buffer.pop(0)
    
    def compute_order_flow_signal(self) -> Array1D:
        """
        Convert order book events into continuous-time signal for S4 processing.
        
        Uses volume-weighted price impact as the signal.
        """
        if not self.order_flow_buffer:
            return np.array([0.0], dtype=self.config.dtype)
        
        signals = []
        for price_delta, volume in self.order_flow_buffer:
            # Volume-weighted price impact
            signal = price_delta * np.sqrt(volume)
            signals.append(signal)
        
        return np.array(signals[-self.config.max_seq_len:], dtype=self.config.dtype)
    
    def process_order_flow(self) -> Array1D:
        """Process the current order flow buffer through S4 kernel."""
        signal = self.compute_order_flow_signal()
        if len(signal) == 0:
            return np.array([0.0], dtype=self.config.dtype)
        return self.process_sequence(signal)


def benchmark_s4_performance() -> None:
    """Benchmark S4 kernel performance for validation."""
    import time
    
    config = S4Config(hidden_dim=64, state_dim=16, max_seq_len=2048)
    kernel = S4Kernel(config)
    
    # Test batch processing
    test_seq = np.random.randn(1024).astype(np.float64)
    
    start = time.perf_counter()
    for _ in range(100):
        _ = kernel.process_sequence(test_seq)
    batch_time = (time.perf_counter() - start) / 100
    
    # Test streaming processing
    start = time.perf_counter()
    for _ in range(100):
        for val in test_seq:
            _ = kernel.process_tick_streaming(val)
    stream_time = (time.perf_counter() - start) / 100
    
    print(f"S4 Batch Processing (1024 seq): {batch_time*1000:.2f}ms")
    print(f"S4 Streaming Processing (1024 ticks): {stream_time*1000:.2f}ms")
    print(f"Memory Usage Estimate: {kernel.estimate_memory_usage() / 1024:.2f} KB")


if __name__ == "__main__":
    benchmark_s4_performance()
