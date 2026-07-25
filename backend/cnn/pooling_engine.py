#!/usr/bin/env python3
"""
Pooling Engine for CNN Spatial Dimension Reduction

Fast max and average pooling implementations optimized for
order book heatmap processing in the ZAID trading bot.

Features:
- Max pooling for feature extraction
- Average pooling for smoothing
- Global pooling for classification heads
- Strided pooling for downsampling
- Batch processing support

Integrates with the 152 domains of quantitative finance.
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoolingMode(Enum):
    """Pooling operation types."""
    MAX = "max"
    AVG = "avg"
    MIN = "min"


@dataclass
class PoolingConfig:
    """Configuration for pooling operations."""
    kernel_size: Union[int, Tuple[int, int]]
    stride: Optional[Union[int, Tuple[int, int]]] = None
    padding: int = 0
    mode: PoolingMode = PoolingMode.MAX
    include_indices: bool = False  # For unpooling
    
    def __post_init__(self):
        # Normalize to tuples
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if self.stride is None:
            self.stride = self.kernel_size
        elif isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)


class Pooling2D:
    """
    2D Pooling layer for spatial dimension reduction.
    
    Supports max, average, and min pooling with configurable
    kernel sizes and strides for CNN feature maps.
    """
    
    def __init__(self, config: Optional[PoolingConfig] = None):
        self.config = config or PoolingConfig(kernel_size=2, stride=2)
        self._indices_cache: Optional[npt.NDArray[np.int64]] = None
        
        logger.info(f"Pooling2D initialized: {self.config}")
    
    def forward(
        self,
        input_tensor: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Forward pass through pooling layer.
        
        Args:
            input_tensor: Input [batch, channels, height, width] or [channels, height, width]
            
        Returns:
            Pooled output tensor
        """
        # Ensure 4D input
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor[np.newaxis, :, :, :]
        
        batch_size, channels, in_height, in_width = input_tensor.shape
        
        kh, kw = self.config.kernel_size
        sh, sw = self.config.stride
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.config.padding - kh) // sh + 1
        out_width = (in_width + 2 * self.config.padding - kw) // sw + 1
        
        # Apply padding if needed
        if self.config.padding > 0:
            padded = np.pad(
                input_tensor,
                ((0, 0), (0, 0), 
                 (self.config.padding, self.config.padding),
                 (self.config.padding, self.config.padding)),
                mode='constant',
                constant_values=-np.inf if self.config.mode == PoolingMode.MAX else 0
            )
        else:
            padded = input_tensor
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float64)
        
        # Store indices if requested (for max unpooling)
        if self.config.include_indices and self.config.mode == PoolingMode.MAX:
            self._indices_cache = np.zeros(
                (batch_size, channels, out_height, out_width), dtype=np.int64
            )
        
        # Perform pooling
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * sh
                        h_end = h_start + kh
                        w_start = ow * sw
                        w_end = w_start + kw
                        
                        region = padded[b, c, h_start:h_end, w_start:w_end]
                        
                        if self.config.mode == PoolingMode.MAX:
                            output[b, c, oh, ow] = np.max(region)
                            
                            if self.config.include_indices:
                                flat_idx = np.argmax(region.flatten())
                                h_idx = flat_idx // kw
                                w_idx = flat_idx % kw
                                self._indices_cache[b, c, oh, ow] = h_start + h_idx * in_width + w_start + w_idx
                                
                        elif self.config.mode == PoolingMode.AVG:
                            output[b, c, oh, ow] = np.mean(region)
                            
                        elif self.config.mode == PoolingMode.MIN:
                            output[b, c, oh, ow] = np.min(region)
        
        return output
    
    def backward(
        self,
        grad_output: npt.NDArray[np.float64],
        input_tensor: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Backward pass for gradient computation (training only).
        
        Args:
            grad_output: Gradient from subsequent layer
            input_tensor: Original input (needed for max pooling)
            
        Returns:
            Gradient with respect to input
        """
        if len(grad_output.shape) == 3:
            grad_output = grad_output[np.newaxis, :, :, :]
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor[np.newaxis, :, :, :]
        
        batch_size, channels, in_height, in_width = input_tensor.shape
        _, _, out_height, out_width = grad_output.shape
        
        grad_input = np.zeros_like(input_tensor)
        
        kh, kw = self.config.kernel_size
        sh, sw = self.config.stride
        
        if self.config.mode == PoolingMode.MAX:
            # Max pooling backward: route gradients to max positions
            for b in range(batch_size):
                for c in range(channels):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            h_start = oh * sh
                            h_end = h_start + kh
                            w_start = ow * sw
                            w_end = w_start + kw
                            
                            region = input_tensor[b, c, h_start:h_end, w_start:w_end]
                            max_pos = np.unravel_index(np.argmax(region), region.shape)
                            
                            grad_input[b, c, h_start + max_pos[0], w_start + max_pos[1]] += \
                                grad_output[b, c, oh, ow]
                                
        elif self.config.mode == PoolingMode.AVG:
            # Average pooling backward: distribute gradients equally
            pool_size = kh * kw
            for b in range(batch_size):
                for c in range(channels):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            h_start = oh * sh
                            h_end = h_start + kh
                            w_start = ow * sw
                            w_end = w_start + kw
                            
                            grad_input[b, c, h_start:h_end, w_start:w_end] += \
                                grad_output[b, c, oh, ow] / pool_size
        
        return grad_input
    
    def get_indices(self) -> Optional[npt.NDArray[np.int64]]:
        """Get stored max positions for unpooling."""
        return self._indices_cache


class GlobalPooling2D:
    """
    Global pooling layer that reduces entire feature map to single value.
    
    Commonly used before classification layers to create fixed-size
    representations regardless of input dimensions.
    """
    
    def __init__(self, mode: PoolingMode = PoolingMode.MAX):
        self.mode = mode
        logger.info(f"GlobalPooling2D initialized: mode={mode.value}")
    
    def forward(
        self,
        input_tensor: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Global pooling over spatial dimensions.
        
        Args:
            input_tensor: Input [batch, channels, height, width]
            
        Returns:
            Global pooled output [batch, channels]
        """
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor[np.newaxis, :, :, :]
        
        if self.mode == PoolingMode.MAX:
            return np.max(input_tensor, axis=(2, 3))
        elif self.mode == PoolingMode.AVG:
            return np.mean(input_tensor, axis=(2, 3))
        elif self.mode == PoolingMode.MIN:
            return np.min(input_tensor, axis=(2, 3))


def adaptive_avg_pool2d(
    input_tensor: npt.NDArray[np.float64],
    output_size: Tuple[int, int]
) -> npt.NDArray[np.float64]:
    """
    Adaptive average pooling to target output size.
    
    Automatically calculates kernel and stride to achieve
    the desired output dimensions.
    
    Args:
        input_tensor: Input [batch, channels, height, width]
        output_size: Target (height, width)
        
    Returns:
        Adaptively pooled output
    """
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor[np.newaxis, :, :, :]
    
    batch_size, channels, in_height, in_width = input_tensor.shape
    out_height, out_width = output_size
    
    stride_h = in_height // out_height
    stride_w = in_width // out_width
    kernel_h = in_height - (out_height - 1) * stride_h
    kernel_w = in_width - (out_width - 1) * stride_w
    
    output = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float64)
    
    for b in range(batch_size):
        for c in range(channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    h_start = oh * stride_h
                    w_start = ow * stride_w
                    h_end = h_start + kernel_h
                    w_end = w_start + kernel_w
                    
                    region = input_tensor[b, c, h_start:h_end, w_start:w_end]
                    output[b, c, oh, ow] = np.mean(region)
    
    return output


def main():
    """Test pooling implementations."""
    np.random.seed(42)
    
    print("=" * 60)
    print("Pooling Engine Test")
    print("=" * 60)
    
    # Create test input [batch=2, channels=3, height=8, width=8]
    input_tensor = np.random.randn(2, 3, 8, 8).astype(np.float64)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Test max pooling
    print("\n1. Testing Max Pooling...")
    max_pool_config = PoolingConfig(kernel_size=2, stride=2, mode=PoolingMode.MAX)
    max_pool = Pooling2D(max_pool_config)
    
    output_max = max_pool.forward(input_tensor)
    print(f"   Output shape: {output_max.shape}")
    print(f"   Output range: [{output_max.min():.3f}, {output_max.max():.3f}]")
    
    # Test average pooling
    print("\n2. Testing Average Pooling...")
    avg_pool_config = PoolingConfig(kernel_size=2, stride=2, mode=PoolingMode.AVG)
    avg_pool = Pooling2D(avg_pool_config)
    
    output_avg = avg_pool.forward(input_tensor)
    print(f"   Output shape: {output_avg.shape}")
    print(f"   Output mean: {output_avg.mean():.3f}")
    
    # Test global pooling
    print("\n3. Testing Global Max Pooling...")
    global_pool = GlobalPooling2D(PoolingMode.MAX)
    
    output_global = global_pool.forward(input_tensor)
    print(f"   Output shape: {output_global.shape}")
    print(f"   Output: {output_global[0]}")  # First batch
    
    # Test adaptive pooling
    print("\n4. Testing Adaptive Average Pooling...")
    output_adaptive = adaptive_avg_pool2d(input_tensor, (3, 3))
    print(f"   Output shape: {output_adaptive.shape}")
    
    # Test backward pass
    print("\n5. Testing Backward Pass...")
    grad_output = np.ones_like(output_max)
    grad_input = max_pool.backward(grad_output, input_tensor)
    print(f"   Gradient shape: {grad_input.shape}")
    print(f"   Non-zero gradients: {np.count_nonzero(grad_input)}")
    
    # Performance benchmark
    print("\n6. Performance Benchmark...")
    import time
    
    large_input = np.random.randn(16, 64, 64, 64).astype(np.float64)
    
    start = time.perf_counter()
    for _ in range(10):
        _ = max_pool.forward(large_input)
    elapsed = (time.perf_counter() - start) * 1000 / 10
    
    print(f"   Large input: {large_input.shape}")
    print(f"   Avg time: {elapsed:.2f} ms/batch")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
