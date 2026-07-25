#!/usr/bin/env python3
"""
Array Operations Module for ZAID Trading Bot
Wraps Rust SIMD functions for rapid Python array scaling
Provides C-extension interfaces for zero-copy data transfer

This module bridges Python numpy arrays with Rust SIMD operations
for high-performance price mathematics on AMD Ryzen AI 5.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Union
import ctypes
from dataclasses import dataclass

# Type aliases for strict typing
PriceArray = Union[np.ndarray, List[float]]
FloatLike = Union[float, np.floating]


@dataclass
class SimdResult:
    """Container for SIMD operation results with metadata"""
    values: np.ndarray
    execution_time_ns: int
    used_simd: bool
    lane_count: int


class ArrayOperations:
    """
    High-performance array operations wrapper for Rust SIMD functions.
    
    Uses ctypes to call into compiled Rust shared libraries for AVX2
    accelerated operations while maintaining Python ergonomics.
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize array operations with optional Rust library path.
        
        Args:
            lib_path: Path to compiled Rust SIMD library (.so/.dll)
        """
        self.lib_path = lib_path
        self._lib = None
        self._load_library()
    
    def _load_library(self) -> None:
        """Load the Rust SIMD library if available"""
        if self.lib_path:
            try:
                self._lib = ctypes.CDLL(self.lib_path)
                self._setup_ctypes_signatures()
            except OSError:
                # Library not available, will use pure Python fallbacks
                self._lib = None
    
    def _setup_ctypes_signatures(self) -> None:
        """Configure ctypes function signatures for Rust FFI"""
        if self._lib is None:
            return
        
        # scale_prices: fn(*mut f64, usize, f64)
        self._lib.scale_prices.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double
        ]
        self._lib.scale_prices.restype = None
        
        # compute_dot_product: fn(*const f64, *const f64, usize) -> f64
        self._lib.compute_dot_product.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t
        ]
        self._lib.compute_dot_product.restype = ctypes.c_double
        
        # find_min_max: fn(*const f64, usize, *mut f64, *mut f64)
        self._lib.find_min_max.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]
        self._lib.find_min_max.restype = None
    
    def scale_array(self, 
                    arr: PriceArray, 
                    scalar: FloatLike,
                    inplace: bool = False) -> SimdResult:
        """
        Scale all elements in array by scalar using SIMD acceleration.
        
        Args:
            arr: Input price array (list or numpy array)
            scalar: Multiplication factor
            inplace: Whether to modify input array
            
        Returns:
            SimdResult with scaled values and performance metadata
        """
        import time
        
        # Convert to numpy for contiguous memory layout
        if isinstance(arr, list):
            np_arr = np.array(arr, dtype=np.float64)
        else:
            np_arr = np.asarray(arr, dtype=np.float64)
        
        # Ensure contiguous memory for safe FFI
        if not np_arr.flags['C_CONTIGUOUS']:
            np_arr = np.ascontiguousarray(np_arr)
        
        target = np_arr if inplace else np_arr.copy()
        start_ns = time.perf_counter_ns()
        used_simd = False
        
        # Try Rust SIMD path
        if self._lib is not None and len(target) > 0:
            try:
                ptr = target.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                self._lib.scale_prices(ptr, len(target), float(scalar))
                used_simd = True
            except Exception:
                # Fallback to numpy
                target *= scalar
        else:
            # Pure Python/numpy fallback
            target *= scalar
        
        elapsed_ns = time.perf_counter_ns() - start_ns
        
        return SimdResult(
            values=target,
            execution_time_ns=elapsed_ns,
            used_simd=used_simd,
            lane_count=4 if used_simd else 1
        )
    
    def add_arrays(self, 
                   a: PriceArray, 
                   b: PriceArray,
                   out: Optional[np.ndarray] = None) -> SimdResult:
        """
        Element-wise addition of two price arrays using SIMD.
        
        Args:
            a: First input array
            b: Second input array
            out: Optional output buffer
            
        Returns:
            SimdResult with sum and performance metadata
        """
        import time
        
        np_a = np.asarray(a, dtype=np.float64)
        np_b = np.asarray(b, dtype=np.float64)
        
        if np_a.shape != np_b.shape:
            raise ValueError("Array shapes must match for addition")
        
        if out is None:
            result = np.empty_like(np_a)
        else:
            if out.shape != np_a.shape:
                raise ValueError("Output buffer shape mismatch")
            result = out
        
        start_ns = time.perf_counter_ns()
        used_simd = False
        
        # Try Rust SIMD path
        if self._lib is not None and len(result) >= 4:
            try:
                ptr_a = np_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                ptr_b = np_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                ptr_out = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                
                # Note: This would need corresponding Rust FFI function
                # For now, use numpy which also uses SIMD internally
                np.add(np_a, np_b, out=result)
            except Exception:
                np.add(np_a, np_b, out=result)
        else:
            np.add(np_a, np_b, out=result)
        
        elapsed_ns = time.perf_counter_ns() - start_ns
        
        return SimdResult(
            values=result,
            execution_time_ns=elapsed_ns,
            used_simd=used_simd,
            lane_count=4 if used_simd else 1
        )
    
    def dot_product(self, a: PriceArray, b: PriceArray) -> Tuple[float, SimdResult]:
        """
        Compute dot product of two arrays using SIMD acceleration.
        
        Args:
            a: First input array
            b: Second input array
            
        Returns:
            Tuple of (dot_product_value, SimdResult metadata)
        """
        import time
        
        np_a = np.asarray(a, dtype=np.float64)
        np_b = np.asarray(b, dtype=np.float64)
        
        if np_a.shape != np_b.shape:
            raise ValueError("Array shapes must match for dot product")
        
        start_ns = time.perf_counter_ns()
        used_simd = False
        result = 0.0
        
        # Try Rust SIMD path
        if self._lib is not None and len(np_a) >= 4:
            try:
                ptr_a = np_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                ptr_b = np_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                result = self._lib.compute_dot_product(ptr_a, ptr_b, len(np_a))
                used_simd = True
            except Exception:
                result = float(np.dot(np_a, np_b))
        else:
            result = float(np.dot(np_a, np_b))
        
        elapsed_ns = time.perf_counter_ns() - start_ns
        
        metadata = SimdResult(
            values=np.array([result]),
            execution_time_ns=elapsed_ns,
            used_simd=used_simd,
            lane_count=4 if used_simd else 1
        )
        
        return result, metadata
    
    def min_max(self, arr: PriceArray) -> Tuple[float, float, SimdResult]:
        """
        Find minimum and maximum values using SIMD comparison.
        
        Args:
            arr: Input price array
            
        Returns:
            Tuple of (min_val, max_val, SimdResult metadata)
        """
        import time
        
        np_arr = np.asarray(arr, dtype=np.float64)
        
        if len(np_arr) == 0:
            return 0.0, 0.0, SimdResult(
                values=np.array([]),
                execution_time_ns=0,
                used_simd=False,
                lane_count=1
            )
        
        start_ns = time.perf_counter_ns()
        used_simd = False
        min_val, max_val = 0.0, 0.0
        
        # Try Rust SIMD path
        if self._lib is not None and len(np_arr) >= 4:
            try:
                ptr_in = np_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                min_ptr = ctypes.c_double()
                max_ptr = ctypes.c_double()
                
                self._lib.find_min_max(ptr_in, len(np_arr), 
                                       ctypes.byref(min_ptr), 
                                       ctypes.byref(max_ptr))
                min_val = min_ptr.value
                max_val = max_ptr.value
                used_simd = True
            except Exception:
                min_val = float(np.min(np_arr))
                max_val = float(np.max(np_arr))
        else:
            min_val = float(np.min(np_arr))
            max_val = float(np.max(np_arr))
        
        elapsed_ns = time.perf_counter_ns() - start_ns
        
        metadata = SimdResult(
            values=np.array([min_val, max_val]),
            execution_time_ns=elapsed_ns,
            used_simd=used_simd,
            lane_count=4 if used_simd else 1
        )
        
        return min_val, max_val, metadata
    
    def normalize(self, arr: PriceArray, 
                  epsilon: float = 1e-10) -> SimdResult:
        """
        Min-max normalization of price array to [0, 1] range.
        
        Args:
            arr: Input price array
            epsilon: Small value to prevent division by zero
            
        Returns:
            SimdResult with normalized values
        """
        np_arr = np.asarray(arr, dtype=np.float64).copy()
        
        if len(np_arr) == 0:
            return SimdResult(
                values=np_arr,
                execution_time_ns=0,
                used_simd=False,
                lane_count=1
            )
        
        min_val, max_val, _ = self.min_max(np_arr)
        range_val = max_val - min_val
        
        if range_val > epsilon:
            # Scale by inverse range
            scaled = self.scale_array(np_arr, 1.0 / range_val, inplace=True)
            # Subtract normalized minimum
            scaled.values -= min_val / range_val
            return scaled
        else:
            # All values are essentially the same
            np_arr.fill(0.5)
            return SimdResult(
                values=np_arr,
                execution_time_ns=0,
                used_simd=False,
                lane_count=1
            )


# Convenience functions for direct usage
def scale_prices(prices: PriceArray, scalar: FloatLike) -> np.ndarray:
    """Quick price scaling without object instantiation"""
    ops = ArrayOperations()
    return ops.scale_array(prices, scalar).values


def compute_correlation(a: PriceArray, b: PriceArray) -> float:
    """Compute correlation coefficient between two price series"""
    ops = ArrayOperations()
    dot_prod, _ = ops.dot_product(a, b)
    
    norm_a = ops.normalize(a)
    norm_b = ops.normalize(b)
    
    # Correlation is dot product of normalized vectors
    corr, _ = ops.dot_product(norm_a.values, norm_b.values)
    return corr


if __name__ == "__main__":
    # Self-test
    test_prices = [100.0, 102.5, 98.7, 105.2, 103.8]
    
    ops = ArrayOperations()
    
    # Test scaling
    scaled = ops.scale_array(test_prices, 1.1)
    print(f"Scaled prices: {scaled.values}")
    print(f"SIMD used: {scaled.used_simd}, Time: {scaled.execution_time_ns}ns")
    
    # Test min/max
    min_p, max_p, meta = ops.min_max(test_prices)
    print(f"Min: {min_p}, Max: {max_p}, SIMD: {meta.used_simd}")
    
    # Test normalization
    normalized = ops.normalize(test_prices)
    print(f"Normalized: {normalized.values}")
