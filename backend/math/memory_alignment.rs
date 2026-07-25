//! Memory Alignment Module for ZAID Trading Bot
//! Ensures 32-byte alignment to prevent CPU cache misses and AVX2 segfaults
//! Critical for stable SIMD operations on AMD Ryzen AI 5
//!
//! This module provides allocation primitives that guarantee proper
//! memory alignment for vectorized operations, preventing crashes
//! and maximizing cache efficiency under the 8GB RAM constraint.

use std::alloc::{self, Layout};
use std::ptr;
use std::slice;

/// Required alignment for AVX2 operations (256-bit = 32 bytes)
pub const AVX2_ALIGNMENT: usize = 32;
/// Cache line size for optimal L1 cache utilization
pub const CACHE_LINE_SIZE: usize = 64;
/// Prefetch distance in elements for streaming operations
const PREFETCH_DISTANCE: usize = 8;

/// Custom allocator wrapper ensuring 32-byte alignment
#[derive(Debug)]
pub struct AlignedAllocator {
    alignment: usize,
}

impl AlignedAllocator {
    /// Create a new aligned allocator with specified alignment
    pub fn new(alignment: usize) -> Self {
        // Ensure alignment is at least AVX2 requirement
        let effective_alignment = alignment.max(AVX2_ALIGNMENT);
        Self {
            alignment: effective_alignment,
        }
    }
    
    /// Allocate aligned memory for f64 array
    /// 
    /// # Safety
    /// Caller must ensure proper deallocation using aligned_dealloc
    pub unsafe fn alloc_f64_array(&self, len: usize) -> *mut f64 {
        let byte_size = len * std::mem::size_of::<f64>();
        
        // Create layout with proper alignment
        let layout = Layout::from_size_align(byte_size, self.alignment)
            .expect("Invalid layout parameters");
        
        let ptr = alloc::alloc(layout);
        
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        
        ptr as *mut f64
    }
    
    /// Deallocate previously allocated aligned memory
    /// 
    /// # Safety
    /// Must only be called on pointers allocated by this allocator
    pub unsafe fn dealloc_f64_array(&self, ptr: *mut f64, len: usize) {
        let byte_size = len * std::mem::size_of::<f64>();
        let layout = Layout::from_size_align(byte_size, self.alignment)
            .expect("Invalid layout parameters");
        
        alloc::dealloc(ptr as *mut u8, layout);
    }
    
    /// Allocate and initialize aligned memory for f64 array
    pub fn alloc_init_f64_array(&self, len: usize, init_val: f64) -> AlignedF64Vec {
        unsafe {
            let ptr = self.alloc_f64_array(len);
            
            // Initialize all values
            for i in 0..len {
                ptr.add(i).write(init_val);
            }
            
            AlignedF64Vec {
                ptr,
                len,
                allocator: self.alignment,
            }
        }
    }
}

/// RAII wrapper for aligned f64 arrays
pub struct AlignedF64Vec {
    ptr: *mut f64,
    len: usize,
    allocator: usize,
}

impl AlignedF64Vec {
    /// Create a new aligned vector with specified length
    pub fn new(len: usize) -> Self {
        let allocator = AlignedAllocator::new(AVX2_ALIGNMENT);
        allocator.alloc_init_f64_array(len, 0.0)
    }
    
    /// Get immutable slice reference
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        unsafe {
            slice::from_raw_parts(self.ptr, self.len)
        }
    }
    
    /// Get mutable slice reference
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr, self.len)
        }
    }
    
    /// Get raw pointer for FFI/SIMD operations
    #[inline]
    pub fn as_ptr(&self) -> *const f64 {
        self.ptr
    }
    
    /// Get mutable raw pointer for SIMD operations
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.ptr
    }
    
    /// Get length of the vector
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Verify alignment is correct
    #[inline]
    pub fn is_properly_aligned(&self) -> bool {
        (self.ptr as usize) % self.allocator == 0
    }
}

impl Drop for AlignedF64Vec {
    fn drop(&mut self) {
        unsafe {
            let allocator = AlignedAllocator::new(self.allocator);
            allocator.dealloc_f64_array(self.ptr, self.len);
        }
    }
}

/// Padding utility for ensuring proper alignment of existing data
pub struct PaddingCalculator {
    alignment: usize,
}

impl PaddingCalculator {
    pub const fn new() -> Self {
        Self {
            alignment: AVX2_ALIGNMENT,
        }
    }
    
    /// Calculate required padding to reach alignment
    #[inline]
    pub const fn calculate_padding(&self, current_size: usize) -> usize {
        let remainder = current_size % self.alignment;
        if remainder == 0 {
            0
        } else {
            self.alignment - remainder
        }
    }
    
    /// Calculate padded size including original data
    #[inline]
    pub const fn padded_size(&self, original_size: usize) -> usize {
        original_size + self.calculate_padding(original_size)
    }
    
    /// Check if size is already properly aligned
    #[inline]
    pub const fn is_aligned(&self, size: usize) -> bool {
        size % self.alignment == 0
    }
}

impl Default for PaddingCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy view into aligned memory region
/// Useful for creating views into larger aligned buffers
pub struct AlignedView<'a> {
    data: &'a [f64],
    offset: usize,
    stride: usize,
}

impl<'a> AlignedView<'a> {
    /// Create a new aligned view
    /// 
    /// # Panics
    /// Panics if the underlying data is not properly aligned
    pub fn new(data: &'a [f64]) -> Self {
        assert!(
            (data.as_ptr() as usize) % AVX2_ALIGNMENT == 0,
            "Data must be aligned to {} bytes",
            AVX2_ALIGNMENT
        );
        
        Self {
            data,
            offset: 0,
            stride: 1,
        }
    }
    
    /// Create a strided view for interleaved data access
    pub fn with_stride(data: &'a [f64], stride: usize) -> Self {
        Self {
            data,
            offset: 0,
            stride,
        }
    }
    
    /// Get element at index with stride applied
    #[inline]
    pub fn get(&self, index: usize) -> Option<f64> {
        let actual_index = self.offset + index * self.stride;
        if actual_index < self.data.len() {
            Some(self.data[actual_index])
        } else {
            None
        }
    }
    
    /// Get underlying slice
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }
    
    /// Verify the view maintains alignment
    #[inline]
    pub fn check_alignment(&self) -> bool {
        (self.data.as_ptr() as usize) % AVX2_ALIGNMENT == 0
    }
}

/// Prefetch helper for optimizing cache utilization
#[inline]
pub fn prefetch_read<T>(ptr: *const T, offset: usize) {
    unsafe {
        // Use LLVM's prefetch intrinsic via inline assembly
        // This hints to the CPU to load data into cache before it's needed
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_T0;
            
            let addr = ptr.add(offset) as *const i8;
            _mm_prefetch(addr, _MM_HINT_T0);
        }
    }
}

/// Prefetch for write operations
#[inline]
pub fn prefetch_write<T>(ptr: *mut T, offset: usize) {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_ET0;
            
            let addr = ptr.add(offset) as *const i8;
            _mm_prefetch(addr, _MM_HINT_ET0);
        }
    }
}

/// Utility function to align a pointer to 32-byte boundary
/// Returns the aligned pointer and the number of elements skipped
#[inline]
pub fn align_pointer<T>(ptr: *mut T, count: usize) -> (*mut T, usize) {
    let addr = ptr as usize;
    let element_size = std::mem::size_of::<T>();
    
    let misalignment = addr % AVX2_ALIGNMENT;
    if misalignment == 0 {
        return (ptr, 0);
    }
    
    let bytes_to_skip = AVX2_ALIGNMENT - misalignment;
    let elements_to_skip = (bytes_to_skip + element_size - 1) / element_size;
    
    if elements_to_skip >= count {
        // Cannot align within available data
        (ptr, 0)
    } else {
        unsafe {
            (ptr.add(elements_to_skip), elements_to_skip)
        }
    }
}

/// Stack-allocated aligned buffer for small fixed-size arrays
/// Avoids heap allocation entirely for performance-critical paths
#[macro_export]
macro_rules! aligned_stack_array {
    ($t:ty, $size:expr) => {{
        const SIZE: usize = $size;
        const PADDED: usize = ((${SIZE} * std::mem::size_of::<$t>() + AVX2_ALIGNMENT - 1) 
                               / AVX2_ALIGNMENT) * AVX2_ALIGNMENT;
        
        static mut BUFFER: [u8; PADDED] = [0; PADDED];
        
        unsafe {
            let ptr = BUFFER.as_mut_ptr();
            let aligned_ptr = align_pointer(ptr as *mut $t, SIZE).0;
            slice::from_raw_parts_mut(aligned_ptr, SIZE)
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_padding_calculator() {
        let calc = PaddingCalculator::new();
        
        // Already aligned
        assert_eq!(calc.calculate_padding(32), 0);
        assert_eq!(calc.calculate_padding(64), 0);
        
        // Needs padding
        assert_eq!(calc.calculate_padding(1), 31);
        assert_eq!(calc.calculate_padding(33), 31);
        assert_eq!(calc.calculate_padding(10), 22);
    }
    
    #[test]
    fn test_aligned_allocation() {
        let allocator = AlignedAllocator::new(AVX2_ALIGNMENT);
        
        unsafe {
            let ptr = allocator.alloc_f64_array(100);
            assert!(!ptr.is_null());
            assert_eq!((ptr as usize) % AVX2_ALIGNMENT, 0);
            
            // Write some values
            for i in 0..100 {
                ptr.add(i).write(i as f64);
            }
            
            // Read back
            for i in 0..100 {
                assert_eq!(ptr.add(i).read(), i as f64);
            }
            
            allocator.dealloc_f64_array(ptr, 100);
        }
    }
    
    #[test]
    fn test_aligned_vec() {
        let vec = AlignedF64Vec::new(50);
        assert!(vec.is_properly_aligned());
        assert_eq!(vec.len(), 50);
        assert!(!vec.is_empty());
        
        // Test mutation
        let slice = vec.as_mut_slice();
        slice[0] = 42.0;
        assert_eq!(vec.as_slice()[0], 42.0);
    }
    
    #[test]
    fn test_aligned_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let view = AlignedView::new(&data);
        
        assert_eq!(view.get(0), Some(1.0));
        assert_eq!(view.get(2), Some(3.0));
        assert_eq!(view.get(10), None);
    }
    
    #[test]
    fn test_pointer_alignment() {
        let mut buffer = vec![0.0f64; 100];
        let ptr = buffer.as_mut_ptr();
        
        let (aligned_ptr, skip) = align_pointer(ptr, 100);
        
        if skip > 0 {
            assert!((aligned_ptr as usize) % AVX2_ALIGNMENT == 0);
        }
    }
}
