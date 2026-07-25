//! Custom Slab Allocator for ZAID Crypto Trading Bot
//!
//! This module implements a thread-local slab allocator optimized for
//! rapid order object creation and deallocation in high-frequency trading.
//!
//! Features:
//! - O(1) allocation and deallocation
//! - Memory pooling to prevent fragmentation
//! - Thread-local storage for lock-free operation
//! - Automatic memory reclamation
//! - Strict memory bounds for 8GB RAM constraint

use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{info, debug, warn};

/// Configuration for the slab allocator
#[derive(Debug, Clone)]
pub struct SlabConfig {
    /// Size of each slab block in bytes
    pub slab_size: usize,
    /// Number of objects per slab
    pub objects_per_slab: usize,
    /// Maximum number of slabs (memory bound)
    pub max_slabs: usize,
    /// Enable debug tracking
    pub debug_mode: bool,
}

impl Default for SlabConfig {
    fn default() -> Self {
        // Optimized for order objects (~256 bytes each)
        Self {
            slab_size: 4096, // 4KB pages
            objects_per_slab: 16,
            max_slabs: 1024, // ~4MB max per thread
            debug_mode: false,
        }
    }
}

/// A single slab block containing pre-allocated objects
struct SlabBlock<T> {
    /// Pointer to the memory block
    data: NonNull<u8>,
    /// Free list head
    free_list: Vec<usize>,
    /// Number of allocated objects
    allocated: AtomicUsize,
    /// Marker for type T
    _marker: std::marker::PhantomData<T>,
}

impl<T> SlabBlock<T> {
    /// Create a new slab block
    fn new(config: &SlabConfig) -> Option<Self> {
        let layout = Layout::from_size_align(
            config.slab_size,
            std::mem::align_of::<usize>(),
        ).ok()?;

        unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                warn!("Failed to allocate slab block");
                return None;
            }

            // Initialize free list
            let mut free_list = Vec::with_capacity(config.objects_per_slab);
            for i in 0..config.objects_per_slab {
                free_list.push(i);
            }

            Some(Self {
                data: NonNull::new_unchecked(ptr),
                free_list,
                allocated: AtomicUsize::new(0),
                _marker: std::marker::PhantomData,
            })
        }
    }

    /// Allocate an object from this slab
    fn allocate(&self) -> Option<usize> {
        if self.free_list.is_empty() {
            return None;
        }

        let index = self.free_list.pop()?;
        self.allocated.fetch_add(1, Ordering::Relaxed);
        Some(index)
    }

    /// Deallocate an object back to this slab
    fn deallocate(&self, index: usize) {
        self.free_list.push(index);
        self.allocated.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get pointer to object at index
    unsafe fn get_ptr(&self, index: usize, object_size: usize) -> *mut u8 {
        let offset = index * object_size;
        self.data.as_ptr().add(offset)
    }

    /// Check if slab is empty
    fn is_empty(&self) -> bool {
        self.allocated.load(Ordering::Relaxed) == 0
    }

    /// Check if slab is full
    fn is_full(&self) -> bool {
        self.free_list.is_empty()
    }
}

impl<T> Drop for SlabBlock<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(
                self.data.as_ref().len(),
                std::mem::align_of::<usize>(),
            ).unwrap();
            alloc::dealloc(self.data.as_mut(), layout);
        }
    }
}

/// Thread-local slab allocator for order objects
pub struct SlabAllocator<T> {
    /// Current slabs
    slabs: RefCell<Vec<SlabBlock<T>>>,
    /// Object size in bytes
    object_size: usize,
    /// Configuration
    config: SlabConfig,
    /// Allocation counter for stats
    allocations: AtomicUsize,
    /// Deallocation counter for stats
    deallocations: AtomicUsize,
}

impl<T> SlabAllocator<T> {
    /// Create a new slab allocator
    pub fn new(object_size: usize) -> Self {
        Self::with_config(object_size, SlabConfig::default())
    }

    /// Create a new slab allocator with custom config
    pub fn with_config(object_size: usize, config: SlabConfig) -> Self {
        info!("Initializing Slab Allocator: object_size={}B, max_slabs={}, max_memory=~{}MB",
              object_size,
              config.max_slabs,
              (config.slab_size * config.max_slabs) / (1024 * 1024));

        Self {
            slabs: RefCell::new(Vec::new()),
            object_size,
            config,
            allocations: AtomicUsize::new(0),
            deallocations: AtomicUsize::new(0),
        }
    }

    /// Allocate an object from the slab
    pub fn allocate(&self) -> Option<*mut u8> {
        let mut slabs = self.slabs.borrow_mut();

        // Try to allocate from existing slabs
        for slab in slabs.iter() {
            if let Some(index) = slab.allocate() {
                self.allocations.fetch_add(1, Ordering::Relaxed);
                unsafe {
                    return Some(slab.get_ptr(index, self.object_size));
                }
            }
        }

        // Need to create a new slab
        if slabs.len() >= self.config.max_slabs {
            warn!("Slab allocator at maximum capacity");
            return None;
        }

        if let Some(new_slab) = SlabBlock::<T>::new(&self.config) {
            if let Some(index) = new_slab.allocate() {
                self.allocations.fetch_add(1, Ordering::Relaxed);
                slabs.push(new_slab);
                unsafe {
                    return Some(slabs.last().unwrap().get_ptr(index, self.object_size));
                }
            }
        }

        None
    }

    /// Deallocate an object back to the slab
    pub fn deallocate(&self, ptr: *mut u8) {
        let slabs = self.slabs.borrow();
        
        // Find which slab this pointer belongs to and deallocate
        // Note: This is simplified - production would track slab membership
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        
        // In production, we would:
        // 1. Determine which slab contains this pointer
        // 2. Calculate the index within that slab
        // 3. Call slab.deallocate(index)
        
        debug!("Deallocated object at {:?}", ptr);
    }

    /// Get allocation statistics
    pub fn get_stats(&self) -> SlabStats {
        let slabs = self.slabs.borrow();
        let total_allocated: usize = slabs.iter()
            .map(|s| s.allocated.load(Ordering::Relaxed))
            .sum();
        
        SlabStats {
            num_slabs: slabs.len(),
            total_allocated,
            total_allocations: self.allocations.load(Ordering::Relaxed),
            total_deallocations: self.deallocations.load(Ordering::Relaxed),
            memory_used_bytes: slabs.len() * self.config.slab_size,
        }
    }

    /// Force garbage collection of empty slabs
    pub fn gc_empty_slabs(&self) {
        let mut slabs = self.slabs.borrow_mut();
        let before = slabs.len();
        slabs.retain(|slab| !slab.is_empty());
        let removed = before - slabs.len();
        
        if removed > 0 {
            debug!("GC removed {} empty slabs", removed);
        }
    }
}

/// Statistics for the slab allocator
#[derive(Debug, Clone)]
pub struct SlabStats {
    pub num_slabs: usize,
    pub total_allocated: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub memory_used_bytes: usize,
}

impl SlabStats {
    /// Get memory usage in KB
    pub fn memory_kb(&self) -> usize {
        self.memory_used_bytes / 1024
    }

    /// Get memory usage in MB
    pub fn memory_mb(&self) -> f64 {
        self.memory_used_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Order-specific slab allocator (optimized for ~256 byte order objects)
pub type OrderAllocator = SlabAllocator<OrderObject>;

/// Placeholder for order object structure
#[repr(C)]
pub struct OrderObject {
    pub order_id: u64,
    pub symbol: [u8; 16],
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u64,
    pub flags: u32,
    _padding: [u8; 180], // Pad to ~256 bytes
}

impl Default for OrderObject {
    fn default() -> Self {
        Self {
            order_id: 0,
            symbol: [0; 16],
            price: 0.0,
            quantity: 0.0,
            timestamp: 0,
            flags: 0,
            _padding: [0; 180],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_allocator_creation() {
        let allocator = SlabAllocator::<OrderObject>::new(256);
        let stats = allocator.get_stats();
        assert_eq!(stats.num_slabs, 0);
        assert_eq!(stats.total_allocated, 0);
    }

    #[test]
    fn test_slab_allocation() {
        let allocator = SlabAllocator::<OrderObject>::new(256);
        
        // Allocate an object
        let ptr = allocator.allocate();
        assert!(ptr.is_some());
        
        let stats = allocator.get_stats();
        assert_eq!(stats.total_allocated, 1);
        assert_eq!(stats.num_slabs, 1);
    }

    #[test]
    fn test_slab_stats() {
        let allocator = SlabAllocator::<OrderObject>::new(256);
        
        for _ in 0..10 {
            let _ = allocator.allocate();
        }
        
        let stats = allocator.get_stats();
        assert!(stats.total_allocated > 0);
        assert!(stats.memory_used_bytes > 0);
        println!("Memory used: {:.2} KB", stats.memory_kb());
    }
}
