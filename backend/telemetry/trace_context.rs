//! Trace Context Propagation for Distributed Tracing
//! 
//! This module implements a zero-cost abstraction for propagating trace IDs
//! across Rust async boundaries and FFI crossings with Python.
//! 
//! Key Features:
//! - O(1) trace ID generation using atomic counters
//! - Thread-local storage for context propagation
//! - FFI-safe structures for Python interoperability
//! - Zero-allocation context cloning
//! 
//! Designed for the ZAID Personal Crypto Trading Bot to maintain
//! microsecond observability across the entire execution pipeline.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Global atomic counter for generating unique trace IDs
static TRACE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Thread-local storage for the current trace context
thread_local! {
    static CURRENT_CONTEXT: RefCell<Option<TraceContext>> = RefCell::new(None);
}

/// Represents a unique trace identifier that survives FFI boundaries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)] // Ensures C-compatible memory layout for FFI
pub struct TraceId(u64);

impl TraceId {
    /// Generate a new unique trace ID in O(1) time
    #[inline]
    pub fn new() -> Self {
        let id = TRACE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        TraceId(id)
    }

    /// Create a trace ID from an existing u64 (useful for FFI)
    #[inline]
    pub fn from_u64(id: u64) -> Self {
        TraceId(id)
    }

    /// Get the underlying u64 value
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Convert to a hex string for logging
    #[inline]
    pub fn to_hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

impl Default for TraceId {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a span within a trace
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SpanId(u64);

impl SpanId {
    #[inline]
    pub fn new() -> Self {
        let id = TRACE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        SpanId(id)
    }

    #[inline]
    pub fn from_u64(id: u64) -> Self {
        SpanId(id)
    }

    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl Default for SpanId {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Trace context containing all necessary information for distributed tracing
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TraceContext {
    trace_id: TraceId,
    parent_span_id: Option<SpanId>,
    current_span_id: SpanId,
    start_time: Instant,
    flags: u8, // Bit flags for sampling, debug mode, etc.
}

impl TraceContext {
    /// Create a new root trace context
    #[inline]
    pub fn root() -> Self {
        let trace_id = TraceId::new();
        let span_id = SpanId::new();
        
        TraceContext {
            trace_id,
            parent_span_id: None,
            current_span_id: span_id,
            start_time: Instant::now(),
            flags: 0,
        }
    }

    /// Create a child span from the current context
    #[inline]
    pub fn child(&self) -> Self {
        TraceContext {
            trace_id: self.trace_id,
            parent_span_id: Some(self.current_span_id),
            current_span_id: SpanId::new(),
            start_time: Instant::now(),
            flags: self.flags,
        }
    }

    /// Get the trace ID
    #[inline]
    pub fn trace_id(&self) -> TraceId {
        self.trace_id
    }

    /// Get the current span ID
    #[inline]
    pub fn span_id(&self) -> SpanId {
        self.current_span_id
    }

    /// Get the elapsed time since the span started
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get elapsed time in microseconds (optimized for telemetry)
    #[inline]
    pub fn elapsed_micros(&self) -> u64 {
        self.elapsed().as_micros() as u64
    }

    /// Set a flag (e.g., sampling enabled)
    #[inline]
    pub fn set_flag(&mut self, flag: u8) {
        self.flags |= flag;
    }

    /// Check if a flag is set
    #[inline]
    pub fn has_flag(&self, flag: u8) -> bool {
        (self.flags & flag) != 0
    }

    /// Serialize context to bytes for FFI (zero-copy compatible)
    #[inline]
    pub fn to_bytes(&self) -> [u8; 33] {
        let mut bytes = [0u8; 33];
        bytes[0..8].copy_from_slice(&self.trace_id.as_u64().to_le_bytes());
        bytes[8..16].copy_from_slice(&self.current_span_id.as_u64().to_le_bytes());
        if let Some(parent) = self.parent_span_id {
            bytes[16] = 1; // Indicator that parent exists
            bytes[17..25].copy_from_slice(&parent.as_u64().to_le_bytes());
        } else {
            bytes[16] = 0;
        }
        bytes[25..33].copy_from_slice(&(self.elapsed_micros() as u64).to_le_bytes());
        bytes[32] = self.flags;
        bytes
    }

    /// Deserialize context from bytes (FFI safe)
    #[inline]
    pub fn from_bytes(bytes: &[u8; 33]) -> Option<Self> {
        if bytes.len() < 33 {
            return None;
        }

        let trace_id = TraceId::from_u64(u64::from_le_bytes(bytes[0..8].try_into().ok()?));
        let current_span_id = SpanId::from_u64(u64::from_le_bytes(bytes[8..16].try_into().ok()?));
        
        let parent_span_id = if bytes[16] == 1 {
            Some(SpanId::from_u64(u64::from_le_bytes(bytes[17..25].try_into().ok()?)))
        } else {
            None
        };

        let elapsed_micros = u64::from_le_bytes(bytes[25..33].try_into().ok()?);
        let flags = bytes[32];

        // Note: We can't reconstruct the original Instant, so we use now()
        // The elapsed time is preserved for reference
        Some(TraceContext {
            trace_id,
            parent_span_id,
            current_span_id,
            start_time: Instant::now() - Duration::from_micros(elapsed_micros),
            flags,
        })
    }
}

/// Manages the current thread's trace context
pub struct TraceContextManager;

impl TraceContextManager {
    /// Initialize a new root context for this thread
    #[inline]
    pub fn init_root() -> TraceContext {
        let ctx = TraceContext::root();
        CURRENT_CONTEXT.with(|cell| {
            *cell.borrow_mut() = Some(ctx);
        });
        ctx
    }

    /// Get the current context, or create a root if none exists
    #[inline]
    pub fn get_or_init() -> TraceContext {
        CURRENT_CONTEXT.with(|cell| {
            if let Some(ctx) = *cell.borrow() {
                ctx
            } else {
                let ctx = TraceContext::root();
                *cell.borrow_mut() = Some(ctx);
                ctx
            }
        })
    }

    /// Enter a child span, returning the new context
    #[inline]
    pub fn enter_child() -> TraceContext {
        let parent = Self::get_or_init();
        let child = parent.child();
        CURRENT_CONTEXT.with(|cell| {
            *cell.borrow_mut() = Some(child);
        });
        child
    }

    /// Set the current context explicitly (useful for FFI)
    #[inline]
    pub fn set_context(ctx: TraceContext) {
        CURRENT_CONTEXT.with(|cell| {
            *cell.borrow_mut() = Some(ctx);
        });
    }

    /// Take the current context, leaving None
    #[inline]
    pub fn take() -> Option<TraceContext> {
        CURRENT_CONTEXT.with(|cell| cell.borrow_mut().take())
    }

    /// Get elapsed time of the current span in microseconds
    #[inline]
    pub fn current_elapsed_micros() -> u64 {
        Self::get_or_init().elapsed_micros()
    }
}

/// RAII guard for automatically managing span lifecycle
pub struct SpanGuard {
    context: TraceContext,
    name: &'static str,
}

impl SpanGuard {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        let context = TraceContextManager::enter_child();
        SpanGuard { context, name }
    }

    #[inline]
    pub fn context(&self) -> TraceContext {
        self.context
    }

    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }
}

impl Drop for SpanGuard {
    #[inline]
    fn drop(&mut self) {
        // Automatically restore parent context when span ends
        if let Some(parent_id) = self.context.parent_span_id {
            // In a full implementation, we'd restore the parent context here
            // For now, we just log the span completion
            let elapsed = self.context.elapsed_micros();
            if elapsed > 100 { // Log spans taking > 100μs
                eprintln!(
                    "SPAN_COMPLETE: name={} trace_id={} span_id={} elapsed_us={}",
                    self.name,
                    self.context.trace_id().to_hex(),
                    self.context.span_id().as_u64(),
                    elapsed
                );
            }
        }
    }
}

/// FFI exports for Python integration
#[cfg(feature = "ffi")]
pub mod ffi {
    use super::*;
    use std::os::raw::c_char;
    use std::ffi::CStr;

    /// Create a new trace context and return it as bytes
    #[no_mangle]
    pub extern "C" fn ffi_trace_context_create() -> [u8; 33] {
        let ctx = TraceContext::root();
        ctx.to_bytes()
    }

    /// Create a child context from parent bytes
    #[no_mangle]
    pub extern "C" fn ffi_trace_context_child(parent_bytes: &[u8; 33]) -> [u8; 33] {
        if let Some(parent) = TraceContext::from_bytes(parent_bytes) {
            let child = parent.child();
            child.to_bytes()
        } else {
            TraceContext::root().to_bytes()
        }
    }

    /// Get elapsed time in microseconds from context bytes
    #[no_mangle]
    pub extern "C" fn ffi_trace_context_elapsed_micros(ctx_bytes: &[u8; 33]) -> u64 {
        if let Some(ctx) = TraceContext::from_bytes(ctx_bytes) {
            ctx.elapsed_micros()
        } else {
            0
        }
    }

    /// Extract trace ID as hex string (caller must free)
    #[no_mangle]
    pub extern "C" fn ffi_trace_context_get_trace_id_hex(ctx_bytes: &[u8; 33]) -> *mut c_char {
        if let Some(ctx) = TraceContext::from_bytes(ctx_bytes) {
            let hex = ctx.trace_id().to_hex();
            let c_str = std::ffi::CString::new(hex).unwrap();
            c_str.into_raw()
        } else {
            std::ffi::CString::new("0000000000000000").unwrap().into_raw()
        }
    }

    /// Free a C string allocated by this library
    #[no_mangle]
    pub extern "C" fn ffi_free_string(ptr: *mut c_char) {
        unsafe {
            if !ptr.is_null() {
                let _ = std::ffi::CString::from_raw(ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_id_generation() {
        let id1 = TraceId::new();
        let id2 = TraceId::new();
        assert_ne!(id1.as_u64(), id2.as_u64());
    }

    #[test]
    fn test_context_serialization() {
        let ctx = TraceContext::root();
        let bytes = ctx.to_bytes();
        let restored = TraceContext::from_bytes(&bytes).unwrap();
        
        assert_eq!(ctx.trace_id(), restored.trace_id());
        assert_eq!(ctx.span_id(), restored.span_id());
    }

    #[test]
    fn test_span_guard() {
        let _guard = SpanGuard::new("test_operation");
        // Span automatically logged on drop
    }

    #[test]
    fn test_child_context() {
        let root = TraceContext::root();
        let child = root.child();
        
        assert_eq!(root.trace_id(), child.trace_id());
        assert_ne!(root.span_id(), child.span_id());
        assert_eq!(child.parent_span_id, Some(root.span_id()));
    }
}
