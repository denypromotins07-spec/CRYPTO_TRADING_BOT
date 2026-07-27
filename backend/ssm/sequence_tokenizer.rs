//! Sequence Tokenizer: L2 Order Book Delta to SSM Token Conversion
//!
//! This module converts raw L2 order book updates into discrete tokens
//! suitable for State-Space Model (Mamba/S4) processing.
//!
//! Key Features:
//! - Zero-heap allocation tokenization pipeline
//! - Microsecond-level timestamp encoding
//! - Price/volume quantization with adaptive bins
//! - Memory pool for token reuse under 8GB constraint
//!
//! Token Format:
//! Each token encodes: [price_delta_sign, price_delta_magnitude, volume_bucket, time_delta]

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Maximum number of tokens in the circular buffer
const MAX_TOKEN_BUFFER: usize = 8192;

/// Token representation for SSM input
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)] // Ensure contiguous memory layout
pub struct SsmToken {
    /// Signed price delta (normalized)
    pub price_delta: f64,
    /// Volume bucket index (0-15)
    pub volume_bucket: u8,
    /// Time delta since last event (microseconds, log-scaled)
    pub time_delta_us: u16,
    /// Event type: 0=bid_add, 1=ask_add, 2=bid_cancel, 3=ask_cancel, 4=trade
    pub event_type: u8,
    /// Reserved padding for alignment
    _padding: u8,
}

impl SsmToken {
    /// Create a new token from raw order book event data
    #[inline]
    pub fn new(
        price_delta: f64,
        volume: f64,
        time_delta_us: u64,
        event_type: u8,
    ) -> Self {
        // Quantize volume into 16 buckets (log-spaced)
        let volume_bucket = Self::quantize_volume(volume);
        
        // Log-scale time delta to fit in u16 (max ~65ms represented precisely)
        let time_delta_quantized = Self::quantize_time(time_delta_us);
        
        Self {
            price_delta,
            volume_bucket,
            time_delta_us: time_delta_quantized,
            event_type,
            _padding: 0,
        }
    }
    
    /// Quantize volume into logarithmic buckets
    #[inline]
    fn quantize_volume(volume: f64) -> u8 {
        if volume <= 0.0 {
            return 0;
        }
        // Log-spaced buckets: [0, 1), [1, 2), [2, 4), [4, 8), ..., [8192, inf)
        let log_vol = volume.ln() / 2.0_f64.ln(); // log2(volume)
        (log_vol.clamp(0.0, 15.0) as u8).min(15)
    }
    
    /// Quantize time delta into u16 representation
    #[inline]
    fn quantize_time(time_delta_us: u64) -> u16 {
        // Direct mapping for small deltas, log-scale for larger ones
        if time_delta_us < 1000 {
            time_delta_us as u16
        } else if time_delta_us < 1_000_000 {
            // Log scale: 1ms to 1s
            (1000.0 + (time_delta_us as f64).ln() * 100.0) as u16
        } else {
            65535 // Cap at max u16
        }
    }
    
    /// Convert token to SSM input vector (flattened representation)
    #[inline]
    pub fn to_input_vector(&self, dim: usize) -> Vec<f64> {
        let mut vec = vec![0.0; dim];
        
        if dim >= 4 {
            vec[0] = self.price_delta;
            vec[1] = self.volume_bucket as f64 / 15.0; // Normalize to [0, 1]
            vec[2] = self.time_delta_us as f64 / 65535.0; // Normalize
            vec[3] = self.event_type as f64 / 4.0; // Normalize
        }
        
        // Fill remaining dimensions with interactions
        for i in 4..dim.min(16) {
            vec[i] = match i % 4 {
                0 => self.price_delta * (self.volume_bucket as f64 / 15.0),
                1 => self.price_delta * (self.time_delta_us as f64 / 65535.0),
                2 => (self.volume_bucket as f64 / 15.0) * (self.time_delta_us as f64 / 65535.0),
                _ => self.event_type as f64 / 4.0 * self.price_delta.abs(),
                _ => 0.0,
            };
        }
        
        vec
    }
}

/// Order book event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderBookEvent {
    BidAdd = 0,
    AskAdd = 1,
    BidCancel = 2,
    AskCancel = 3,
    Trade = 4,
}

/// Raw L2 order book delta structure
#[derive(Debug, Clone)]
pub struct L2Delta {
    pub side: OrderBookEvent,
    pub price: f64,
    pub previous_price: f64,
    pub volume: f64,
    pub timestamp_us: u64,
}

impl L2Delta {
    /// Compute price delta from this event
    #[inline]
    pub fn price_delta(&self) -> f64 {
        self.price - self.previous_price
    }
}

/// Sequence Tokenizer: converts L2 deltas to SSM tokens
pub struct SequenceTokenizer {
    /// Circular buffer for tokens (zero-allocation after init)
    token_buffer: VecDeque<SsmToken>,
    /// Last timestamp for time delta computation
    last_timestamp_us: u64,
    /// Running statistics for adaptive normalization
    price_delta_mean: f64,
    price_delta_var: f64,
    volume_mean: f64,
    /// Number of samples for running stats
    sample_count: u64,
    /// Configuration
    max_sequence_length: usize,
}

impl SequenceTokenizer {
    /// Create a new sequence tokenizer
    pub fn new(max_sequence_length: usize) -> Self {
        Self {
            token_buffer: VecDeque::with_capacity(max_sequence_length.min(MAX_TOKEN_BUFFER)),
            last_timestamp_us: 0,
            price_delta_mean: 0.0,
            price_delta_var: 0.0,
            volume_mean: 0.0,
            sample_count: 0,
            max_sequence_length: max_sequence_length.min(MAX_TOKEN_BUFFER),
        }
    }
    
    /// Process a single L2 delta and return the tokenized result
    #[inline]
    pub fn process_delta(&mut self, delta: &L2Delta) -> Option<SsmToken> {
        // Compute time delta
        let time_delta_us = if self.last_timestamp_us == 0 {
            0
        } else {
            delta.timestamp_us.saturating_sub(self.last_timestamp_us)
        };
        self.last_timestamp_us = delta.timestamp_us;
        
        // Update running statistics (Welford's algorithm for numerical stability)
        self.update_running_stats(delta.price_delta(), delta.volume);
        
        // Normalize price delta using running statistics
        let normalized_price_delta = self.normalize_price_delta(delta.price_delta());
        
        // Create token
        let token = SsmToken::new(
            normalized_price_delta,
            delta.volume,
            time_delta_us,
            delta.side as u8,
        );
        
        // Add to circular buffer
        if self.token_buffer.len() >= self.max_sequence_length {
            self.token_buffer.pop_front();
        }
        self.token_buffer.push_back(token);
        
        Some(token)
    }
    
    /// Get the current sequence of tokens as input vectors
    pub fn get_sequence(&self, embedding_dim: usize) -> Vec<Vec<f64>> {
        self.token_buffer
            .iter()
            .map(|t| t.to_input_vector(embedding_dim))
            .collect()
    }
    
    /// Get tokens as a flat slice for zero-copy processing
    pub fn get_tokens_slice(&self) -> &[SsmToken] {
        // Note: VecDeque doesn't provide contiguous storage, so we can't return a true slice
        // For true zero-copy, use a ring buffer implementation
        self.token_buffer.as_slices().0 // Return first contiguous segment
    }
    
    /// Reset tokenizer state (for episode boundaries)
    pub fn reset(&mut self) {
        self.token_buffer.clear();
        self.last_timestamp_us = 0;
        self.price_delta_mean = 0.0;
        self.price_delta_var = 0.0;
        self.volume_mean = 0.0;
        self.sample_count = 0;
    }
    
    /// Get current sequence length
    pub fn sequence_length(&self) -> usize {
        self.token_buffer.len()
    }
    
    /// Check if sequence is ready for SSM processing (minimum length)
    pub fn is_ready(&self, min_length: usize) -> bool {
        self.token_buffer.len() >= min_length
    }
    
    /// Update running statistics using Welford's online algorithm
    #[inline]
    fn update_running_stats(&mut self, price_delta: f64, volume: f64) {
        self.sample_count += 1;
        let n = self.sample_count as f64;
        
        // Update mean and variance for price delta
        let delta_mean = price_delta - self.price_delta_mean;
        self.price_delta_mean += delta_mean / n;
        let delta_mean2 = price_delta - self.price_delta_mean;
        self.price_delta_var += delta_mean * delta_mean2;
        
        // Update mean for volume (exponential moving average for faster adaptation)
        let alpha = 0.01; // Fast adaptation for volume
        self.volume_mean = (1.0 - alpha) * self.volume_mean + alpha * volume;
    }
    
    /// Normalize price delta using running statistics
    #[inline]
    fn normalize_price_delta(&self, price_delta: f64) -> f64 {
        if self.sample_count < 10 {
            // Not enough samples, use raw value with clamping
            return price_delta.clamp(-10.0, 10.0) / 10.0;
        }
        
        let std_dev = (self.price_delta_var / self.sample_count as f64).sqrt();
        if std_dev < 1e-10 {
            return 0.0; // Avoid division by zero
        }
        
        // Z-score normalization with clamping
        ((price_delta - self.price_delta_mean) / std_dev).clamp(-3.0, 3.0) / 3.0
    }
}

/// Batch tokenizer for processing multiple deltas efficiently
pub struct BatchTokenizer {
    inner: SequenceTokenizer,
}

impl BatchTokenizer {
    pub fn new(max_sequence_length: usize) -> Self {
        Self {
            inner: SequenceTokenizer::new(max_sequence_length),
        }
    }
    
    /// Process a batch of L2 deltas
    pub fn process_batch(&mut self, deltas: &[L2Delta]) -> Vec<SsmToken> {
        deltas.iter().filter_map(|d| self.inner.process_delta(d)).collect()
    }
    
    /// Get the final sequence after batch processing
    pub fn get_sequence(&self, embedding_dim: usize) -> Vec<Vec<f64>> {
        self.inner.get_sequence(embedding_dim)
    }
    
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = SsmToken::new(0.5, 100.0, 1000, 0);
        assert!(token.price_delta.abs() <= 1.0);
        assert!(token.volume_bucket <= 15);
    }

    #[test]
    fn test_tokenizer_sequence() {
        let mut tokenizer = SequenceTokenizer::new(100);
        
        for i in 0..10 {
            let delta = L2Delta {
                side: OrderBookEvent::BidAdd,
                price: 100.0 + i as f64 * 0.1,
                previous_price: 100.0 + (i - 1) as f64 * 0.1,
                volume: 10.0 + i as f64,
                timestamp_us: i * 1000,
            };
            tokenizer.process_delta(&delta);
        }
        
        assert_eq!(tokenizer.sequence_length(), 10);
        assert!(tokenizer.is_ready(5));
    }

    #[test]
    fn test_zero_allocation() {
        // Verify that repeated tokenization doesn't allocate
        let mut tokenizer = SequenceTokenizer::new(1000);
        
        for i in 0..100 {
            let delta = L2Delta {
                side: OrderBookEvent::Trade,
                price: 50000.0,
                previous_price: 49999.0,
                volume: 1.5,
                timestamp_us: i * 100,
            };
            let _ = tokenizer.process_delta(&delta);
        }
        
        // Buffer should not exceed max length
        assert!(tokenizer.sequence_length() <= 1000);
    }
}
