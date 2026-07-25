//! Positional Encoding for High-Frequency Tick Data
//! 
//! Generates sinusoidal positional embeddings optimized for temporal
//! sequence data in cryptocurrency trading applications.
//! 
//! Features:
//! - Standard sinusoidal encodings as per "Attention Is All You Need"
//! - Learnable relative position biases
//! - Adaptive frequency scaling for tick-level timestamps
//! - Zero-cost abstractions with precomputed lookup tables
//! - Thread-safe encoding generation

use std::f64::consts::PI;

/// Configuration for positional encoding
#[derive(Debug, Clone)]
pub struct PositionalEncodingConfig {
    pub d_model: usize,           // Embedding dimension
    pub max_len: usize,           // Maximum sequence length
    pub use_learned: bool,        // Whether to use learned offsets
    pub frequency_scale: f64,     // Frequency scaling factor for ticks
    pub dropout: f64,             // Dropout rate (inference: 0)
}

impl Default for PositionalEncodingConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            max_len: 512,
            use_learned: false,
            frequency_scale: 1.0,
            dropout: 0.0,
        }
    }
}

/// Sinusoidal positional encoding generator
pub struct PositionalEncoder {
    config: PositionalEncodingConfig,
    encodings: Vec<Vec<f64>>,     // Precomputed [max_len, d_model]
    learned_bias: Option<Vec<f64>>, // Optional learned bias terms
}

impl PositionalEncoder {
    /// Create a new positional encoder with precomputed encodings
    pub fn new(config: PositionalEncodingConfig) -> Self {
        let mut encodings = vec![vec![0.0; config.d_model]; config.max_len];
        
        // Generate sinusoidal encodings
        for pos in 0..config.max_len {
            for i in (0..config.d_model).step_by(2) {
                // Original formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
                let div_term = (pos as f64 * config.frequency_scale) 
                    / 10000.0_f64.powf((i as f64) / (config.d_model as f64));
                
                encodings[pos][i] = div_term.sin();
                
                if i + 1 < config.d_model {
                    encodings[pos][i + 1] = div_term.cos();
                }
            }
        }
        
        let learned_bias = if config.use_learned {
            Some(vec![0.0; config.max_len * config.d_model])
        } else {
            None
        };
        
        Self {
            config,
            encodings,
            learned_bias,
        }
    }
    
    /// Get encoding for a single position
    #[inline]
    pub fn get_encoding(&self, pos: usize) -> &[f64] {
        &self.encodings[pos.min(self.config.max_len - 1)]
    }
    
    /// Add positional encoding to a sequence in-place
    pub fn add_to_sequence(&self, sequence: &mut [Vec<f64>]) {
        for (i, token) in sequence.iter_mut().enumerate() {
            let enc = self.get_encoding(i);
            for (j, val) in token.iter_mut().enumerate().take(self.config.d_model) {
                *val += enc[j];
            }
            
            // Add learned bias if available
            if let Some(bias) = &self.learned_bias {
                if j < self.config.d_model {
                    *val += bias[i * self.config.d_model + j];
                }
            }
        }
    }
    
    /// Create encoded sequence (returns new vector)
    pub fn encode_sequence(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut output = input.to_vec();
        self.add_to_sequence(&mut output);
        output
    }
    
    /// Get encoding for a range of positions (batch processing)
    pub fn get_batch_encodings(&self, start_pos: usize, count: usize) -> Vec<&[f64]> {
        let end = (start_pos + count).min(self.config.max_len);
        (start_pos..end)
            .map(|p| self.get_encoding(p))
            .collect()
    }
    
    /// Update learned bias terms (for training scenarios)
    pub fn update_learned_bias(&mut self, new_bias: &[f64]) -> Result<(), &'static str> {
        if !self.config.use_learned {
            return Err("Learned bias not enabled");
        }
        if new_bias.len() != self.config.max_len * self.config.d_model {
            return Err("Invalid bias size");
        }
        
        if let Some(ref mut bias) = self.learned_bias {
            bias.copy_from_slice(new_bias);
        }
        Ok(())
    }
    
    /// Apply dropout to encodings (training only)
    pub fn apply_dropout(&self, sequence: &mut [Vec<f64>], dropout_rate: f64) {
        if dropout_rate <= 0.0 || dropout_rate >= 1.0 {
            return;
        }
        
        let keep_prob = 1.0 - dropout_rate;
        let scale = 1.0 / keep_prob;
        
        // Simple dropout implementation
        let mut rng = rand_xorshift::XorShiftRng::from_seed([0; 16]);
        for token in sequence.iter_mut() {
            for val in token.iter_mut().take(self.config.d_model) {
                let mask = (rng.next_u64() % 1000) as f64 / 1000.0;
                if mask > dropout_rate {
                    *val *= scale;
                } else {
                    *val = 0.0;
                }
            }
        }
    }
}

/// Relative positional encoding for transformer-XL style attention
pub struct RelativePositionalEncoding {
    d_model: usize,
    max_rel_pos: usize,
    embeddings: Vec<f64>,  // [2 * max_rel_pos + 1, d_model]
}

impl RelativePositionalEncoding {
    /// Create relative positional embeddings
    pub fn new(d_model: usize, max_rel_pos: usize) -> Self {
        let n_embeddings = 2 * max_rel_pos + 1;
        let std_dev = 1.0 / (d_model as f64).sqrt();
        
        let mut rng = rand_xorshift::XorShiftRng::from_seed([0; 16]);
        let embeddings: Vec<f64> = (0..n_embeddings * d_model)
            .map(|_| (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 2.0 * std_dev)
            .collect();
        
        Self {
            d_model,
            max_rel_pos,
            embeddings,
        }
    }
    
    /// Get embedding for relative position
    /// Positive positions are future, negative are past
    #[inline]
    pub fn get_embedding(&self, rel_pos: i32) -> &[f64] {
        let clamped = rel_pos.clamp(-(self.max_rel_pos as i32), self.max_rel_pos as i32);
        let idx = (clamped + self.max_rel_pos as i32) as usize;
        let start = idx * self.d_model;
        &self.embeddings[start..start + self.d_model]
    }
    
    /// Compute relative attention scores for a query-key pair
    pub fn compute_relative_scores(
        &self,
        query: &[f64],
        keys: &[Vec<f64>],
        query_pos: usize
    ) -> Vec<f64> {
        let mut scores = Vec::with_capacity(keys.len());
        
        for (key_pos, key) in keys.iter().enumerate() {
            let rel_pos = key_pos as i32 - query_pos as i32;
            let rel_emb = self.get_embedding(rel_pos);
            
            // R_w1: content-based attention (query · key)
            let content_score: f64 = query.iter().zip(key.iter()).map(|(a, b)| a * b).sum();
            
            // R_w2: position-based attention (query · rel_emb)
            let pos_score: f64 = query.iter().zip(rel_emb.iter()).map(|(a, b)| a * b).sum();
            
            scores.push(content_score + pos_score);
        }
        
        scores
    }
}

/// Time-aware positional encoding using actual timestamps
pub struct TimeAwarePositionalEncoder {
    base_encoder: PositionalEncoder,
    tick_interval_ns: u64,  // Expected tick interval in nanoseconds
}

impl TimeAwarePositionalEncoder {
    /// Create time-aware encoder
    pub fn new(config: PositionalEncodingConfig, tick_interval_ns: u64) -> Self {
        Self {
            base_encoder: PositionalEncoder::new(config),
            tick_interval_ns,
        }
    }
    
    /// Generate encoding based on actual timestamp differences
    pub fn encode_with_timestamps(
        &self,
        input: &[Vec<f64>],
        timestamps: &[u64]  // Nanosecond timestamps
    ) -> Vec<Vec<f64>> {
        let mut output = input.to_vec();
        
        if timestamps.len() != input.len() {
            // Fall back to standard encoding
            return self.base_encoder.encode_sequence(input);
        }
        
        // Calculate relative time positions
        let base_time = timestamps[0];
        
        for (i, token) in output.iter_mut().enumerate() {
            let time_diff = timestamps[i] - base_time;
            let normalized_pos = (time_diff / self.tick_interval_ns) as usize;
            
            let enc = self.base_encoder.get_encoding(normalized_pos);
            for (j, val) in token.iter_mut().enumerate().take(self.base_encoder.config.d_model) {
                *val += enc[j];
            }
        }
        
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_positional_encoder_basic() {
        let config = PositionalEncodingConfig::default();
        let encoder = PositionalEncoder::new(config);
        
        // Test single encoding retrieval
        let enc_0 = encoder.get_encoding(0);
        assert_eq!(enc_0.len(), 32);
        
        // First position should have specific pattern
        assert!((enc_0[0] - 0.0).abs() < 1e-6);  // sin(0) = 0
        
        // Test sequence encoding
        let mut sequence = vec![vec![1.0; 32]; 10];
        encoder.add_to_sequence(&mut sequence);
        
        // Verify encodings were added
        assert_ne!(sequence[0][0], 1.0);  // Should be modified
    }
    
    #[test]
    fn test_relative_positional_encoding() {
        let rel_enc = RelativePositionalEncoding::new(32, 16);
        
        // Test various relative positions
        let emb_neg = rel_enc.get_embedding(-5);
        let emb_zero = rel_enc.get_embedding(0);
        let emb_pos = rel_enc.get_embedding(5);
        
        assert_eq!(emb_neg.len(), 32);
        assert_eq!(emb_zero.len(), 32);
        assert_eq!(emb_pos.len(), 32);
        
        // Different positions should have different embeddings
        assert_ne!(emb_neg, emb_pos);
    }
    
    #[test]
    fn test_time_aware_encoding() {
        let config = PositionalEncodingConfig::default();
        let encoder = TimeAwarePositionalEncoder::new(config, 1_000_000); // 1ms ticks
        
        let input = vec![vec![0.5; 32]; 5];
        let timestamps = vec![0, 1_000_000, 2_000_000, 3_000_000, 4_000_000];
        
        let encoded = encoder.encode_with_timestamps(&input, &timestamps);
        
        assert_eq!(encoded.len(), 5);
        assert_eq!(encoded[0].len(), 32);
    }
    
    #[test]
    fn test_encoding_uniqueness() {
        let config = PositionalEncodingConfig::default();
        let encoder = PositionalEncoder::new(config);
        
        // Verify different positions have different encodings
        let mut seen = std::collections::HashSet::new();
        for i in 0..100 {
            let enc = encoder.get_encoding(i);
            let hash = format!("{:?}", enc);
            assert!(!seen.contains(&hash), "Duplicate encoding at position {}", i);
            seen.insert(hash);
        }
    }
}
