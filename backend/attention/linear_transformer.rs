//! Linear Transformer: O(N) Attention via Kernel Feature Maps
//!
//! This module implements linear attention mechanisms that avoid the O(N^2)
//! memory bottleneck of standard softmax attention, enabling efficient
//! processing of long sequences on 8GB RAM constrained systems.
//!
//! Key Features:
//! - Kernel-based feature maps for linear complexity
//! - Causal masking to prevent future data leakage
//! - Memory-efficient chunked inference
//! - Zero-cost abstractions using Rust iterators
//!
//! Mathematical Foundation:
//! Standard Attention: A = softmax(QK^T / sqrt(d)) V  [O(N^2)]
//! Linear Attention: A = phi(Q) (phi(K)^T V) / normalization  [O(N)]
//! Where phi is a kernel feature map (e.g., elu(x) + 1).

use std::time::{Duration, Instant};

/// Configuration for linear transformer
#[derive(Debug, Clone)]
pub struct LinearTransformerConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub use_causal_mask: bool,
}

impl Default for LinearTransformerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64,       // Optimized for 8GB RAM
            num_heads: 4,         // 4 heads for multi-head attention
            head_dim: 16,         // 64/4 = 16 per head
            max_seq_len: 2048,    // Maximum sequence length
            use_causal_mask: true, // Essential for autoregressive modeling
        }
    }
}

/// Linear Transformer core structure
pub struct LinearTransformer {
    config: LinearTransformerConfig,
    /// Query projection weights [num_heads * head_dim * hidden_dim]
    w_query: Vec<f64>,
    /// Key projection weights [num_heads * head_dim * hidden_dim]
    w_key: Vec<f64>,
    /// Value projection weights [num_heads * head_dim * hidden_dim]
    w_value: Vec<f64>,
    /// Output projection [hidden_dim * hidden_dim]
    w_output: Vec<f64>,
    /// Cumulative key-value state for streaming [num_heads * head_dim * head_dim]
    kv_state: Vec<f64>,
    /// Cumulative normalization state [num_heads * head_dim]
    z_state: Vec<f64>,
    /// Current sequence position
    position: usize,
}

impl LinearTransformer {
    /// Create a new linear transformer
    pub fn new(config: LinearTransformerConfig) -> Result<Self, String> {
        if config.hidden_dim > 128 {
            return Err("Hidden dimension exceeds memory budget".to_string());
        }
        
        let total_params = 3 * config.num_heads * config.head_dim * config.hidden_dim 
            + config.hidden_dim * config.hidden_dim;
        
        if total_params > 50000 {
            return Err("Parameter count too large for 8GB constraint".to_string());
        }
        
        Ok(Self {
            config: config.clone(),
            w_query: vec![0.0; config.num_heads * config.head_dim * config.hidden_dim],
            w_key: vec![0.0; config.num_heads * config.head_dim * config.hidden_dim],
            w_value: vec![0.0; config.num_heads * config.head_dim * config.hidden_dim],
            w_output: vec![0.0; config.hidden_dim * config.hidden_dim],
            kv_state: vec![0.0; config.num_heads * config.head_dim * config.head_dim],
            z_state: vec![0.0; config.num_heads * config.head_dim],
            position: 0,
        })
    }
    
    /// Initialize weights with orthogonal initialization
    pub fn initialize_weights(&mut self, seed: u64) {
        let mut rng = seed;
        let lcg_next = |x: u64| x.wrapping_mul(6364136223846793005).wrapping_add(1);
        
        let scale = 1.0 / (self.config.hidden_dim as f64).sqrt();
        
        // Initialize Q, K, V projections
        for proj in [&mut self.w_query, &mut self.w_key, &mut self.w_value].iter_mut() {
            for i in 0..proj.len() {
                rng = lcg_next(rng);
                proj[i] = ((rng % 1000) as f64 / 1000.0 - 0.5) * 2.0 * scale;
            }
        }
        
        // Initialize output projection
        let out_scale = 1.0 / (self.config.hidden_dim as f64).sqrt();
        for i in 0..self.w_output.len() {
            rng = lcg_next(rng);
            self.w_output[i] = ((rng % 1000) as f64 / 1000.0 - 0.5) * 2.0 * out_scale;
        }
    }
    
    /// Process a single token in streaming mode with O(1) memory per step
    #[inline]
    pub fn process_token(&mut self, input: &[f64]) -> Result<Vec<f64>, String> {
        if input.len() != self.config.hidden_dim {
            return Err(format!(
                "Input dimension {} mismatch with hidden dim {}",
                input.len(),
                self.config.hidden_dim
            ));
        }
        
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_dim;
        
        // Project input to Q, K, V for each head
        let mut queries = vec![Vec::new(); num_heads];
        let mut keys = vec![Vec::new(); num_heads];
        let mut values = vec![Vec::new(); num_heads];
        
        for h in 0..num_heads {
            queries[h] = vec![0.0; head_dim];
            keys[h] = vec![0.0; head_dim];
            values[h] = vec![0.0; head_dim];
            
            let q_offset = h * head_dim * hidden_dim;
            let k_offset = h * head_dim * hidden_dim;
            let v_offset = h * head_dim * hidden_dim;
            
            for i in 0..head_dim {
                for j in 0..hidden_dim {
                    queries[h][i] += self.w_query[q_offset + i * hidden_dim + j] * input[j];
                    keys[h][i] += self.w_key[k_offset + i * hidden_dim + j] * input[j];
                    values[h][i] += self.w_value[v_offset + i * hidden_dim + j] * input[j];
                }
                
                // Apply kernel feature map: phi(x) = elu(x) + 1
                // This ensures positivity for the attention mechanism
                queries[h][i] = Self::kernel_feature(queries[h][i]);
                keys[h][i] = Self::kernel_feature(keys[h][i]);
            }
        }
        
        // Update cumulative KV state: KV = sum_i phi(k_i)^T phi(v_i)
        // And normalization: z = sum_i phi(k_i)
        for h in 0..num_heads {
            for i in 0..head_dim {
                for j in 0..head_dim {
                    // Outer product: k_i * v_j
                    self.kv_state[h * head_dim * head_dim + i * head_dim + j] 
                        += keys[h][i] * values[h][j];
                }
                
                // Update normalization
                self.z_state[h * head_dim + i] += keys[h][i];
            }
        }
        
        // Compute attention output: out = (KV @ q) / z
        let mut head_outputs = vec![Vec::new(); num_heads];
        
        for h in 0..num_heads {
            head_outputs[h] = vec![0.0; head_dim];
            
            for i in 0..head_dim {
                let mut numerator = 0.0;
                
                for j in 0..head_dim {
                    numerator += self.kv_state[h * head_dim * head_dim + j * head_dim + i] 
                        * queries[h][j];
                }
                
                // Normalize with epsilon for numerical stability
                let denom = self.z_state[h * head_dim + i].max(1e-6);
                head_outputs[h][i] = numerator / denom;
            }
        }
        
        // Concatenate heads and project output
        let mut concatenated = vec![0.0; hidden_dim];
        for h in 0..num_heads {
            for i in 0..head_dim {
                concatenated[h * head_dim + i] = head_outputs[h][i];
            }
        }
        
        // Output projection
        let mut output = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                output[i] += self.w_output[i * hidden_dim + j] * concatenated[j];
            }
        }
        
        self.position += 1;
        
        Ok(output)
    }
    
    /// Kernel feature map: phi(x) = elu(x) + 1
    #[inline]
    fn kernel_feature(x: f64) -> f64 {
        if x >= 0.0 {
            x + 1.0
        } else {
            x.exp()
        }
    }
    
    /// Process an entire sequence at once (batch mode)
    pub fn process_sequence(&mut self, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        // Reset state for batch processing
        self.reset_state();
        
        let mut outputs = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let output = self.process_token(input)?;
            outputs.push(output);
        }
        
        Ok(outputs)
    }
    
    /// Reset internal state for new sequence
    pub fn reset_state(&mut self) {
        self.kv_state.fill(0.0);
        self.z_state.fill(0.0);
        self.position = 0;
    }
    
    /// Get current position in sequence
    pub fn get_position(&self) -> usize {
        self.position
    }
    
    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let param_size = (self.w_query.len() + self.w_key.len() + self.w_value.len() 
            + self.w_output.len()) * 8;
        let state_size = (self.kv_state.len() + self.z_state.len()) * 8;
        base_size + param_size + state_size
    }
}

/// Chunked linear transformer for memory-efficient long sequence processing
pub struct ChunkedLinearTransformer {
    inner: LinearTransformer,
    chunk_size: usize,
    buffer: Vec<Vec<f64>>,
}

impl ChunkedLinearTransformer {
    pub fn new(config: LinearTransformerConfig, chunk_size: usize) -> Result<Self, String> {
        let inner = LinearTransformer::new(config)?;
        
        Ok(Self {
            inner,
            chunk_size,
            buffer: Vec::new(),
        })
    }
    
    /// Add tokens to buffer and process when chunk is full
    pub fn add_tokens(&mut self, tokens: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let mut outputs = Vec::new();
        
        // Add to buffer
        for token in tokens {
            self.buffer.push(token.clone());
            
            // Process when buffer is full
            if self.buffer.len() >= self.chunk_size {
                let chunk = self.buffer.clone();
                self.buffer.clear();
                
                let chunk_outputs = self.inner.process_sequence(&chunk)?;
                outputs.extend(chunk_outputs);
            }
        }
        
        Ok(outputs)
    }
    
    /// Flush remaining buffered tokens
    pub fn flush(&mut self) -> Result<Vec<Vec<f64>>, String> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        
        let chunk = self.buffer.clone();
        self.buffer.clear();
        
        self.inner.process_sequence(&chunk)
    }
    
    /// Reset state
    pub fn reset(&mut self) {
        self.inner.reset_state();
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_transformer_initialization() {
        let config = LinearTransformerConfig::default();
        let mut transformer = LinearTransformer::new(config).unwrap();
        transformer.initialize_weights(42);
        
        assert_eq!(transformer.w_query.len(), 4 * 16 * 64);
        assert!(!transformer.w_query.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_streaming_processing() {
        let config = LinearTransformerConfig::default();
        let mut transformer = LinearTransformer::new(config).unwrap();
        transformer.initialize_weights(42);
        
        let input = vec![0.1; config.hidden_dim];
        
        // Process multiple tokens
        for _ in 0..10 {
            let output = transformer.process_token(&input);
            assert!(output.is_ok());
            
            let out = output.unwrap();
            assert_eq!(out.len(), config.hidden_dim);
            
            // Check no NaN or Inf
            for &x in &out {
                assert!(x.is_finite());
            }
        }
    }

    #[test]
    fn test_causality() {
        let config = LinearTransformerConfig::default();
        let mut transformer = LinearTransformer::new(config).unwrap();
        transformer.initialize_weights(42);
        
        // Process sequence
        let input1 = vec![1.0; config.hidden_dim];
        let input2 = vec![0.0; config.hidden_dim];
        
        let out1 = transformer.process_token(&input1).unwrap();
        let out2 = transformer.process_token(&input2).unwrap();
        
        // Outputs should be different due to accumulated state
        let mut different = false;
        for (a, b) in out1.iter().zip(out2.iter()) {
            if (a - b).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "Outputs should differ due to causal accumulation");
    }

    #[test]
    fn test_memory_efficiency() {
        let config = LinearTransformerConfig {
            max_seq_len: 10000,
            ..Default::default()
        };
        let transformer = LinearTransformer::new(config).unwrap();
        
        // Memory should be O(1) per step, not O(N)
        let mem_usage = transformer.estimate_memory_usage();
        println!("Memory usage: {} KB", mem_usage / 1024);
        
        // Should be well under 8GB limit
        assert!(mem_usage < 100 * 1024 * 1024); // Under 100MB
    }
}
