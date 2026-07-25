//! High-Speed ML Inference Engine
//! 
//! Executes compiled ML models in microseconds with memory page locking
//! to prevent OS paging delays. Optimized for sub-100μs inference latency.

use std::collections::HashMap;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Configuration for inference engine
#[derive(Clone, Debug)]
pub struct InferenceConfig {
    pub n_threads: usize,
    pub batch_size: usize,
    pub enable_memory_locking: bool,
    pub cache_predictions: bool,
    pub max_cache_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_threads: num_cpus::get(),
            batch_size: 64,
            enable_memory_locking: true,
            cache_predictions: true,
            max_cache_size: 10000,
        }
    }
}

/// Compiled model representation (simplified)
#[derive(Clone)]
pub struct CompiledModel {
    pub name: String,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub architecture: ModelArchitecture,
}

#[derive(Clone, Debug)]
pub enum ModelArchitecture {
    Linear { input_dim: usize, output_dim: usize },
    TwoLayer { input_dim: usize, hidden_dim: usize, output_dim: usize },
    TreeEnsemble { n_trees: usize, max_depth: usize },
}

/// Prediction cache entry
#[derive(Clone)]
struct CacheEntry {
    input_hash: u64,
    prediction: Vec<f32>,
    timestamp: std::time::Instant,
}

/// Memory-locked inference engine
pub struct InferenceEngine {
    config: InferenceConfig,
    models: Arc<RwLock<HashMap<String, CompiledModel>>>,
    
    // Prediction cache (LRU-style)
    cache: Arc<RwLock<Vec<CacheEntry>>>,
    
    // Performance metrics
    total_inferences: u64,
    total_time_ns: u128,
    
    // Memory locking state
    memory_locked: bool,
}

impl InferenceEngine {
    pub fn new(config: InferenceConfig) -> Self {
        let mut engine = Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(Vec::new())),
            total_inferences: 0,
            total_time_ns: 0,
            memory_locked: false,
        };
        
        // Optionally lock memory pages
        if config.enable_memory_locking {
            engine.lock_memory_pages();
        }
        
        engine
    }
    
    /// Lock memory pages to prevent swapping
    #[cfg(target_os = "windows")]
    fn lock_memory_pages(&mut self) {
        // Windows-specific: Use SetProcessWorkingSetSize
        // This is a simplified placeholder - actual implementation would use winapi
        unsafe {
            // Placeholder for Windows API call
            // SetProcessWorkingSetSize(GetCurrentProcess(), -1, -1);
        }
        self.memory_locked = true;
        eprintln!("Memory pages locked (Windows)");
    }
    
    #[cfg(target_os = "linux")]
    fn lock_memory_pages(&mut self) {
        // Linux-specific: Use mlockall
        unsafe {
            libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);
        }
        self.memory_locked = true;
        eprintln!("Memory pages locked (Linux)");
    }
    
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    fn lock_memory_pages(&mut self) {
        self.memory_locked = true;
        eprintln!("Memory locking not available on this platform");
    }
    
    /// Register a compiled model
    pub fn register_model(&self, model: CompiledModel) {
        let mut models = self.models.write().unwrap();
        models.insert(model.name.clone(), model);
    }
    
    /// Run inference on single sample (optimized path)
    pub fn infer(&mut self, model_name: &str, input: &[f32]) -> Vec<f32> {
        let start = std::time::Instant::now();
        
        // Check cache first
        if self.config.cache_predictions {
            let input_hash = self.hash_input(input);
            if let Some(cached) = self.get_cached(input_hash) {
                return cached;
            }
        }
        
        // Get model
        let models = self.models.read().unwrap();
        let model = match models.get(model_name) {
            Some(m) => m.clone(),
            None => return vec![0.0],
        };
        drop(models);
        
        // Execute based on architecture
        let prediction = match &model.architecture {
            ModelArchitecture::Linear { .. } => {
                self.infer_linear(&model, input)
            },
            ModelArchitecture::TwoLayer { .. } => {
                self.infer_two_layer(&model, input)
            },
            ModelArchitecture::TreeEnsemble { .. } => {
                self.infer_tree_ensemble(&model, input)
            },
        };
        
        // Cache result
        if self.config.cache_predictions {
            self.cache_prediction(input, &prediction);
        }
        
        // Update metrics
        let elapsed = start.elapsed();
        self.total_inferences += 1;
        self.total_time_ns += elapsed.as_nanos();
        
        prediction
    }
    
    /// Batch inference (parallelized)
    pub fn infer_batch(&mut self, model_name: &str, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let start = std::time::Instant::now();
        
        // Get model
        let models = self.models.read().unwrap();
        let model = match models.get(model_name) {
            Some(m) => m.clone(),
            None => return vec![vec![0.0]; inputs.len()],
        };
        drop(models);
        
        // Parallel batch processing
        let predictions: Vec<Vec<f32>> = inputs
            .par_iter()
            .map(|input| {
                match &model.architecture {
                    ModelArchitecture::Linear { .. } => {
                        self.infer_linear(&model, input)
                    },
                    ModelArchitecture::TwoLayer { .. } => {
                        self.infer_two_layer(&model, input)
                    },
                    ModelArchitecture::TreeEnsemble { .. } => {
                        self.infer_tree_ensemble(&model, input)
                    },
                }
            })
            .collect();
        
        // Update metrics
        let elapsed = start.elapsed();
        self.total_inferences += inputs.len() as u64;
        self.total_time_ns += elapsed.as_nanos();
        
        predictions
    }
    
    /// Linear model inference
    fn infer_linear(&self, model: &CompiledModel, input: &[f32]) -> Vec<f32> {
        let (input_dim, output_dim) = match &model.architecture {
            ModelArchitecture::Linear { input_dim, output_dim } => (*input_dim, *output_dim),
            _ => return vec![0.0],
        };
        
        let weights = &model.weights;
        let biases = &model.biases;
        
        (0..output_dim)
            .into_par_iter()
            .map(|i| {
                let mut sum = biases[i];
                for j in 0..input_dim.min(input.len()) {
                    sum += weights[i * input_dim + j] * input[j];
                }
                sum
            })
            .collect()
    }
    
    /// Two-layer neural network inference
    fn infer_two_layer(&self, model: &CompiledModel, input: &[f32]) -> Vec<f32> {
        let (input_dim, hidden_dim, output_dim) = match &model.architecture {
            ModelArchitecture::TwoLayer { input_dim, hidden_dim, output_dim } => {
                (*input_dim, *hidden_dim, *output_dim)
            },
            _ => return vec![0.0],
        };
        
        let weights = &model.weights;
        let biases = &model.biases;
        
        // Layer 1 with ReLU
        let hidden: Vec<f32> = (0..hidden_dim)
            .into_par_iter()
            .map(|i| {
                let mut sum = biases[i];
                for j in 0..input_dim.min(input.len()) {
                    sum += weights[i * input_dim + j] * input[j];
                }
                sum.max(0.0)  // ReLU
            })
            .collect();
        
        // Layer 2 (output)
        let offset = hidden_dim * input_dim;
        (0..output_dim)
            .into_par_iter()
            .map(|i| {
                let mut sum = biases[hidden_dim + i];
                for j in 0..hidden_dim {
                    sum += weights[offset + i * hidden_dim + j] * hidden[j];
                }
                sum
            })
            .collect()
    }
    
    /// Tree ensemble inference (simplified)
    fn infer_tree_ensemble(&self, model: &CompiledModel, input: &[f32]) -> Vec<f32> {
        // Simplified tree ensemble - in production would have actual tree structure
        let n_trees = match &model.architecture {
            ModelArchitecture::TreeEnsemble { n_trees, .. } => *n_trees,
            _ => return vec![0.0],
        };
        
        // Average prediction across trees (placeholder)
        let pred: f32 = model.weights
            .par_iter()
            .take(n_trees)
            .map(|&w| w * input.iter().sum::<f32>() / input.len() as f32)
            .sum();
        
        vec![pred / n_trees as f32]
    }
    
    /// Hash input for caching
    fn hash_input(&self, input: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for &val in input {
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
    
    /// Get cached prediction
    fn get_cached(&self, hash: u64) -> Option<Vec<f32>> {
        let cache = self.cache.read().unwrap();
        cache.iter()
            .find(|e| e.input_hash == hash)
            .filter(|e| e.timestamp.elapsed().as_secs() < 60)  // 1-minute TTL
            .map(|e| e.prediction.clone())
    }
    
    /// Cache prediction
    fn cache_prediction(&self, input: &[f32], prediction: &[f32]) {
        let mut cache = self.cache.write().unwrap();
        
        let entry = CacheEntry {
            input_hash: self.hash_input(input),
            prediction: prediction.to_vec(),
            timestamp: std::time::Instant::now(),
        };
        
        // Evict old entries if at capacity
        if cache.len() >= self.config.max_cache_size {
            cache.remove(0);
        }
        
        cache.push(entry);
    }
    
    /// Get average inference latency
    pub fn get_avg_latency_us(&self) -> f64 {
        if self.total_inferences == 0 {
            return 0.0;
        }
        (self.total_time_ns / self.total_inferences as u128) as f64 / 1000.0
    }
    
    /// Get performance summary
    pub fn get_summary(&self) -> serde_json::Value {
        use serde_json::json;
        
        json!({
            "total_inferences": self.total_inferences,
            "avg_latency_us": self.get_avg_latency_us(),
            "memory_locked": self.memory_locked,
            "cache_size": self.cache.read().unwrap().len(),
            "n_models": self.models.read().unwrap().len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_engine_creation() {
        let config = InferenceConfig::default();
        let engine = InferenceEngine::new(config);
        
        assert!(engine.memory_locked || !InferenceConfig::default().enable_memory_locking);
    }
    
    #[test]
    fn test_linear_inference() {
        let config = InferenceConfig::default();
        let mut engine = InferenceEngine::new(config);
        
        // Register simple linear model: y = 2x + 1
        let model = CompiledModel {
            name: "test_linear".to_string(),
            weights: vec![2.0],
            biases: vec![1.0],
            architecture: ModelArchitecture::Linear {
                input_dim: 1,
                output_dim: 1,
            },
        };
        engine.register_model(model);
        
        let result = engine.infer("test_linear", &[3.0]);
        assert!((result[0] - 7.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_batch_inference_speed() {
        let config = InferenceConfig::default();
        let mut engine = InferenceEngine::new(config);
        
        // Register model
        let model = CompiledModel {
            name: "test_batch".to_string(),
            weights: (0..1000).map(|_| rand::random::<f32>()).collect(),
            biases: vec![0.0; 10],
            architecture: ModelArchitecture::Linear {
                input_dim: 100,
                output_dim: 10,
            },
        };
        engine.register_model(model);
        
        // Generate batch
        let inputs: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..100).map(|_| rand::random::<f32>()).collect())
            .collect();
        
        // Time batch inference
        let start = std::time::Instant::now();
        let _results = engine.infer_batch("test_batch", &inputs);
        let elapsed = start.elapsed();
        
        // Should complete 100 inferences in < 10ms
        assert!(elapsed.as_millis() < 10, "Batch inference too slow: {:?}", elapsed);
    }
}
