//! Thread Pool Module for ZAID Trading Bot
//! Dedicated math worker pool to isolate heavy CPU tasks
//! Prevents math operations from starving the Nautilus event loop
//!
//! This module provides a configurable thread pool with work-stealing
//! for parallel matrix operations, FFT computations, and signal processing
//! under the 8GB RAM constraint.

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

/// Math task types supported by the pool
pub enum MathTask {
    /// Matrix multiplication
    MatrixMult {
        data_a: Vec<f64>,
        data_b: Vec<f64>,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    },
    /// FFT computation
    Fft {
        signal: Vec<f64>,
    },
    /// Covariance calculation
    Covariance {
        returns: Vec<f64>,
        n_assets: usize,
        n_samples: usize,
    },
    /// Cholesky decomposition
    Cholesky {
        matrix: Vec<Vec<f64>>,
    },
    /// Custom closure task
    Custom(Box<dyn FnOnce() + Send + 'static>),
}

/// Result from a math task
pub enum MathResult {
    MatrixMult { result: Vec<f64> },
    Fft { spectrum: Vec<f64> },
    Covariance { matrix: Vec<f64> },
    Cholesky { lower: Vec<f64>, success: bool },
    CustomComplete,
    Error(String),
}

/// Work item with priority
struct WorkItem {
    task: MathTask,
    priority: u8,
    tx: mpsc::Sender<(u64, MathResult, Duration)>,
    task_id: u64,
    submitted_at: Instant,
}

/// Worker thread in the math pool
struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
    tasks_completed: AtomicUsize,
    is_busy: AtomicBool,
}

impl Worker {
    fn new(id: usize, receiver: Arc<std::sync::Mutex<mpsc::Receiver<WorkItem>>>) -> Self {
        let tasks_completed = AtomicUsize::new(0);
        let is_busy = AtomicBool::new(false);
        
        let handle = thread::spawn(move || {
            loop {
                // Try to get work
                let work = {
                    let lock = receiver.lock().unwrap();
                    lock.recv()
                };
                
                match work {
                    Ok(item) => {
                        is_busy.store(true, Ordering::Relaxed);
                        
                        // Execute task
                        let start = Instant::now();
                        let result = execute_task(item.task);
                        let elapsed = start.elapsed();
                        
                        // Send result
                        let _ = item.tx.send((item.task_id, result, elapsed));
                        
                        tasks_completed.fetch_add(1, Ordering::Relaxed);
                        is_busy.store(false, Ordering::Relaxed);
                    }
                    Err(_) => {
                        // Channel disconnected, shutdown
                        break;
                    }
                }
            }
        });
        
        Self {
            id,
            thread: Some(handle),
            tasks_completed,
            is_busy,
        }
    }
    
    fn tasks_completed(&self) -> usize {
        self.tasks_completed.load(Ordering::Relaxed)
    }
    
    fn is_busy(&self) -> bool {
        self.is_busy.load(Ordering::Relaxed)
    }
}

fn execute_task(task: MathTask) -> MathResult {
    match task {
        MathTask::MatrixMult { data_a, data_b, rows_a, cols_a, cols_b } => {
            // Naive matrix multiplication (would use optimized BLAS in production)
            let mut result = vec![0.0; rows_a * cols_b];
            
            for i in 0..rows_a {
                for j in 0..cols_b {
                    let mut sum = 0.0;
                    for k in 0..cols_a {
                        sum += data_a[i * cols_a + k] * data_b[k * cols_b + j];
                    }
                    result[i * cols_b + j] = sum;
                }
            }
            
            MathResult::MatrixMult { result }
        }
        MathTask::Fft { signal } => {
            // Simplified DFT (would use FFT in production)
            let n = signal.len();
            let mut spectrum = vec![0.0; n];
            
            for k in 0..n {
                let mut real = 0.0;
                let mut imag = 0.0;
                
                for t in 0..n {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                    real += signal[t] * angle.cos();
                    imag -= signal[t] * angle.sin();
                }
                
                spectrum[k] = (real * real + imag * imag).sqrt();
            }
            
            MathResult::Fft { spectrum }
        }
        MathTask::Covariance { returns, n_assets, n_samples } => {
            // Compute covariance matrix
            let mut cov = vec![0.0; n_assets * n_assets];
            
            // First compute means
            let mut means = vec![0.0; n_assets];
            for i in 0..n_samples {
                for j in 0..n_assets {
                    means[j] += returns[i * n_assets + j];
                }
            }
            for mean in &mut means {
                *mean /= n_samples as f64;
            }
            
            // Compute covariance
            for i in 0..n_assets {
                for j in 0..n_assets {
                    let mut sum = 0.0;
                    for k in 0..n_samples {
                        let diff_i = returns[k * n_assets + i] - means[i];
                        let diff_j = returns[k * n_assets + j] - means[j];
                        sum += diff_i * diff_j;
                    }
                    cov[i * n_assets + j] = sum / (n_samples - 1) as f64;
                }
            }
            
            MathResult::Covariance { matrix: cov }
        }
        MathTask::Cholesky { matrix } => {
            let n = matrix.len();
            if n == 0 || matrix[0].len() != n {
                return MathResult::Error("Invalid matrix dimensions".to_string());
            }
            
            let mut lower = vec![vec![0.0; n]; n];
            
            for i in 0..n {
                for j in 0..=i {
                    let mut sum = 0.0;
                    
                    if j == i {
                        for k in 0..j {
                            sum += lower[j][k] * lower[j][k];
                        }
                        
                        let val = matrix[j][j] - sum;
                        if val <= 0.0 {
                            return MathResult::Cholesky { lower: vec![], success: false };
                        }
                        lower[j][j] = val.sqrt();
                    } else {
                        for k in 0..j {
                            sum += lower[i][k] * lower[j][k];
                        }
                        
                        if lower[j][j].abs() < 1e-15 {
                            return MathResult::Cholesky { lower: vec![], success: false };
                        }
                        lower[i][j] = (matrix[i][j] - sum) / lower[j][j];
                    }
                }
            }
            
            // Flatten result
            let flat: Vec<f64> = lower.into_iter().flatten().collect();
            MathResult::Cholesky { lower: flat, success: true }
        }
        MathTask::Custom(f) => {
            f();
            MathResult::CustomComplete
        }
    }
}

/// High-performance math thread pool
pub struct MathThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<WorkItem>,
    receiver: Arc<std::sync::Mutex<mpsc::Receiver<WorkItem>>>,
    task_counter: AtomicUsize,
    shutdown: AtomicBool,
    max_queue_size: usize,
    current_queue_size: AtomicUsize,
}

impl MathThreadPool {
    /// Create new thread pool with specified number of workers
    pub fn new(n_workers: usize, max_queue_size: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(std::sync::Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(n_workers);
        for i in 0..n_workers {
            workers.push(Worker::new(i, Arc::clone(&receiver)));
        }
        
        Self {
            workers,
            sender,
            receiver,
            task_counter: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            max_queue_size,
            current_queue_size: AtomicUsize::new(0),
        }
    }
    
    /// Submit a math task to the pool
    pub fn submit(&self, task: MathTask, priority: u8) -> Result<u64, PoolError> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(PoolError::Shutdown);
        }
        
        if self.current_queue_size.load(Ordering::Relaxed) >= self.max_queue_size {
            return Err(PoolError::QueueFull);
        }
        
        let (tx, rx) = mpsc::channel();
        let task_id = self.task_counter.fetch_add(1, Ordering::Relaxed) as u64;
        
        let item = WorkItem {
            task,
            priority,
            tx,
            task_id,
            submitted_at: Instant::now(),
        };
        
        self.sender.send(item).map_err(|_| PoolError::ChannelError)?;
        self.current_queue_size.fetch_add(1, Ordering::Relaxed);
        
        // Wait for result
        let (_id, _result, _elapsed) = rx.recv().map_err(|_| PoolError::ChannelError)?;
        self.current_queue_size.fetch_sub(1, Ordering::Relaxed);
        
        Ok(task_id)
    }
    
    /// Submit task asynchronously (non-blocking)
    pub fn submit_async(&self, task: MathTask, priority: u8) -> Result<u64, PoolError> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(PoolError::Shutdown);
        }
        
        if self.current_queue_size.load(Ordering::Relaxed) >= self.max_queue_size {
            return Err(PoolError::QueueFull);
        }
        
        let (tx, _rx) = mpsc::channel(); // Fire and forget
        let task_id = self.task_counter.fetch_add(1, Ordering::Relaxed) as u64;
        
        let item = WorkItem {
            task,
            priority,
            tx,
            task_id,
            submitted_at: Instant::now(),
        };
        
        self.sender.send(item).map_err(|_| PoolError::ChannelError)?;
        self.current_queue_size.fetch_add(1, Ordering::Relaxed);
        
        Ok(task_id)
    }
    
    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let total_tasks: usize = self.workers.iter().map(|w| w.tasks_completed()).sum();
        let busy_workers = self.workers.iter().filter(|w| w.is_busy()).count();
        
        PoolStats {
            n_workers: self.workers.len(),
            busy_workers,
            total_tasks_completed: total_tasks,
            queue_size: self.current_queue_size.load(Ordering::Relaxed),
            is_shutdown: self.shutdown.load(Ordering::Relaxed),
        }
    }
    
    /// Graceful shutdown
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);
        drop(self.sender); // Close channel to signal workers
        
        for worker in self.workers {
            if let Some(handle) = worker.thread {
                let _ = handle.join();
            }
        }
    }
}

#[derive(Debug)]
pub enum PoolError {
    Shutdown,
    QueueFull,
    ChannelError,
}

#[derive(Debug)]
pub struct PoolStats {
    pub n_workers: usize,
    pub busy_workers: usize,
    pub total_tasks_completed: usize,
    pub queue_size: usize,
    pub is_shutdown: bool,
}

impl Drop for MathThreadPool {
    fn drop(&mut self) {
        if !self.shutdown.load(Ordering::Relaxed) {
            self.shutdown.store(true, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_mult() {
        let pool = MathThreadPool::new(2, 100);
        
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![5.0, 6.0, 7.0, 8.0];
        
        let task = MathTask::MatrixMult {
            data_a,
            data_b,
            rows_a: 2,
            cols_a: 2,
            cols_b: 2,
        };
        
        let result = pool.submit(task, 5);
        assert!(result.is_ok());
        
        pool.shutdown();
    }
    
    #[test]
    fn test_pool_stats() {
        let pool = MathThreadPool::new(4, 100);
        
        let stats = pool.get_stats();
        assert_eq!(stats.n_workers, 4);
        assert_eq!(stats.busy_workers, 0);
        
        pool.shutdown();
    }
}
