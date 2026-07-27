//! Runtime Integrity Checker - Computes SHA-256 hashes of core binaries
//! 
//! This module performs runtime integrity verification by computing and comparing
//! cryptographic hashes of critical application components. It detects tampering,
//! unauthorized modifications, and code injection attempts.
//! 
//! Security Features:
//! - SHA-256 hashing of Rust binaries and Python bytecode
//! - Embedded baseline hashes compiled into the binary
//! - Continuous monitoring during execution
//! - Immediate termination on integrity failure
//! 
//! Optimized for zero-cost abstractions and minimal memory footprint.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::thread;

/// Maximum file size to hash (100MB limit for safety)
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Check interval in milliseconds
const CHECK_INTERVAL_MS: u64 = 5000;

/// Atomic flag indicating integrity status
static INTEGRITY_OK: AtomicBool = AtomicBool::new(false);

/// Timestamp of last successful check
static LAST_CHECK_TIMESTAMP: AtomicU64 = AtomicU64::new(0);

/// Result of an integrity check
#[derive(Debug, Clone)]
pub struct IntegrityResult {
    /// Path that was checked
    pub path: PathBuf,
    /// Expected hash (hex encoded)
    pub expected_hash: String,
    /// Actual hash (hex encoded)
    pub actual_hash: String,
    /// Whether the check passed
    pub is_valid: bool,
    /// Time taken for the check
    pub check_duration_ms: u64,
}

/// IntegrityChecker manages runtime verification of application integrity
pub struct IntegrityChecker {
    /// Baseline hashes embedded at compile time
    baseline_hashes: HashMap<String, String>,
    /// Paths to monitor
    monitored_paths: Vec<PathBuf>,
    /// Whether continuous monitoring is active
    is_monitoring: bool,
    /// Stop signal for monitoring thread
    stop_monitoring: AtomicBool,
}

impl IntegrityChecker {
    /// Create a new integrity checker with embedded baselines
    pub fn new() -> Self {
        // In production, these would be embedded at build time
        let mut baseline_hashes = HashMap::new();
        
        // Example baseline hashes (would be generated during secure build)
        baseline_hashes.insert(
            "backend/security/key_vault.rs".to_string(),
            "PLACEHOLDER_HASH_KEY_VAULT".to_string()
        );
        baseline_hashes.insert(
            "backend/security/memory_guard.py".to_string(),
            "PLACEHOLDER_HASH_MEMORY_GUARD".to_string()
        );
        baseline_hashes.insert(
            "backend/security/secure_enclave.py".to_string(),
            "PLACEHOLDER_HASH_SECURE_ENCLAVE".to_string()
        );
        
        IntegrityChecker {
            baseline_hashes,
            monitored_paths: Vec::new(),
            is_monitoring: false,
            stop_monitoring: AtomicBool::new(false),
        }
    }

    /// Add a path to be monitored
    pub fn add_path(&mut self, path: &str) -> &mut Self {
        self.monitored_paths.push(PathBuf::from(path));
        self
    }

    /// Compute SHA-256 hash of a file
    pub fn compute_hash<P: AsRef<Path>>(&self, path: P) -> Result<String, IntegrityError> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(IntegrityError::FileNotFound(path.to_path_buf()));
        }

        let metadata = fs::metadata(path)?;
        
        // Check file size limit
        if metadata.len() > MAX_FILE_SIZE {
            return Err(IntegrityError::FileTooLarge(metadata.len()));
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let result = hasher.finalize();
        Ok(hex::encode(result))
    }

    /// Verify a single file against its baseline hash
    pub fn verify_file<P: AsRef<Path>>(&self, path: P) -> Result<IntegrityResult, IntegrityError> {
        let path = path.as_ref();
        let start = Instant::now();
        
        let path_str = path.to_string_lossy().to_string();
        let expected_hash = self.baseline_hashes.get(&path_str)
            .ok_or_else(|| IntegrityError::NoBaselineHash(path.to_path_buf()))?;
        
        let actual_hash = self.compute_hash(path)?;
        let duration = start.elapsed();

        let result = IntegrityResult {
            path: path.to_path_buf(),
            expected_hash: expected_hash.clone(),
            actual_hash: actual_hash.clone(),
            is_valid: expected_hash == &actual_hash,
            check_duration_ms: duration.as_millis() as u64,
        };

        Ok(result)
    }

    /// Verify all monitored files
    pub fn verify_all(&self) -> Result<Vec<IntegrityResult>, IntegrityError> {
        let mut results = Vec::with_capacity(self.monitored_paths.len());
        let mut all_valid = true;

        for path in &self.monitored_paths {
            match self.verify_file(path) {
                Ok(result) => {
                    if !result.is_valid {
                        all_valid = false;
                    }
                    results.push(result);
                }
                Err(e) => {
                    all_valid = false;
                    results.push(IntegrityResult {
                        path: path.clone(),
                        expected_hash: String::new(),
                        actual_hash: String::new(),
                        is_valid: false,
                        check_duration_ms: 0,
                    });
                }
            }
        }

        // Update global status
        INTEGRITY_OK.store(all_valid, Ordering::SeqCst);
        if all_valid {
            LAST_CHECK_TIMESTAMP.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                Ordering::SeqCst
            );
        }

        Ok(results)
    }

    /// Start continuous integrity monitoring in a background thread
    pub fn start_monitoring(&mut self, callback: Box<dyn Fn(Vec<IntegrityResult>) + Send + 'static>) {
        if self.is_monitoring {
            return;
        }

        self.is_monitoring = true;
        self.stop_monitoring.store(false, Ordering::SeqCst);

        let paths = self.monitored_paths.clone();
        let baseline = self.baseline_hashes.clone();
        let stop_signal = self.stop_monitoring.clone();

        thread::spawn(move || {
            let checker = IntegrityChecker {
                baseline_hashes: baseline,
                monitored_paths: paths,
                is_monitoring: true,
                stop_monitoring: stop_signal.clone(),
            };

            while !stop_signal.load(Ordering::SeqCst) {
                match checker.verify_all() {
                    Ok(results) => {
                        callback(results);
                        
                        // If any check failed, trigger immediate action
                        if results.iter().any(|r| !r.is_valid) {
                            eprintln!("CRITICAL: Integrity check failed! Terminating...");
                            std::process::exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("Integrity check error: {:?}", e);
                        std::process::exit(1);
                    }
                }

                thread::sleep(Duration::from_millis(CHECK_INTERVAL_MS));
            }
        });
    }

    /// Stop continuous monitoring
    pub fn stop_monitoring(&mut self) {
        self.stop_monitoring.store(true, Ordering::SeqCst);
        self.is_monitoring = false;
    }

    /// Get current integrity status
    pub fn is_integrity_ok(&self) -> bool {
        INTEGRITY_OK.load(Ordering::SeqCst)
    }

    /// Get timestamp of last successful check
    pub fn last_check_timestamp(&self) -> u64 {
        LAST_CHECK_TIMESTAMP.load(Ordering::SeqCst)
    }

    /// Generate baseline hashes for all monitored files
    /// This should only be run during secure build/deployment
    pub fn generate_baselines(&mut self) -> Result<HashMap<String, String>, IntegrityError> {
        let mut baselines = HashMap::new();

        for path in &self.monitored_paths {
            if path.exists() {
                let hash = self.compute_hash(path)?;
                baselines.insert(path.to_string_lossy().to_string(), hash);
            }
        }

        // Update internal baselines
        self.baseline_hashes.extend(baselines.clone());
        
        Ok(baselines)
    }

    /// Verify Python bytecode files (.pyc)
    pub fn verify_python_bytecode<P: AsRef<Path>>(&self, py_path: P) -> Result<IntegrityResult, IntegrityError> {
        let py_path = py_path.as_ref();
        
        // Construct .pyc path (simplified, real implementation would handle __pycache__)
        let pyc_path = py_path.with_extension("pyc");
        
        if !pyc_path.exists() {
            return Err(IntegrityError::FileNotFound(pyc_path));
        }

        self.verify_file(pyc_path)
    }

    /// Emergency shutdown if integrity is compromised
    pub fn emergency_shutdown(&self, reason: &str) -> ! {
        eprintln!("EMERGENCY SHUTDOWN: {}", reason);
        eprintln!("Reason: Integrity check failed");
        
        // Securely zero sensitive memory before exit
        // (handled by Drop implementations)
        
        std::process::exit(1);
    }
}

impl Default for IntegrityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IntegrityChecker {
    fn drop(&mut self) {
        self.stop_monitoring();
    }
}

/// Integrity check errors
#[derive(Debug)]
pub enum IntegrityError {
    Io(std::io::Error),
    FileNotFound(PathBuf),
    FileTooLarge(u64),
    NoBaselineHash(PathBuf),
    HashMismatch { expected: String, actual: String },
}

impl From<std::io::Error> for IntegrityError {
    fn from(err: std::io::Error) -> Self {
        IntegrityError::Io(err)
    }
}

impl std::fmt::Display for IntegrityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegrityError::Io(e) => write!(f, "IO error: {}", e),
            IntegrityError::FileNotFound(p) => write!(f, "File not found: {:?}", p),
            IntegrityError::FileTooLarge(size) => write!(f, "File too large: {} bytes", size),
            IntegrityError::NoBaselineHash(p) => write!(f, "No baseline hash for: {:?}", p),
            IntegrityError::HashMismatch { expected, actual } => {
                write!(f, "Hash mismatch - expected: {}, actual: {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for IntegrityError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_compute_hash() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        
        // Create test file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"Hello, World!").unwrap();
        drop(file);

        let checker = IntegrityChecker::new();
        let hash = checker.compute_hash(&file_path).unwrap();
        
        // Verify it's a valid hex string of correct length
        assert_eq!(hash.len(), 64); // SHA-256 produces 64 hex characters
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_verify_all() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        
        // Create test file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"Test content").unwrap();
        drop(file);

        let mut checker = IntegrityChecker::new();
        checker.add_path(file_path.to_str().unwrap());
        
        // Generate baseline
        let baselines = checker.generate_baselines().unwrap();
        assert_eq!(baselines.len(), 1);
        
        // Verify (should pass)
        let results = checker.verify_all().unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_valid);
        
        // Modify file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"Modified content").unwrap();
        drop(file);
        
        // Verify again (should fail)
        let results = checker.verify_all().unwrap();
        assert!(!results[0].is_valid);
    }
}
