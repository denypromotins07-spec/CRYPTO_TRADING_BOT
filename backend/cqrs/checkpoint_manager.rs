//! Checkpoint Manager - Executes atomic disk writes for memory snapshots.
//!
//! This module handles the persistence of state snapshots to disk using
//! atomic write operations to ensure data integrity. Critical for crash
//! recovery where partial or corrupted snapshots could lead to incorrect
//! state reconstruction.
//!
//! Features:
//! - Atomic file writes using rename semantics
//! - Checksum validation for data integrity
//! - Compression for storage efficiency
//! - Multi-version checkpoint retention
//! - Async I/O for non-blocking operation

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;

/// Default directory for checkpoints
const DEFAULT_CHECKPOINT_DIR: &str = "./checkpoints";

/// Maximum number of checkpoints to retain
const MAX_CHECKPOINTS: usize = 10;

/// Result of a checkpoint operation
#[derive(Debug, Clone)]
pub struct CheckpointResult {
    pub success: bool,
    pub checkpoint_path: String,
    pub sequence_number: u64,
    pub size_bytes: u64,
    pub duration_ms: f64,
    pub checksum: String,
    pub error_message: Option<String>,
}

impl CheckpointResult {
    pub fn success(
        checkpoint_path: String,
        sequence_number: u64,
        size_bytes: u64,
        duration_ms: f64,
        checksum: String,
    ) -> Self {
        CheckpointResult {
            success: true,
            checkpoint_path,
            sequence_number,
            size_bytes,
            duration_ms,
            checksum,
            error_message: None,
        }
    }

    pub fn failure(error: &str) -> Self {
        CheckpointResult {
            success: false,
            checkpoint_path: String::new(),
            sequence_number: 0,
            size_bytes: 0,
            duration_ms: 0.0,
            checksum: String::new(),
            error_message: Some(error.to_string()),
        }
    }
}

/// Metadata stored with each checkpoint
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    pub sequence_number: u64,
    pub timestamp_us: u64,
    pub version: u16,
    pub compression: String,
    pub original_size: u64,
    pub compressed_size: u64,
    pub checksum: String,
    pub hostname: String,
    pub process_id: u32,
}

/// Checkpoint manager for atomic state persistence
pub struct CheckpointManager {
    /// Base directory for checkpoints
    checkpoint_dir: PathBuf,
    /// Current checkpoint sequence
    current_sequence: AtomicU64,
    /// Checkpoint in progress flag
    checkpoint_in_progress: AtomicBool,
    /// List of valid checkpoint sequences
    valid_checkpoints: Arc<RwLock<Vec<u64>>>,
    /// Compression level (0-9)
    compression_level: u8,
}

unsafe impl Send for CheckpointManager {}
unsafe impl Sync for CheckpointManager {}

impl CheckpointManager {
    /// Create a new CheckpointManager
    pub fn new(checkpoint_dir: Option<&str>) -> std::io::Result<Self> {
        let dir = PathBuf::from(checkpoint_dir.unwrap_or(DEFAULT_CHECKPOINT_DIR));
        
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(&dir)?;
        
        let manager = CheckpointManager {
            checkpoint_dir: dir,
            current_sequence: AtomicU64::new(0),
            checkpoint_in_progress: AtomicBool::new(false),
            valid_checkpoints: Arc::new(RwLock::new(Vec::new())),
            compression_level: 6, // Default compression
        };
        
        // Scan for existing checkpoints
        manager.scan_existing_checkpoints();
        
        Ok(manager)
    }

    /// Scan directory for existing valid checkpoints
    pub fn scan_existing_checkpoints(&self) {
        let mut sequences = Vec::new();
        
        if let Ok(entries) = fs::read_dir(&self.checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("chk") {
                    if let Some(seq) = self.extract_sequence_from_path(&path) {
                        sequences.push(seq);
                    }
                }
            }
        }
        
        sequences.sort();
        *self.valid_checkpoints.write() = sequences;
        
        if let Some(&max_seq) = sequences.last() {
            self.current_sequence.store(max_seq, Ordering::Relaxed);
        }
    }

    /// Extract sequence number from checkpoint filename
    fn extract_sequence_from_path(&self, path: &Path) -> Option<u64> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|name| name.strip_prefix("checkpoint_"))
            .and_then(|seq_str| seq_str.parse::<u64>().ok())
    }

    /// Create an atomic checkpoint of state data
    /// 
    /// Uses atomic rename semantics to ensure checkpoint is either
    /// fully written or not present (no partial states).
    pub fn create_checkpoint<T: serde::Serialize>(
        &self,
        state: &T,
        sequence: u64,
    ) -> CheckpointResult {
        // Prevent concurrent checkpoints
        if self.checkpoint_in_progress.swap(true, Ordering::SeqCst) {
            return CheckpointResult::failure("Checkpoint already in progress");
        }

        let start_time = Instant::now();
        
        // Generate checkpoint paths
        let temp_path = self.get_temp_checkpoint_path(sequence);
        let final_path = self.get_checkpoint_path(sequence);
        let metadata_path = self.get_metadata_path(sequence);

        // Serialize state to bytes
        let serialized = match bincode::serialize(state) {
            Ok(data) => data,
            Err(e) => {
                self.checkpoint_in_progress.store(false, Ordering::SeqCst);
                return CheckpointResult::failure(&format!("Serialization failed: {}", e));
            }
        };

        let original_size = serialized.len() as u64;

        // Compress data (optional, based on compression level)
        let compressed = if self.compression_level > 0 {
            match self.compress_data(&serialized) {
                Ok(data) => data,
                Err(e) => {
                    self.checkpoint_in_progress.store(false, Ordering::SeqCst);
                    return CheckpointResult::failure(&format!("Compression failed: {}", e));
                }
            }
        } else {
            serialized
        };

        let compressed_size = compressed.len() as u64;

        // Calculate checksum
        let checksum = self.calculate_checksum(&compressed);

        // Write to temporary file first
        let write_result = self.write_atomic(&temp_path, &compressed);
        
        if let Err(e) = write_result {
            let _ = fs::remove_file(&temp_path);
            self.checkpoint_in_progress.store(false, Ordering::SeqCst);
            return CheckpointResult::failure(&format!("Write failed: {}", e));
        }

        // Write metadata
        let metadata = CheckpointMetadata {
            sequence_number: sequence,
            timestamp_us: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            version: 1,
            compression: if self.compression_level > 0 { "lz4" } else { "none" }.to_string(),
            original_size,
            compressed_size,
            checksum: checksum.clone(),
            hostname: gethostname::gethostname().to_string_lossy().into_owned(),
            process_id: std::process::id(),
        };

        let metadata_json = match serde_json::to_string_pretty(&metadata) {
            Ok(json) => json,
            Err(e) => {
                let _ = fs::remove_file(&temp_path);
                self.checkpoint_in_progress.store(false, Ordering::SeqCst);
                return CheckpointResult::failure(&format!("Metadata serialization failed: {}", e));
            }
        };

        if let Err(e) = fs::write(&metadata_path, metadata_json) {
            let _ = fs::remove_file(&temp_path);
            self.checkpoint_in_progress.store(false, Ordering::SeqCst);
            return CheckpointResult::failure(&format!("Metadata write failed: {}", e));
        }

        // Atomic rename: temp -> final
        if let Err(e) = fs::rename(&temp_path, &final_path) {
            let _ = fs::remove_file(&temp_path);
            let _ = fs::remove_file(&metadata_path);
            self.checkpoint_in_progress.store(false, Ordering::SeqCst);
            return CheckpointResult::failure(&format!("Atomic rename failed: {}", e));
        }

        let duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Update tracking
        self.current_sequence.store(sequence, Ordering::Relaxed);
        self.valid_checkpoints.write().push(sequence);
        
        // Cleanup old checkpoints
        self.cleanup_old_checkpoints();

        self.checkpoint_in_progress.store(false, Ordering::SeqCst);

        CheckpointResult::success(
            final_path.to_string_lossy().into_owned(),
            sequence,
            compressed_size,
            duration_ms,
            checksum,
        )
    }

    /// Load state from a checkpoint
    pub fn load_checkpoint<T: serde::de::DeserializeOwned>(
        &self,
        sequence: u64,
    ) -> Result<T, CheckpointError> {
        let checkpoint_path = self.get_checkpoint_path(sequence);
        let metadata_path = self.get_metadata_path(sequence);

        // Verify checkpoint exists
        if !checkpoint_path.exists() {
            return Err(CheckpointError::NotFound(format!(
                "Checkpoint {} not found",
                sequence
            )));
        }

        // Load and verify metadata
        let metadata_bytes = fs::read(&metadata_path).map_err(|e| {
            CheckpointError::IoError(format!("Failed to read metadata: {}", e))
        })?;

        let metadata: CheckpointMetadata = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| CheckpointError::ParseError(format!("Invalid metadata: {}", e)))?;

        // Load checkpoint data
        let mut file = File::open(&checkpoint_path).map_err(|e| {
            CheckpointError::IoError(format!("Failed to open checkpoint: {}", e))
        })?;

        let mut compressed_data = Vec::new();
        file.read_to_end(&mut compressed_data).map_err(|e| {
            CheckpointError::IoError(format!("Failed to read checkpoint: {}", e))
        })?;

        // Verify checksum
        let actual_checksum = self.calculate_checksum(&compressed_data);
        if actual_checksum != metadata.checksum {
            return Err(CheckpointError::ChecksumMismatch(format!(
                "Expected {}, got {}",
                metadata.checksum, actual_checksum
            )));
        }

        // Decompress if needed
        let data = if metadata.compression == "lz4" {
            self.decompress_data(&compressed_data)?
        } else {
            compressed_data
        };

        // Deserialize state
        let state: T = bincode::deserialize(&data)
            .map_err(|e| CheckpointError::ParseError(format!("Deserialization failed: {}", e)))?;

        Ok(state)
    }

    /// Get the latest valid checkpoint sequence
    pub fn get_latest_sequence(&self) -> Option<u64> {
        self.valid_checkpoints.read().last().copied()
    }

    /// Get all valid checkpoint sequences
    pub fn get_valid_sequences(&self) -> Vec<u64> {
        self.valid_checkpoints.read().clone()
    }

    /// Delete a specific checkpoint
    pub fn delete_checkpoint(&self, sequence: u64) -> std::io::Result<()> {
        let checkpoint_path = self.get_checkpoint_path(sequence);
        let metadata_path = self.get_metadata_path(sequence);

        let mut removed = false;

        if checkpoint_path.exists() {
            fs::remove_file(&checkpoint_path)?;
            removed = true;
        }

        if metadata_path.exists() {
            fs::remove_file(&metadata_path)?;
            removed = true;
        }

        if removed {
            self.valid_checkpoints
                .write()
                .retain(|&s| s != sequence);
        }

        Ok(())
    }

    /// Cleanup old checkpoints, keeping only the most recent MAX_CHECKPOINTS
    fn cleanup_old_checkpoints(&self) {
        let mut sequences = self.valid_checkpoints.write();
        
        while sequences.len() > MAX_CHECKPOINTS {
            if let Some(oldest) = sequences.first().copied() {
                sequences.remove(0);
                drop(sequences); // Release lock before deletion
                
                let _ = self.delete_checkpoint(oldest);
                
                sequences = self.valid_checkpoints.write();
            } else {
                break;
            }
        }
    }

    /// Write data atomically using a temporary file
    fn write_atomic(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer
        writer.write_all(data)?;
        writer.flush()?;

        // Ensure data is synced to disk
        writer.get_ref().sync_all()?;

        Ok(())
    }

    /// Calculate SHA256 checksum of data
    fn calculate_checksum(&self, data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    /// Compress data using LZ4
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        #[cfg(feature = "lz4")]
        {
            lz4_flex::compress_prepend_size(data)
                .map_err(|e| format!("LZ4 compression failed: {}", e))
        }
        #[cfg(not(feature = "lz4"))]
        {
            Ok(data.to_vec()) // No compression if feature disabled
        }
    }

    /// Decompress LZ4 data
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, CheckpointError> {
        #[cfg(feature = "lz4")]
        {
            lz4_flex::decompress_size_prepended(data)
                .map_err(|e| CheckpointError::DecompressionError(format!("LZ4 decompression failed: {}", e)))
        }
        #[cfg(not(feature = "lz4"))]
        {
            Ok(data.to_vec())
        }
    }

    fn get_checkpoint_path(&self, sequence: u64) -> PathBuf {
        self.checkpoint_dir.join(format!("checkpoint_{:020}.chk", sequence))
    }

    fn get_temp_checkpoint_path(&self, sequence: u64) -> PathBuf {
        self.checkpoint_dir.join(format!(".tmp_checkpoint_{:020}.chk", sequence))
    }

    fn get_metadata_path(&self, sequence: u64) -> PathBuf {
        self.checkpoint_dir.join(format!("checkpoint_{:020}.meta.json", sequence))
    }
}

/// Errors that can occur during checkpoint operations
#[derive(Debug)]
pub enum CheckpointError {
    NotFound(String),
    IoError(String),
    ParseError(String),
    ChecksumMismatch(String),
    DecompressionError(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::NotFound(msg) => write!(f, "Checkpoint not found: {}", msg),
            CheckpointError::IoError(msg) => write!(f, "IO error: {}", msg),
            CheckpointError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            CheckpointError::ChecksumMismatch(msg) => write!(f, "Checksum mismatch: {}", msg),
            CheckpointError::DecompressionError(msg) => write!(f, "Decompression error: {}", msg),
        }
    }
}

impl std::error::Error for CheckpointError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct TestState {
        value: u64,
        data: Vec<u8>,
    }

    #[test]
    fn test_checkpoint_create_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(Some(temp_dir.path().to_str().unwrap())).unwrap();

        let state = TestState {
            value: 42,
            data: vec![1, 2, 3, 4, 5],
        };

        let result = manager.create_checkpoint(&state, 100);
        
        assert!(result.success);
        assert_eq!(result.sequence_number, 100);
        assert!(result.size_bytes > 0);

        // Load and verify
        let loaded: TestState = manager.load_checkpoint(100).unwrap();
        assert_eq!(loaded, state);
    }

    #[test]
    fn test_checkpoint_checksum_verification() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(Some(temp_dir.path().to_str().unwrap())).unwrap();

        let state = TestState {
            value: 100,
            data: vec![10, 20, 30],
        };

        let result = manager.create_checkpoint(&state, 200);
        assert!(result.success);

        // Corrupt the checkpoint file
        let checkpoint_path = manager.get_checkpoint_path(200);
        let mut data = fs::read(&checkpoint_path).unwrap();
        data[0] ^= 0xFF; // Flip bits
        fs::write(&checkpoint_path, data).unwrap();

        // Load should fail due to checksum mismatch
        let load_result: Result<TestState, _> = manager.load_checkpoint(200);
        assert!(matches!(load_result, Err(CheckpointError::ChecksumMismatch(_))));
    }

    #[test]
    fn test_checkpoint_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(Some(temp_dir.path().to_str().unwrap())).unwrap();

        // Create more than MAX_CHECKPOINTS
        for i in 0..MAX_CHECKPOINTS + 5 {
            let state = TestState { value: i, data: vec![] };
            manager.create_checkpoint(&state, i as u64).unwrap();
        }

        // Should only have MAX_CHECKPOINTS remaining
        let sequences = manager.get_valid_sequences();
        assert!(sequences.len() <= MAX_CHECKPOINTS);
    }
}
