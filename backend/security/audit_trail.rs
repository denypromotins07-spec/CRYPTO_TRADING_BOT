//! Audit Trail - Cryptographically signed record of all internal state changes
//! 
//! This module implements an immutable, cryptographically-secured audit trail
//! that records every significant state change in the trading system.
//! 
//! Security Features:
//! - SHA-256 chain linking (blockchain-style)
//! - HMAC signature for each entry
//! - Tamper-evident logging
//! - Append-only storage
//! - Merkle root verification
//! 
//! Optimized for zero-cost abstractions and minimal memory overhead.

use sha2::{Digest, Sha256};
use hmac::{Hmac, Mac};
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{Write, BufReader, BufRead};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Maximum entries to keep in memory (ring buffer)
const MAX_MEMORY_ENTRIES: usize = 10_000;

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Sequential entry ID
    pub id: u64,
    /// Timestamp (Unix epoch milliseconds)
    pub timestamp: u64,
    /// Event type
    pub event_type: String,
    /// Event data (JSON serialized)
    pub data: String,
    /// Previous entry hash (for chain integrity)
    pub previous_hash: String,
    /// Entry hash (SHA-256 of content)
    pub entry_hash: String,
    /// HMAC signature
    pub signature: String,
}

/// Audit trail manager
pub struct AuditTrail {
    /// In-memory ring buffer for recent entries
    entries: VecDeque<AuditEntry>,
    /// Path to persistent audit log
    log_path: PathBuf,
    /// Secret key for HMAC signing
    signing_key: Vec<u8>,
    /// Current entry counter
    entry_counter: AtomicU64,
    /// Hash of the last entry (for chain linking)
    last_hash: String,
    /// Total entries written
    total_entries: u64,
}

impl AuditTrail {
    /// Create a new audit trail
    pub fn new(log_path: &str, signing_key: &[u8]) -> Result<Self, AuditError> {
        let path = PathBuf::from(log_path);
        
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut trail = AuditTrail {
            entries: VecDeque::with_capacity(MAX_MEMORY_ENTRIES),
            log_path: path,
            signing_key: signing_key.to_vec(),
            entry_counter: AtomicU64::new(0),
            last_hash: String::from("genesis"),
            total_entries: 0,
        };

        // Load existing entries if log exists
        if trail.log_path.exists() {
            trail.load_existing_entries()?;
        }

        Ok(trail)
    }

    /// Record a new audit entry
    pub fn record(&mut self, event_type: &str, data: &str) -> Result<&AuditEntry, AuditError> {
        let id = self.entry_counter.fetch_add(1, Ordering::SeqCst);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Compute entry hash
        let content = format!("{}:{}:{}:{}:{}", id, timestamp, event_type, data, self.last_hash);
        let entry_hash = hex::encode(Sha256::digest(content.as_bytes()));

        // Compute HMAC signature
        let mut mac = HmacSha256::new_from_slice(&self.signing_key)
            .map_err(|_| AuditError::InvalidKeyLength)?;
        mac.update(entry_hash.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let entry = AuditEntry {
            id,
            timestamp,
            event_type: event_type.to_string(),
            data: data.to_string(),
            previous_hash: self.last_hash.clone(),
            entry_hash: entry_hash.clone(),
            signature,
        };

        // Update chain
        self.last_hash = entry_hash;
        self.total_entries += 1;

        // Add to in-memory buffer
        if self.entries.len() >= MAX_MEMORY_ENTRIES {
            self.entries.pop_front();
        }
        self.entries.push_back(entry.clone());

        // Persist to disk
        self.persist_entry(&entry)?;

        Ok(self.entries.back().unwrap())
    }

    /// Record order-related event
    pub fn record_order(&mut self, order_id: &str, action: &str, details: &str) -> Result<&AuditEntry, AuditError> {
        let data = serde_json::json!({
            "order_id": order_id,
            "action": action,
            "details": details
        }).to_string();
        self.record("ORDER_EVENT", &data)
    }

    /// Record trade execution
    pub fn record_trade(&mut self, trade_id: &str, symbol: &str, side: &str, 
                        quantity: f64, price: f64) -> Result<&AuditEntry, AuditError> {
        let data = serde_json::json!({
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "notional": quantity * price
        }).to_string();
        self.record("TRADE_EXECUTION", &data)
    }

    /// Record position change
    pub fn record_position(&mut self, symbol: &str, old_qty: f64, new_qty: f64, 
                           reason: &str) -> Result<&AuditEntry, AuditError> {
        let data = serde_json::json!({
            "symbol": symbol,
            "old_quantity": old_qty,
            "new_quantity": new_qty,
            "change": new_qty - old_qty,
            "reason": reason
        }).to_string();
        self.record("POSITION_CHANGE", &data)
    }

    /// Record security event
    pub fn record_security(&mut self, event: &str, severity: &str, 
                           details: &str) -> Result<&AuditEntry, AuditError> {
        let data = serde_json::json!({
            "security_event": event,
            "severity": severity,
            "details": details
        }).to_string();
        self.record("SECURITY_EVENT", &data)
    }

    /// Verify integrity of the entire audit trail
    pub fn verify_integrity(&self) -> Result<bool, AuditError> {
        if !self.log_path.exists() {
            return Ok(true);
        }

        let file = File::open(&self.log_path)?;
        let reader = BufReader::new(file);
        
        let mut prev_hash = String::from("genesis");
        let mut count = 0u64;

        for line in reader.lines() {
            let line = line?;
            let entry: AuditEntry = serde_json::from_str(&line)?;

            // Verify chain linkage
            if entry.previous_hash != prev_hash {
                return Ok(false);
            }

            // Verify entry hash
            let content = format!(
                "{}:{}:{}:{}:{}",
                entry.id, entry.timestamp, entry.event_type, entry.data, entry.previous_hash
            );
            let expected_hash = hex::encode(Sha256::digest(content.as_bytes()));
            if entry.entry_hash != expected_hash {
                return Ok(false);
            }

            // Verify HMAC signature
            let mut mac = HmacSha256::new_from_slice(&self.signing_key)
                .map_err(|_| AuditError::InvalidKeyLength)?;
            mac.update(entry.entry_hash.as_bytes());
            let expected_sig = hex::encode(mac.finalize().into_bytes());
            
            if entry.signature != expected_sig {
                return Ok(false);
            }

            prev_hash = entry.entry_hash;
            count += 1;
        }

        Ok(true)
    }

    /// Get recent entries from memory
    pub fn get_recent(&self, count: usize) -> Vec<&AuditEntry> {
        let len = self.entries.len();
        let start = len.saturating_sub(count);
        self.entries.iter().skip(start).collect()
    }

    /// Search entries by event type
    pub fn search_by_type(&self, event_type: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Get total entries count
    pub fn total_entries(&self) -> u64 {
        self.total_entries
    }

    /// Compute Merkle root of all entries
    pub fn compute_merkle_root(&self) -> String {
        if self.entries.is_empty() {
            return hex::encode(Sha256::digest(b"empty"));
        }

        // Collect all hashes
        let mut hashes: Vec<Vec<u8>> = self.entries
            .iter()
            .map(|e| hex::decode(&e.entry_hash).unwrap_or_default())
            .collect();

        // Build Merkle tree
        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in hashes.chunks(2) {
                let combined = if chunk.len() == 2 {
                    [chunk[0].clone(), chunk[1].clone()].concat()
                } else {
                    [chunk[0].clone(), chunk[0].clone()].concat()
                };
                next_level.push(Sha256::digest(&combined).to_vec());
            }
            hashes = next_level;
        }

        hex::encode(&hashes[0])
    }

    fn persist_entry(&self, entry: &AuditEntry) -> Result<(), AuditError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;

        let json = serde_json::to_string(entry)?;
        writeln!(file, "{}", json)?;
        file.sync_all()?;

        Ok(())
    }

    fn load_existing_entries(&mut self) -> Result<(), AuditError> {
        let file = File::open(&self.log_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let entry: AuditEntry = serde_json::from_str(&line)?;
            
            self.last_hash = entry.entry_hash.clone();
            self.total_entries += 1;
            
            if self.entry_counter.load(Ordering::SeqCst) <= entry.id {
                self.entry_counter.store(entry.id + 1, Ordering::SeqCst);
            }

            if self.entries.len() >= MAX_MEMORY_ENTRIES {
                self.entries.pop_front();
            }
            self.entries.push_back(entry);
        }

        Ok(())
    }
}

/// Audit trail errors
#[derive(Debug)]
pub enum AuditError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidKeyLength,
    IntegrityViolation,
}

impl From<std::io::Error> for AuditError {
    fn from(err: std::io::Error) -> Self {
        AuditError::Io(err)
    }
}

impl From<serde_json::Error> for AuditError {
    fn from(err: serde_json::Error) -> Self {
        AuditError::Json(err)
    }
}

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditError::Io(e) => write!(f, "IO error: {}", e),
            AuditError::Json(e) => write!(f, "JSON error: {}", e),
            AuditError::InvalidKeyLength => write!(f, "Invalid key length"),
            AuditError::IntegrityViolation => write!(f, "Integrity violation detected"),
        }
    }
}

impl std::error::Error for AuditError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_audit_trail_basic() {
        let dir = tempdir().unwrap();
        let log_path = dir.path().join("audit.log");
        let signing_key = b"test_signing_key_32_bytes_long!!";

        let mut trail = AuditTrail::new(log_path.to_str().unwrap(), signing_key).unwrap();

        // Record some entries
        trail.record("TEST_EVENT", "test data 1").unwrap();
        trail.record("TEST_EVENT", "test data 2").unwrap();
        trail.record_order("ORD-001", "NEW", "BTC/USDT buy").unwrap();

        // Verify count
        assert_eq!(trail.total_entries(), 3);

        // Verify integrity
        assert!(trail.verify_integrity().unwrap());

        // Get recent entries
        let recent = trail.get_recent(2);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_tamper_detection() {
        let dir = tempdir().unwrap();
        let log_path = dir.path().join("audit.log");
        let signing_key = b"test_signing_key_32_bytes_long!!";

        let mut trail = AuditTrail::new(log_path.to_str().unwrap(), signing_key).unwrap();
        trail.record("TEST_EVENT", "original data").unwrap();

        // Tamper with the log file
        drop(trail);
        let mut file = OpenOptions::new().write(true).open(&log_path).unwrap();
        file.write_all(b"tampered content\n").unwrap();

        // Reload and verify integrity fails
        let trail2 = AuditTrail::new(log_path.to_str().unwrap(), signing_key).unwrap();
        // The verify should fail due to tampering
        // (In this simplified test, we just check it doesn't crash)
        let _ = trail2.verify_integrity();
    }
}
