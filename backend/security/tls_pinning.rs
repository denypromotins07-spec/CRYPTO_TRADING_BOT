//! TLS Certificate Pinning - Hardcoded certificate fingerprints to prevent MITM
//! 
//! This module implements TLS certificate pinning to prevent man-in-the-middle
//! attacks by validating server certificates against known, hardcoded fingerprints.
//! 
//! Security Features:
//! - SHA-256 fingerprint validation of server certificates
//! - Hardcoded pins for Binance and other exchange APIs
//! - Immediate connection termination on pin mismatch
//! - Support for backup pins for certificate rotation
//! 
//! Optimized for zero-cost abstractions and minimal latency impact.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Certificate pinning result
#[derive(Debug, Clone)]
pub struct PinValidationResult {
    /// Hostname that was validated
    pub hostname: String,
    /// Whether validation succeeded
    pub is_valid: bool,
    /// Fingerprint that was checked
    pub expected_fingerprint: String,
    /// Actual fingerprint received
    pub actual_fingerprint: String,
    /// Time taken for validation
    pub validation_duration_ms: u64,
}

/// Certificate pin entry with support for rotation
#[derive(Debug, Clone)]
pub struct CertificatePin {
    /// Primary fingerprint (current certificate)
    pub primary: String,
    /// Backup fingerprint (for rotation)
    pub backup: Option<String>,
    /// Expiration timestamp (Unix epoch)
    pub expires_at: u64,
    /// Whether this pin is currently active
    pub is_active: AtomicBool,
}

impl CertificatePin {
    /// Check if pin is expired
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now > self.expires_at
    }

    /// Validate a fingerprint against this pin
    pub fn validate(&self, fingerprint: &str) -> bool {
        if !self.is_active.load(Ordering::SeqCst) {
            return false;
        }

        if self.is_expired() {
            self.is_active.store(false, Ordering::SeqCst);
            return false;
        }

        fingerprint == self.primary || 
        self.backup.as_ref().map_or(false, |b| fingerprint == b)
    }
}

/// TlsPinner manages certificate pinning for exchange connections
pub struct TlsPinner {
    /// Pinned certificates by hostname
    pinned_certs: HashMap<String, CertificatePin>,
    /// Whether to fail closed on unknown hosts
    fail_closed: bool,
    /// Validation statistics
    validations_count: u64,
    failures_count: u64,
}

impl TlsPinner {
    /// Create a new TLS pinner with default exchange pins
    pub fn new() -> Self {
        let mut pinned_certs = HashMap::new();

        // Binance API certificates (example fingerprints - update with real values)
        // These are SHA-256 fingerprints of the Subject Public Key Info (SPKI)
        pinned_certs.insert(
            "api.binance.com".to_string(),
            CertificatePin {
                primary: "PLACEHOLDER_BINANCE_PRIMARY_FINGERPRINT".to_string(),
                backup: Some("PLACEHOLDER_BINANCE_BACKUP_FINGERPRINT".to_string()),
                expires_at: 1735689600, // Example: Jan 1, 2025
                is_active: AtomicBool::new(true),
            }
        );

        pinned_certs.insert(
            "testnet.binance.vision".to_string(),
            CertificatePin {
                primary: "PLACEHOLDER_TESTNET_PRIMARY_FINGERPRINT".to_string(),
                backup: None,
                expires_at: 1735689600,
                is_active: AtomicBool::new(true),
            }
        );

        // Add more exchanges as needed
        // Coinbase, Kraken, etc.

        TlsPinner {
            pinned_certs,
            fail_closed: true,
            validations_count: 0,
            failures_count: 0,
        }
    }

    /// Enable fail-closed mode (reject unknown hosts)
    pub fn set_fail_closed(&mut self, enabled: bool) {
        self.fail_closed = enabled;
    }

    /// Add a certificate pin for a hostname
    pub fn add_pin(
        &mut self,
        hostname: &str,
        primary_fingerprint: &str,
        backup_fingerprint: Option<&str>,
        expires_at: u64,
    ) {
        self.pinned_certs.insert(
            hostname.to_string(),
            CertificatePin {
                primary: primary_fingerprint.to_string(),
                backup: backup_fingerprint.map(String::from),
                expires_at,
                is_active: AtomicBool::new(true),
            }
        );
    }

    /// Compute SHA-256 fingerprint from DER-encoded certificate
    pub fn compute_fingerprint(&self, der_data: &[u8]) -> String {
        let hash = Sha256::digest(der_data);
        hex::encode(hash)
    }

    /// Validate a certificate against pinned fingerprints
    pub fn validate_certificate(
        &self,
        hostname: &str,
        der_data: &[u8],
    ) -> Result<PinValidationResult, PinError> {
        let start = Instant::now();

        let fingerprint = self.compute_fingerprint(der_data);
        self.validations_count += 1;

        let pin = self.pinned_certs.get(hostname).ok_or_else(|| {
            if self.fail_closed {
                PinError::UnknownHost(hostname.to_string())
            } else {
                // In fail-open mode, allow unknown hosts
                return Ok(PinValidationResult {
                    hostname: hostname.to_string(),
                    is_valid: true,
                    expected_fingerprint: String::new(),
                    actual_fingerprint: fingerprint.clone(),
                    validation_duration_ms: start.elapsed().as_millis() as u64,
                });
            }
        })?;

        let is_valid = pin.validate(&fingerprint);

        if !is_valid {
            self.failures_count += 1;
        }

        let expected = pin.primary.clone();

        Ok(PinValidationResult {
            hostname: hostname.to_string(),
            is_valid,
            expected_fingerprint: expected,
            actual_fingerprint: fingerprint,
            validation_duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Get validation statistics
    pub fn get_stats(&self) -> (u64, u64) {
        (self.validations_count, self.failures_count)
    }

    /// Check if any pins are expiring soon (within specified days)
    pub fn check_expiring_soon(&self, days: u32) -> Vec<String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let threshold = days as u64 * 24 * 60 * 60;

        self.pinned_certs
            .iter()
            .filter_map(|(hostname, pin)| {
                if pin.expires_at > now && pin.expires_at - now < threshold {
                    Some(hostname.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Emergency shutdown on pin failure
    pub fn emergency_shutdown(&self, hostname: &str, result: &PinValidationResult) -> ! {
        eprintln!("CRITICAL: TLS Pin Validation Failed!");
        eprintln!("Hostname: {}", hostname);
        eprintln!("Expected: {}", result.expected_fingerprint);
        eprintln!("Received: {}", result.actual_fingerprint);
        eprintln!("Possible MITM attack detected!");
        eprintln!("Terminating connection immediately...");

        std::process::exit(1);
    }
}

impl Default for TlsPinner {
    fn default() -> Self {
        Self::new()
    }
}

/// Pin validation errors
#[derive(Debug)]
pub enum PinError {
    UnknownHost(String),
    InvalidCertificate(String),
    ExpiredPin(String),
    Io(std::io::Error),
}

impl From<std::io::Error> for PinError {
    fn from(err: std::io::Error) -> Self {
        PinError::Io(err)
    }
}

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinError::UnknownHost(h) => write!(f, "Unknown host: {}", h),
            PinError::InvalidCertificate(c) => write!(f, "Invalid certificate: {}", c),
            PinError::ExpiredPin(p) => write!(f, "Expired pin: {}", p),
            PinError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for PinError {}

/// Helper to extract certificate from TLS stream (integration point)
/// This would be called during TLS handshake in a real implementation
pub fn extract_cert_from_stream(stream: &impl std::io::Read) -> Result<Vec<u8>, PinError> {
    // In production, this would integrate with rustls or native-tls
    // to extract the peer certificate during handshake
    // 
    // Example integration with rustls:
    // let cert = conn.peer_certificate()?;
    // Ok(cert.der().to_vec())
    
    Err(PinError::InvalidCertificate(
        "Certificate extraction requires TLS integration".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_validation() {
        let mut pinner = TlsPinner::new();
        
        // Add test pin
        pinner.add_pin(
            "test.example.com",
            "abcd1234",
            Some("efgh5678"),
            1735689600,
        );

        // Valid primary fingerprint
        let fake_cert = b"fake certificate data";
        // In real test, we'd compute actual fingerprint
        
        // Test unknown host in fail-closed mode
        assert!(pinner.validate_certificate("unknown.com", fake_cert).is_err());
        
        // Disable fail-closed
        let mut pinner_open = TlsPinner::new();
        pinner_open.set_fail_closed(false);
        assert!(pinner_open.validate_certificate("unknown.com", fake_cert).is_ok());
    }

    #[test]
    fn test_expiration_check() {
        let pinner = TlsPinner::new();
        
        // Check for pins expiring within 30 days
        let expiring = pinner.check_expiring_soon(30);
        
        // Should return hosts expiring soon
        println!("Expiring pins: {:?}", expiring);
    }
}
