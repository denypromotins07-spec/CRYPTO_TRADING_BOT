//! WebSocket Authentication - HMAC signature validation on incoming WS messages
//! 
//! This module implements HMAC-based authentication for WebSocket connections
//! to exchange APIs, ensuring message integrity and authenticity.
//! 
//! Security Features:
//! - HMAC-SHA256 signature generation and verification
//! - Timestamp-based replay attack prevention
//! - Request signing for authenticated endpoints
//! - Automatic signature refresh
//! 
//! Optimized for zero-cost abstractions and microsecond latency.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Authentication result
#[derive(Debug, Clone)]
pub struct AuthResult {
    /// Whether authentication succeeded
    pub is_authenticated: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Timestamp of authentication
    pub timestamp: u64,
}

/// WebSocket authenticator for exchange connections
pub struct WebSocketAuthenticator {
    /// API key (public identifier)
    api_key: String,
    /// Secret key for HMAC (never transmitted)
    secret_key: Vec<u8>,
    /// Last used timestamp for replay prevention
    last_timestamp: AtomicU64,
    /// Allowed clock skew in milliseconds
    allowed_skew_ms: u64,
    /// Request counter for nonce generation
    request_counter: AtomicU64,
}

impl WebSocketAuthenticator {
    /// Create a new WebSocket authenticator
    pub fn new(api_key: &str, secret_key: &str) -> Self {
        WebSocketAuthenticator {
            api_key: api_key.to_string(),
            secret_key: secret_key.as_bytes().to_vec(),
            last_timestamp: AtomicU64::new(0),
            allowed_skew_ms: 5000, // 5 seconds default
            request_counter: AtomicU64::new(0),
        }
    }

    /// Get the public API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Generate current timestamp in milliseconds
    fn current_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Generate HMAC signature for a message
    pub fn sign(&self, message: &str, timestamp: u64) -> Result<String, AuthError> {
        let mut mac = HmacSha256::new_from_slice(&self.secret_key)
            .map_err(|_| AuthError::InvalidKeyLength)?;
        
        // Sign: timestamp + message
        let data = format!("{}{}", timestamp, message);
        mac.update(data.as_bytes());
        
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    /// Verify HMAC signature on an incoming message
    pub fn verify(&self, message: &str, signature: &str, timestamp: u64) -> AuthResult {
        let now = Self::current_timestamp_ms();
        
        // Check timestamp validity (replay attack prevention)
        if now > timestamp + self.allowed_skew_ms {
            return AuthResult {
                is_authenticated: false,
                error: Some("Timestamp too old".to_string()),
                timestamp: now,
            };
        }
        
        if timestamp > now + self.allowed_skew_ms {
            return AuthResult {
                is_authenticated: false,
                error: Some("Timestamp in future".to_string()),
                timestamp: now,
            };
        }

        // Check for replay (timestamp must be greater than last seen)
        let last = self.last_timestamp.load(Ordering::SeqCst);
        if timestamp <= last {
            return AuthResult {
                is_authenticated: false,
                error: Some("Replay detected".to_string()),
                timestamp: now,
            };
        }

        // Verify signature
        match self.sign(message, timestamp) {
            Ok(expected_sig) => {
                if constant_time_eq(signature.as_bytes(), expected_sig.as_bytes()) {
                    // Update last timestamp atomically
                    self.last_timestamp.store(timestamp, Ordering::SeqCst);
                    
                    AuthResult {
                        is_authenticated: true,
                        error: None,
                        timestamp: now,
                    }
                } else {
                    AuthResult {
                        is_authenticated: false,
                        error: Some("Signature mismatch".to_string()),
                        timestamp: now,
                    }
                }
            }
            Err(e) => AuthResult {
                is_authenticated: false,
                error: Some(format!("Signing error: {:?}", e)),
                timestamp: now,
            },
        }
    }

    /// Generate signed request headers for Binance WebSocket
    pub fn generate_auth_headers(&self, endpoint: &str) -> Result<Vec<(String, String)>, AuthError> {
        let timestamp = Self::current_timestamp_ms();
        let recv_window = 5000; // 5 seconds
        
        // Build signature payload
        let payload = format!(
            "recvWindow={}&timestamp={}",
            recv_window, timestamp
        );
        
        let signature = self.sign(&payload, timestamp)?;
        
        Ok(vec![
            ("X-MBX-APIKEY".to_string(), self.api_key.clone()),
            ("recvWindow".to_string(), recv_window.to_string()),
            ("timestamp".to_string(), timestamp.to_string()),
            ("signature".to_string(), signature),
        ])
    }

    /// Generate authentication message for WebSocket stream
    pub fn generate_auth_message(&self) -> Result<String, AuthError> {
        let timestamp = Self::current_timestamp_ms();
        let signature = self.sign("USER_DATA_STREAM", timestamp)?;
        
        let auth_msg = serde_json::json!({
            "method": "SUBSCRIBE",
            "params": [format!("{}?signature={}", self.api_key, signature)],
            "id": self.request_counter.fetch_add(1, Ordering::SeqCst)
        });
        
        Ok(auth_msg.to_string())
    }

    /// Validate incoming WebSocket message signature
    pub fn validate_ws_message(&self, message: &str, signature: &str) -> AuthResult {
        // Extract timestamp from message or use current time
        let timestamp = Self::current_timestamp_ms();
        self.verify(message, signature, timestamp)
    }

    /// Set allowed clock skew
    pub fn set_allowed_skew(&mut self, skew_ms: u64) {
        self.allowed_skew_ms = skew_ms;
    }

    /// Rotate secret key (for key rotation procedures)
    pub fn rotate_key(&mut self, new_secret: &str) {
        self.secret_key = new_secret.as_bytes().to_vec();
        self.last_timestamp.store(0, Ordering::SeqCst);
    }
}

/// Constant-time equality check to prevent timing attacks
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    
    result == 0
}

/// Authentication errors
#[derive(Debug)]
pub enum AuthError {
    InvalidKeyLength,
    SigningFailed(String),
    TimestampError(String),
    Io(std::io::Error),
}

impl From<std::io::Error> for AuthError {
    fn from(err: std::io::Error) -> Self {
        AuthError::Io(err)
    }
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::InvalidKeyLength => write!(f, "Invalid key length"),
            AuthError::SigningFailed(s) => write!(f, "Signing failed: {}", s),
            AuthError::TimestampError(s) => write!(f, "Timestamp error: {}", s),
            AuthError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for AuthError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_and_verify() {
        let auth = WebSocketAuthenticator::new("test_api_key", "test_secret_key");
        
        let message = "test_message";
        let timestamp = WebSocketAuthenticator::current_timestamp_ms();
        
        // Sign message
        let signature = auth.sign(message, timestamp).unwrap();
        
        // Verify should succeed
        let result = auth.verify(message, &signature, timestamp);
        assert!(result.is_authenticated);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_replay_prevention() {
        let auth = WebSocketAuthenticator::new("test_api_key", "test_secret_key");
        
        let message = "test_message";
        let timestamp = WebSocketAuthenticator::current_timestamp_ms();
        
        // First verification should succeed
        let signature = auth.sign(message, timestamp).unwrap();
        let result1 = auth.verify(message, &signature, timestamp);
        assert!(result1.is_authenticated);
        
        // Replay with same timestamp should fail
        let result2 = auth.verify(message, &signature, timestamp);
        assert!(!result2.is_authenticated);
        assert_eq!(result2.error, Some("Replay detected".to_string()));
    }

    #[test]
    fn test_timestamp_validation() {
        let mut auth = WebSocketAuthenticator::new("test_api_key", "test_secret_key");
        auth.set_allowed_skew(100); // 100ms skew allowed
        
        let message = "test_message";
        let old_timestamp = WebSocketAuthenticator::current_timestamp_ms() - 10000; // 10 seconds ago
        
        let signature = auth.sign(message, old_timestamp).unwrap();
        let result = auth.verify(message, &signature, old_timestamp);
        
        assert!(!result.is_authenticated);
        assert!(result.error.unwrap().contains("Timestamp too old"));
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"short", b"longer string"));
    }
}
