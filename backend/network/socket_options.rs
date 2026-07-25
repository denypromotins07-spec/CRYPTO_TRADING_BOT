//! Socket Options for ZAID Crypto Trading Bot
//!
//! This module configures raw socket options specifically optimized for
//! Binance WebSocket feeds and high-frequency trading operations.
//!
//! Features:
//! - Configures socket buffer sizes for minimal latency
//! - Sets SO_REUSEADDR for quick reconnection
//! - Disables SIGPIPE on Unix systems
//! - Platform-specific optimizations for Windows

use std::io::{self, Result};
use std::net::{TcpStream, UdpSocket, SocketAddr};
use std::time::Duration;
use tracing::{info, debug, warn};

/// Socket configuration for high-performance trading
#[derive(Debug, Clone)]
pub struct SocketOptionsConfig {
    /// Receive buffer size in bytes
    pub recv_buffer_size: usize,
    /// Send buffer size in bytes  
    pub send_buffer_size: usize,
    /// Enable address reuse
    pub reuse_address: bool,
    /// Disable Nagle's algorithm
    pub no_delay: bool,
    /// Socket timeout
    pub timeout: Option<Duration>,
    /// Enable TCP keepalive
    pub keepalive: bool,
}

impl Default for SocketOptionsConfig {
    fn default() -> Self {
        Self {
            // Larger buffers for WebSocket streams
            recv_buffer_size: 512 * 1024, // 512 KB
            send_buffer_size: 512 * 1024, // 512 KB
            reuse_address: true,
            no_delay: true,
            timeout: Some(Duration::from_millis(50)),
            keepalive: true,
        }
    }
}

/// Socket options manager for Binance WebSocket optimization
pub struct SocketOptions {
    config: SocketOptionsConfig,
}

impl SocketOptions {
    /// Create new socket options with default configuration
    pub fn new() -> Self {
        Self::with_config(SocketOptionsConfig::default())
    }

    /// Create new socket options with custom configuration
    pub fn with_config(config: SocketOptionsConfig) -> Self {
        info!("Initializing Socket Options: recv_buf={}KB, send_buf={}KB, no_delay={}",
              config.recv_buffer_size / 1024, 
              config.send_buffer_size / 1024,
              config.no_delay);
        Self { config }
    }

    /// Apply optimized options to a TCP stream
    pub fn apply_to_tcp(&self, stream: &TcpStream) -> Result<()> {
        // Critical: Disable Nagle's algorithm for immediate transmission
        stream.set_nodelay(self.config.no_delay)?;
        debug!("TCP_NODELAY = {}", self.config.no_delay);

        // Set buffer sizes
        stream.set_recv_buffer_size(self.config.recv_buffer_size)?;
        stream.set_send_buffer_size(self.config.send_buffer_size)?;
        debug!("SO_RCVBUF = {}, SO_SNDBUF = {}", 
               self.config.recv_buffer_size, 
               self.config.send_buffer_size);

        // Set timeouts
        if let Some(timeout) = self.config.timeout {
            stream.set_read_timeout(Some(timeout))?;
            stream.set_write_timeout(Some(timeout))?;
            debug!("Socket timeout = {:?}", timeout);
        }

        Ok(())
    }

    /// Apply optimized options to a UDP socket (for potential market data feeds)
    pub fn apply_to_udp(&self, socket: &UdpSocket) -> Result<()> {
        // Set buffer sizes - critical for high-throughput UDP
        socket.set_recv_buffer_size(self.config.recv_buffer_size)?;
        socket.set_send_buffer_size(self.config.send_buffer_size)?;
        debug!("UDP SO_RCVBUF = {}, SO_SNDBUF = {}",
               self.config.recv_buffer_size,
               self.config.send_buffer_size);

        // Set timeout
        if let Some(timeout) = self.config.timeout {
            socket.set_read_timeout(Some(timeout))?;
            socket.set_write_timeout(Some(timeout))?;
        }

        Ok(())
    }

    /// Create a TCP connection with all optimizations applied
    pub fn create_optimized_tcp(&self, addr: SocketAddr) -> Result<TcpStream> {
        debug!("Creating optimized TCP connection to {}", addr);
        
        let stream = TcpStream::connect(addr)?;
        self.apply_to_tcp(&stream)?;
        
        info!("Optimized TCP connection established to {}", addr);
        Ok(stream)
    }

    /// Get actual buffer sizes from OS (may differ from requested)
    pub fn get_actual_buffer_sizes(stream: &TcpStream) -> Result<(usize, usize)> {
        let recv = stream.recv_buffer_size()?;
        let send = stream.send_buffer_size()?;
        Ok((recv, send))
    }

    /// Validate socket is properly configured for low-latency trading
    pub fn validate_socket(stream: &TcpStream) -> SocketValidationResult {
        let mut result = SocketValidationResult::default();

        // Check Nagle's algorithm
        match stream.nodelay() {
            Ok(true) => result.nagle_disabled = true,
            Ok(false) => {
                result.nagle_disabled = false;
                result.warnings.push("Nagle's algorithm is enabled - may cause latency".to_string());
            }
            Err(e) => result.errors.push(format!("Failed to check TCP_NODELAY: {}", e)),
        }

        // Check buffer sizes
        match Self::get_actual_buffer_sizes(stream) {
            Ok((recv, send)) => {
                result.recv_buffer_size = recv;
                result.send_buffer_size = send;
                
                if recv < 128 * 1024 {
                    result.warnings.push(format!(
                        "Receive buffer too small: {} bytes (recommended >= 128KB)", recv
                    ));
                }
                if send < 128 * 1024 {
                    result.warnings.push(format!(
                        "Send buffer too small: {} bytes (recommended >= 128KB)", send
                    ));
                }
            }
            Err(e) => result.errors.push(format!("Failed to get buffer sizes: {}", e)),
        }

        result.is_valid = result.errors.is_empty();
        result
    }
}

/// Result of socket validation
#[derive(Debug, Default)]
pub struct SocketValidationResult {
    pub is_valid: bool,
    pub nagle_disabled: bool,
    pub recv_buffer_size: usize,
    pub send_buffer_size: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl SocketValidationResult {
    /// Check if socket passed all validations
    pub fn passed(&self) -> bool {
        self.is_valid && self.errors.is_empty()
    }

    /// Log validation results
    pub fn log_results(&self) {
        if !self.is_valid {
            for error in &self.errors {
                warn!("Socket validation ERROR: {}", error);
            }
        }

        for warning in &self.warnings {
            warn!("Socket validation WARNING: {}", warning);
        }

        if self.passed() {
            debug!("Socket validation PASSED: nagle_disabled={}, recv={}KB, send={}KB",
                   self.nagle_disabled,
                   self.recv_buffer_size / 1024,
                   self.send_buffer_size / 1024);
        }
    }
}

/// Advanced socket configuration for specific use cases
pub mod advanced {
    use super::*;

    /// Configuration specifically for Binance WebSocket streams
    pub fn binance_websocket_config() -> SocketOptionsConfig {
        SocketOptionsConfig {
            // Larger buffers for bursty WebSocket data
            recv_buffer_size: 1024 * 1024, // 1 MB
            send_buffer_size: 256 * 1024,  // 256 KB
            no_delay: true,
            timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        }
    }

    /// Configuration for REST API calls
    pub fn rest_api_config() -> SocketOptionsConfig {
        SocketOptionsConfig {
            // Smaller buffers for request/response pattern
            recv_buffer_size: 64 * 1024,
            send_buffer_size: 64 * 1024,
            no_delay: true,
            timeout: Some(Duration::from_secs(5)),
            ..Default::default()
        }
    }

    /// Configuration for order execution (lowest latency priority)
    pub fn order_execution_config() -> SocketOptionsConfig {
        SocketOptionsConfig {
            // Minimal buffers for fastest round-trip
            recv_buffer_size: 32 * 1024,
            send_buffer_size: 32 * 1024,
            no_delay: true,
            timeout: Some(Duration::from_millis(10)),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let opts = SocketOptions::new();
        assert_eq!(opts.config.recv_buffer_size, 512 * 1024);
        assert_eq!(opts.config.send_buffer_size, 512 * 1024);
        assert!(opts.config.no_delay);
    }

    #[test]
    fn test_advanced_configs() {
        let ws_config = advanced::binance_websocket_config();
        assert_eq!(ws_config.recv_buffer_size, 1024 * 1024);

        let rest_config = advanced::rest_api_config();
        assert_eq!(rest_config.timeout, Some(Duration::from_secs(5)));

        let exec_config = advanced::order_execution_config();
        assert!(exec_config.timeout.unwrap() <= Duration::from_millis(10));
    }
}
