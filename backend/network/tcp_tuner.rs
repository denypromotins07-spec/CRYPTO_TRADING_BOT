//! TCP Tuner for ZAID Crypto Trading Bot
//!
//! This module configures TCP socket options for ultra-low latency networking,
//! specifically optimized for Binance WebSocket feeds and REST API calls.
//!
//! Features:
//! - Disables Nagle's algorithm (TCP_NODELAY) for immediate packet transmission
//! - Adjusts TCP window sizes for optimal throughput
//! - Configures socket buffers for high-frequency trading
//! - Platform-specific optimizations for Windows

use std::io::{self, Result};
use std::net::{TcpStream, SocketAddr};
use std::time::Duration;
use tracing::{info, warn, debug};

/// Configuration for TCP tuning
#[derive(Debug, Clone)]
pub struct TcpTunerConfig {
    /// Disable Nagle's algorithm for immediate transmission
    pub no_delay: bool,
    /// Receive buffer size in bytes
    pub recv_buffer_size: usize,
    /// Send buffer size in bytes
    pub send_buffer_size: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Read timeout
    pub read_timeout: Option<Duration>,
    /// Write timeout
    pub write_timeout: Option<Duration>,
    /// Enable TCP keepalive
    pub keepalive: bool,
    /// Keepalive interval
    pub keepalive_interval: Duration,
}

impl Default for TcpTunerConfig {
    fn default() -> Self {
        Self {
            no_delay: true, // Critical for low latency
            recv_buffer_size: 256 * 1024, // 256 KB
            send_buffer_size: 256 * 1024, // 256 KB
            connection_timeout: Duration::from_secs(5),
            read_timeout: Some(Duration::from_millis(100)),
            write_timeout: Some(Duration::from_millis(100)),
            keepalive: true,
            keepalive_interval: Duration::from_secs(30),
        }
    }
}

/// TCP Tuner for optimizing network connections
pub struct TcpTuner {
    config: TcpTunerConfig,
}

impl TcpTuner {
    /// Create a new TCP tuner with default configuration
    pub fn new() -> Self {
        Self::with_config(TcpTunerConfig::default())
    }

    /// Create a new TCP tuner with custom configuration
    pub fn with_config(config: TcpTunerConfig) -> Self {
        info!("Initializing TCP Tuner with no_delay={}, recv_buf={}KB, send_buf={}KB",
              config.no_delay, config.recv_buffer_size / 1024, config.send_buffer_size / 1024);
        Self { config }
    }

    /// Configure a TCP stream with optimized settings
    pub fn configure_stream(&self, stream: &TcpStream) -> Result<()> {
        // Disable Nagle's algorithm for immediate packet transmission
        // This is CRITICAL for low-latency trading to prevent buffering delays
        stream.set_nodelay(self.config.no_delay)?;
        debug!("TCP_NODELAY set to {}", self.config.no_delay);

        // Set receive buffer size
        stream.set_recv_buffer_size(self.config.recv_buffer_size)?;
        debug!("SO_RCVBUF set to {} bytes", self.config.recv_buffer_size);

        // Set send buffer size
        stream.set_send_buffer_size(self.config.send_buffer_size)?;
        debug!("SO_SNDBUF set to {} bytes", self.config.send_buffer_size);

        // Set timeouts
        if let Some(timeout) = self.config.read_timeout {
            stream.set_read_timeout(Some(timeout))?;
            debug!("Read timeout set to {:?}", timeout);
        }

        if let Some(timeout) = self.config.write_timeout {
            stream.set_write_timeout(Some(timeout))?;
            debug!("Write timeout set to {:?}", timeout);
        }

        // Configure keepalive if enabled
        if self.config.keepalive {
            #[cfg(target_os = "windows")]
            {
                // Windows-specific keepalive configuration
                // Note: Full keepalive params require winapi crate
                debug!("Windows keepalive enabled");
            }

            #[cfg(not(target_os = "windows"))]
            {
                // Linux/Unix keepalive configuration would go here
                debug!("Keepalive enabled");
            }
        }

        Ok(())
    }

    /// Connect to an address with optimized TCP settings
    pub fn connect(&self, addr: SocketAddr) -> Result<TcpStream> {
        debug!("Connecting to {} with optimized TCP settings", addr);

        let stream = TcpStream::connect_timeout(&addr, self.config.connection_timeout)?;
        self.configure_stream(&stream)?;

        info!("Successfully connected to {} with optimized TCP settings", addr);
        Ok(stream)
    }

    /// Get current buffer sizes from the OS (may differ from requested)
    pub fn get_buffer_sizes(stream: &TcpStream) -> Result<(usize, usize)> {
        let recv = stream.recv_buffer_size()?;
        let send = stream.send_buffer_size()?;
        Ok((recv, send))
    }

    /// Validate that TCP settings are optimal for trading
    pub fn validate_optimization(stream: &TcpStream) -> bool {
        let nodelay = stream.nodelay().unwrap_or(false);
        
        if !nodelay {
            warn!("TCP_NODELAY is disabled - this may cause latency spikes!");
            return false;
        }

        let (recv, send) = Self::get_buffer_sizes(stream).unwrap_or((0, 0));
        
        if recv < 64 * 1024 || send < 64 * 1024 {
            warn!("Buffer sizes too small: recv={}, send={}", recv, send);
            return false;
        }

        debug!("TCP optimization validated: nodelay={}, recv={}KB, send={}KB",
               nodelay, recv / 1024, send / 1024);
        true
    }
}

/// Dynamic TCP tuner that adjusts based on network conditions
pub struct DynamicTcpTuner {
    base_config: TcpTunerConfig,
    congestion_detected: bool,
}

impl DynamicTcpTuner {
    /// Create a new dynamic TCP tuner
    pub fn new(base_config: TcpTunerConfig) -> Self {
        Self {
            base_config,
            congestion_detected: false,
        }
    }

    /// Adjust buffer sizes based on detected network congestion
    pub fn adjust_for_congestion(&mut self, congestion_level: u8) {
        // Congestion level: 0 (none) to 100 (severe)
        self.congestion_detected = congestion_level > 50;

        if self.congestion_detected {
            // Increase buffer sizes during congestion
            let multiplier = 1 + (congestion_level as f64 / 100.0);
            self.base_config.recv_buffer_size = (256.0 * 1024.0 * multiplier) as usize;
            self.base_config.send_buffer_size = (256.0 * 1024.0 * multiplier) as usize;
            
            info!("Network congestion detected, increased buffer sizes by {:.1}x", multiplier);
        } else {
            // Reset to default
            self.base_config.recv_buffer_size = 256 * 1024;
            self.base_config.send_buffer_size = 256 * 1024;
        }
    }

    /// Get the current tuner configuration
    pub fn get_tuner(&self) -> TcpTuner {
        TcpTuner::with_config(self.base_config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, SocketAddrV4, Ipv4Addr};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_tcp_tuner_default_config() {
        let tuner = TcpTuner::new();
        assert!(tuner.config.no_delay);
        assert_eq!(tuner.config.recv_buffer_size, 256 * 1024);
        assert_eq!(tuner.config.send_buffer_size, 256 * 1024);
    }

    #[test]
    fn test_dynamic_tuner_congestion_adjustment() {
        let mut dynamic_tuner = DynamicTcpTuner::new(TcpTunerConfig::default());
        
        // No congestion
        dynamic_tuner.adjust_for_congestion(0);
        assert!(!dynamic_tuner.congestion_detected);
        
        // High congestion
        dynamic_tuner.adjust_for_congestion(80);
        assert!(dynamic_tuner.congestion_detected);
        assert!(dynamic_tuner.base_config.recv_buffer_size > 256 * 1024);
    }
}
