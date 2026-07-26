//! Prometheus Metrics Exporter with Zero-Copy HTTP Server
//!
//! This module implements a lightweight Prometheus metrics exporter that
//! serves metrics via HTTP without interfering with the main Nautilus
//! execution loop.
//!
//! Key Features:
//! - Zero-copy metric serialization where possible
//! - Non-blocking HTTP server using Tokio
//! - Memory-efficient metric storage
//! - Automatic metric registration and collection
//! - Thread-safe metric updates
//!
//! Designed for the ZAID Personal Crypto Trading Bot to provide
//! observability without impacting trading performance.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::net::SocketAddr;
use tokio::sync::RwLock;
use tokio::net::TcpListener;
use tokio::io::{AsyncWriteExt, AsyncBufReadExt, BufReader};
use serde::{Serialize, Deserialize};

/// Metric types supported by the exporter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

impl MetricType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
        }
    }
}

/// A single metric value with labels
#[derive(Debug, Clone)]
pub struct MetricValue {
    pub value: f64,
    pub labels: Vec<(String, String)>,
    pub timestamp_ms: u64,
}

impl MetricValue {
    pub fn new(value: f64) -> Self {
        MetricValue {
            value,
            labels: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.push((key.to_string(), value.to_string()));
        self
    }

    /// Format labels for Prometheus output
    pub fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            return String::new();
        }
        let formatted: Vec<String> = self.labels
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect();
        format!("{{{}}}", formatted.join(","))
    }
}

/// Metric definition
#[derive(Debug)]
pub struct Metric {
    pub name: String,
    pub help: String,
    pub metric_type: MetricType,
    pub values: RwLock<Vec<MetricValue>>,
}

impl Metric {
    pub fn new(name: &str, help: &str, metric_type: MetricType) -> Self {
        Metric {
            name: name.to_string(),
            help: help.to_string(),
            metric_type,
            values: RwLock::new(Vec::new()),
        }
    }

    /// Add or update a metric value
    pub async fn record(&self, value: f64, labels: Vec<(String, String)>) {
        let mut values = self.values.write().await;
        
        // Check if labels already exist
        let existing = values.iter_mut().find(|v| v.labels == labels);
        
        if let Some(existing_value) = existing {
            existing_value.value = value;
            existing_value.timestamp_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
        } else {
            let mut new_value = MetricValue::new(value);
            new_value.labels = labels;
            values.push(new_value);
        }
    }

    /// Increment a counter
    pub async fn increment(&self, delta: f64) {
        let mut values = self.values.write().await;
        if values.is_empty() {
            values.push(MetricValue::new(delta));
        } else {
            values[0].value += delta;
        }
    }

    /// Format metric for Prometheus output
    pub async fn format_prometheus(&self) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("# HELP {} {}\n", self.name, self.help));
        output.push_str(&format!("# TYPE {} {}\n", self.name, self.metric_type.as_str()));
        
        let values = self.values.read().await;
        for value in values.iter() {
            let labels = value.format_labels();
            output.push_str(&format!("{}{} {} {}\n", self.name, labels, value.value, value.timestamp_ms));
        }
        
        output
    }
}

/// Metrics registry containing all registered metrics
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    metrics: RwLock<Vec<Arc<Metric>>>,
    start_time: u64,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        MetricsRegistry {
            metrics: RwLock::new(Vec::new()),
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Register a new metric
    pub async fn register(&self, metric: Arc<Metric>) {
        let mut metrics = self.metrics.write().await;
        metrics.push(metric);
    }

    /// Get or create a counter metric
    pub async fn counter(&self, name: &str, help: &str) -> Arc<Metric> {
        let metrics = self.metrics.read().await;
        if let Some(existing) = metrics.iter().find(|m| m.name == name) {
            return Arc::clone(existing);
        }
        drop(metrics);

        let metric = Arc::new(Metric::new(name, help, MetricType::Counter));
        self.register(Arc::clone(&metric)).await;
        metric
    }

    /// Get or create a gauge metric
    pub async fn gauge(&self, name: &str, help: &str) -> Arc<Metric> {
        let metrics = self.metrics.read().await;
        if let Some(existing) = metrics.iter().find(|m| m.name == name) {
            return Arc::clone(existing);
        }
        drop(metrics);

        let metric = Arc::new(Metric::new(name, help, MetricType::Gauge));
        self.register(Arc::clone(&metric)).await;
        metric
    }

    /// Get or create a histogram metric
    pub async fn histogram(&self, name: &str, help: &str) -> Arc<Metric> {
        let metrics = self.metrics.read().await;
        if let Some(existing) = metrics.iter().find(|m| m.name == name) {
            return Arc::clone(existing);
        }
        drop(metrics);

        let metric = Arc::new(Metric::new(name, help, MetricType::Histogram));
        self.register(Arc::clone(&metric)).await;
        metric
    }

    /// Export all metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let mut output = String::with_capacity(8192);
        
        // Add uptime metric
        let uptime = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() - self.start_time;
        
        output.push_str("# HELP bot_uptime_seconds Bot uptime in seconds\n");
        output.push_str("# TYPE bot_uptime_seconds counter\n");
        output.push_str(&format!("bot_uptime_seconds {}\n\n", uptime));

        // Export all registered metrics
        let metrics = self.metrics.read().await;
        for metric in metrics.iter() {
            output.push_str(&metric.format_prometheus().await);
            output.push('\n');
        }

        output
    }

    /// Get metric count
    pub async fn metric_count(&self) -> usize {
        self.metrics.read().await.len()
    }
}

/// Prometheus exporter server configuration
#[derive(Debug, Clone)]
pub struct ExporterConfig {
    pub host: String,
    pub port: u16,
    pub enabled: bool,
}

impl Default for ExporterConfig {
    fn default() -> Self {
        ExporterConfig {
            host: "0.0.0.0".to_string(),
            port: 9090,
            enabled: true,
        }
    }
}

/// Statistics for the exporter
#[derive(Debug, Default)]
pub struct ExporterStats {
    pub total_scrapes: AtomicU64,
    pub total_errors: AtomicU64,
    pub last_scrape_time: AtomicU64,
    pub avg_response_size: AtomicU64,
}

impl ExporterStats {
    pub fn record_scrape(&self, response_size: usize) {
        self.total_scrapes.fetch_add(1, Ordering::Relaxed);
        self.last_scrape_time.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed,
        );
        
        // Simple moving average
        let current_avg = self.avg_response_size.load(Ordering::Relaxed);
        let new_avg = ((current_avg * 9) + (response_size as u64)) / 10;
        self.avg_response_size.store(new_avg, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "total_scrapes": self.total_scrapes.load(Ordering::Relaxed),
            "total_errors": self.total_errors.load(Ordering::Relaxed),
            "last_scrape_time": self.last_scrape_time.load(Ordering::Relaxed),
            "avg_response_size": self.avg_response_size.load(Ordering::Relaxed),
        })
    }
}

/// Prometheus metrics exporter with HTTP server
pub struct PrometheusExporter {
    config: ExporterConfig,
    registry: Arc<MetricsRegistry>,
    stats: Arc<ExporterStats>,
    shutdown: Arc<AtomicBool>,
    server_handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(config: ExporterConfig, registry: Arc<MetricsRegistry>) -> Self {
        PrometheusExporter {
            config,
            registry,
            stats: Arc::new(ExporterStats::default()),
            shutdown: Arc::new(AtomicBool::new(false)),
            server_handle: tokio::sync::Mutex::new(None),
        }
    }

    /// Start the HTTP server
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.config.enabled {
            return Ok(());
        }

        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        let listener = TcpListener::bind(addr).await?;
        println!("Prometheus exporter listening on http://{}", addr);

        let shutdown = Arc::clone(&self.shutdown);
        let registry = Arc::clone(&self.registry);
        let stats = Arc::clone(&self.stats);

        let handle = tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                let (mut socket, _) = tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok(conn) => conn,
                            Err(_) => continue,
                        }
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                        continue;
                    }
                };

                let registry = Arc::clone(&registry);
                let stats = Arc::clone(&stats);

                tokio::spawn(async move {
                    let (reader, mut writer) = socket.split();
                    let mut reader = BufReader::new(reader);

                    // Read HTTP request
                    let mut request = String::new();
                    if reader.read_line(&mut request).await.is_ok() {
                        if request.starts_with("GET /metrics") {
                            // Generate Prometheus metrics
                            let metrics = registry.export_prometheus().await;
                            
                            // Build HTTP response
                            let response = format!(
                                "HTTP/1.1 200 OK\r\n\
                                 Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n\
                                 Content-Length: {}\r\n\
                                 Connection: close\r\n\
                                 \r\n\
                                 {}",
                                metrics.len(),
                                metrics
                            );

                            if writer.write_all(response.as_bytes()).await.is_ok() {
                                stats.record_scrape(metrics.len());
                            } else {
                                stats.record_error();
                            }
                        } else if request.starts_with("GET /health") {
                            // Health check endpoint
                            let response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                            let _ = writer.write_all(response.as_bytes()).await;
                        } else if request.starts_with("GET /stats") {
                            // Exporter statistics endpoint
                            let stats_json = serde_json::to_string_pretty(&stats.to_dict()).unwrap_or_default();
                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                stats_json.len(),
                                stats_json
                            );
                            let _ = writer.write_all(response.as_bytes()).await;
                        } else {
                            // 404 for other paths
                            let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found";
                            let _ = writer.write_all(response.as_bytes()).await;
                        }
                    }
                    
                    let _ = writer.flush().await;
                });
            }
        });

        *self.server_handle.lock().await = Some(handle);
        Ok(())
    }

    /// Stop the HTTP server
    pub async fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        
        if let Some(handle) = self.server_handle.lock().await.take() {
            handle.abort();
        }
    }

    /// Get exporter statistics
    pub fn get_stats(&self) -> serde_json::Value {
        self.stats.to_dict()
    }

    /// Get the metrics registry
    pub fn registry(&self) -> Arc<MetricsRegistry> {
        Arc::clone(&self.registry)
    }

    /// Check if exporter is running
    pub fn is_running(&self) -> bool {
        !self.shutdown.load(Ordering::Relaxed)
    }
}

impl Drop for PrometheusExporter {
    fn drop(&mut self) {
        // Note: Can't await in drop, so we just set shutdown flag
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_registry() {
        let registry = Arc::new(MetricsRegistry::new());
        
        // Create counter
        let counter = registry.counter("test_counter", "Test counter help").await;
        counter.increment(1.0).await;
        counter.increment(2.5).await;
        
        // Create gauge
        let gauge = registry.gauge("test_gauge", "Test gauge help").await;
        gauge.record(42.0, vec![]).await;
        
        // Export
        let output = registry.export_prometheus().await;
        
        assert!(output.contains("test_counter"));
        assert!(output.contains("test_gauge"));
        assert!(output.contains("3.5")); // Counter value
        assert!(output.contains("42")); // Gauge value
    }

    #[tokio::test]
    async fn test_metric_with_labels() {
        let registry = Arc::new(MetricsRegistry::new());
        
        let gauge = registry.gauge("labeled_gauge", "Gauge with labels").await;
        gauge.record(100.0, vec![("host".to_string(), "server1".to_string())]).await;
        gauge.record(200.0, vec![("host".to_string(), "server2".to_string())]).await;
        
        let output = gauge.format_prometheus().await;
        
        assert!(output.contains("host=\"server1\""));
        assert!(output.contains("host=\"server2\""));
        assert!(output.contains("100"));
        assert!(output.contains("200"));
    }

    #[test]
    fn test_metric_value_formatting() {
        let value = MetricValue::new(42.0)
            .with_label("method", "GET")
            .with_label("status", "200");
        
        let labels = value.format_labels();
        assert!(labels.contains("method=\"GET\""));
        assert!(labels.contains("status=\"200\""));
    }
}
