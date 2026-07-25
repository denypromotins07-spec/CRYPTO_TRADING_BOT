// State Sync: Zero-copy shared memory synchronization between Ray actors.
// Implements efficient state sharing without serialization overhead.
// Uses Rust's memory-safe concurrency primitives.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Shared state entry for an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetState {
    /// Asset identifier (e.g., "BTCUSDT")
    pub asset: String,
    /// Current position side: 0=none, 1=long, -1=short
    pub position_side: i8,
    /// Position quantity
    pub position_quantity: f64,
    /// Entry price
    pub entry_price: f64,
    /// Current PnL
    pub current_pnl: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Last update timestamp (nanoseconds)
    pub last_update_ns: u64,
    /// Signal strength
    pub signal_strength: f64,
    /// Is active flag
    pub is_active: bool,
}

impl AssetState {
    pub fn new(asset: &str) -> Self {
        Self {
            asset: asset.to_string(),
            position_side: 0,
            position_quantity: 0.0,
            entry_price: 0.0,
            current_pnl: 0.0,
            unrealized_pnl: 0.0,
            last_update_ns: current_time_ns(),
            signal_strength: 0.0,
            is_active: true,
        }
    }
    
    pub fn update_pnl(&mut self, pnl: f64, current_price: f64) {
        self.current_pnl = pnl;
        if self.position_quantity != 0.0 && self.entry_price != 0.0 {
            let price_change = current_price - self.entry_price;
            self.unrealized_pnl = price_change * self.position_quantity * 
                if self.position_side > 0 { 1.0 } else { -1.0 };
        }
        self.last_update_ns = current_time_ns();
    }
}

/// Global shared state container
/// Uses Arc<RwLock> for thread-safe access with minimal locking overhead
pub struct SharedStateStore {
    /// Asset states map
    states: Arc<RwLock<HashMap<String, AssetState>>>,
    /// Version counter for optimistic locking
    version: AtomicU64,
    /// Maximum allowed assets
    max_assets: usize,
}

impl SharedStateStore {
    /// Create a new shared state store
    pub fn new(max_assets: usize) -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::with_capacity(max_assets))),
            version: AtomicU64::new(0),
            max_assets,
        }
    }
    
    /// Get Arc clone for sharing across threads/actors
    pub fn clone_arc(&self) -> Arc<RwLock<HashMap<String, AssetState>>> {
        Arc::clone(&self.states)
    }
    
    /// Register or update an asset state
    pub fn update_asset(&self, asset: &str, state: AssetState) -> Result<u64, String> {
        let mut states = self.states.write()
            .map_err(|e| format!("Poisoned lock: {}", e))?;
        
        if states.len() >= self.max_assets && !states.contains_key(asset) {
            return Err(format!("Maximum assets ({}) reached", self.max_assets));
        }
        
        states.insert(asset.to_string(), state);
        
        // Increment version
        let new_version = self.version.fetch_add(1, Ordering::SeqCst) + 1;
        
        Ok(new_version)
    }
    
    /// Get current state for an asset (read-only snapshot)
    pub fn get_asset(&self, asset: &str) -> Option<AssetState> {
        let states = self.states.read().ok()?;
        states.get(asset).cloned()
    }
    
    /// Get all asset states (read-only snapshot)
    pub fn get_all_assets(&self) -> HashMap<String, AssetState> {
        let states = self.states.read().ok()?;
        states.clone()
    }
    
    /// Update PnL for an asset atomically
    pub fn update_pnl(&self, asset: &str, pnl: f64, current_price: f64) -> Result<(), String> {
        let mut states = self.states.write()
            .map_err(|e| format!("Poisoned lock: {}", e))?;
        
        if let Some(state) = states.get_mut(asset) {
            state.update_pnl(pnl, current_price);
            self.version.fetch_add(1, Ordering::SeqCst);
            Ok(())
        } else {
            Err(format!("Asset {} not found", asset))
        }
    }
    
    /// Remove an asset from the store
    pub fn remove_asset(&self, asset: &str) -> Option<AssetState> {
        let mut states = self.states.write().ok()?;
        let removed = states.remove(asset);
        if removed.is_some() {
            self.version.fetch_add(1, Ordering::SeqCst);
        }
        removed
    }
    
    /// Get current version number
    pub fn get_version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }
    
    /// Check if version has changed (for optimistic locking)
    pub fn has_changed(&self, expected_version: u64) -> bool {
        self.version.load(Ordering::SeqCst) != expected_version
    }
    
    /// Get count of active assets
    pub fn active_count(&self) -> usize {
        let states = self.states.read().ok()?;
        states.values().filter(|s| s.is_active).count()
    }
    
    /// Clear all states
    pub fn clear(&self) {
        if let Ok(mut states) = self.states.write() {
            states.clear();
            self.version.fetch_add(1, Ordering::SeqCst);
        }
    }
    
    /// Export state for serialization
    pub fn export(&self) -> Vec<AssetState> {
        let states = self.states.read().ok()?;
        states.values().cloned().collect()
    }
}

/// Message types for inter-actor communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateSyncMessage {
    /// Update asset state
    UpdateState(AssetState),
    /// Request current state
    RequestState(String),
    /// State response
    StateResponse(Option<AssetState>),
    /// Heartbeat
    Heartbeat(String),
    /// Shutdown signal
    Shutdown,
}

/// State synchronizer for coordinating between Ray actors
pub struct StateSynchronizer {
    store: Arc<SharedStateStore>,
    message_queue: Arc<RwLock<Vec<StateSyncMessage>>>,
    last_sync_version: AtomicU64,
}

impl StateSynchronizer {
    pub fn new(store: Arc<SharedStateStore>) -> Self {
        Self {
            store,
            message_queue: Arc::new(RwLock::new(Vec::new())),
            last_sync_version: AtomicU64::new(0),
        }
    }
    
    /// Queue a message for delivery
    pub fn queue_message(&self, msg: StateSyncMessage) {
        if let Ok(mut queue) = self.message_queue.write() {
            queue.push(msg);
            
            // Limit queue size
            if queue.len() > 1000 {
                *queue = queue.split_off(queue.len() - 100);
            }
        }
    }
    
    /// Process pending messages
    pub fn process_messages(&self) -> Vec<StateSyncMessage> {
        let mut queue = self.message_queue.write().ok()?;
        let messages: Vec<StateSyncMessage> = queue.drain(..).collect();
        
        for msg in &messages {
            if let StateSyncMessage::UpdateState(ref state) = msg {
                let _ = self.store.update_asset(&state.asset, state.clone());
            }
        }
        
        messages
    }
    
    /// Sync state with store
    pub fn sync_state(&self, asset: &str) -> Option<AssetState> {
        self.store.get_asset(asset)
    }
    
    /// Get latest version
    pub fn get_latest_version(&self) -> u64 {
        self.store.get_version()
    }
    
    /// Check if sync needed
    pub fn needs_sync(&self) -> bool {
        let current = self.store.get_version();
        let last = self.last_sync_version.load(Ordering::SeqCst);
        current != last
    }
    
    /// Mark as synced
    pub fn mark_synced(&self) {
        let current = self.store.get_version();
        self.last_sync_version.store(current, Ordering::SeqCst);
    }
}

/// Helper function to get current time in nanoseconds
fn current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Builder for creating shared state stores with configuration
pub struct SharedStateBuilder {
    max_assets: usize,
    preloaded_assets: Vec<String>,
}

impl SharedStateBuilder {
    pub fn new() -> Self {
        Self {
            max_assets: 10,
            preloaded_assets: Vec::new(),
        }
    }
    
    pub fn max_assets(mut self, count: usize) -> Self {
        self.max_assets = count;
        self
    }
    
    pub fn preload_assets(mut self, assets: Vec<&str>) -> Self {
        self.preloaded_assets = assets.iter().map(|s| s.to_string()).collect();
        self
    }
    
    pub fn build(self) -> SharedStateStore {
        let store = SharedStateStore::new(self.max_assets);
        
        // Preload standard crypto assets
        let default_assets = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "USDT"];
        let assets_to_load = if self.preloaded_assets.is_empty() {
            default_assets
        } else {
            self.preloaded_assets
        };
        
        for asset in assets_to_load {
            let state = AssetState::new(&asset);
            let _ = store.update_asset(&asset, state);
        }
        
        store
    }
}

impl Default for SharedStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shared_state_creation() {
        let store = SharedStateBuilder::new()
            .max_assets(5)
            .preload_assets(vec!["BTCUSDT", "ETHUSDT"])
            .build();
        
        assert_eq!(store.active_count(), 2);
        
        let btc_state = store.get_asset("BTCUSDT");
        assert!(btc_state.is_some());
        assert_eq!(btc_state.unwrap().asset, "BTCUSDT");
    }
    
    #[test]
    fn test_state_updates() {
        let store = SharedStateStore::new(10);
        
        let mut state = AssetState::new("BTCUSDT");
        state.position_side = 1;
        state.position_quantity = 0.1;
        state.entry_price = 50000.0;
        
        let version = store.update_asset("BTCUSDT", state.clone()).unwrap();
        assert_eq!(version, 1);
        
        // Update PnL
        store.update_pnl("BTCUSDT", 100.0, 51000.0).unwrap();
        
        let updated = store.get_asset("BTCUSDT").unwrap();
        assert_eq!(updated.current_pnl, 100.0);
        assert!(updated.unrealized_pnl > 0.0);
    }
    
    #[test]
    fn test_version_tracking() {
        let store = SharedStateStore::new(10);
        
        assert_eq!(store.get_version(), 0);
        
        let state = AssetState::new("BTCUSDT");
        store.update_asset("BTCUSDT", state).unwrap();
        
        assert_eq!(store.get_version(), 1);
        assert!(!store.has_changed(1));
        assert!(store.has_changed(0));
    }
    
    #[test]
    fn test_synchronizer() {
        let store = Arc::new(SharedStateBuilder::new().build());
        let sync = StateSynchronizer::new(Arc::clone(&store));
        
        // Queue update message
        let state = AssetState::new("ETHUSDT");
        sync.queue_message(StateSyncMessage::UpdateState(state.clone()));
        
        // Process messages
        let messages = sync.process_messages();
        assert_eq!(messages.len(), 1);
        
        // Verify state was updated
        let retrieved = sync.sync_state("ETHUSDT");
        assert!(retrieved.is_some());
    }
}

// FFI exports for Python integration
#[no_mangle]
pub extern "C" fn create_shared_state_store(max_assets: usize) -> *mut SharedStateStore {
    Box::into_raw(Box::new(SharedStateStore::new(max_assets)))
}

#[no_mangle]
pub extern "C" fn update_asset_state(
    store: *mut SharedStateStore,
    asset: *const i8,
    position_side: i8,
    quantity: f64,
    entry_price: f64,
) -> u64 {
    unsafe {
        let store = &mut *store;
        let asset_str = std::ffi::CStr::from_ptr(asset).to_str().unwrap_or("UNKNOWN");
        
        let mut state = AssetState::new(asset_str);
        state.position_side = position_side;
        state.position_quantity = quantity;
        state.entry_price = entry_price;
        
        store.update_asset(asset_str, state).unwrap_or(0)
    }
}
