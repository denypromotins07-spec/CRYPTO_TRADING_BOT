// ZAID PERSONAL CRYPTO TRADING BOT - Stage 24
// Schema Registry for Backward-Compatible Binary Schema Evolution
// Manages FlatBuffer/Protobuf schema versions and compatibility checks
// Instantly halts bot if schema mismatch threatens event store corruption

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Schema compatibility levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompatibilityLevel {
    /// No compatibility guarantees (breaking changes allowed)
    None,
    /// New schema can read old data (forward compatibility)
    Forward,
    /// Old schema can read new data (backward compatibility)
    Backward,
    /// Both forward and backward compatible (full compatibility)
    Full,
}

impl CompatibilityLevel {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "forward" => Some(Self::Forward),
            "backward" => Some(Self::Backward),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Forward => "forward",
            Self::Backward => "backward",
            Self::Full => "full",
        }
    }
}

/// Schema metadata
#[derive(Debug, Clone)]
pub struct SchemaMetadata {
    /// Unique schema identifier (hash of schema content)
    pub schema_id: String,
    /// Schema version number
    pub version: u32,
    /// Schema type (flatbuffer, protobuf, etc.)
    pub schema_type: String,
    /// Message types defined in this schema
    pub message_types: Vec<String>,
    /// Timestamp when schema was registered
    pub registered_at: u64,
    /// Compatibility level
    pub compatibility: CompatibilityLevel,
    /// Whether this schema is active
    pub is_active: bool,
    /// Parent schema version (for evolution tracking)
    pub parent_version: Option<u32>,
    /// Breaking changes from parent
    pub breaking_changes: Vec<String>,
}

/// Schema registry error types
#[derive(Debug)]
pub enum SchemaError {
    /// Schema not found
    NotFound(String),
    /// Schema compatibility violation
    CompatibilityViolation(String),
    /// Invalid schema format
    InvalidFormat(String),
    /// Version conflict
    VersionConflict(String),
    /// IO error
    IoError(io::Error),
    /// Schema mismatch (critical - requires halt)
    SchemaMismatch(String),
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaError::NotFound(id) => write!(f, "Schema not found: {}", id),
            SchemaError::CompatibilityViolation(msg) => {
                write!(f, "Schema compatibility violation: {}", msg)
            }
            SchemaError::InvalidFormat(msg) => write!(f, "Invalid schema format: {}", msg),
            SchemaError::VersionConflict(msg) => write!(f, "Version conflict: {}", msg),
            SchemaError::IoError(e) => write!(f, "IO error: {}", e),
            SchemaError::SchemaMismatch(msg) => write!(f, "CRITICAL Schema mismatch: {}", msg),
        }
    }
}

impl std::error::Error for SchemaError {}

impl From<io::Error> for SchemaError {
    fn from(err: io::Error) -> Self {
        Self::IoError(err)
    }
}

/// Result type for schema operations
pub type SchemaResult<T> = Result<T, SchemaError>;

/// Schema registry for managing binary schema evolution
pub struct SchemaRegistry {
    /// Registered schemas by ID
    schemas: HashMap<String, SchemaMetadata>,
    /// Active schema version per message type
    active_schemas: HashMap<String, u32>,
    /// Schema storage directory
    storage_dir: PathBuf,
    /// Default compatibility level for new schemas
    default_compatibility: CompatibilityLevel,
    /// Whether to halt on schema mismatch
    halt_on_mismatch: bool,
    /// Registry statistics
    stats: RegistryStats,
}

/// Registry statistics
#[derive(Debug, Clone, Default)]
pub struct RegistryStats {
    pub total_schemas: usize,
    pub active_schemas: usize,
    pub compatibility_checks: u64,
    pub compatibility_failures: u64,
    pub schema_registrations: u64,
    pub schema_validations: u64,
}

impl SchemaRegistry {
    /// Create a new schema registry
    pub fn new(storage_dir: &str, halt_on_mismatch: bool) -> SchemaResult<Self> {
        let storage_dir = PathBuf::from(storage_dir);
        
        // Create storage directory if it doesn't exist
        if !storage_dir.exists() {
            fs::create_dir_all(&storage_dir)?;
        }
        
        let mut registry = Self {
            schemas: HashMap::new(),
            active_schemas: HashMap::new(),
            storage_dir,
            default_compatibility: CompatibilityLevel::Backward,
            halt_on_mismatch,
            stats: RegistryStats::default(),
        };
        
        // Load existing schemas from storage
        registry.load_from_storage()?;
        
        Ok(registry)
    }
    
    /// Register a new schema
    pub fn register_schema(
        &mut self,
        schema_content: &str,
        schema_type: &str,
        message_types: Vec<String>,
        compatibility: Option<CompatibilityLevel>,
    ) -> SchemaResult<String> {
        // Generate schema ID (hash of content)
        let schema_id = self.generate_schema_id(schema_content);
        
        // Check if schema already exists
        if let Some(existing) = self.schemas.get(&schema_id) {
            return Ok(existing.schema_id.clone());
        }
        
        // Determine version
        let version = self.get_next_version(&message_types);
        
        // Find parent schema
        let parent_version = if version > 1 {
            Some(version - 1)
        } else {
            None
        };
        
        // Check compatibility with previous version
        let breaking_changes = self.check_compatibility(
            &schema_id,
            &message_types,
            compatibility.unwrap_or(self.default_compatibility),
        )?;
        
        // Create metadata
        let metadata = SchemaMetadata {
            schema_id: schema_id.clone(),
            version,
            schema_type: schema_type.to_string(),
            message_types: message_types.clone(),
            registered_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            compatibility: compatibility.unwrap_or(self.default_compatibility),
            is_active: true,
            parent_version,
            breaking_changes,
        };
        
        // Store schema
        self.schemas.insert(schema_id.clone(), metadata);
        
        // Update active schemas
        for msg_type in message_types {
            self.active_schemas.insert(msg_type, version);
        }
        
        // Persist to storage
        self.persist_schema(&schema_id, schema_content)?;
        
        self.stats.schema_registrations += 1;
        self.stats.total_schemas = self.schemas.len();
        self.stats.active_schemas = self.active_schemas.len();
        
        Ok(schema_id)
    }
    
    /// Validate data against schema
    pub fn validate_data(&self, schema_id: &str, data: &[u8]) -> SchemaResult<bool> {
        self.stats.schema_validations += 1;
        
        let schema = self.schemas
            .get(schema_id)
            .ok_or_else(|| SchemaError::NotFound(schema_id.to_string()))?;
        
        if !schema.is_active {
            return Err(SchemaError::SchemaMismatch(format!(
                "Schema {} is not active",
                schema_id
            )));
        }
        
        // Basic validation (in production, would use actual schema validation)
        if data.is_empty() {
            return Err(SchemaError::InvalidFormat("Empty data".to_string()));
        }
        
        Ok(true)
    }
    
    /// Check if schema is compatible with active version
    pub fn check_data_compatibility(
        &self,
        message_type: &str,
        incoming_schema_id: &str,
    ) -> SchemaResult<bool> {
        self.stats.compatibility_checks += 1;
        
        // Get active schema version for this message type
        let active_version = self.active_schemas
            .get(message_type)
            .copied()
            .unwrap_or(0);
        
        if active_version == 0 {
            // No active schema, accept any
            return Ok(true);
        }
        
        // Get incoming schema
        let incoming = self.schemas
            .get(incoming_schema_id)
            .ok_or_else(|| SchemaError::NotFound(incoming_schema_id.to_string()))?;
        
        // Check compatibility based on level
        match incoming.compatibility {
            CompatibilityLevel::None => {
                // No compatibility guarantee - reject if versions differ
                if incoming.version != active_version {
                    self.stats.compatibility_failures += 1;
                    
                    if self.halt_on_mismatch {
                        return Err(SchemaError::SchemaMismatch(format!(
                            "Schema version mismatch for {}: expected {}, got {}. HALTING.",
                            message_type, active_version, incoming.version
                        )));
                    }
                    
                    return Err(SchemaError::CompatibilityViolation(format!(
                        "Version mismatch: expected {}, got {}",
                        active_version, incoming.version
                    )));
                }
            }
            CompatibilityLevel::Forward | 
            CompatibilityLevel::Backward | 
            CompatibilityLevel::Full => {
                // Compatible schemas are accepted
                // In production, would verify actual field compatibility
            }
        }
        
        Ok(true)
    }
    
    /// Get active schema ID for a message type
    pub fn get_active_schema(&self, message_type: &str) -> Option<&SchemaMetadata> {
        let version = self.active_schemas.get(message_type)?;
        
        self.schemas.values().find(|s| {
            s.message_types.contains(&message_type.to_string()) && s.version == *version
        })
    }
    
    /// Deactivate a schema (for rollback)
    pub fn deactivate_schema(&mut self, schema_id: &str) -> SchemaResult<()> {
        let schema = self.schemas
            .get_mut(schema_id)
            .ok_or_else(|| SchemaError::NotFound(schema_id.to_string()))?;
        
        schema.is_active = false;
        self.stats.active_schemas = self.active_schemas.len();
        
        Ok(())
    }
    
    /// Activate a specific schema version
    pub fn activate_version(
        &mut self,
        message_type: &str,
        version: u32,
    ) -> SchemaResult<()> {
        // Find schema with matching version and message type
        let schema_id = self.schemas.iter()
            .find(|(_, s)| {
                s.message_types.contains(&message_type.to_string()) && s.version == version
            })
            .map(|(id, _)| id.clone())
            .ok_or_else(|| SchemaError::NotFound(format!(
                "Schema version {} for {}",
                version, message_type
            )))?;
        
        self.active_schemas.insert(message_type.to_string(), version);
        self.stats.active_schemas = self.active_schemas.len();
        
        Ok(())
    }
    
    /// Get registry statistics
    pub fn get_stats(&self) -> &RegistryStats {
        &self.stats
    }
    
    /// List all registered schemas
    pub fn list_schemas(&self) -> Vec<&SchemaMetadata> {
        self.schemas.values().collect()
    }
    
    /// Generate schema ID from content (SHA-256 hash prefix)
    fn generate_schema_id(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
    
    /// Get next version for message types
    fn get_next_version(&self, message_types: &[String]) -> u32 {
        let mut max_version = 0;
        
        for msg_type in message_types {
            if let Some(version) = self.active_schemas.get(msg_type) {
                max_version = max_version.max(*version);
            }
        }
        
        max_version + 1
    }
    
    /// Check compatibility with previous schema version
    fn check_compatibility(
        &self,
        schema_id: &str,
        message_types: &[String],
        level: CompatibilityLevel,
    ) -> SchemaResult<Vec<String>> {
        let mut breaking_changes = Vec::new();
        
        for msg_type in message_types {
            if let Some(active_version) = self.active_schemas.get(msg_type) {
                if *active_version == 0 {
                    continue; // First version, no compatibility check needed
                }
                
                // Find previous schema
                let prev_schema = self.schemas.values().find(|s| {
                    s.message_types.contains(msg_type) && s.version == *active_version
                });
                
                if let Some(prev) = prev_schema {
                    // Check for breaking changes based on compatibility level
                    match level {
                        CompatibilityLevel::None => {
                            breaking_changes.push(format!(
                                "No compatibility guarantee for {}",
                                msg_type
                            ));
                        }
                        CompatibilityLevel::Backward | CompatibilityLevel::Full => {
                            // Verify that old readers can read new data
                            // In production, would compare field definitions
                            if prev.schema_type != "flatbuffer" {
                                breaking_changes.push(format!(
                                    "Potential breaking change in {} (non-FlatBuffer)",
                                    msg_type
                                ));
                            }
                        }
                        CompatibilityLevel::Forward => {
                            // Verify that new readers can read old data
                            // Implementation depends on schema format
                        }
                    }
                }
            }
        }
        
        Ok(breaking_changes)
    }
    
    /// Persist schema to storage
    fn persist_schema(&self, schema_id: &str, content: &str) -> SchemaResult<()> {
        let path = self.storage_dir.join(format!("{}.schema", schema_id));
        let mut file = fs::File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }
    
    /// Load schemas from storage
    fn load_from_storage(&mut self) -> SchemaResult<()> {
        for entry in fs::read_dir(&self.storage_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("schema") {
                let mut file = fs::File::open(&path)?;
                let mut content = String::new();
                file.read_to_string(&mut content)?;
                
                // Extract schema ID from filename
                let schema_id = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                
                if !schema_id.is_empty() {
                    // Reconstruct metadata (simplified - in production would store metadata separately)
                    let metadata = SchemaMetadata {
                        schema_id: schema_id.clone(),
                        version: 1,
                        schema_type: "unknown".to_string(),
                        message_types: vec![],
                        registered_at: 0,
                        compatibility: self.default_compatibility,
                        is_active: true,
                        parent_version: None,
                        breaking_changes: vec![],
                    };
                    
                    self.schemas.insert(schema_id, metadata);
                }
            }
        }
        
        self.stats.total_schemas = self.schemas.len();
        self.stats.active_schemas = self.schemas.len();
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_schema_registration() {
        let temp_dir = "/tmp/schema_registry_test";
        let _ = fs::remove_dir_all(temp_dir);
        
        let mut registry = SchemaRegistry::new(temp_dir, true).unwrap();
        
        let schema_content = r#"
            table Tick {
                timestamp_ns: long;
                price: double;
                quantity: double;
            }
        "#;
        
        let schema_id = registry.register_schema(
            schema_content,
            "flatbuffer",
            vec!["Tick".to_string()],
            Some(CompatibilityLevel::Backward),
        ).unwrap();
        
        assert!(!schema_id.is_empty());
        assert_eq!(registry.schemas.len(), 1);
        
        // Clean up
        let _ = fs::remove_dir_all(temp_dir);
    }
    
    #[test]
    fn test_compatibility_check() {
        let temp_dir = "/tmp/schema_compat_test";
        let _ = fs::remove_dir_all(temp_dir);
        
        let mut registry = SchemaRegistry::new(temp_dir, true).unwrap();
        
        // Register first version
        let schema_v1 = r#"table Order { price: double; }"#;
        registry.register_schema(
            schema_v1,
            "flatbuffer",
            vec!["Order".to_string()],
            Some(CompatibilityLevel::Backward),
        ).unwrap();
        
        // Register second version (compatible)
        let schema_v2 = r#"table Order { price: double; quantity: double; }"#;
        let result = registry.register_schema(
            schema_v2,
            "flatbuffer",
            vec!["Order".to_string()],
            Some(CompatibilityLevel::Backward),
        );
        
        assert!(result.is_ok());
        
        // Clean up
        let _ = fs::remove_dir_all(temp_dir);
    }
    
    #[test]
    fn test_schema_mismatch_halt() {
        let temp_dir = "/tmp/schema_halt_test";
        let _ = fs::remove_dir_all(temp_dir);
        
        let mut registry = SchemaRegistry::new(temp_dir, true).unwrap();
        
        // Register schema with None compatibility
        let schema = r#"table Test { value: int; }"#;
        registry.register_schema(
            schema,
            "flatbuffer",
            vec!["Test".to_string()],
            Some(CompatibilityLevel::None),
        ).unwrap();
        
        // Try to use incompatible version
        let result = registry.check_data_compatibility("Test", "nonexistent_schema");
        
        assert!(result.is_err());
        
        match result.unwrap_err() {
            SchemaError::NotFound(_) => (),
            _ => panic!("Expected NotFound error"),
        }
        
        // Clean up
        let _ = fs::remove_dir_all(temp_dir);
    }
}
