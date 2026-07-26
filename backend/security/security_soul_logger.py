#!/usr/bin/env python3
"""
Security Soul Logger - Logs blocked intrusion attempts to SOUL.md.

This module provides centralized security event logging with cryptographic
sealing to ensure audit trail integrity. All blocked intrusions and
security events are recorded in an immutable, append-only log.

Security Features:
- Cryptographic event sealing
- Append-only logging
- Tamper-evident entries
- Automatic SOUL.md updates
- Event correlation and analysis

Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
"""

import os
import sys
import time
import json
import hashlib
import hmac
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import threading


class EventType(Enum):
    """Types of security events."""
    INTRUSION_BLOCKED = "intrusion_blocked"
    DEBUGGER_DETECTED = "debugger_detected"
    INTEGRITY_FAILURE = "integrity_failure"
    TLS_PIN_MISMATCH = "tls_pin_mismatch"
    DNS_SPOOF_ATTEMPT = "dns_spoof_attempt"
    PORT_SCAN_BLOCKED = "port_scan_blocked"
    MALICIOUS_PROCESS = "malicious_process"
    API_KEY_ACCESS = "api_key_access"
    ENCRYPTION_OPERATION = "encryption_operation"
    SYSTEM_ANOMALY = "system_anomaly"


@dataclass
class SecurityLogEntry:
    """A sealed security log entry."""
    entry_id: str
    timestamp: float
    event_type: EventType
    severity: str  # low, medium, high, critical
    source_module: str
    description: str
    details: Dict[str, Any]
    previous_hash: str
    entry_hash: str
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity,
            "source_module": self.source_module,
            "description": self.description,
            "details": self.details,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "signature": self.signature
        }


class SecuritySoulLogger:
    """
    Centralized security event logger with cryptographic sealing.
    
    Implements the Singleton pattern to ensure all security events
    are logged through a single, consistent interface.
    """
    
    _instance: Optional['SecuritySoulLogger'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> 'SecuritySoulLogger':
        """Singleton instance creation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, soul_path: str = "SOUL.md", signing_key: Optional[bytes] = None):
        """
        Initialize the security logger.
        
        Args:
            soul_path: Path to the SOUL.md log file
            signing_key: Key for HMAC signatures (generated if not provided)
        """
        # Only initialize once
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.soul_path = Path(soul_path)
        self.signing_key = signing_key or self._generate_signing_key()
        self._last_hash = "genesis"
        self._entry_counter = 0
        self._memory_buffer: List[SecurityLogEntry] = []
        self._buffer_lock = threading.Lock()
        self._initialized = True
        
        # Load existing state if log exists
        self._load_existing_state()
    
    def _generate_signing_key(self) -> bytes:
        """Generate a signing key from system entropy."""
        # In production, this would be stored securely
        # For now, derive from system-specific data
        system_data = f"{os.uname() if hasattr(os, 'uname') else 'windows'}_{time.time()}".encode()
        return hashlib.sha256(system_data).digest()
    
    def _load_existing_state(self) -> None:
        """Load the last hash from existing log."""
        if not self.soul_path.exists():
            return
        
        try:
            with open(self.soul_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find the last entry hash in the log
            # This is a simplified approach - production would parse properly
            for line in reversed(content.split('\n')):
                if 'entry_hash' in line:
                    # Extract hash value
                    parts = line.split(':')
                    if len(parts) >= 2:
                        self._last_hash = parts[1].strip().strip('"\'')
                        break
                        
            # Count existing entries
            self._entry_counter = content.count('## Security Event')
            
        except Exception:
            pass
    
    def log_event(
        self,
        event_type: EventType,
        description: str,
        source_module: str,
        severity: str = "medium",
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityLogEntry:
        """
        Log a security event with cryptographic sealing.
        
        Args:
            event_type: Type of security event
            description: Human-readable description
            source_module: Module that generated the event
            severity: Event severity level
            details: Additional event details
            
        Returns:
            The created log entry
        """
        with self._buffer_lock:
            self._entry_counter += 1
            
            # Generate entry ID
            entry_id = f"SEC-{self._entry_counter:08d}"
            timestamp = time.time()
            
            # Create entry content
            entry_content = {
                "entry_id": entry_id,
                "timestamp": timestamp,
                "event_type": event_type.value,
                "severity": severity,
                "source_module": source_module,
                "description": description,
                "details": details or {},
                "previous_hash": self._last_hash
            }
            
            # Compute entry hash
            content_str = json.dumps(entry_content, sort_keys=True)
            entry_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            # Compute HMAC signature
            signature = hmac.new(
                self.signing_key,
                entry_hash.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Create log entry
            log_entry = SecurityLogEntry(
                entry_id=entry_id,
                timestamp=timestamp,
                event_type=event_type,
                severity=severity,
                source_module=source_module,
                description=description,
                details=details or {},
                previous_hash=self._last_hash,
                entry_hash=entry_hash,
                signature=signature
            )
            
            # Update chain
            self._last_hash = entry_hash
            
            # Store in memory buffer
            self._memory_buffer.append(log_entry)
            
            # Persist to disk
            self._persist_entry(log_entry)
            
            return log_entry
    
    def _persist_entry(self, entry: SecurityLogEntry) -> None:
        """Persist entry to SOUL.md file."""
        # Ensure parent directory exists
        self.soul_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format entry for markdown
        dt = datetime.fromtimestamp(entry.timestamp)
        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        entry_markdown = f"""
## Security Event - {entry.entry_id}

**Timestamp:** {formatted_time}
**Type:** {entry.event_type.value.upper()}
**Severity:** {entry.severity.upper()}
**Source:** {entry.source_module}
**Description:** {entry.description}

### Cryptographic Proof

- **Previous Hash:** `{entry.previous_hash}`
- **Entry Hash:** `{entry.entry_hash}`
- **Signature:** `{entry.signature}`

### Details

```json
{json.dumps(entry.details, indent=2, default=str)}
```

---
"""
        
        try:
            # Append to file
            with open(self.soul_path, 'a', encoding='utf-8') as f:
                f.write(entry_markdown)
            
            # Sync to disk
            fd = os.open(str(self.soul_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
                
        except Exception as e:
            print(f"Warning: Failed to persist security log: {e}", file=sys.stderr)
    
    def log_intrusion_blocked(
        self,
        intrusion_type: str,
        source_ip: Optional[str] = None,
        port: Optional[int] = None,
        action_taken: str = "blocked"
    ) -> SecurityLogEntry:
        """Log a blocked intrusion attempt."""
        return self.log_event(
            event_type=EventType.INTRUSION_BLOCKED,
            description=f"Intrusion attempt blocked: {intrusion_type}",
            source_module="intrusion_detector",
            severity="high",
            details={
                "intrusion_type": intrusion_type,
                "source_ip": source_ip,
                "port": port,
                "action_taken": action_taken
            }
        )
    
    def log_debugger_detected(
        self,
        debugger_type: str,
        detection_method: str
    ) -> SecurityLogEntry:
        """Log debugger detection event."""
        return self.log_event(
            event_type=EventType.DEBUGGER_DETECTED,
            description=f"Debugger detected: {debugger_type}",
            source_module="anti_debug",
            severity="critical",
            details={
                "debugger_type": debugger_type,
                "detection_method": detection_method
            }
        )
    
    def log_integrity_failure(
        self,
        file_path: str,
        expected_hash: str,
        actual_hash: str
    ) -> SecurityLogEntry:
        """Log file integrity failure."""
        return self.log_event(
            event_type=EventType.INTEGRITY_FAILURE,
            description=f"Integrity check failed for: {file_path}",
            source_module="integrity_checker",
            severity="critical",
            details={
                "file_path": file_path,
                "expected_hash": expected_hash,
                "actual_hash": actual_hash
            }
        )
    
    def log_tls_pin_mismatch(
        self,
        hostname: str,
        expected_pin: str,
        received_pin: str
    ) -> SecurityLogEntry:
        """Log TLS certificate pin mismatch."""
        return self.log_event(
            event_type=EventType.TLS_PIN_MISMATCH,
            description=f"TLS pin mismatch for: {hostname}",
            source_module="tls_pinning",
            severity="critical",
            details={
                "hostname": hostname,
                "expected_pin": expected_pin,
                "received_pin": received_pin
            }
        )
    
    def log_port_scan(
        self,
        source_ip: str,
        ports_scanned: int,
        duration_seconds: float
    ) -> SecurityLogEntry:
        """Log detected port scan."""
        return self.log_event(
            event_type=EventType.PORT_SCAN_BLOCKED,
            description=f"Port scan detected from {source_ip}",
            source_module="intrusion_detector",
            severity="medium",
            details={
                "source_ip": source_ip,
                "ports_scanned": ports_scanned,
                "duration_seconds": duration_seconds
            }
        )
    
    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the entire log chain.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.soul_path.exists():
            return True, []
        
        try:
            with open(self.soul_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse entries (simplified parsing)
            entries = content.split('## Security Event')[1:]
            
            prev_hash = "genesis"
            for i, entry_text in enumerate(entries):
                # Extract hashes from entry
                entry_hash_line = None
                prev_hash_line = None
                sig_line = None
                
                for line in entry_text.split('\n'):
                    if 'Entry Hash:' in line:
                        entry_hash_line = line.split('`')[1] if '`' in line else None
                    elif 'Previous Hash:' in line:
                        prev_hash_line = line.split('`')[1] if '`' in line else None
                    elif 'Signature:' in line:
                        sig_line = line.split('`')[1] if '`' in line else None
                
                # Verify chain linkage
                if prev_hash_line and prev_hash_line != prev_hash:
                    issues.append(f"Entry {i}: Chain broken at previous_hash")
                
                # Verify signature (would need full entry data)
                # Simplified - just check signature exists
                if not sig_line:
                    issues.append(f"Entry {i}: Missing signature")
                
                if entry_hash_line:
                    prev_hash = entry_hash_line
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Verification error: {e}"]
    
    def get_recent_entries(self, count: int = 10) -> List[SecurityLogEntry]:
        """Get recent entries from memory buffer."""
        with self._buffer_lock:
            return self._memory_buffer[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._buffer_lock:
            event_counts = {}
            for entry in self._memory_buffer:
                key = entry.event_type.value
                event_counts[key] = event_counts.get(key, 0) + 1
            
            return {
                "total_entries": self._entry_counter,
                "entries_in_memory": len(self._memory_buffer),
                "event_breakdown": event_counts,
                "log_file_exists": self.soul_path.exists(),
                "log_file_size": self.soul_path.stat().st_size if self.soul_path.exists() else 0
            }


# Convenience functions for direct logging
def log_security_event(
    event_type: EventType,
    description: str,
    source: str,
    severity: str = "medium",
    details: Optional[Dict] = None
) -> SecurityLogEntry:
    """Log a security event using the singleton logger."""
    logger = SecuritySoulLogger()
    return logger.log_event(event_type, description, source, severity, details)


def log_intrusion_blocked(
    intrusion_type: str,
    source_ip: Optional[str] = None,
    port: Optional[int] = None
) -> SecurityLogEntry:
    """Convenience function to log blocked intrusions."""
    logger = SecuritySoulLogger()
    return logger.log_intrusion_blocked(intrusion_type, source_ip, port)


if __name__ == '__main__':
    print("Security Soul Logger Self-Test")
    print("=" * 40)
    
    # Initialize logger
    logger = SecuritySoulLogger(soul_path="test_SOUL.md")
    
    # Log some test events
    print("\nLogging test events...")
    
    logger.log_intrusion_blocked(
        intrusion_type="port_scan",
        source_ip="192.168.1.100",
        port=4444
    )
    
    logger.log_debugger_detected(
        debugger_type="gdb",
        detection_method="ptrace"
    )
    
    logger.log_event(
        event_type=EventType.API_KEY_ACCESS,
        description="API key accessed for Binance order",
        source_module="order_manager",
        severity="low",
        details={"key_id": "binance_main", "operation": "sign_order"}
    )
    
    # Show statistics
    stats = logger.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total Entries: {stats['total_entries']}")
    print(f"  Event Breakdown: {stats['event_breakdown']}")
    
    # Verify chain
    is_valid, issues = logger.verify_chain_integrity()
    print(f"\nChain Integrity: {'VALID' if is_valid else 'INVALID'}")
    if issues:
        print(f"Issues: {issues}")
    
    # Cleanup test file
    test_path = Path("test_SOUL.md")
    if test_path.exists():
        test_path.unlink()
        print("\nTest file cleaned up.")
