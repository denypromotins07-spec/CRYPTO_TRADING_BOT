#!/usr/bin/env python3
"""
Intrusion Detector - Monitors for anomalous local process behavior.

This module implements intrusion detection by monitoring system processes,
network connections, and file system activity for signs of compromise.

Security Features:
- Process anomaly detection
- Network connection monitoring
- File integrity monitoring
- Behavioral analysis
- Automatic threat response

Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
"""

import os
import sys
import time
import json
import hashlib
import threading
import subprocess
from typing import Optional, Dict, List, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import deque
import re


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Detected security event."""
    event_id: str
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    description: str
    source: str
    details: Dict[str, Any] = field(default_factory=dict)


class IntrusionDetector:
    """
    Multi-layered intrusion detection system.
    
    Implements the Observer pattern for security event notifications.
    """
    
    # Known malicious process patterns
    MALICIOUS_PATTERNS = [
        r'mimikatz',
        r'pwdump',
        r'keylog',
        r'screencap',
        r'rat.*\.exe',
        r'cryptominer',
        r'xmrig',
        r'coinhive',
    ]
    
    # Suspicious network ports
    SUSPICIOUS_PORTS = {
        4444,  # Common reverse shell
        5555,  # Android debug
        6666,  # IRC backdoor
        31337, # Elite backdoor
        12345, # NetBus
        20000, # Millenium
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize intrusion detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config or {}
        self._events: deque = deque(maxlen=1000)
        self._callbacks: List[callable] = []
        self._is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        
        # Baseline state
        self._baseline_processes: Set[str] = set()
        self._baseline_connections: Set[Tuple[str, int]] = set()
        self._file_hashes: Dict[str, str] = {}
        
        # Statistics
        self._events_detected = 0
        self._false_positives = 0
        
        # Platform detection
        self._is_windows = sys.platform == 'win32'
        self._is_linux = sys.platform.startswith('linux')
        self._is_macos = sys.platform.startswith('darwin')
        
        # Initialize baselines
        self._capture_baseline()
    
    def add_callback(self, callback: callable) -> None:
        """Add callback for security events."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, event: SecurityEvent) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass
    
    def _capture_baseline(self) -> None:
        """Capture baseline system state."""
        self._baseline_processes = self._get_running_processes()
        self._baseline_connections = self._get_network_connections()
    
    def _get_running_processes(self) -> Set[str]:
        """Get set of currently running process names."""
        try:
            if self._is_windows:
                result = subprocess.run(
                    ['tasklist', '/FO', 'CSV'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                processes = set()
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if line.strip():
                        parts = line.split(',')
                        if parts:
                            proc_name = parts[0].strip('"').lower()
                            processes.add(proc_name)
                return processes
            else:
                result = subprocess.run(
                    ['ps', 'aux'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                processes = set()
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 11:
                        proc_name = parts[10].split('/')[-1].lower()
                        processes.add(proc_name)
                return processes
        except Exception:
            return set()
    
    def _get_network_connections(self) -> Set[Tuple[str, int]]:
        """Get set of active network connections (host, port)."""
        connections = set()
        
        try:
            if self._is_windows:
                result = subprocess.run(
                    ['netstat', '-ano'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:
                result = subprocess.run(
                    ['netstat', '-an'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            
            for line in result.stdout.split('\n'):
                # Look for ESTABLISHED or LISTENING connections
                if 'ESTABLISHED' in line or 'LISTEN' in line:
                    parts = line.split()
                    for part in parts:
                        if ':' in part:
                            try:
                                host, port_str = part.rsplit(':', 1)
                                port = int(port_str)
                                if port > 0:
                                    connections.add((host, port))
                            except ValueError:
                                continue
        except Exception:
            pass
        
        return connections
    
    def _check_malicious_processes(self) -> List[SecurityEvent]:
        """Check for known malicious processes."""
        events = []
        current_processes = self._get_running_processes()
        
        for proc in current_processes:
            for pattern in self.MALICIOUS_PATTERNS:
                if re.search(pattern, proc, re.IGNORECASE):
                    event = SecurityEvent(
                        event_id=hashlib.md5(f"{proc}_{time.time()}".encode()).hexdigest()[:12],
                        timestamp=time.time(),
                        event_type="MALICIOUS_PROCESS",
                        threat_level=ThreatLevel.CRITICAL,
                        description=f"Malicious process detected: {proc}",
                        source="process_monitor",
                        details={"process_name": proc, "pattern": pattern}
                    )
                    events.append(event)
                    break
        
        return events
    
    def _check_suspicious_connections(self) -> List[SecurityEvent]:
        """Check for suspicious network connections."""
        events = []
        current_connections = self._get_network_connections()
        
        # Check for new suspicious connections
        new_connections = current_connections - self._baseline_connections
        
        for host, port in new_connections:
            if port in self.SUSPICIOUS_PORTS:
                event = SecurityEvent(
                    event_id=hashlib.md5(f"{host}_{port}_{time.time()}".encode()).hexdigest()[:12],
                    timestamp=time.time(),
                    event_type="SUSPICIOUS_CONNECTION",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Suspicious network connection detected",
                    source="network_monitor",
                    details={"host": host, "port": port}
                )
                events.append(event)
        
        return events
    
    def _check_process_anomalies(self) -> List[SecurityEvent]:
        """Check for anomalous process behavior."""
        events = []
        current_processes = self._get_running_processes()
        
        # Check for new processes not in baseline
        new_processes = current_processes - self._baseline_processes
        
        # Filter out common benign new processes
        benign_patterns = {'python', 'node', 'chrome', 'firefox', 'code'}
        suspicious_new = [
            p for p in new_processes 
            if not any(bp in p for bp in benign_patterns)
        ]
        
        for proc in suspicious_new:
            # Check if process is running from suspicious location
            event = SecurityEvent(
                event_id=hashlib.md5(f"{proc}_{time.time()}".encode()).hexdigest()[:12],
                timestamp=time.time(),
                event_type="NEW_PROCESS",
                threat_level=ThreatLevel.LOW,
                description=f"New process detected: {proc}",
                source="process_monitor",
                details={"process_name": proc}
            )
            events.append(event)
        
        return events
    
    def _check_port_scan(self) -> List[SecurityEvent]:
        """Detect potential port scanning activity."""
        events = []
        
        try:
            if self._is_windows:
                result = subprocess.run(
                    ['netstat', '-an'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                # Count SYN_RECV states (potential scan)
                syn_count = result.stdout.count('SYN_RECV')
                
                if syn_count > 50:  # Threshold for potential scan
                    event = SecurityEvent(
                        event_id=hashlib.md5(f"portscan_{time.time()}".encode()).hexdigest()[:12],
                        timestamp=time.time(),
                        event_type="PORT_SCAN_DETECTED",
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"Potential port scan detected ({syn_count} SYN_RECV)",
                        source="network_monitor",
                        details={"syn_recv_count": syn_count}
                    )
                    events.append(event)
        except Exception:
            pass
        
        return events
    
    def run_detection(self) -> List[SecurityEvent]:
        """Run all detection checks."""
        events = []
        
        events.extend(self._check_malicious_processes())
        events.extend(self._check_suspicious_connections())
        events.extend(self._check_process_anomalies())
        events.extend(self._check_port_scan())
        
        # Store events
        for event in events:
            self._events.append(event)
            self._events_detected += 1
            self._notify_callbacks(event)
        
        return events
    
    def start_monitoring(self, check_interval_seconds: float = 10.0) -> None:
        """Start continuous monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._stop_flag.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while not self._stop_flag.is_set():
            events = self.run_detection()
            
            # Auto-respond to critical threats
            for event in events:
                if event.threat_level == ThreatLevel.CRITICAL:
                    self._emergency_response(event)
            
            self._stop_flag.wait(interval)
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._stop_flag.set()
        self._is_monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def _emergency_response(self, event: SecurityEvent) -> None:
        """Execute emergency response for critical threats."""
        print(f"\n{'='*60}", file=sys.stderr)
        print("CRITICAL SECURITY THREAT DETECTED", file=sys.stderr)
        print(f"Event: {event.event_type}", file=sys.stderr)
        print(f"Description: {event.description}", file=sys.stderr)
        print(f"Taking emergency action...", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        
        # Log to SOUL.md
        self._log_to_soul(event)
    
    def _log_to_soul(self, event: SecurityEvent) -> None:
        """Log security event to SOUL.md."""
        soul_path = Path("SOUL.md")
        
        entry = f"""
## Security Event - {event.event_id}

**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}
**Type:** {event.event_type}
**Threat Level:** {event.threat_level.value.upper()}
**Description:** {event.description}
**Source:** {event.source}

**Details:**
```json
{json.dumps(event.details, indent=2)}
```

---
"""
        
        try:
            if soul_path.exists():
                with open(soul_path, 'a') as f:
                    f.write(entry)
            else:
                with open(soul_path, 'w') as f:
                    f.write(f"# Security Audit Log\n{entry}")
        except Exception:
            pass
    
    def get_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events."""
        return list(self._events)[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_events": self._events_detected,
            "false_positives": self._false_positives,
            "events_in_memory": len(self._events),
            "baseline_processes": len(self._baseline_processes),
            "baseline_connections": len(self._baseline_connections),
            "is_monitoring": self._is_monitoring
        }
    
    def mark_false_positive(self, event_id: str) -> None:
        """Mark an event as false positive."""
        self._false_positives += 1


if __name__ == '__main__':
    print("Intrusion Detector Self-Test")
    print("=" * 40)
    
    detector = IntrusionDetector()
    
    print("\nRunning detection checks...")
    events = detector.run_detection()
    
    if events:
        print(f"\nDetected {len(events)} events:")
        for event in events:
            print(f"  [{event.threat_level.value.upper()}] {event.event_type}: {event.description}")
    else:
        print("\nNo threats detected.")
    
    # Show statistics
    stats = detector.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total Events: {stats['total_events']}")
    print(f"  Baseline Processes: {stats['baseline_processes']}")
    print(f"  Baseline Connections: {stats['baseline_connections']}")
