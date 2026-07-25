#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 24
Serialization Profiler for Measuring Nanosecond Overhead
Tracks serialization/deserialization performance across all protocols
Identifies bottlenecks and memory spikes in real-time
"""

from __future__ import annotations

import time
import statistics
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SerializationMetrics:
    """Metrics for a single serialization operation."""
    operation_type: str  # 'serialize' or 'deserialize'
    protocol: str  # 'flatbuffer', 'protobuf', 'json', etc.
    message_type: str  # 'Tick', 'OrderBook', etc.
    size_bytes: int
    duration_ns: int
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ProtocolStats:
    """Aggregated statistics for a protocol."""
    protocol: str
    total_operations: int = 0
    total_bytes: int = 0
    total_time_ns: int = 0
    min_time_ns: int = 0
    max_time_ns: int = 0
    times: List[int] = field(default_factory=list)
    errors: int = 0
    
    @property
    def avg_time_ns(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return self.total_time_ns / self.total_operations
    
    @property
    def throughput_mbps(self) -> float:
        if self.total_time_ns == 0:
            return 0.0
        return (self.total_bytes / self.total_time_ns) * 1e9 / (1024 * 1024)
    
    @property
    def std_dev_ns(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return statistics.stdev(self.times)
    
    @property
    def p50_ns(self) -> int:
        if not self.times:
            return 0
        return int(statistics.median(self.times))
    
    @property
    def p95_ns(self) -> int:
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    @property
    def p99_ns(self) -> int:
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'protocol': self.protocol,
            'total_operations': self.total_operations,
            'total_bytes': self.total_bytes,
            'avg_time_ns': round(self.avg_time_ns, 2),
            'min_time_ns': self.min_time_ns,
            'max_time_ns': self.max_time_ns,
            'p50_ns': self.p50_ns,
            'p95_ns': self.p95_ns,
            'p99_ns': self.p99_ns,
            'std_dev_ns': round(self.std_dev_ns, 2),
            'throughput_mbps': round(self.throughput_mbps, 2),
            'error_count': self.errors,
            'error_rate': round(self.errors / max(1, self.total_operations) * 100, 4),
        }


class SerializationProfiler:
    """
    High-precision profiler for serialization operations.
    Measures nanosecond-level overhead and identifies bottlenecks.
    """
    
    # Maximum samples to keep per protocol (circular buffer)
    MAX_SAMPLES_PER_PROTOCOL = 10000
    
    # Thresholds for alerts (nanoseconds)
    WARNING_THRESHOLD_NS = 500  # 500ns warning
    CRITICAL_THRESHOLD_NS = 1000  # 1μs critical
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-protocol statistics
        self.protocol_stats: Dict[str, ProtocolStats] = {}
        
        # Per-message-type statistics
        self.message_stats: Dict[str, ProtocolStats] = {}
        
        # Recent metrics for analysis
        self.recent_metrics: List[SerializationMetrics] = []
        
        # Alert history
        self.alerts: List[Dict[str, Any]] = []
        
        # Start time for session tracking
        self.session_start = datetime.now()
        
        logger.info("Serialization profiler initialized")
    
    def record_operation(
        self,
        protocol: str,
        message_type: str,
        operation_type: str,
        size_bytes: int,
        duration_ns: int,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Record a serialization operation.
        
        Returns alert dict if thresholds exceeded, None otherwise.
        """
        now = datetime.now()
        
        # Create metrics record
        metrics = SerializationMetrics(
            operation_type=operation_type,
            protocol=protocol,
            message_type=message_type,
            size_bytes=size_bytes,
            duration_ns=duration_ns,
            timestamp=now,
            success=success,
            error_message=error_message,
        )
        
        # Update protocol stats
        if protocol not in self.protocol_stats:
            self.protocol_stats[protocol] = ProtocolStats(protocol=protocol)
        
        pstats = self.protocol_stats[protocol]
        pstats.total_operations += 1
        pstats.total_bytes += size_bytes
        pstats.total_time_ns += duration_ns
        
        if pstats.min_time_ns == 0 or duration_ns < pstats.min_time_ns:
            pstats.min_time_ns = duration_ns
        if duration_ns > pstats.max_time_ns:
            pstats.max_time_ns = duration_ns
        
        # Keep recent samples for percentile calculation
        pstats.times.append(duration_ns)
        if len(pstats.times) > self.MAX_SAMPLES_PER_PROTOCOL:
            pstats.times.pop(0)
        
        # Update message type stats
        msg_key = f"{protocol}:{message_type}"
        if msg_key not in self.message_stats:
            self.message_stats[msg_key] = ProtocolStats(protocol=msg_key)
        
        mstats = self.message_stats[msg_key]
        mstats.total_operations += 1
        mstats.total_bytes += size_bytes
        mstats.total_time_ns += duration_ns
        
        if not success:
            pstats.errors += 1
            mstats.errors += 1
        
        # Store recent metrics
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > self.MAX_SAMPLES_PER_PROTOCOL:
            self.recent_metrics.pop(0)
        
        # Check thresholds and generate alerts
        alert = None
        if duration_ns >= self.CRITICAL_THRESHOLD_NS:
            alert = {
                'level': 'CRITICAL',
                'protocol': protocol,
                'message_type': message_type,
                'operation': operation_type,
                'duration_ns': duration_ns,
                'threshold_ns': self.CRITICAL_THRESHOLD_NS,
                'timestamp': now.isoformat(),
            }
            self.alerts.append(alert)
            logger.warning(
                f"CRITICAL: {protocol}/{message_type} {operation_type} took {duration_ns}ns "
                f"(threshold: {self.CRITICAL_THRESHOLD_NS}ns)"
            )
        elif duration_ns >= self.WARNING_THRESHOLD_NS:
            alert = {
                'level': 'WARNING',
                'protocol': protocol,
                'message_type': message_type,
                'operation': operation_type,
                'duration_ns': duration_ns,
                'threshold_ns': self.WARNING_THRESHOLD_NS,
                'timestamp': now.isoformat(),
            }
            self.alerts.append(alert)
        
        return alert
    
    def profile_serialization(
        self,
        protocol: str,
        message_type: str,
        serialize_func,
        data: Any,
    ) -> Tuple[bytes, Optional[Dict[str, Any]]]:
        """
        Profile a serialization function and record metrics.
        
        Args:
            protocol: Protocol name
            message_type: Message type name
            serialize_func: Function that serializes data to bytes
            data: Data to serialize
            
        Returns:
            Tuple of (serialized_bytes, alert_if_any)
        """
        start = time.perf_counter_ns()
        alert = None
        
        try:
            result = serialize_func(data)
            duration = time.perf_counter_ns() - start
            size = len(result) if isinstance(result, bytes) else 0
            
            alert = self.record_operation(
                protocol=protocol,
                message_type=message_type,
                operation_type='serialize',
                size_bytes=size,
                duration_ns=duration,
                success=True,
            )
            
            return result, alert
            
        except Exception as e:
            duration = time.perf_counter_ns() - start
            alert = self.record_operation(
                protocol=protocol,
                message_type=message_type,
                operation_type='serialize',
                size_bytes=0,
                duration_ns=duration,
                success=False,
                error_message=str(e),
            )
            raise
    
    def profile_deserialization(
        self,
        protocol: str,
        message_type: str,
        deserialize_func,
        data: bytes,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Profile a deserialization function and record metrics.
        
        Returns:
            Tuple of (deserialized_data, alert_if_any)
        """
        start = time.perf_counter_ns()
        alert = None
        
        try:
            result = deserialize_func(data)
            duration = time.perf_counter_ns() - start
            size = len(data)
            
            alert = self.record_operation(
                protocol=protocol,
                message_type=message_type,
                operation_type='deserialize',
                size_bytes=size,
                duration_ns=duration,
                success=True,
            )
            
            return result, alert
            
        except Exception as e:
            duration = time.perf_counter_ns() - start
            alert = self.record_operation(
                protocol=protocol,
                message_type=message_type,
                operation_type='deserialize',
                size_bytes=size,
                duration_ns=duration,
                success=False,
                error_message=str(e),
            )
            raise
    
    def get_protocol_stats(self, protocol: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific protocol."""
        if protocol not in self.protocol_stats:
            return None
        return self.protocol_stats[protocol].to_dict()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all protocols."""
        return {
            protocol: stats.to_dict()
            for protocol, stats in self.protocol_stats.items()
        }
    
    def get_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identify the slowest protocol/message combinations."""
        all_stats = [
            {**stats.to_dict(), 'key': key}
            for key, stats in self.message_stats.items()
        ]
        
        # Sort by average time descending
        all_stats.sort(key=lambda x: x['avg_time_ns'], reverse=True)
        
        return all_stats[:top_n]
    
    def get_memory_impact(self) -> Dict[str, Any]:
        """Estimate memory impact of serialization buffers."""
        total_bytes = sum(s.total_bytes for s in self.protocol_stats.values())
        total_ops = sum(s.total_operations for s in self.protocol_stats.values())
        
        return {
            'total_bytes_processed': total_bytes,
            'total_operations': total_ops,
            'average_message_size': total_bytes / max(1, total_ops),
            'session_duration_seconds': (datetime.now() - self.session_start).total_seconds(),
            'bytes_per_second': total_bytes / max(1, (datetime.now() - self.session_start).total_seconds()),
        }
    
    def export_report(self, filepath: Optional[str] = None) -> str:
        """Export profiling report to JSON file."""
        if filepath is None:
            if self.output_dir:
                filepath = str(self.output_dir / f"serialization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            else:
                filepath = f"serialization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'session_start': self.session_start.isoformat(),
            'protocol_statistics': self.get_all_stats(),
            'bottlenecks': self.get_bottlenecks(10),
            'memory_impact': self.get_memory_impact(),
            'alert_count': len(self.alerts),
            'recent_alerts': self.alerts[-20:],  # Last 20 alerts
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Serialization report exported to {filepath}")
        return filepath
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.protocol_stats.clear()
        self.message_stats.clear()
        self.recent_metrics.clear()
        self.alerts.clear()
        self.session_start = datetime.now()
        logger.info("Serialization profiler reset")


def main():
    """Example usage and testing."""
    print("Serialization Profiler for ZAID Trading Bot")
    print("=" * 60)
    
    profiler = SerializationProfiler()
    
    # Simulate some serialization operations
    import random
    
    protocols = ['flatbuffer', 'protobuf', 'json', 'msgpack']
    message_types = ['Tick', 'OrderBook', 'Trade', 'Signal']
    
    print("\nSimulating 1000 serialization operations...")
    
    for i in range(1000):
        protocol = random.choice(protocols)
        msg_type = random.choice(message_types)
        
        # Simulate variable serialization times
        base_time = random.randint(50, 300)  # Base time in ns
        if protocol == 'json':
            base_time *= 3  # JSON is slower
        elif protocol == 'flatbuffer':
            base_time = base_time // 2  # FlatBuffer is faster
        
        # Add some outliers
        if random.random() < 0.01:
            base_time *= 5  # 1% chance of slow operation
        
        size = random.randint(64, 4096)
        
        profiler.record_operation(
            protocol=protocol,
            message_type=msg_type,
            operation_type='serialize',
            size_bytes=size,
            duration_ns=base_time,
            success=random.random() > 0.001,  # 0.1% failure rate
        )
    
    # Print summary
    print("\nProtocol Statistics:")
    print("-" * 60)
    
    for protocol, stats in profiler.get_all_stats().items():
        print(f"\n{protocol}:")
        print(f"  Operations: {stats['total_operations']}")
        print(f"  Avg Time: {stats['avg_time_ns']:.0f} ns")
        print(f"  P50: {stats['p50_ns']} ns")
        print(f"  P95: {stats['p95_ns']} ns")
        print(f"  P99: {stats['p99_ns']} ns")
        print(f"  Throughput: {stats['throughput_mbps']:.2f} MB/s")
        print(f"  Error Rate: {stats['error_rate']:.4f}%")
    
    # Show bottlenecks
    print("\nTop Bottlenecks:")
    print("-" * 60)
    for i, bottleneck in enumerate(profiler.get_bottlenecks(5), 1):
        print(f"{i}. {bottleneck['key']}: {bottleneck['avg_time_ns']:.0f} ns avg")
    
    # Memory impact
    print("\nMemory Impact:")
    print("-" * 60)
    mem_impact = profiler.get_memory_impact()
    print(f"Total Bytes: {mem_impact['total_bytes_processed']:,}")
    print(f"Avg Message Size: {mem_impact['average_message_size']:.0f} bytes")
    print(f"Throughput: {mem_impact['bytes_per_second']:,.0f} bytes/sec")
    
    print("\nSerialization profiler ready!")


if __name__ == "__main__":
    main()
