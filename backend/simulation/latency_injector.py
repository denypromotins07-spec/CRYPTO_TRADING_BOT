"""
Latency Injector for Realistic Exchange Simulation

Models network latency and exchange processing delays:
- TCP round-trip times to Binance AWS servers
- API rate limiting delays
- Order processing queue times
- Jitter and packet loss simulation

Optimized for 8GB RAM constraint.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NetworkCondition(Enum):
    """Network quality levels."""
    EXCELLENT = "excellent"   # < 10ms RTT
    GOOD = "good"             # 10-50ms RTT
    NORMAL = "normal"         # 50-150ms RTT
    POOR = "poor"             # 150-300ms RTT
    DEGRADED = "degraded"     # > 300ms RTT


@dataclass
class LatencyConfig:
    """Configuration for latency injection."""
    base_rtt_ms: float          # Base round-trip time
    jitter_std_ms: float        # Standard deviation of jitter
    packet_loss_rate: float     # Probability of packet loss (0-1)
    queue_delay_mean_ms: float  # Mean queue processing delay
    queue_delay_std_ms: float   # Queue delay variation
    burst_probability: float    # Probability of latency burst
    burst_multiplier: float     # Multiplier during bursts
    
    @classmethod
    def excellent(cls) -> 'LatencyConfig':
        return cls(
            base_rtt_ms=5.0,
            jitter_std_ms=1.0,
            packet_loss_rate=0.0001,
            queue_delay_mean_ms=1.0,
            queue_delay_std_ms=0.5,
            burst_probability=0.001,
            burst_multiplier=2.0,
        )
    
    @classmethod
    def good(cls) -> 'LatencyConfig':
        return cls(
            base_rtt_ms=25.0,
            jitter_std_ms=5.0,
            packet_loss_rate=0.001,
            queue_delay_mean_ms=2.0,
            queue_delay_std_ms=1.0,
            burst_probability=0.005,
            burst_multiplier=3.0,
        )
    
    @classmethod
    def normal(cls) -> 'LatencyConfig':
        """Typical internet connection to AWS."""
        return cls(
            base_rtt_ms=80.0,
            jitter_std_ms=20.0,
            packet_loss_rate=0.005,
            queue_delay_mean_ms=5.0,
            queue_delay_std_ms=3.0,
            burst_probability=0.01,
            burst_multiplier=5.0,
        )
    
    @classmethod
    def poor(cls) -> 'LatencyConfig':
        return cls(
            base_rtt_ms=200.0,
            jitter_std_ms=50.0,
            packet_loss_rate=0.02,
            queue_delay_mean_ms=20.0,
            queue_delay_std_ms=10.0,
            burst_probability=0.05,
            burst_multiplier=10.0,
        )


@dataclass
class LatencyEvent:
    """Record of a latency event."""
    timestamp_ns: int
    operation: str
    base_latency_ns: int
    jitter_ns: int
    queue_delay_ns: int
    total_latency_ns: int
    was_dropped: bool
    was_retried: bool


class LatencyInjector:
    """
    Inject realistic latency into order submissions and responses.
    
    Models:
    - Network RTT with jitter
    - Exchange API processing queues
    - Rate limiting backoff
    - Packet loss and retries
    """
    
    def __init__(self, config: Optional[LatencyConfig] = None, seed: int = 42):
        self.config = config or LatencyConfig.normal()
        self.rng = np.random.default_rng(seed)
        
        # State tracking
        self.events: List[LatencyEvent] = []
        self.current_network_condition = NetworkCondition.NORMAL
        
        # Rate limiting state
        self.request_timestamps: List[int] = []
        self.rate_limit_window_ns = 1_000_000_000  # 1 second
        self.max_requests_per_second = 10  # Conservative limit
        
        # Retry state
        self.pending_retries: List[Tuple[int, str, int]] = []  # (timestamp, op, retry_count)
    
    def inject_latency(self, timestamp_ns: int, operation: str) -> Tuple[int, bool]:
        """
        Inject latency for an operation.
        
        Args:
            timestamp_ns: Operation timestamp
            operation: Type of operation (order_submit, cancel, query, etc.)
            
        Returns:
            (total_latency_ns, was_dropped)
        """
        # Check for packet loss
        if self.rng.random() < self.config.packet_loss_rate:
            event = LatencyEvent(
                timestamp_ns=timestamp_ns,
                operation=operation,
                base_latency_ns=int(self.config.base_rtt_ms * 1_000_000),
                jitter_ns=0,
                queue_delay_ns=0,
                total_latency_ns=0,
                was_dropped=True,
                was_retried=False,
            )
            self.events.append(event)
            return 0, True
        
        # Calculate base RTT
        base_rtt_ns = int(self.config.base_rtt_ms * 1_000_000)
        
        # Add jitter (Gaussian)
        jitter_ms = self.rng.normal(0, self.config.jitter_std_ms)
        jitter_ns = int(max(jitter_ms, -self.config.base_rtt_ms * 0.5) * 1_000_000)
        
        # Add queue delay (exponential distribution)
        queue_delay_ms = self.rng.exponential(self.config.queue_delay_mean_ms)
        queue_delay_ns = int(queue_delay_ms * 1_000_000)
        
        # Check for latency burst
        multiplier = 1.0
        if self.rng.random() < self.config.burst_probability:
            multiplier = self.config.burst_multiplier
        
        # Rate limiting backoff
        rate_limit_delay = self._check_rate_limit(timestamp_ns)
        
        # Total latency
        total_latency_ns = int((base_rtt_ns + jitter_ns + queue_delay_ns) * multiplier) + rate_limit_delay
        total_latency_ns = max(total_latency_ns, 0)  # Ensure non-negative
        
        event = LatencyEvent(
            timestamp_ns=timestamp_ns,
            operation=operation,
            base_latency_ns=base_rtt_ns,
            jitter_ns=jitter_ns,
            queue_delay_ns=queue_delay_ns,
            total_latency_ns=total_latency_ns,
            was_dropped=False,
            was_retried=rate_limit_delay > 0,
        )
        self.events.append(event)
        
        return total_latency_ns, False
    
    def _check_rate_limit(self, timestamp_ns: int) -> int:
        """Check rate limiting and return additional delay if needed."""
        # Remove old timestamps
        cutoff = timestamp_ns - self.rate_limit_window_ns
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        
        if len(self.request_timestamps) >= self.max_requests_per_second:
            # Need to wait
            oldest_in_window = min(self.request_timestamps)
            wait_until = oldest_in_window + self.rate_limit_window_ns
            delay = max(wait_until - timestamp_ns, 0)
            logger.debug(f"Rate limited, delaying {delay}ns")
            return delay
        
        self.request_timestamps.append(timestamp_ns)
        return 0
    
    def simulate_order_round_trip(
        self, 
        timestamp_ns: int, 
        order_type: str
    ) -> Dict[str, int]:
        """
        Simulate complete order submission round trip.
        
        Returns dict with timing breakdown.
        """
        # Submit phase
        submit_latency, dropped = self.inject_latency(timestamp_ns, f"{order_type}_submit")
        
        if dropped:
            # Retry after backoff
            retry_delay = int(self.config.base_rtt_ms * 2 * 1_000_000)
            retry_timestamp = timestamp_ns + retry_delay
            submit_latency, dropped = self.inject_latency(retry_timestamp, f"{order_type}_retry")
        
        # Processing phase (exchange side)
        processing_latency = int(self.rng.exponential(5) * 1_000_000)  # ~5ms mean
        
        # Response phase
        response_latency, _ = self.inject_latency(
            timestamp_ns + submit_latency + processing_latency,
            f"{order_type}_response"
        )
        
        return {
            'submit_latency_ns': submit_latency,
            'processing_latency_ns': processing_latency,
            'response_latency_ns': response_latency,
            'total_latency_ns': submit_latency + processing_latency + response_latency,
            'was_dropped_initially': dropped,
        }
    
    def set_network_condition(self, condition: NetworkCondition) -> None:
        """Update network condition dynamically."""
        self.current_network_condition = condition
        
        config_map = {
            NetworkCondition.EXCELLENT: LatencyConfig.excellent(),
            NetworkCondition.GOOD: LatencyConfig.good(),
            NetworkCondition.NORMAL: LatencyConfig.normal(),
            NetworkCondition.POOR: LatencyConfig.poor(),
        }
        self.config = config_map.get(condition, LatencyConfig.normal())
        logger.info(f"Network condition changed to {condition}")
    
    def get_latency_statistics(self, window_ms: int = 1000) -> Dict[str, float]:
        """Calculate latency statistics over recent window."""
        if not self.events:
            return {}
        
        # Filter recent events
        now = self.events[-1].timestamp_ns if self.events else 0
        window_ns = window_ms * 1_000_000
        recent = [e for e in self.events if e.timestamp_ns > now - window_ns]
        
        if not recent:
            return {}
        
        latencies = [e.total_latency_ns for e in recent if not e.was_dropped]
        
        if not latencies:
            return {'mean_latency_ms': 0, 'dropped_rate': 1.0}
        
        dropped_count = sum(1 for e in recent if e.was_dropped)
        
        return {
            'mean_latency_ms': np.mean(latencies) / 1_000_000,
            'std_latency_ms': np.std(latencies) / 1_000_000,
            'min_latency_ms': min(latencies) / 1_000_000,
            'max_latency_ms': max(latencies) / 1_000_000,
            'p50_latency_ms': np.percentile(latencies, 50) / 1_000_000,
            'p95_latency_ms': np.percentile(latencies, 95) / 1_000_000,
            'p99_latency_ms': np.percentile(latencies, 99) / 1_000_000,
            'dropped_rate': dropped_count / len(recent),
            'retry_rate': sum(1 for e in recent if e.was_retried) / len(recent),
        }
    
    def reset_statistics(self) -> None:
        """Clear event history."""
        self.events.clear()
        self.request_timestamps.clear()
        self.pending_retries.clear()
    
    def calibrate_to_binance_aws(self, region: str = "ap-northeast-1") -> None:
        """
        Calibrate latency settings for specific AWS region.
        
        Binance uses Tokyo (ap-northeast-1) primarily.
        """
        # Approximate RTTs from various locations to Tokyo AWS
        region_rtts = {
            "ap-northeast-1": 80,   # Tokyo - baseline
            "ap-southeast-1": 100,  # Singapore
            "us-east-1": 150,       # Virginia
            "us-west-2": 180,       # Oregon
            "eu-west-1": 220,       # Ireland
            "eu-central-1": 250,    # Frankfurt
        }
        
        base_rtt = region_rtts.get(region, 100)
        self.config = LatencyConfig(
            base_rtt_ms=base_rtt,
            jitter_std_ms=base_rtt * 0.2,
            packet_loss_rate=0.001,
            queue_delay_mean_ms=5.0,
            queue_delay_std_ms=2.0,
            burst_probability=0.01,
            burst_multiplier=5.0,
        )
        logger.info(f"Calibrated for {region}: base_rtt={base_rtt}ms")


# Example usage
if __name__ == "__main__":
    injector = LatencyInjector(LatencyConfig.normal(), seed=42)
    
    # Simulate series of orders
    print("Simulating order submissions...")
    latencies = []
    for i in range(100):
        timestamp_ns = i * 100_000_000  # 100ms apart
        result = injector.simulate_order_round_trip(timestamp_ns, "limit_order")
        latencies.append(result['total_latency_ns'] / 1_000_000)  # Convert to ms
    
    stats = injector.get_latency_statistics()
    print(f"Latency statistics: {stats}")
    
    # Test network degradation
    print("\nSimulating network degradation...")
    injector.set_network_condition(NetworkCondition.POOR)
    
    degraded_latencies = []
    for i in range(50):
        timestamp_ns = (100 + i) * 100_000_000
        result = injector.simulate_order_round_trip(timestamp_ns, "market_order")
        degraded_latencies.append(result['total_latency_ns'] / 1_000_000)
    
    print(f"Degraded mean latency: {np.mean(degraded_latencies):.2f}ms")
    print(f"Normal mean latency: {np.mean(latencies):.2f}ms")
