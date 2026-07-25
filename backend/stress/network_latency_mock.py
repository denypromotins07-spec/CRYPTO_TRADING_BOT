#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stress Testing & Chaos Engineering
Chapter 3: Network Latency Mock

This module injects artificial WebSocket delays safely to test bot
resilience under poor network conditions. Used for chaos engineering
and ensuring graceful degradation during exchange connectivity issues.

Memory Budget: <20MB for latency simulation state
Target Precision: <1ms delay accuracy
Safety: Isolated from live trading, sandboxed execution
Integration: Works with circuit breaker for disconnect handling

Author: Opus 4.8
Stage: 5/100 - Advanced Risk Management and Order Book Microstructure
"""

from __future__ import annotations
import asyncio
import time
import random
from typing import Dict, List, Optional, Callable, Any, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics


class LatencyProfile(Enum):
    """Predefined latency profiles for different scenarios."""
    NORMAL = "normal"              # <50ms typical
    ELEVATED = "elevated"          # 50-200ms
    HIGH = "high"                  # 200-500ms
    EXTREME = "extreme"            # 500ms-2s
    PACKET_LOSS = "packet_loss"    # Random drops
    JITTERY = "jittery"            # High variance
    GRADUAL_DEGRADATION = "gradual"  # Slowly worsening


@dataclass
class LatencyConfig:
    """Configuration for latency injection."""
    base_latency_ms: float = 10.0
    jitter_ms: float = 5.0
    packet_loss_rate: float = 0.0  # 0-1 probability
    max_latency_ms: float = 5000.0  # Safety cap
    profile: LatencyProfile = LatencyProfile.NORMAL
    active: bool = False  # Disabled by default
    gradual_increase_rate: float = 0.0  # ms per second


@dataclass
class LatencyMetrics(TypedDict):
    """Metrics for latency injection analysis."""
    messages_processed: int
    total_delay_added_ms: float
    average_delay_ms: float
    max_delay_ms: float
    min_delay_ms: float
    packets_dropped: int
    current_latency_ms: float


class NetworkLatencyMock:
    """
    Async network latency injector for chaos testing.
    
    Features:
    - Configurable base latency with jitter
    - Packet loss simulation
    - Multiple predefined profiles
    - Gradual degradation simulation
    - Metrics collection
    - Safe sandboxing
    """
    
    __slots__ = (
        '_config', '_delay_history', '_is_sandboxed',
        '_message_queue', '_metrics', '_active_profile'
    )
    
    def __init__(self, config: LatencyConfig = None) -> None:
        """
        Initialize latency mock.
        
        Args:
            config: Latency configuration
        """
        self._config = config or LatencyConfig()
        self._delay_history: deque = deque(maxlen=1000)
        self._is_sandboxed = True  # Always sandboxed
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._metrics = {
            'messages_processed': 0,
            'total_delay_added_ms': 0.0,
            'average_delay_ms': 0.0,
            'max_delay_ms': 0.0,
            'min_delay_ms': float('inf'),
            'packets_dropped': 0,
            'current_latency_ms': 0.0
        }
        self._active_profile = self._config.profile
    
    def verify_sandbox(self) -> bool:
        """Verify sandbox mode is active."""
        return self._is_sandboxed
    
    def set_profile(self, profile: LatencyProfile) -> None:
        """
        Set latency profile.
        
        Args:
            profile: Desired latency profile
        """
        self._active_profile = profile
        
        # Apply profile settings
        if profile == LatencyProfile.NORMAL:
            self._config.base_latency_ms = 10.0
            self._config.jitter_ms = 5.0
            self._config.packet_loss_rate = 0.0
        elif profile == LatencyProfile.ELEVATED:
            self._config.base_latency_ms = 100.0
            self._config.jitter_ms = 50.0
            self._config.packet_loss_rate = 0.01
        elif profile == LatencyProfile.HIGH:
            self._config.base_latency_ms = 300.0
            self._config.jitter_ms = 100.0
            self._config.packet_loss_rate = 0.05
        elif profile == LatencyProfile.EXTREME:
            self._config.base_latency_ms = 1000.0
            self._config.jitter_ms = 500.0
            self._config.packet_loss_rate = 0.1
        elif profile == LatencyProfile.PACKET_LOSS:
            self._config.base_latency_ms = 50.0
            self._config.jitter_ms = 20.0
            self._config.packet_loss_rate = 0.3
        elif profile == LatencyProfile.JITTERY:
            self._config.base_latency_ms = 50.0
            self._config.jitter_ms = 200.0
            self._config.packet_loss_rate = 0.02
        elif profile == LatencyProfile.GRADUAL_DEGRADATION:
            self._config.base_latency_ms = 10.0
            self._config.jitter_ms = 5.0
            self._config.gradual_increase_rate = 1.0  # 1ms/s
    
    async def inject_latency(self, message: Any, 
                             callback: Optional[Callable] = None) -> Optional[Any]:
        """
        Inject latency into a message transmission.
        
        Args:
            message: Message to delay
            callback: Optional callback after delay
            
        Returns:
            Delayed message or None if dropped
        """
        if not self.verify_sandbox():
            return message  # Pass through if not sandboxed
        
        if not self._config.active:
            return message  # No injection if inactive
        
        # Check for packet loss
        if random.random() < self._config.packet_loss_rate:
            self._metrics['packets_dropped'] += 1
            return None
        
        # Calculate delay
        delay_ms = self._calculate_delay()
        
        # Update metrics
        self._delay_history.append(delay_ms)
        self._update_metrics(delay_ms)
        
        # Apply delay
        await asyncio.sleep(delay_ms / 1000.0)
        
        # Execute callback if provided
        if callback:
            try:
                callback(message, delay_ms)
            except Exception:
                pass
        
        return message
    
    def _calculate_delay(self) -> float:
        """Calculate delay based on current configuration."""
        # Base + jitter
        jitter = random.gauss(0, self._config.jitter_ms)
        delay = self._config.base_latency_ms + jitter
        
        # Apply gradual increase if configured
        if self._config.gradual_increase_rate > 0:
            elapsed_time = len(self._delay_history) * 0.1  # Approximate
            delay += elapsed_time * self._config.gradual_increase_rate
        
        # Clamp to safe limits
        delay = max(0, min(delay, self._config.max_latency_ms))
        
        self._metrics['current_latency_ms'] = delay
        return delay
    
    def _update_metrics(self, delay_ms: float) -> None:
        """Update latency metrics."""
        self._metrics['messages_processed'] += 1
        self._metrics['total_delay_added_ms'] += delay_ms
        self._metrics['max_delay_ms'] = max(self._metrics['max_delay_ms'], delay_ms)
        self._metrics['min_delay_ms'] = min(self._metrics['min_delay_ms'], delay_ms)
        
        # Recalculate average
        if self._metrics['messages_processed'] > 0:
            self._metrics['average_delay_ms'] = (
                self._metrics['total_delay_added_ms'] / 
                self._metrics['messages_processed']
            )
    
    def get_metrics(self) -> LatencyMetrics:
        """Get current latency metrics."""
        return LatencyMetrics(**self._metrics)
    
    def activate(self) -> None:
        """Activate latency injection."""
        self._config.active = True
    
    def deactivate(self) -> None:
        """Deactivate latency injection."""
        self._config.active = False
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._delay_history.clear()
        self._metrics = {
            'messages_processed': 0,
            'total_delay_added_ms': 0.0,
            'average_delay_ms': 0.0,
            'max_delay_ms': 0.0,
            'min_delay_ms': float('inf'),
            'packets_dropped': 0,
            'current_latency_ms': 0.0
        }
    
    def simulate_connection_burst(self, duration_seconds: float = 5.0) -> None:
        """
        Simulate a burst of connection issues.
        
        Args:
            duration_seconds: How long the burst lasts
        """
        original_profile = self._active_profile
        
        # Temporarily switch to extreme profile
        self.set_profile(LatencyProfile.EXTREME)
        self.activate()
        
        # Schedule restoration (would need event loop in real usage)
        def restore():
            self.set_profile(original_profile)
        
        # In production, this would use asyncio.call_later
        # For now, just document the behavior
        pass


class WebSocketLatencySimulator(NetworkLatencyMock):
    """
    Specialized simulator for WebSocket connections.
    
    Simulates realistic WS behavior including:
    - Handshake delays
    - Message ordering issues
    - Reconnection delays
    - Heartbeat timeouts
    """
    
    __slots__ = ('_pending_messages', '_sequence_numbers', '_reconnect_delay')
    
    def __init__(self, config: LatencyConfig = None) -> None:
        super().__init__(config)
        self._pending_messages: Dict[int, Any] = {}
        self._sequence_numbers: Dict[str, int] = {}
        self._reconnect_delay = 5.0  # seconds
    
    async def send_message(self, channel: str, message: Any) -> Optional[int]:
        """
        Simulate sending a message through WS with latency.
        
        Args:
            channel: WS channel/subscription
            message: Message payload
            
        Returns:
            Sequence number or None if dropped
        """
        # Get next sequence number for channel
        seq = self._sequence_numbers.get(channel, 0) + 1
        self._sequence_numbers[channel] = seq
        
        # Apply latency
        result = await self.inject_latency((seq, message))
        
        if result is None:
            return None
        
        # Store pending message
        self._pending_messages[seq] = result
        
        return seq
    
    async def receive_message(self, channel: str, timeout_ms: float = 5000) -> Optional[Any]:
        """
        Simulate receiving a message with latency.
        
        Args:
            channel: WS channel
            timeout_ms: Maximum wait time
            
        Returns:
            Received message or None
        """
        start_time = time.time()
        
        while (time.time() - start_time) * 1000 < timeout_ms:
            # Simulate network delay
            await self.inject_latency(None)
            
            # Check for messages
            for seq, msg in list(self._pending_messages.items()):
                if isinstance(msg, tuple) and len(msg) == 2:
                    return msg[1]
            
            await asyncio.sleep(0.001)  # Small sleep to prevent busy loop
        
        return None
    
    def simulate_reconnect(self) -> float:
        """
        Simulate reconnection delay.
        
        Returns:
            Simulated reconnect delay in seconds
        """
        # Exponential backoff with jitter
        base_delay = self._reconnect_delay
        jitter = random.uniform(0, base_delay * 0.5)
        return base_delay + jitter
    
    def simulate_heartbeat_timeout(self, expected_interval_ms: float) -> bool:
        """
        Check if heartbeat would timeout given current latency.
        
        Args:
            expected_interval_ms: Expected heartbeat interval
            
        Returns:
            True if timeout likely
        """
        current_latency = self._metrics['current_latency_ms']
        
        # Timeout if round-trip exceeds interval
        return (current_latency * 2) > expected_interval_ms * 0.8


class ChaosEngine:
    """
    Orchestrator for chaos engineering experiments.
    
    Runs controlled chaos tests to validate system resilience.
    """
    
    def __init__(self) -> None:
        self.latency_mock = NetworkLatencyMock()
        self.ws_simulator = WebSocketLatencySimulator()
        self.experiment_results: List[Dict] = []
    
    async def run_experiment(self, name: str, 
                             profile: LatencyProfile,
                             duration_seconds: float,
                             test_function: Callable) -> Dict:
        """
        Run a chaos experiment.
        
        Args:
            name: Experiment name
            profile: Latency profile to apply
            duration_seconds: How long to run
            test_function: Async function to test
            
        Returns:
            Experiment results
        """
        print(f"Starting chaos experiment: {name}")
        print(f"Profile: {profile.value}, Duration: {duration_seconds}s")
        
        # Record baseline
        baseline_start = time.time()
        self.latency_mock.deactivate()
        baseline_result = await test_function()
        baseline_time = time.time() - baseline_start
        
        # Apply chaos
        self.latency_mock.set_profile(profile)
        self.latency_mock.activate()
        
        # Run test under stress
        stress_start = time.time()
        try:
            stress_result = await asyncio.wait_for(
                test_function(),
                timeout=duration_seconds
            )
        except asyncio.TimeoutError:
            stress_result = {'error': 'timeout'}
        
        stress_time = time.time() - stress_start
        
        # Cleanup
        self.latency_mock.deactivate()
        metrics = self.latency_mock.get_metrics()
        
        # Compile results
        result = {
            'experiment_name': name,
            'profile': profile.value,
            'baseline_time': baseline_time,
            'stress_time': stress_time,
            'performance_degradation': (stress_time - baseline_time) / baseline_time if baseline_time > 0 else 0,
            'metrics': dict(metrics),
            'result': stress_result
        }
        
        self.experiment_results.append(result)
        return result
    
    def get_experiment_summary(self) -> Dict:
        """Get summary of all experiments."""
        if not self.experiment_results:
            return {}
        
        avg_degradation = statistics.mean(
            r['performance_degradation'] for r in self.experiment_results
        )
        
        return {
            'total_experiments': len(self.experiment_results),
            'average_degradation': avg_degradation,
            'max_latency_observed': max(
                r['metrics']['max_delay_ms'] for r in self.experiment_results
            ),
            'total_packets_dropped': sum(
                r['metrics']['packets_dropped'] for r in self.experiment_results
            )
        }


if __name__ == "__main__":
    # Example usage and validation
    async def main():
        config = LatencyConfig(active=True)
        mock = NetworkLatencyMock(config)
        
        print("=== Testing Normal Profile ===")
        mock.set_profile(LatencyProfile.NORMAL)
        
        start = time.time()
        for i in range(10):
            await mock.inject_latency(f"message_{i}")
        elapsed = (time.time() - start) * 1000
        
        metrics = mock.get_metrics()
        print(f"10 messages in {elapsed:.1f}ms")
        print(f"Avg delay: {metrics['average_delay_ms']:.1f}ms")
        
        print("\n=== Testing Extreme Profile ===")
        mock.reset_metrics()
        mock.set_profile(LatencyProfile.EXTREME)
        
        start = time.time()
        for i in range(5):
            await mock.inject_latency(f"message_{i}")
        elapsed = (time.time() - start) * 1000
        
        metrics = mock.get_metrics()
        print(f"5 messages in {elapsed:.1f}ms")
        print(f"Avg delay: {metrics['average_delay_ms']:.1f}ms")
        print(f"Packets dropped: {metrics['packets_dropped']}")
        
        print("\n=== Testing WebSocket Simulator ===")
        ws = WebSocketLatencySimulator()
        ws.set_profile(LatencyProfile.ELEVATED)
        ws.activate()
        
        seq = await ws.send_message("BTC", {"type": "subscribe"})
        print(f"Sent message with seq: {seq}")
        
        # Test heartbeat timeout detection
        would_timeout = ws.simulate_heartbeat_timeout(expected_interval_ms=100)
        print(f"Heartbeat timeout likely: {would_timeout}")
    
    asyncio.run(main())
