#!/usr/bin/env python3
"""
Circuit Breaker Telemetry Module

This module tracks circuit breaker state transitions and provides
telemetry for the trading system's protective mechanisms.

Key Features:
- State transition tracking with timestamps
- Failure rate monitoring
- Automatic state progression (Closed -> Open -> Half-Open -> Closed)
- Metrics export for observability
- Thread-safe state management

Designed for the ZAID Personal Crypto Trading Bot to ensure
circuit breakers operate correctly during market stress.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum
import json


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Tripped, blocking operations
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass(slots=True)
class StateTransition:
    """Records a state transition event."""
    timestamp: float
    from_state: CircuitState
    to_state: CircuitState
    reason: str
    failure_count: int = 0
    success_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'from_state': self.from_state.value,
            'to_state': self.to_state.value,
            'reason': self.reason,
            'failure_count': self.failure_count,
            'success_count': self.success_count
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    name: str
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes before closing from half-open
    timeout_seconds: float = 30.0       # Time in open state before half-open
    half_open_max_calls: int = 3        # Max calls allowed in half-open state
    window_size: int = 100              # Sliding window for failure rate
    failure_rate_threshold: float = 0.5 # 50% failure rate triggers open


class CircuitBreakerTelemetry:
    """
    Telemetry system for circuit breakers.
    
    Tracks state transitions, failure rates, and provides
    metrics for monitoring circuit breaker health.
    """
    
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        
        # Counters
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        
        # Sliding window for recent results
        self._recent_results: deque = deque(maxlen=config.window_size)
        
        # Transition history
        self._transitions: List[StateTransition] = []
        
        # Timing
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._created_at = time.time()
        
        # Callbacks
        self._on_transition: Optional[Callable[[StateTransition], None]] = None
        
        # Statistics
        self._total_transitions = 0
        self._total_operations = 0
        self._total_failures = 0
        self._total_successes = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_timeout()
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking operations)."""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN
    
    def _check_timeout(self) -> None:
        """Check if timeout has elapsed to transition from OPEN to HALF_OPEN."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.time() - self._opened_at
            if elapsed >= self.config.timeout_seconds:
                self._transition_to(
                    CircuitState.HALF_OPEN,
                    f"Timeout elapsed ({elapsed:.1f}s >= {self.config.timeout_seconds}s)"
                )
    
    def _transition_to(self, new_state: CircuitState, reason: str) -> None:
        """Perform state transition."""
        old_state = self._state
        if old_state == new_state:
            return
        
        transition = StateTransition(
            timestamp=time.time(),
            from_state=old_state,
            to_state=new_state,
            reason=reason,
            failure_count=self._failure_count,
            success_count=self._success_count
        )
        
        self._state = new_state
        self._transitions.append(transition)
        self._total_transitions += 1
        
        # Reset counters based on new state
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._recent_results.clear()
        elif new_state == CircuitState.OPEN:
            self._opened_at = time.time()
        
        # Notify callback
        if self._on_transition:
            try:
                self._on_transition(transition)
            except Exception as e:
                print(f"Transition callback error: {e}")
        
        print(f"[CIRCUIT BREAKER] {self.config.name}: {old_state.value} -> {new_state.value} ({reason})")
    
    def record_success(self) -> bool:
        """
        Record a successful operation.
        
        Returns:
            True if operation should proceed, False if blocked
        """
        with self._lock:
            self._check_timeout()
            
            if self._state == CircuitState.OPEN:
                return False
            
            self._success_count += 1
            self._total_successes += 1
            self._total_operations += 1
            self._recent_results.append(True)
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(
                        CircuitState.CLOSED,
                        f"Success threshold reached ({self._success_count} >= {self.config.success_threshold})"
                    )
                elif self._half_open_calls >= self.config.half_open_max_calls:
                    # Max calls reached but not enough successes, stay half-open or reopen
                    if self._success_count < self._half_open_calls // 2:
                        self._transition_to(
                            CircuitState.OPEN,
                            "Half-open testing failed"
                        )
            
            return True
    
    def record_failure(self) -> bool:
        """
        Record a failed operation.
        
        Returns:
            True if operation was attempted, False if blocked
        """
        with self._lock:
            self._check_timeout()
            
            self._failure_count += 1
            self._total_failures += 1
            self._total_operations += 1
            self._last_failure_time = time.time()
            self._recent_results.append(False)
            
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(
                        CircuitState.OPEN,
                        f"Failure threshold reached ({self._failure_count} >= {self.config.failure_threshold})"
                    )
                elif self._get_failure_rate() >= self.config.failure_rate_threshold:
                    self._transition_to(
                        CircuitState.OPEN,
                        f"Failure rate exceeded ({self._get_failure_rate():.2%} >= {self.config.failure_rate_threshold:.2%})"
                    )
            
            elif self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                self._transition_to(
                    CircuitState.OPEN,
                    f"Failure during half-open testing"
                )
            
            return self._state != CircuitState.OPEN
    
    def _get_failure_rate(self) -> float:
        """Calculate failure rate over recent results."""
        if not self._recent_results:
            return 0.0
        
        failures = sum(1 for r in self._recent_results if not r)
        return failures / len(self._recent_results)
    
    def can_execute(self) -> bool:
        """Check if an operation can be executed."""
        with self._lock:
            self._check_timeout()
            
            if self._state == CircuitState.OPEN:
                return False
            
            if self._state == CircuitState.HALF_OPEN:
                return self._half_open_calls < self.config.half_open_max_calls
            
            return True
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
        """
        if not self.can_execute():
            raise CircuitOpenError(f"Circuit breaker {self.config.name} is open")
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def set_transition_callback(self, callback: Callable[[StateTransition], None]) -> None:
        """Set callback for state transitions."""
        self._on_transition = callback
    
    def get_transition_history(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent state transitions."""
        return [t.to_dict() for t in self._transitions[-count:]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self._lock:
            uptime = time.time() - self._created_at
            
            return {
                'name': self.config.name,
                'state': self._state.value,
                'uptime_seconds': uptime,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'failure_rate': self._get_failure_rate(),
                'total_transitions': self._total_transitions,
                'total_operations': self._total_operations,
                'total_failures': self._total_failures,
                'total_successes': self._total_successes,
                'half_open_calls': self._half_open_calls,
                'last_failure_time': self._last_failure_time,
                'opened_at': self._opened_at,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout_seconds': self.config.timeout_seconds,
                    'failure_rate_threshold': self.config.failure_rate_threshold,
                }
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                self._transition_to(CircuitState.CLOSED, "Manual reset")
            else:
                self._failure_count = 0
                self._success_count = 0
                self._recent_results.clear()


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides centralized management and telemetry for all
    circuit breakers in the trading system.
    """
    
    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreakerTelemetry] = {}
        self._lock = threading.RLock()
    
    def create_breaker(self, config: CircuitBreakerConfig) -> CircuitBreakerTelemetry:
        """Create and register a new circuit breaker."""
        with self._lock:
            if config.name in self._breakers:
                return self._breakers[config.name]
            
            breaker = CircuitBreakerTelemetry(config)
            self._breakers[config.name] = breaker
            return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreakerTelemetry]:
        """Get a circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_statistics()
                for name, breaker in self._breakers.items()
            }
    
    def get_open_breakers(self) -> List[str]:
        """Get names of all open circuit breakers."""
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.is_open
            ]
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry instance
_registry: Optional[CircuitBreakerRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> CircuitBreakerRegistry:
    """Get or create the global circuit breaker registry."""
    global _registry
    
    with _registry_lock:
        if _registry is None:
            _registry = CircuitBreakerRegistry()
        return _registry


# Pre-configured circuit breakers for common use cases
def create_trading_circuit_breaker(name: str = "trading") -> CircuitBreakerTelemetry:
    """Create a circuit breaker for trading operations."""
    config = CircuitBreakerConfig(
        name=name,
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=60.0,
        failure_rate_threshold=0.4
    )
    return get_registry().create_breaker(config)


def create_exchange_circuit_breaker(exchange: str) -> CircuitBreakerTelemetry:
    """Create a circuit breaker for exchange connectivity."""
    config = CircuitBreakerConfig(
        name=f"exchange_{exchange}",
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=30.0,
        failure_rate_threshold=0.5
    )
    return get_registry().create_breaker(config)


if __name__ == '__main__':
    # Example usage and testing
    print("Initializing Circuit Breaker Telemetry...")
    
    config = CircuitBreakerConfig(
        name="test_breaker",
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=2.0
    )
    
    breaker = CircuitBreakerTelemetry(config)
    
    # Set transition callback
    def on_transition(t: StateTransition) -> None:
        print(f"  Transition: {t.from_state.value} -> {t.to_state.value}: {t.reason}")
    
    breaker.set_transition_callback(on_transition)
    
    print("\nSimulating operations...")
    
    # Successful operations
    print("  Success x3:")
    for i in range(3):
        breaker.record_success()
        print(f"    State: {breaker.state.value}, Can execute: {breaker.can_execute()}")
    
    # Failures to trip circuit
    print("\n  Failure x3 (should trip):")
    for i in range(3):
        breaker.record_failure()
        print(f"    State: {breaker.state.value}, Can execute: {breaker.can_execute()}")
    
    # Try to execute while open
    print(f"\n  Trying to execute while open: {breaker.can_execute()}")
    
    # Wait for timeout
    print(f"\n  Waiting {config.timeout_seconds}s for timeout...")
    time.sleep(config.timeout_seconds + 0.5)
    
    print(f"  State after timeout: {breaker.state.value}")
    
    # Successful operations in half-open
    print("\n  Success x2 in half-open (should close):")
    for i in range(2):
        breaker.record_success()
        print(f"    State: {breaker.state.value}")
    
    print("\nStatistics:")
    stats = breaker.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\nTransition History:")
    for t in breaker.get_transition_history():
        print(f"  {t['from_state']} -> {t['to_state']}: {t['reason']}")
    
    print("\nCircuit Breaker Telemetry test complete.")
