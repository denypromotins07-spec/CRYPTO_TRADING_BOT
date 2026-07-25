"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
Chapter 2: NautilusTrader Backtesting Engine

File: backend/backtest/latency_simulator.py
Purpose: Inject historical network delays and slippage into backtests.
Features:
    - Realistic latency distribution modeling
    - Slippage based on order size and market depth
    - Exchange-specific delay profiles
    - Memory-efficient streaming injection
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LatencyProfile(Enum):
    """Predefined latency profiles for different scenarios."""
    NORMAL = "normal"           # Typical market conditions
    HIGH_VOLATILITY = "high_volatility"  # During news/events
    FLASH_CRASH = "flash_crash"          # Extreme stress
    EXCHANGE_OUTAGE = "outage"           # Partial connectivity
    WEEKEND = "weekend"                  # Low liquidity periods


@dataclass
class LatencyConfig:
    """Configuration for latency simulation."""
    profile: LatencyProfile = LatencyProfile.NORMAL
    
    # Base latency in milliseconds
    base_latency_ms: float = 50.0
    
    # Latency jitter (standard deviation)
    jitter_ms: float = 10.0
    
    # Probability of extreme latency spike
    spike_probability: float = 0.01
    
    # Maximum latency during spikes
    max_spike_latency_ms: float = 5000.0
    
    # Packet loss probability
    packet_loss_prob: float = 0.001
    
    # Reorder probability (packets arriving out of order)
    reorder_prob: float = 0.005


@dataclass
class SlippageConfig:
    """Configuration for slippage simulation."""
    # Base slippage as percentage of price
    base_slippage_pct: float = 0.0005  # 0.05%
    
    # Slippage increase per unit of order size relative to ADTV
    size_impact_factor: float = 0.001
    
    # Volatility multiplier on slippage
    volatility_multiplier: float = 2.0
    
    # Minimum slippage (exchange fees, etc.)
    min_slippage_pct: float = 0.0001
    
    # Maximum slippage cap
    max_slippage_pct: float = 0.02  # 2%


@dataclass
class SimulatedOrder:
    """Represents an order with simulated latency/slippage."""
    original_price: float
    original_timestamp: float
    quantity: float
    side: str  # 'buy' or 'sell'
    symbol: str
    
    # Simulated values
    executed_price: Optional[float] = None
    executed_timestamp: Optional[float] = None
    latency_ms: Optional[float] = None
    slippage_pct: Optional[float] = None
    was_dropped: bool = False
    was_reordered: bool = False


class LatencySimulator:
    """
    Simulates realistic network latency for backtesting.
    Uses statistical models based on historical exchange latency data.
    """
    
    __slots__ = [
        'config',
        'slippage_config',
        'latency_history',
        'current_volatility',
        'random_state'
    ]
    
    def __init__(
        self,
        config: LatencyConfig = None,
        slippage_config: SlippageConfig = None,
        seed: int = None
    ):
        self.config = config or LatencyConfig()
        self.slippage_config = slippage_config or SlippageConfig()
        self.latency_history: List[float] = []
        self.current_volatility: float = 1.0
        self.random_state = np.random.RandomState(seed)
        
        self._configure_profile()
    
    def _configure_profile(self) -> None:
        """Configure parameters based on latency profile."""
        profile_configs = {
            LatencyProfile.NORMAL: LatencyConfig(
                base_latency_ms=50.0,
                jitter_ms=10.0,
                spike_probability=0.01,
                max_spike_latency_ms=5000.0,
                packet_loss_prob=0.001,
                reorder_prob=0.005
            ),
            LatencyProfile.HIGH_VOLATILITY: LatencyConfig(
                base_latency_ms=150.0,
                jitter_ms=50.0,
                spike_probability=0.05,
                max_spike_latency_ms=10000.0,
                packet_loss_prob=0.005,
                reorder_prob=0.02
            ),
            LatencyProfile.FLASH_CRASH: LatencyConfig(
                base_latency_ms=500.0,
                jitter_ms=200.0,
                spike_probability=0.2,
                max_spike_latency_ms=30000.0,
                packet_loss_prob=0.02,
                reorder_prob=0.1
            ),
            LatencyProfile.EXCHANGE_OUTAGE: LatencyConfig(
                base_latency_ms=2000.0,
                jitter_ms=1000.0,
                spike_probability=0.5,
                max_spike_latency_ms=60000.0,
                packet_loss_prob=0.1,
                reorder_prob=0.2
            ),
            LatencyProfile.WEEKEND: LatencyConfig(
                base_latency_ms=100.0,
                jitter_ms=30.0,
                spike_probability=0.02,
                max_spike_latency_ms=8000.0,
                packet_loss_prob=0.002,
                reorder_prob=0.01
            ),
        }
        
        if self.config.profile in profile_configs:
            profile_config = profile_configs[self.config.profile]
            # Override non-profile settings
            self.config.base_latency_ms = profile_config.base_latency_ms
            self.config.jitter_ms = profile_config.jitter_ms
            self.config.spike_probability = profile_config.spike_probability
            self.config.max_spike_latency_ms = profile_config.max_spike_latency_ms
            self.config.packet_loss_prob = profile_config.packet_loss_prob
            self.config.reorder_prob = profile_config.reorder_prob
    
    def simulate_latency(self, timestamp: float) -> Tuple[float, bool, bool]:
        """
        Simulate network latency for a given timestamp.
        Returns: (delayed_timestamp, was_dropped, was_reordered)
        """
        # Check for packet loss
        if self.random_state.random() < self.config.packet_loss_prob:
            logger.debug("Packet lost in simulation")
            return timestamp + self.config.max_spike_latency_ms, True, False
        
        # Check for reorder
        was_reordered = self.random_state.random() < self.config.reorder_prob
        
        # Base latency with jitter (log-normal distribution for realism)
        base_latency = self.random_state.lognormal(
            mean=np.log(self.config.base_latency_ms),
            sigma=np.log(self.config.jitter_ms / self.config.base_latency_ms + 1)
        )
        
        # Check for latency spike
        if self.random_state.random() < self.config.spike_probability:
            spike_latency = self.random_state.uniform(
                self.config.base_latency_ms * 10,
                self.config.max_spike_latency_ms
            )
            latency = max(base_latency, spike_latency)
            logger.debug(f"Latency spike detected: {latency:.2f}ms")
        else:
            latency = base_latency
        
        # Apply volatility multiplier
        latency *= self.current_volatility
        
        delayed_timestamp = timestamp + (latency / 1000.0)  # Convert ms to seconds
        
        # Record for statistics
        self.latency_history.append(latency)
        
        return delayed_timestamp, False, was_reordered
    
    def simulate_slippage(
        self,
        price: float,
        quantity: float,
        side: str,
        market_depth: float = 1.0,
        current_volatility: float = 1.0
    ) -> float:
        """
        Simulate execution slippage based on order characteristics.
        Returns the slippage-adjusted price.
        """
        # Base slippage
        base_slippage = self.slippage_config.base_slippage_pct
        
        # Size impact: larger orders cause more slippage
        # market_depth represents average daily volume normalization
        size_ratio = quantity / max(market_depth, 1.0)
        size_impact = size_ratio * self.slippage_config.size_impact_factor
        
        # Volatility impact
        vol_impact = current_volatility * self.slippage_config.volatility_multiplier * 0.0001
        
        # Total slippage percentage
        total_slippage_pct = base_slippage + size_impact + vol_impact
        
        # Apply bounds
        total_slippage_pct = max(
            self.slippage_config.min_slippage_pct,
            min(total_slippage_pct, self.slippage_config.max_slippage_pct)
        )
        
        # Apply slippage based on side
        if side.lower() == 'buy':
            # Buy orders slip upward
            slippage_amount = price * total_slippage_pct
            executed_price = price + slippage_amount
        else:
            # Sell orders slip downward
            slippage_amount = price * total_slippage_pct
            executed_price = price - slippage_amount
        
        return executed_price
    
    def process_order(
        self,
        order: SimulatedOrder,
        market_depth: float = 1.0
    ) -> SimulatedOrder:
        """
        Process an order through latency and slippage simulation.
        Returns the order with simulated execution details.
        """
        # Simulate latency
        executed_ts, dropped, reordered = self.simulate_latency(
            order.original_timestamp
        )
        
        order.executed_timestamp = executed_ts
        order.was_dropped = dropped
        order.was_reordered = reordered
        
        if dropped:
            logger.warning(f"Order dropped due to simulated packet loss: {order.symbol}")
            return order
        
        # Simulate slippage
        executed_price = self.simulate_slippage(
            price=order.original_price,
            quantity=order.quantity,
            side=order.side,
            market_depth=market_depth,
            current_volatility=self.current_volatility
        )
        
        order.executed_price = executed_price
        order.slippage_pct = abs(executed_price - order.original_price) / order.original_price
        
        return order
    
    def update_volatility(self, volatility: float) -> None:
        """Update current market volatility estimate."""
        self.current_volatility = max(0.1, min(volatility, 10.0))
        logger.debug(f"Volatility updated to {self.current_volatility}")
    
    def get_statistics(self) -> Dict:
        """Get latency simulation statistics."""
        if not self.latency_history:
            return {"error": "No latency data recorded"}
        
        latencies = np.array(self.latency_history)
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'max_latency_ms': float(np.max(latencies)),
            'sample_count': len(latencies),
            'current_volatility': self.current_volatility,
            'profile': self.config.profile.value
        }
    
    def reset(self) -> None:
        """Reset simulator state."""
        self.latency_history = []
        self.current_volatility = 1.0
        self._configure_profile()
        logger.info("Latency simulator reset")


class LatencyInjectionMiddleware:
    """
    Middleware class for injecting latency into backtest streams.
    Can be used as a wrapper around data feeds or order execution.
    """
    
    def __init__(
        self,
        simulator: LatencySimulator,
        buffer_size: int = 1000
    ):
        self.simulator = simulator
        self.buffer: List[SimulatedOrder] = []
        self.buffer_size = buffer_size
    
    def inject(self, order_dict: Dict) -> Optional[Dict]:
        """Inject latency into an order dictionary."""
        order = SimulatedOrder(
            original_price=order_dict.get('price', 0.0),
            original_timestamp=order_dict.get('timestamp', 0.0),
            quantity=order_dict.get('quantity', 1.0),
            side=order_dict.get('side', 'buy'),
            symbol=order_dict.get('symbol', 'UNKNOWN')
        )
        
        processed = self.simulator.process_order(order)
        
        if processed.was_dropped:
            return None
        
        return {
            **order_dict,
            'executed_price': processed.executed_price,
            'executed_timestamp': processed.executed_timestamp,
            'latency_ms': processed.latency_ms,
            'slippage_pct': processed.slippage_pct,
            'was_reordered': processed.was_reordered
        }
    
    def inject_batch(self, orders: List[Dict]) -> List[Dict]:
        """Inject latency into a batch of orders."""
        results = []
        for order_dict in orders:
            result = self.inject(order_dict)
            if result is not None:
                results.append(result)
        return results


def main():
    """Example usage of the latency simulator."""
    print("="*60)
    print("LATENCY SIMULATOR DEMO")
    print("="*60)
    
    # Create simulator with high volatility profile
    config = LatencyConfig(profile=LatencyProfile.HIGH_VOLATILITY)
    simulator = LatencySimulator(config=config, seed=42)
    
    # Simulate some orders
    test_orders = [
        SimulatedOrder(
            original_price=50000.0,
            original_timestamp=1000.0 + i * 0.1,
            quantity=0.1 * (i % 10 + 1),
            side='buy' if i % 2 == 0 else 'sell',
            symbol='BTCUSDT'
        )
        for i in range(20)
    ]
    
    print("\nProcessing orders...")
    for order in test_orders:
        processed = simulator.process_order(order, market_depth=100.0)
        
        status = "DROPPED" if processed.was_dropped else "OK"
        if processed.slippage_pct:
            status += f" | Slippage: {processed.slippage_pct*100:.4f}%"
        
        print(f"  {order.symbol} {order.side}: ${processed.original_price:.2f} -> "
              f"${processed.executed_price:.2f if processed.executed_price else 'N/A'} "
              f"[{status}]")
    
    # Print statistics
    stats = simulator.get_statistics()
    print("\n" + "="*60)
    print("LATENCY STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    main()
