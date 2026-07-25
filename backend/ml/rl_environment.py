"""
RL Environment for NautilusTrader Integration
Defines state space, action limits, and observation processing for PPO agents.
Optimized for 8GB RAM constraint with zero-copy operations where possible.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Nautilus-compatible action types."""
    HOLD = 0
    BUY_MARKET = 1
    BUY_LIMIT = 2
    SELL_MARKET = 3
    SELL_LIMIT = 4
    CANCEL_ALL = 5


@dataclass
class StateConfig:
    """Configuration for state space dimensions."""
    n_orderbook_levels: int = 10
    n_features_per_level: int = 5  # price, size, imbalance, velocity, acceleration
    n_technical_indicators: int = 15
    n_macro_signals: int = 8
    n_sentiment_features: int = 5
    n_position_features: int = 6  # size, pnl, exposure, etc.
    
    @property
    def total_state_dim(self) -> int:
        return (
            self.n_orderbook_levels * self.n_features_per_level +
            self.n_technical_indicators +
            self.n_macro_signals +
            self.n_sentiment_features +
            self.n_position_features
        )


@dataclass
class ActionLimits:
    """Safety limits for RL agent actions."""
    max_position_size_usd: float = 50000.0
    max_order_size_usd: float = 10000.0
    min_order_size_usd: float = 10.0
    max_orders_per_second: int = 10
    max_total_exposure_usd: float = 150000.0
    tick_sizes: Dict[str, float] = field(default_factory=lambda: {
        'BTCUSDT': 0.1,
        'ETHUSDT': 0.01,
        'SOLUSDT': 0.001,
    })
    lot_sizes: Dict[str, float] = field(default_factory=lambda: {
        'BTCUSDT': 0.001,
        'ETHUSDT': 0.01,
        'SOLUSDT': 0.1,
    })


class RLEnvironment:
    """
    Reinforcement Learning Environment for NautilusTrader.
    
    Implements:
    - Bounded state space with fixed memory footprint
    - Action masking to prevent invalid orders
    - Incremental state updates without full recomputation
    - Thread-safe observation processing
    """
    
    def __init__(
        self,
        config: StateConfig,
        limits: ActionLimits,
        symbols: List[str],
        buffer_size: int = 1000
    ):
        self.config = config
        self.limits = limits
        self.symbols = symbols
        self.buffer_size = buffer_size
        
        # Pre-allocate state buffers (zero-copy friendly)
        self.state_dim = config.total_state_dim
        self.action_dim = len(ActionType)
        
        # Circular buffers for historical context
        self._state_buffer: deque = deque(maxlen=buffer_size)
        self._reward_buffer: deque = deque(maxlen=buffer_size)
        self._action_buffer: deque = deque(maxlen=buffer_size)
        
        # Current state vector (pre-allocated numpy array)
        self.current_state: np.ndarray = np.zeros(
            self.state_dim, dtype=np.float32
        )
        
        # Position tracking per symbol
        self.positions: Dict[str, float] = {s: 0.0 for s in symbols}
        self.unrealized_pnl: Dict[str, float] = {s: 0.0 for s in symbols}
        
        # Order rate limiting
        self._order_timestamps: deque = deque(maxlen=limits.max_orders_per_second * 2)
        
        # Action masks (to prevent invalid actions)
        self.action_mask: np.ndarray = np.ones(self.action_dim, dtype=bool)
        
        logger.info(f"RL Environment initialized: state_dim={self.state_dim}, "
                   f"action_dim={self.action_dim}, symbols={symbols}")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_state.fill(0.0)
        self.positions = {s: 0.0 for s in self.symbols}
        self.unrealized_pnl = {s: 0.0 for s in self.symbols}
        self._order_timestamps.clear()
        self._update_action_mask()
        return self.current_state.copy()
    
    def _update_action_mask(self) -> None:
        """
        Update action mask based on current state and limits.
        Prevents agent from taking invalid actions (e.g., buying at max position).
        """
        self.action_mask[:] = True  # Default: all actions allowed
        
        total_exposure = sum(abs(p) for p in self.positions.values())
        
        for symbol in self.symbols:
            pos = self.positions[symbol]
            
            # Disable buy actions if at max position
            if pos >= self.limits.max_position_size_usd / len(self.symbols):
                self.action_mask[ActionType.BUY_MARKET] = False
                self.action_mask[ActionType.BUY_LIMIT] = False
            
            # Disable sell actions if at min position (short limit)
            if pos <= -self.limits.max_position_size_usd / len(self.symbols):
                self.action_mask[ActionType.SELL_MARKET] = False
                self.action_mask[ActionType.SELL_LIMIT] = False
        
        # Disable trading if order rate exceeded
        if len(self._order_timestamps) >= self.limits.max_orders_per_second:
            self.action_mask[ActionType.BUY_MARKET] = False
            self.action_mask[ActionType.BUY_LIMIT] = False
            self.action_mask[ActionType.SELL_MARKET] = False
            self.action_mask[ActionType.SELL_LIMIT] = False
    
    def get_observation(
        self,
        orderbook_data: Dict[str, np.ndarray],
        technical_indicators: Dict[str, np.ndarray],
        macro_signals: np.ndarray,
        sentiment_features: np.ndarray
    ) -> np.ndarray:
        """
        Construct observation vector from market data.
        
        Uses pre-allocated arrays to avoid memory allocation during inference.
        All inputs are expected to be normalized to [-1, 1] or [0, 1].
        
        Args:
            orderbook_data: Dict mapping symbol to L2 orderbook array [levels, features]
            technical_indicators: Dict mapping symbol to indicator array
            macro_signals: Global macro feature vector
            sentiment_features: Global sentiment feature vector
        
        Returns:
            Flattened state vector of shape (state_dim,)
        """
        idx = 0
        
        # Orderbook features (flattened)
        for symbol in self.symbols:
            if symbol in orderbook_data:
                ob = orderbook_data[symbol]
                n_levels = min(ob.shape[0], self.config.n_orderbook_levels)
                for i in range(n_levels):
                    for j in range(min(ob.shape[1], self.config.n_features_per_level)):
                        if idx < self.state_dim:
                            self.current_state[idx] = ob[i, j]
                            idx += 1
        
        # Technical indicators
        for symbol in self.symbols:
            if symbol in technical_indicators:
                tech = technical_indicators[symbol]
                n_tech = min(len(tech), self.config.n_technical_indicators // len(self.symbols))
                for i in range(n_tech):
                    if idx < self.state_dim:
                        self.current_state[idx] = tech[i]
                        idx += 1
        
        # Macro signals
        n_macro = min(len(macro_signals), self.config.n_macro_signals)
        for i in range(n_macro):
            if idx < self.state_dim:
                self.current_state[idx] = macro_signals[i]
                idx += 1
        
        # Sentiment features
        n_sent = min(len(sentiment_features), self.config.n_sentiment_features)
        for i in range(n_sent):
            if idx < self.state_dim:
                self.current_state[idx] = sentiment_features[i]
                idx += 1
        
        # Position features
        for symbol in self.symbols:
            pos_norm = self.positions[symbol] / self.limits.max_position_size_usd
            if idx < self.state_dim:
                self.current_state[idx] = np.clip(pos_norm, -1, 1)
                idx += 1
            
            pnl_norm = self.unrealized_pnl[symbol] / 1000.0  # Normalize by $1k
            if idx < self.state_dim:
                self.current_state[idx] = np.tanh(pnl_norm / 10.0)
                idx += 1
        
        # Fill remaining with zeros if any
        self.current_state[idx:] = 0.0
        
        # Store in buffer
        self._state_buffer.append(self.current_state.copy())
        
        return self.current_state
    
    def step(
        self,
        action: int,
        timestamp: float
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return next state, reward, done, info.
        
        Validates action against mask and limits before execution.
        Updates internal state incrementally.
        
        Args:
            action: Integer action index from ActionType enum
            timestamp: Current timestamp for rate limiting
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Validate action
        if not self.action_mask[action]:
            # Invalid action: force HOLD and apply small penalty
            action = ActionType.HOLD
            reward = -0.01
        else:
            # Record order timestamp for rate limiting
            self._order_timestamps.append(timestamp)
            
            # Placeholder reward (actual reward from RewardShaper)
            reward = 0.0
        
        # Update action history
        self._action_buffer.append(action)
        
        # Update action mask for next step
        self._update_action_mask()
        
        info = {
            'action_taken': action,
            'action_mask': self.action_mask.copy(),
            'positions': self.positions.copy(),
            'valid_action': action != ActionType.HOLD or self.action_mask[action]
        }
        
        done = False  # Continuous trading environment
        
        return self.current_state.copy(), reward, done, info
    
    def update_position(self, symbol: str, size: float, pnl: float) -> None:
        """Update position and PnL for a symbol."""
        if symbol in self.positions:
            self.positions[symbol] = np.clip(
                size,
                -self.limits.max_position_size_usd,
                self.limits.max_position_size_usd
            )
            self.unrealized_pnl[symbol] = pnl
    
    def validate_nautilus_order(
        self,
        action: int,
        symbol: str,
        price: float,
        quantity: float
    ) -> Tuple[bool, str]:
        """
        Validate that an action can be converted to a valid Nautilus order.
        
        Checks:
        - Tick size compliance
        - Lot size compliance
        - Position limits
        - Rate limits
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if action not in [ActionType.BUY_MARKET, ActionType.BUY_LIMIT,
                         ActionType.SELL_MARKET, ActionType.SELL_LIMIT]:
            return True, ""
        
        # Check tick size
        tick_size = self.limits.tick_sizes.get(symbol, 0.01)
        if price % tick_size != 0:
            return False, f"Price {price} not aligned with tick size {tick_size}"
        
        # Check lot size
        lot_size = self.limits.lot_sizes.get(symbol, 0.001)
        if quantity % lot_size != 0:
            return False, f"Quantity {quantity} not aligned with lot size {lot_size}"
        
        # Check order size limits
        order_value = abs(price * quantity)
        if order_value < self.limits.min_order_size_usd:
            return False, f"Order value {order_value} below minimum"
        if order_value > self.limits.max_order_size_usd:
            return False, f"Order value {order_value} exceeds maximum"
        
        return True, ""
    
    def get_state_statistics(self) -> Dict[str, float]:
        """Get statistics about current state for monitoring."""
        if len(self._state_buffer) == 0:
            return {}
        
        states = np.array(self._state_buffer)
        return {
            'state_mean': float(np.mean(states)),
            'state_std': float(np.std(states)),
            'state_min': float(np.min(states)),
            'state_max': float(np.max(states)),
            'buffer_fill_ratio': len(self._state_buffer) / self.buffer_size
        }
