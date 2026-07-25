"""
Inventory Risk Management for Market Making
Applies penalty functions to prevent directional drift and enforce delta-neutrality.
Optimized for 8GB RAM constraint with efficient numpy operations.

This module implements:
- Quadratic inventory penalty (mean-variance optimization)
- Dynamic position limits based on volatility
- Inventory skew adjustment for quote placement
- Real-time risk monitoring with alerts

Target: Prevent naked directional exposure while capturing spreads.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum
import numpy as np
from collections import deque


class RiskLevel(Enum):
    """Risk severity levels for inventory management."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InventoryConfig:
    """Configuration for inventory risk management."""
    # Maximum absolute inventory per asset (in base units)
    max_inventory: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.5,
        "ETH": 5.0,
        "SOL": 50.0,
        "USDT": 10000.0,
    })
    
    # Risk aversion coefficient (gamma in AS model)
    gamma: float = 0.1
    
    # Volatility scaling factor for dynamic limits
    vol_scale_factor: float = 2.0
    
    # Penalty exponent for quadratic cost function
    penalty_exponent: float = 2.0
    
    # Warning threshold (% of max inventory)
    warning_threshold: float = 0.7
    
    # Critical threshold (% of max inventory)
    critical_threshold: float = 0.9
    
    # Lookback window for volatility estimation (seconds)
    vol_lookback: int = 300


@dataclass
class AssetInventory:
    """Tracks inventory state for a single asset."""
    symbol: str
    quantity: float = 0.0
    average_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    last_update_ts: int = 0
    
    def update_position(self, quantity_delta: float, price: float, timestamp: int) -> None:
        """Update position after a fill."""
        old_qty = self.quantity
        
        # Update quantity
        self.quantity += quantity_delta
        
        # Update average entry price using weighted average
        if (old_qty >= 0 and quantity_delta > 0) or (old_qty <= 0 and quantity_delta < 0):
            # Adding to position (same direction)
            total_cost = abs(old_qty) * self.average_entry_price + abs(quantity_delta) * price
            new_total_qty = abs(self.quantity)
            if new_total_qty > 0:
                self.average_entry_price = total_cost / new_total_qty
        else:
            # Reducing or reversing position
            remaining_qty = abs(old_qty) - abs(quantity_delta)
            if remaining_qty <= 0:
                # Position fully closed or reversed
                if self.quantity != 0:
                    self.average_entry_price = price
                else:
                    self.average_entry_price = 0.0
        
        self.last_update_ts = timestamp
    
    def get_notional_value(self, current_price: float) -> float:
        """Get current notional value of inventory."""
        return self.quantity * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.quantity == 0:
            return 0.0
        return (current_price - self.average_entry_price) * self.quantity


class InventoryRiskManager:
    """
    Main inventory risk management engine.
    
    Implements quadratic penalty functions to discourage large inventories
    and dynamically adjusts position limits based on market volatility.
    """
    
    def __init__(self, config: Optional[InventoryConfig] = None):
        self.config = config or InventoryConfig()
        
        # Per-asset inventory tracking
        self.inventories: Dict[str, AssetInventory] = {}
        
        # Price history for volatility estimation (ring buffer)
        self.price_history: Dict[str, deque] = {}
        
        # Initialize inventories for configured assets
        for symbol in self.config.max_inventory.keys():
            self.inventories[symbol] = AssetInventory(symbol=symbol)
            self.price_history[symbol] = deque(maxlen=self.config.vol_lookback)
        
        # Current volatility estimates (annualized)
        self.volatility_estimates: Dict[str, float] = {s: 0.01 for s in self.config.max_inventory}
        
        # Risk state cache
        self._risk_cache: Dict[str, RiskLevel] = {}
    
    def update_price(self, symbol: str, price: float, timestamp: int) -> None:
        """Update price for volatility estimation."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.config.vol_lookback)
        
        self.price_history[symbol].append((timestamp, price))
        
        # Update volatility estimate if enough data points
        if len(self.price_history[symbol]) >= 10:
            self._update_volatility(symbol)
    
    def _update_volatility(self, symbol: str) -> None:
        """Calculate rolling volatility using log returns."""
        prices = self.price_history[symbol]
        if len(prices) < 2:
            return
        
        price_array = np.array([p[1] for p in prices])
        log_returns = np.diff(np.log(price_array))
        
        if len(log_returns) > 0:
            # Annualized volatility (assuming crypto trades 24/7)
            vol = np.std(log_returns) * np.sqrt(365 * 24 * 3600 / np.mean(np.diff([p[0] for p in prices])))
            self.volatility_estimates[symbol] = min(vol, 10.0)  # Cap at 1000% annual vol
    
    def update_position(self, symbol: str, quantity_delta: float, price: float, timestamp: int) -> None:
        """Record a position change from a fill."""
        if symbol not in self.inventories:
            self.inventories[symbol] = AssetInventory(symbol=symbol)
        
        self.inventories[symbol].update_position(quantity_delta, price, timestamp)
        self.update_price(symbol, price, timestamp)
        
        # Clear risk cache for this symbol
        if symbol in self._risk_cache:
            del self._risk_cache[symbol]
    
    def get_inventory_ratio(self, symbol: str) -> float:
        """Get current inventory as a fraction of maximum allowed."""
        if symbol not in self.inventories:
            return 0.0
        
        max_inv = self.config.max_inventory.get(symbol, float('inf'))
        if max_inv == 0:
            return 0.0
        
        current_inv = abs(self.inventories[symbol].quantity)
        return current_inv / max_inv
    
    def get_risk_level(self, symbol: str) -> RiskLevel:
        """Determine risk level based on inventory ratio."""
        if symbol in self._risk_cache:
            return self._risk_cache[symbol]
        
        ratio = self.get_inventory_ratio(symbol)
        
        if ratio >= self.config.critical_threshold:
            level = RiskLevel.CRITICAL
        elif ratio >= self.config.warning_threshold:
            level = RiskLevel.HIGH
        elif ratio >= 0.5:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
        
        self._risk_cache[symbol] = level
        return level
    
    def calculate_inventory_penalty(self, symbol: str, current_price: float) -> float:
        """
        Calculate quadratic penalty for current inventory position.
        
        Penalty = gamma * sigma^2 * |q|^exponent
        
        This penalty is added to the spread calculation to discourage
        holding large inventories during high volatility.
        """
        if symbol not in self.inventories:
            return 0.0
        
        inv = self.inventories[symbol]
        if inv.quantity == 0:
            return 0.0
        
        vol = self.volatility_estimates.get(symbol, 0.01)
        gamma = self.config.gamma
        
        # Quadratic penalty scaled by volatility
        penalty = gamma * (vol ** 2) * (abs(inv.quantity) ** self.config.penalty_exponent)
        
        return penalty
    
    def get_dynamic_position_limit(self, symbol: str) -> float:
        """
        Calculate dynamic position limit based on current volatility.
        
        Higher volatility -> Lower position limits to reduce risk.
        """
        base_limit = self.config.max_inventory.get(symbol, 1.0)
        vol = self.volatility_estimates.get(symbol, 0.01)
        
        # Scale limit inversely with volatility
        # Use vol_scale_factor to control sensitivity
        base_vol = 0.01  # 1% daily vol as baseline
        vol_multiplier = base_vol / max(vol, base_vol * 0.1)  # Cap reduction at 90%
        
        dynamic_limit = base_limit * (1 + self.config.vol_scale_factor * (vol_multiplier - 1))
        
        return max(dynamic_limit, base_limit * 0.1)  # Never go below 10% of base limit
    
    def get_skew_factor(self, symbol: str) -> float:
        """
        Get quote skew factor based on inventory position.
        
        Positive inventory -> Skew quotes downward (encourage sells)
        Negative inventory -> Skew quotes upward (encourage buys)
        
        Returns a multiplier for bid/ask adjustment.
        """
        if symbol not in self.inventories:
            return 0.0
        
        inv = self.inventories[symbol]
        max_inv = self.config.max_inventory.get(symbol, 1.0)
        
        if max_inv == 0:
            return 0.0
        
        # Normalized inventory (-1 to 1)
        norm_inv = inv.quantity / max_inv
        
        # Apply non-linear skew using tanh for smooth saturation
        skew = np.tanh(norm_inv * self.config.gamma * 10)
        
        return skew
    
    def should_reduce_inventory(self, symbol: str) -> Tuple[bool, str]:
        """
        Determine if inventory reduction is needed and recommended action.
        
        Returns: (should_reduce, action_description)
        """
        risk_level = self.get_risk_level(symbol)
        
        if risk_level == RiskLevel.CRITICAL:
            return True, f"CRITICAL: Immediately liquidate {symbol} position"
        elif risk_level == RiskLevel.HIGH:
            inv = self.inventories.get(symbol)
            if inv and inv.quantity > 0:
                return True, f"HIGH: Aggressively sell {symbol}"
            elif inv and inv.quantity < 0:
                return True, f"HIGH: Aggressively buy {symbol}"
        
        return False, ""
    
    def get_portfolio_summary(self, prices: Dict[str, float]) -> Dict:
        """Get comprehensive portfolio risk summary."""
        total_notional = 0.0
        total_unrealized_pnl = 0.0
        riskiest_asset = None
        max_risk_ratio = 0.0
        
        for symbol, inv in self.inventories.items():
            price = prices.get(symbol, 0.0)
            notional = inv.get_notional_value(price)
            pnl = inv.get_unrealized_pnl(price)
            
            total_notional += notional
            total_unrealized_pnl += pnl
            
            risk_ratio = self.get_inventory_ratio(symbol)
            if risk_ratio > max_risk_ratio:
                max_risk_ratio = risk_ratio
                riskiest_asset = symbol
        
        return {
            "total_notional": total_notional,
            "total_unrealized_pnl": total_unrealized_pnl,
            "riskiest_asset": riskiest_asset,
            "max_inventory_ratio": max_risk_ratio,
            "overall_risk_level": self.get_risk_level(riskiest_asset) if riskiest_asset else RiskLevel.LOW,
        }


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Initialize risk manager
    config = InventoryConfig()
    manager = InventoryRiskManager(config)
    
    # Simulate some trades
    base_ts = int(time.time() * 1000)
    
    # Buy 0.1 BTC at 50000
    manager.update_position("BTC", 0.1, 50000.0, base_ts)
    
    # Update prices
    for i in range(20):
        manager.update_price("BTC", 50000.0 + i * 10, base_ts + i * 1000)
    
    # Check risk status
    print(f"BTC Inventory Ratio: {manager.get_inventory_ratio('BTC'):.2%}")
    print(f"BTC Risk Level: {manager.get_risk_level('BTC').value}")
    print(f"Inventory Penalty: {manager.calculate_inventory_penalty('BTC', 50100.0):.4f}")
    print(f"Skew Factor: {manager.get_skew_factor('BTC'):.4f}")
    
    # Portfolio summary
    summary = manager.get_portfolio_summary({"BTC": 50100.0})
    print(f"\nPortfolio Summary:")
    print(f"  Total Notional: ${summary['total_notional']:,.2f}")
    print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
    print(f"  Risk Level: {summary['overall_risk_level'].value}")
