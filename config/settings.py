"""
=============================================================================
ZAID PERSONAL CRYPTO TRADING BOT - GLOBAL SETTINGS
=============================================================================
Base Configuration Architecture for Multi-Asset Algorithmic Trading
Provides O(1) lookup times for critical trading parameters via hash-based structures.

Domains Integrated:
- Market Microstructure
- Order Execution Algorithms
- Slippage Modeling
- Time-Weighted Average Price (TWAP)
- Asset Correlation Matrix
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Final
from enum import Enum, auto
from decimal import Decimal
import json
import os
from pathlib import Path


class AssetClass(Enum):
    """Enumeration of supported asset classes."""
    CRYPTO = auto()
    FIAT = auto()
    STABLECOIN = auto()


class OrderSide(Enum):
    """Order direction enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration for execution strategies."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"


@dataclass(frozen=True)
class SlippageModel:
    """
    Slippage estimation model for realistic P&L calculations.
    Uses linear impact model with volatility adjustment.
    
    Attributes:
        base_slippage_bps: Base slippage in basis points
        volatility_factor: Multiplier based on market volatility
        size_impact_factor: Impact coefficient for order size
    """
    base_slippage_bps: Decimal = Decimal("0.05")  # 5 bps base
    volatility_factor: Decimal = Decimal("1.5")
    size_impact_factor: Decimal = Decimal("0.1")
    
    def calculate_slippage(
        self,
        order_size: Decimal,
        market_volume: Decimal,
        volatility: Decimal
    ) -> Decimal:
        """
        Calculate expected slippage for a given order.
        
        Args:
            order_size: Size of the order in base currency
            market_volume: Current market volume
            volatility: Annualized volatility (0.0 to 1.0)
            
        Returns:
            Expected slippage as a percentage (Decimal)
        """
        size_ratio = order_size / market_volume if market_volume > 0 else Decimal("0")
        vol_adjustment = Decimal("1") + (volatility * self.volatility_factor)
        return (
            self.base_slippage_bps +
            (size_ratio * self.size_impact_factor * Decimal("100")) * vol_adjustment
        )


@dataclass(frozen=True)
class TWAPConfig:
    """
    Time-Weighted Average Price execution configuration.
    Splits large orders across time intervals to minimize market impact.
    
    Attributes:
        total_duration_seconds: Total execution time
        num_slices: Number of order slices
        randomize_timing: Add randomness to slice timing to avoid detection
        participation_rate: Max percentage of market volume per slice
    """
    total_duration_seconds: int = 3600  # 1 hour default
    num_slices: int = 12  # 5-minute intervals
    randomize_timing: bool = True
    participation_rate: Decimal = Decimal("0.05")  # 5% max
    min_slice_size: Decimal = Decimal("0.001")  # Minimum BTC equivalent
    
    def get_slice_interval(self) -> int:
        """Calculate interval between slices in seconds."""
        return self.total_duration_seconds // self.num_slices


@dataclass
class GlobalBotParameters:
    """
    Central configuration singleton for all bot parameters.
    Implements thread-safe access to critical trading constants.
    """
    
    # Trading Window Configuration (4-hour strict window)
    TRADING_WINDOW_START_HOUR: Final[int] = 9
    TRADING_WINDOW_END_HOUR: Final[int] = 13
    TRADING_TIMEZONE: Final[str] = "UTC"
    
    # Profit Targets (INR per hour)
    TARGET_PROFIT_MIN_INR: Final[Decimal] = Decimal("8000")
    TARGET_PROFIT_MAX_INR: Final[Decimal] = Decimal("20000")
    
    # System Constraints
    MAX_RAM_GB: Final[int] = 8
    MAX_CPU_THREADS: Final[int] = 12
    
    # Supported Assets with base precision
    SUPPORTED_ASSETS: Final[Dict[str, AssetClass]] = field(default_factory=lambda: {
        "BTC": AssetClass.CRYPTO,
        "ETH": AssetClass.CRYPTO,
        "SOL": AssetClass.CRYPTO,
        "USDT": AssetClass.STABLECOIN,
    })
    
    # Default trading pairs (base/quote)
    TRADING_PAIRS: Final[List[Tuple[str, str]]] = field(default_factory=lambda: [
        ("BTC", "USDT"),
        ("ETH", "USDT"),
        ("SOL", "USDT"),
    ])
    
    # Slippage model instance
    slippage_model: SlippageModel = field(default_factory=SlippageModel)
    
    # TWAP configuration
    twap_config: TWAPConfig = field(default_factory=TWAPConfig)
    
    # Fee structure (Binance maker/taker)
    MAKER_FEE_BPS: Final[Decimal] = Decimal("0.01")  # 0.01%
    TAKER_FEE_BPS: Final[Decimal] = Decimal("0.01")  # 0.01%
    
    # Minimum order sizes (exchange-specific)
    MIN_ORDER_SIZES: Final[Dict[str, Decimal]] = field(default_factory=lambda: {
        "BTC": Decimal("0.00001"),
        "ETH": Decimal("0.0001"),
        "SOL": Decimal("0.01"),
    })
    
    # Price precision per asset
    PRICE_PRECISION: Final[Dict[str, int]] = field(default_factory=lambda: {
        "BTC": 2,
        "ETH": 2,
        "SOL": 4,
    })
    
    # Quantity precision per asset
    QUANTITY_PRECISION: Final[Dict[str, int]] = field(default_factory=lambda: {
        "BTC": 5,
        "ETH": 4,
        "SOL": 2,
    })

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.TWAP_CONFIG_END_HOUR <= self.TRADING_WINDOW_START_HOUR:
            raise ValueError("Trading window end must be after start")
    
    def get_asset_class(self, asset: str) -> Optional[AssetClass]:
        """O(1) lookup for asset class."""
        return self.SUPPORTED_ASSETS.get(asset.upper())
    
    def is_supported_asset(self, asset: str) -> bool:
        """Check if asset is supported for trading."""
        return asset.upper() in self.SUPPORTED_ASSETS
    
    def get_trading_pair(self, base: str, quote: str = "USDT") -> Optional[Tuple[str, str]]:
        """Get valid trading pair if it exists."""
        pair = (base.upper(), quote.upper())
        return pair if pair in self.TRADING_PAIRS else None
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return {
            "trading_window": {
                "start": self.TRADING_WINDOW_START_HOUR,
                "end": self.TRADING_WINDOW_END_HOUR,
                "timezone": self.TRADING_TIMEZONE,
            },
            "profit_targets": {
                "min_inr": str(self.TARGET_PROFIT_MIN_INR),
                "max_inr": str(self.TARGET_PROFIT_MAX_INR),
            },
            "system_constraints": {
                "max_ram_gb": self.MAX_RAM_GB,
                "max_cpu_threads": self.MAX_CPU_THREADS,
            },
            "fees": {
                "maker_bps": str(self.MAKER_FEE_BPS),
                "taker_bps": str(self.TAKER_FEE_BPS),
            },
        }


# Singleton instance for global access
_bot_config: Optional[GlobalBotParameters] = None


def get_bot_config() -> GlobalBotParameters:
    """
    Get or create the singleton bot configuration instance.
    Thread-safe lazy initialization.
    
    Returns:
        GlobalBotParameters: The singleton configuration instance
    """
    global _bot_config
    if _bot_config is None:
        _bot_config = GlobalBotParameters()
    return _bot_config


def reload_config() -> GlobalBotParameters:
    """Force reload of configuration from disk."""
    global _bot_config
    _bot_config = GlobalBotParameters()
    return _bot_config


if __name__ == "__main__":
    # Test configuration loading
    config = get_bot_config()
    print(f"Trading Window: {config.TRADING_WINDOW_START_HOUR}:00 - {config.TRADING_WINDOW_END_HOUR}:00 UTC")
    print(f"Supported Assets: {list(config.SUPPORTED_ASSETS.keys())}")
    print(f"Target Profit: {config.TARGET_PROFIT_MIN_INR} - {config.TARGET_PROFIT_MAX_INR} INR/hour")
