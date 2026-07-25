"""
Venue Configuration for ZAID Personal Crypto Trading Bot
=========================================================
Chapter 4: NautilusTrader Integration and High-Speed Data Engine Adapters

Defines live venue parameters and tick sizes for Binance.
Contains all instrument specifications for BTC, ETH, SOL, USDT pairs.

Features:
- Venue configuration for testnet/mainnet
- Instrument definitions with correct tick sizes
- Symbol-specific lot size constraints
- Fee structure configuration
- Support for BTC, SOL, ETH, USDT pairs

Author: Opus 4.8
Stage: 2 of 100
"""

from typing import Dict, List, Any, Optional
from decimal import Decimal
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TickSizeConfig:
    """Tick size configuration for a symbol."""
    symbol: str
    price_tick: Decimal
    quantity_tick: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal


@dataclass
class FeeStructure:
    """Trading fee structure."""
    maker_fee_bps: Decimal = Decimal("10")  # 0.10%
    taker_fee_bps: Decimal = Decimal("10")  # 0.10%
    volume_discount_threshold: Decimal = Decimal("50")  # BTC equivalent
    
    @property
    def maker_fee_decimal(self) -> Decimal:
        return self.maker_fee_bps / Decimal("10000")
    
    @property
    def taker_fee_decimal(self) -> Decimal:
        return self.taker_fee_bps / Decimal("10000")


@dataclass
class VenueConfig:
    """Configuration for a trading venue."""
    name: str
    base_url: str
    ws_url: str
    testnet: bool
    rate_limit_weight_per_minute: int = 1200
    orders_per_second: int = 10
    max_open_orders: int = 200
    fee_structure: FeeStructure = field(default_factory=FeeStructure)
    
    @property
    def is_production(self) -> bool:
        return not self.testnet


# Pre-configured tick sizes for major pairs (Binance Spot)
TICK_SIZE_CONFIGS: Dict[str, TickSizeConfig] = {
    "BTCUSDT": TickSizeConfig(
        symbol="BTCUSDT",
        price_tick=Decimal("0.01"),
        quantity_tick=Decimal("0.00001"),
        min_qty=Decimal("0.00001"),
        max_qty=Decimal("9000"),
        min_notional=Decimal("10"),
    ),
    "ETHUSDT": TickSizeConfig(
        symbol="ETHUSDT",
        price_tick=Decimal("0.01"),
        quantity_tick=Decimal("0.0001"),
        min_qty=Decimal("0.0001"),
        max_qty=Decimal("90000"),
        min_notional=Decimal("10"),
    ),
    "SOLUSDT": TickSizeConfig(
        symbol="SOLUSDT",
        price_tick=Decimal("0.01"),
        quantity_tick=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        max_qty=Decimal("900000"),
        min_notional=Decimal("10"),
    ),
    "BNBUSDT": TickSizeConfig(
        symbol="BNBUSDT",
        price_tick=Decimal("0.10"),
        quantity_tick=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("90000"),
        min_notional=Decimal("10"),
    ),
}


# Venue configurations
VENUE_CONFIGS: Dict[str, VenueConfig] = {
    "BINANCE_MAINNET": VenueConfig(
        name="BINANCE",
        base_url="https://api.binance.com",
        ws_url="wss://stream.binance.com:9443/ws",
        testnet=False,
    ),
    "BINANCE_TESTNET": VenueConfig(
        name="BINANCE_TESTNET",
        base_url="https://testnet.binance.vision",
        ws_url="wss://testnet.binance.vision/ws",
        testnet=True,
    ),
}


class VenueManager:
    """
    Manages venue configurations and instrument specifications.
    
    Provides centralized access to all venue parameters.
    """
    
    def __init__(self, venue_name: str = "BINANCE_MAINNET"):
        """
        Initialize venue manager.
        
        Args:
            venue_name: Name of venue configuration to use
        """
        self._venue_config = VENUE_CONFIGS.get(venue_name)
        if not self._venue_config:
            raise ValueError(f"Unknown venue: {venue_name}")
        
        self._tick_configs = TICK_SIZE_CONFIGS.copy()
        self._custom_configs: Dict[str, TickSizeConfig] = {}
        
        logger.info(f"Venue manager initialized for {venue_name}")
    
    @property
    def venue(self) -> VenueConfig:
        """Get current venue configuration."""
        return self._venue_config
    
    def get_tick_config(self, symbol: str) -> Optional[TickSizeConfig]:
        """Get tick size configuration for a symbol."""
        symbol_upper = symbol.upper()
        return self._custom_configs.get(symbol_upper) or self._tick_configs.get(symbol_upper)
    
    def add_custom_tick_config(self, config: TickSizeConfig):
        """Add custom tick configuration for a symbol."""
        self._custom_configs[config.symbol.upper()] = config
        logger.info(f"Added custom tick config for {config.symbol}")
    
    def get_all_symbols(self) -> List[str]:
        """Get all configured symbols."""
        return list(set(list(self._tick_configs.keys()) + list(self._custom_configs.keys())))
    
    def normalize_price(self, symbol: str, price: Decimal) -> Decimal:
        """Normalize price to correct tick size."""
        config = self.get_tick_config(symbol)
        if not config:
            return price
        
        # Round to nearest tick
        ticks = (price / config.price_tick).to_integral_value()
        return ticks * config.price_tick
    
    def normalize_quantity(self, symbol: str, quantity: Decimal) -> Decimal:
        """Normalize quantity to correct tick size."""
        config = self.get_tick_config(symbol)
        if not config:
            return quantity
        
        # Round down to nearest tick
        ticks = int(quantity / config.quantity_tick)
        return Decimal(ticks) * config.quantity_tick
    
    def validate_order(self, symbol: str, side: str, 
                       quantity: Decimal, price: Optional[Decimal] = None) -> tuple:
        """
        Validate order parameters against venue rules.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        config = self.get_tick_config(symbol)
        if not config:
            return True, None  # No config, assume valid
        
        # Check quantity bounds
        if quantity < config.min_qty:
            return False, f"Quantity {quantity} below minimum {config.min_qty}"
        
        if quantity > config.max_qty:
            return False, f"Quantity {quantity} above maximum {config.max_qty}"
        
        # Check notional value for limit orders
        if price:
            notional = quantity * price
            if notional < config.min_notional:
                return False, f"Notional {notional} below minimum {config.min_notional}"
        
        return True, None
    
    def calculate_fees(self, symbol: str, quantity: Decimal, 
                       price: Decimal, is_maker: bool) -> Decimal:
        """Calculate trading fees."""
        notional = quantity * price
        fee_rate = self._venue_config.fee_structure.maker_fee_decimal if is_maker \
                   else self._venue_config.fee_structure.taker_fee_decimal
        return notional * fee_rate
    
    def get_venue_info(self) -> Dict[str, Any]:
        """Get comprehensive venue information."""
        return {
            'name': self._venue_config.name,
            'base_url': self._venue_config.base_url,
            'ws_url': self._venue_config.ws_url,
            'testnet': self._venue_config.testnet,
            'rate_limit': self._venue_config.rate_limit_weight_per_minute,
            'orders_per_second': self._venue_config.orders_per_second,
            'max_open_orders': self._venue_config.max_open_orders,
            'maker_fee_bps': float(self._venue_config.fee_structure.maker_fee_bps),
            'taker_fee_bps': float(self._venue_config.fee_structure.taker_fee_bps),
            'symbols': self.get_all_symbols(),
        }


def get_venue_manager(testnet: bool = True) -> VenueManager:
    """Get venue manager for specified environment."""
    venue_name = "BINANCE_TESTNET" if testnet else "BINANCE_MAINNET"
    return VenueManager(venue_name)


def main():
    """Example usage of venue configuration."""
    print("Venue Configuration - ZAID Personal Crypto Trading Bot")
    print("=" * 60)
    
    # Get testnet venue manager
    manager = get_venue_manager(testnet=True)
    
    # Print venue info
    info = manager.get_venue_info()
    print(f"\nVenue: {info['name']}")
    print(f"URL: {info['base_url']}")
    print(f"Testnet: {info['testnet']}")
    print(f"Rate Limit: {info['rate_limit']} weight/min")
    print(f"Orders/sec: {info['orders_per_second']}")
    print(f"Fees: {info['maker_fee_bps']}/{info['taker_fee_bps']} bps (maker/taker)")
    
    # Print symbol configurations
    print("\nSymbol Configurations:")
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        config = manager.get_tick_config(symbol)
        if config:
            print(f"\n  {symbol}:")
            print(f"    Price Tick: {config.price_tick}")
            print(f"    Qty Tick: {config.quantity_tick}")
            print(f"    Min Qty: {config.min_qty}")
            print(f"    Max Qty: {config.max_qty}")
            print(f"    Min Notional: ${config.min_notional}")
    
    # Test normalization
    print("\nPrice Normalization Tests:")
    test_prices = [
        ("BTCUSDT", Decimal("50000.123")),
        ("ETHUSDT", Decimal("3000.567")),
        ("SOLUSDT", Decimal("100.789")),
    ]
    
    for symbol, price in test_prices:
        normalized = manager.normalize_price(symbol, price)
        print(f"  {symbol}: {price} -> {normalized}")
    
    # Test validation
    print("\nOrder Validation Tests:")
    test_orders = [
        ("BTCUSDT", "BUY", Decimal("0.00001"), Decimal("50000")),
        ("BTCUSDT", "BUY", Decimal("0.000001"), Decimal("50000")),  # Too small
        ("ETHUSDT", "SELL", Decimal("100"), Decimal("3000")),
    ]
    
    for symbol, side, qty, price in test_orders:
        valid, error = manager.validate_order(symbol, side, qty, price)
        status = "✓ Valid" if valid else f"✗ Invalid: {error}"
        print(f"  {symbol} {side} {qty}@{price}: {status}")


if __name__ == "__main__":
    main()
