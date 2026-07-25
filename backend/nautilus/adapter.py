"""
NautilusTrader Adapter for ZAID Personal Crypto Trading Bot
============================================================
Chapter 4: NautilusTrader Integration and High-Speed Data Engine Adapters

Bridges custom WebSocket feeds into Nautilus events.
Maps Binance order types to Nautilus internal enums.

Features:
- Custom data feed adapter for NautilusTrader
- Binance-specific instrument definition
- Order type mapping (Binance <-> Nautilus)
- Real-time quote and trade ingestion
- Support for BTC, SOL, ETH, USDT pairs

Author: Opus 4.8
Stage: 2 of 100
"""

import asyncio
from typing import Dict, List, Optional, Any
from decimal import Decimal
import logging
from datetime import datetime
import uuid

from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import millis_to_nanos
from nautilus_trader.model.data import QuoteTick, TradeTick, BarType
from nautilus_trader.model.enums import AggressorSide, OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoSwap, CryptoSpot

logger = logging.getLogger(__name__)


class BinanceInstrumentFactory:
    """Factory for creating Nautilus instruments from Binance data."""
    
    VENUE = Venue("BINANCE")
    
    @staticmethod
    def create_spot_instrument(
        symbol: str,
        base_asset: str,
        quote_asset: str,
        price_precision: int,
        quantity_precision: int,
        min_qty: Decimal,
        max_qty: Decimal,
        min_notional: Decimal,
    ) -> CryptoSpot:
        """
        Create a crypto spot instrument.
        
        Args:
            symbol: Trading pair symbol
            base_asset: Base asset (e.g., BTC)
            quote_asset: Quote asset (e.g., USDT)
            price_precision: Price decimal places
            quantity_precision: Quantity decimal places
            min_qty: Minimum order quantity
            max_qty: Maximum order quantity
            min_notional: Minimum notional value
        
        Returns:
            CryptoSpot instrument
        """
        instrument_id = InstrumentId(
            symbol=Symbol(symbol.upper()),
            venue=BinanceInstrumentFactory.VENUE,
        )
        
        return CryptoSpot(
            id=instrument_id,
            raw_symbol=Symbol(symbol.upper()),
            base_currency=base_asset.upper(),
            quote_currency=quote_asset.upper(),
            settlement_currency=quote_asset.upper(),
            price_precision=price_precision,
            size_precision=quantity_precision,
            price_increment=Decimal(f"1e-{price_precision}"),
            size_increment=Decimal(f"1e-{quantity_precision}"),
            multiplier=Decimal(1),
            lot_size=min_qty,
            max_quantity=max_qty,
            min_quantity=min_qty,
            min_notional=min_notional,
            info={},
        )
    
    @staticmethod
    def parse_binance_symbol(symbol_info: Dict[str, Any]) -> tuple:
        """Parse Binance symbol info into components."""
        base = symbol_info.get('baseAsset', '')
        quote = symbol_info.get('quoteAsset', '')
        symbol = symbol_info.get('symbol', f"{base}{quote}")
        
        # Get precision from filters
        price_precision = 2
        qty_precision = 8
        
        for f in symbol_info.get('filters', []):
            if f['filterType'] == 'PRICE_FILTER':
                tick_size = Decimal(f.get('tickSize', '0.01'))
                price_precision = abs(int(tick_size.adjusted().as_tuple().exponent))
            elif f['filterType'] == 'LOT_SIZE':
                step_size = Decimal(f.get('stepSize', '0.00001'))
                qty_precision = abs(int(step_size.adjusted().as_tuple().exponent))
        
        return symbol, base, quote, price_precision, qty_precision


class NautilusBinanceAdapter:
    """
    Adapter that converts Binance WebSocket messages to Nautilus events.
    
    Provides seamless integration between custom WS feeds and NautilusTrader core.
    """
    
    def __init__(self):
        self._instruments: Dict[InstrumentId, Any] = {}
        self._quote_subscribers: Dict[InstrumentId, List[callable]] = {}
        self._trade_subscribers: Dict[InstrumentId, List[callable]] = {}
        self._running = False
    
    def register_instrument(self, instrument: CryptoSpot):
        """Register an instrument for data conversion."""
        self._instruments[instrument.id] = instrument
        logger.info(f"Registered instrument: {instrument.id}")
    
    def subscribe_quotes(self, instrument_id: InstrumentId, callback: callable):
        """Subscribe to quote updates for an instrument."""
        if instrument_id not in self._quote_subscribers:
            self._quote_subscribers[instrument_id] = []
        self._quote_subscribers[instrument_id].append(callback)
    
    def subscribe_trades(self, instrument_id: InstrumentId, callback: callable):
        """Subscribe to trade updates for an instrument."""
        if instrument_id not in self._trade_subscribers:
            self._trade_subscribers[instrument_id] = []
        self._trade_subscribers[instrument_id].append(callback)
    
    def on_order_book_update(self, symbol: str, bids: List[tuple], asks: List[tuple], timestamp: int):
        """
        Handle order book update from WebSocket.
        
        Args:
            symbol: Trading pair symbol
            bids: List of (price, quantity) tuples
            asks: List of (price, quantity) tuples
            timestamp: Update timestamp in milliseconds
        """
        instrument_id = InstrumentId(
            symbol=Symbol(symbol.upper()),
            venue=BinanceInstrumentFactory.VENUE,
        )
        
        if instrument_id not in self._instruments:
            logger.debug(f"Instrument not registered: {instrument_id}")
            return
        
        if not bids or not asks:
            return
        
        # Create quote tick
        best_bid_price = Decimal(str(bids[0][0]))
        best_bid_qty = Decimal(str(bids[0][1]))
        best_ask_price = Decimal(str(asks[0][0]))
        best_ask_qty = Decimal(str(asks[0][1]))
        
        quote_tick = QuoteTick(
            instrument_id=instrument_id,
            bid_price=best_bid_price,
            ask_price=best_ask_price,
            bid_size=best_bid_qty,
            ask_size=best_ask_qty,
            ts_event=millis_to_nanos(timestamp),
            ts_init=millis_to_nanos(timestamp),
        )
        
        # Notify subscribers
        for callback in self._quote_subscribers.get(instrument_id, []):
            try:
                callback(quote_tick)
            except Exception as e:
                logger.error(f"Quote callback error: {e}")
    
    def on_trade(self, symbol: str, price: Decimal, quantity: Decimal, 
                 is_buyer_maker: bool, trade_id: int, timestamp: int):
        """
        Handle trade message from WebSocket.
        
        Args:
            symbol: Trading pair symbol
            price: Trade price
            quantity: Trade quantity
            is_buyer_maker: True if buyer is maker
            trade_id: Exchange trade ID
            timestamp: Trade timestamp in milliseconds
        """
        instrument_id = InstrumentId(
            symbol=Symbol(symbol.upper()),
            venue=BinanceInstrumentFactory.VENUE,
        )
        
        # Determine aggressor side
        aggressor_side = AggressorSide.SELL if is_buyer_maker else AggressorSide.BUY
        
        # Create trade tick
        trade_tick = TradeTick(
            instrument_id=instrument_id,
            price=price,
            size=quantity,
            aggressor_side=aggressor_side,
            trade_id=str(trade_id),
            ts_event=millis_to_nanos(timestamp),
            ts_init=millis_to_nanos(timestamp),
        )
        
        # Notify subscribers
        for callback in self._trade_subscribers.get(instrument_id, []):
            try:
                callback(trade_tick)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
    
    def map_order_type(self, binance_type: str) -> OrderType:
        """Map Binance order type to Nautilus enum."""
        mapping = {
            "LIMIT": OrderType.LIMIT,
            "MARKET": OrderType.MARKET,
            "STOP_LOSS": OrderType.STOP_MARKET,
            "STOP_LOSS_LIMIT": OrderType.STOP_LIMIT,
            "TAKE_PROFIT": OrderType.STOP_MARKET,
            "TAKE_PROFIT_LIMIT": OrderType.STOP_LIMIT,
            "LIMIT_MAKER": OrderType.LIMIT,
        }
        return mapping.get(binance_type.upper(), OrderType.MARKET)
    
    def map_order_side(self, binance_side: str) -> OrderSide:
        """Map Binance order side to Nautilus enum."""
        mapping = {
            "BUY": OrderSide.BUY,
            "SELL": OrderSide.SELL,
        }
        return mapping.get(binance_side.upper(), OrderSide.BUY)
    
    def get_instrument(self, symbol: str) -> Optional[Any]:
        """Get registered instrument by symbol."""
        instrument_id = InstrumentId(
            symbol=Symbol(symbol.upper()),
            venue=BinanceInstrumentFactory.VENUE,
        )
        return self._instruments.get(instrument_id)


# Global adapter instance
_adapter: Optional[NautilusBinanceAdapter] = None


def get_adapter() -> NautilusBinanceAdapter:
    """Get or create global adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = NautilusBinanceAdapter()
    return _adapter


async def main():
    """Example usage of Nautilus adapter."""
    adapter = get_adapter()
    
    # Register BTCUSDT instrument
    instrument = BinanceInstrumentFactory.create_spot_instrument(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        price_precision=2,
        quantity_precision=6,
        min_qty=Decimal("0.00001"),
        max_qty=Decimal("9000"),
        min_notional=Decimal("10"),
    )
    adapter.register_instrument(instrument)
    
    # Subscribe to quotes
    def on_quote(quote: QuoteTick):
        print(f"Quote: {quote.bid_price} / {quote.ask_price}")
    
    adapter.subscribe_quotes(instrument.id, on_quote)
    
    # Simulate order book update
    adapter.on_order_book_update(
        symbol="BTCUSDT",
        bids=[("50000.00", "1.5"), ("49999.00", "2.0")],
        asks=[("50001.00", "1.0"), ("50002.00", "2.5")],
        timestamp=int(datetime.now().timestamp() * 1000),
    )
    
    # Simulate trade
    adapter.on_trade(
        symbol="BTCUSDT",
        price=Decimal("50000.50"),
        quantity=Decimal("0.123"),
        is_buyer_maker=False,
        trade_id=123456789,
        timestamp=int(datetime.now().timestamp() * 1000),
    )
    
    # Test order type mapping
    print(f"\nOrder Type Mapping:")
    print(f"  LIMIT -> {adapter.map_order_type('LIMIT')}")
    print(f"  MARKET -> {adapter.map_order_type('MARKET')}")
    print(f"  STOP_LOSS -> {adapter.map_order_type('STOP_LOSS')}")


if __name__ == "__main__":
    asyncio.run(main())
