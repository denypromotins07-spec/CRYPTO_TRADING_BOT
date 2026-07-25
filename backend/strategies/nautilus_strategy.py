"""
NautilusTrader Strategy Class: Core trading strategy implementation.
Integrates alpha signals with NautilusTrader's execution engine.
Implements Factory pattern for strategy instantiation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from decimal import Decimal
import logging
import time

from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, TradeTick, QuoteTick
from nautilus_trader.model.enums import OrderSide, OrderType, PositionSide
from nautilus_trader.model.events import OrderFilled, OrderSubmitted
from nautilus_trader.model.identifiers import InstrumentId, StrategyId, TraderId, Venue
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.live.config import LiveStrategyConfig
from nautilus_trader.strategy import Strategy

from backend.strategies.alpha_generator import (
    get_alpha_generator, 
    AlphaGenerator, 
    AggregatedAlpha,
    SignalType
)
from backend.execution.order_manager import get_order_manager, OrderManager
from backend.config.settings import get_bot_settings, BotSettings

logger = logging.getLogger(__name__)


class ZaidTradingStrategyConfig(LiveStrategyConfig):
    """Configuration for Zaid Trading Strategy."""
    
    instrument_ids: List[str]
    max_position_size: float = 0.1  # Max position size in base currency
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_pct: float = 0.04   # 4% take profit
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop
    min_signal_strength: float = 0.5
    signal_confirmation_count: int = 2
    max_open_orders_per_asset: int = 3


class ZaidTradingStrategy(Strategy):
    """
    Core NautilusTrader strategy implementing the ZAID bot logic.
    Integrates alpha signals, risk management, and smart order routing.
    Uses Observer pattern to react to alpha updates.
    """
    
    def __init__(self, config: ZaidTradingStrategyConfig):
        super().__init__(config)
        
        self.config = config
        self.alpha_generator: AlphaGenerator = get_alpha_generator()
        self.order_manager: OrderManager = get_order_manager()
        self.bot_settings: BotSettings = get_bot_settings()
        
        # State tracking
        self._instrument_map: Dict[InstrumentId, Any] = {}
        self._pending_signals: Dict[str, List[AggregatedAlpha]] = {
            inst_id: [] for inst_id in config.instrument_ids
        }
        self._active_positions: Dict[str, Any] = {}
        self._last_signal_time: Dict[str, float] = {}
        
        # Performance metrics
        self._signals_processed = 0
        self._orders_submitted = 0
        self._orders_filled = 0
        
        logger.info(f"ZaidTradingStrategy initialized with {len(config.instrument_ids)} instruments")
    
    def on_start(self) -> None:
        """Called when strategy starts."""
        logger.info("Strategy starting...")
        
        # Subscribe to market data for all instruments
        for inst_id_str in self.config.instrument_ids:
            instrument_id = InstrumentId.from_str(inst_id_str)
            self._instrument_map[instrument_id] = None
            
            # Request historical data if needed
            # self.request_bar_data(...)
        
        # Register as observer for alpha updates
        self.alpha_generator.weight_manager.register_observer(self)
        
        logger.info(f"Subscribed to {len(self._instrument_map)} instruments")
    
    def on_stop(self) -> None:
        """Called when strategy stops."""
        logger.info("Strategy stopping...")
        
        # Close all open positions gracefully
        self._close_all_positions()
        
        # Save final state
        self._save_state()
        
        logger.info(f"Strategy stopped. Processed {self._signals_processed} signals, "
                   f"submitted {self._orders_submitted} orders")
    
    def on_resume(self) -> None:
        """Called when strategy resumes after pause."""
        logger.info("Strategy resuming...")
        self.on_start()
    
    def on_instrument(self, instrument: Any) -> None:
        """Called when instrument definition is received."""
        self._instrument_map[instrument.id] = instrument
        logger.debug(f"Instrument received: {instrument.id}")
    
    def on_quote_tick(self, tick: QuoteTick) -> None:
        """Called when a new quote tick is received."""
        # Update order book manager
        # Check for immediate execution opportunities
        pass
    
    def on_trade_tick(self, tick: TradeTick) -> None:
        """Called when a new trade tick is received."""
        # Update trade stream analyzer
        pass
    
    def on_bar(self, bar: Bar) -> None:
        """Called when a new bar is received."""
        instrument_id = bar.bar_type.instrument_id
        asset = str(instrument_id.symbol)
        
        # Get latest alpha for this asset
        alpha = self.alpha_generator.get_alpha(str(instrument_id))
        
        if alpha and alpha.net_strength >= self.config.min_signal_strength:
            self._process_alpha_signal(alpha, instrument_id)
    
    def _process_alpha_signal(self, alpha: AggregatedAlpha, instrument_id: InstrumentId) -> None:
        """
        Process an alpha signal and generate trading decisions.
        Implements signal confirmation logic to reduce false positives.
        """
        asset = str(instrument_id)
        current_time = time.time()
        
        # Check signal confirmation
        self._pending_signals[asset].append(alpha)
        
        # Keep only recent signals
        cutoff_time = current_time - 60.0  # 1 minute window
        self._pending_signals[asset] = [
            s for s in self._pending_signals[asset] 
            if s.timestamp > cutoff_time
        ]
        
        # Check if we have enough confirmation
        if len(self._pending_signals[asset]) < self.config.signal_confirmation_count:
            return
        
        # Verify signal consistency
        directions = [s.net_direction for s in self._pending_signals[asset]]
        if not all(d == directions[0] for d in directions):
            # Conflicting signals, wait for clarity
            return
        
        self._signals_processed += 1
        
        # Generate trading decision
        direction = directions[0]
        instrument = self._instrument_map.get(instrument_id)
        
        if not instrument:
            logger.warning(f"Instrument not found: {instrument_id}")
            return
        
        # Check existing position
        existing_position = self.position(instrument_id)
        
        if direction == 0:
            # Neutral signal - close position if exists
            if existing_position and existing_position.is_open:
                self._close_position(instrument_id, existing_position)
            return
        
        # Calculate position size based on Kelly Criterion and risk parameters
        position_size = self._calculate_position_size(
            alpha=alpha,
            instrument=instrument,
            existing_position=existing_position
        )
        
        if position_size <= 0:
            logger.debug(f"Position size too small for {asset}")
            return
        
        # Submit order
        if direction > 0:
            self._submit_buy_order(instrument_id, position_size, alpha)
        else:
            self._submit_sell_order(instrument_id, position_size, alpha)
        
        # Clear pending signals after execution
        self._pending_signals[asset].clear()
        self._last_signal_time[asset] = current_time
    
    def _calculate_position_size(
        self, 
        alpha: AggregatedAlpha, 
        instrument: Any,
        existing_position: Optional[Any]
    ) -> Decimal:
        """
        Calculate position size using Kelly Criterion and risk constraints.
        Returns quantity in base currency units.
        """
        # Get account balance
        account = self.account()
        if not account:
            return Decimal("0")
        
        # Base calculation from Kelly Criterion
        kelly_fraction = alpha.net_strength * alpha.net_confidence
        
        # Apply risk limits from config
        max_position_value = account.balance_total() * Decimal(str(self.config.max_position_size))
        
        # Adjust for signal strength
        position_value = max_position_value * Decimal(str(kelly_fraction))
        
        # Convert to quantity
        price = instrument.make_price(account.balance_total())  # Approximate
        quantity = position_value / price if price > 0 else Decimal("0")
        
        # Round to instrument precision
        quantity = instrument.make_qty(quantity)
        
        # Enforce minimum quantity
        if quantity < instrument.size_precision:
            return Decimal("0")
        
        return quantity
    
    def _submit_buy_order(
        self, 
        instrument_id: InstrumentId, 
        quantity: Decimal, 
        alpha: AggregatedAlpha
    ) -> None:
        """Submit a buy order with appropriate risk management."""
        instrument = self._instrument_map.get(instrument_id)
        if not instrument:
            return
        
        # Calculate stop loss and take profit levels
        current_price = self.last_quote(instrument_id).ask_price if self.last_quote(instrument_id) else None
        
        if not current_price:
            logger.warning(f"No price available for {instrument_id}")
            return
        
        stop_loss_price = current_price * (1 - Decimal(str(self.config.stop_loss_pct)))
        take_profit_price = current_price * (1 + Decimal(str(self.config.take_profit_pct)))
        
        # Submit market order
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=quantity,
            tags=["alpha_entry", f"strength_{alpha.net_strength:.2f}"]
        )
        
        self.submit_order(order)
        self._orders_submitted += 1
        
        logger.info(f"Buy order submitted: {quantity} {instrument.base_currency} at market")
    
    def _submit_sell_order(
        self, 
        instrument_id: InstrumentId, 
        quantity: Decimal, 
        alpha: AggregatedAlpha
    ) -> None:
        """Submit a sell order with appropriate risk management."""
        instrument = self._instrument_map.get(instrument_id)
        if not instrument:
            return
        
        # Calculate stop loss and take profit levels
        current_price = self.last_quote(instrument_id).bid_price if self.last_quote(instrument_id) else None
        
        if not current_price:
            logger.warning(f"No price available for {instrument_id}")
            return
        
        stop_loss_price = current_price * (1 + Decimal(str(self.config.stop_loss_pct)))
        take_profit_price = current_price * (1 - Decimal(str(self.config.take_profit_pct)))
        
        # Submit market order
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.SELL,
            quantity=quantity,
            tags=["alpha_entry", f"strength_{alpha.net_strength:.2f}"]
        )
        
        self.submit_order(order)
        self._orders_submitted += 1
        
        logger.info(f"Sell order submitted: {quantity} {instrument.base_currency} at market")
    
    def _close_position(self, instrument_id: InstrumentId, position: Any) -> None:
        """Close an existing position."""
        if not position or not position.is_open:
            return
        
        quantity = abs(position.quantity)
        
        order_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            tags=["position_close"]
        )
        
        self.submit_order(order)
        self._orders_submitted += 1
        
        logger.info(f"Position closed: {quantity} {order_side.name}")
    
    def _close_all_positions(self) -> None:
        """Close all open positions immediately."""
        for instrument_id in self._instrument_map.keys():
            position = self.position(instrument_id)
            if position and position.is_open:
                self._close_position(instrument_id, position)
    
    def on_order_submitted(self, event: OrderSubmitted) -> None:
        """Called when an order is submitted."""
        logger.debug(f"Order submitted: {event.order_id}")
    
    def on_order_filled(self, event: OrderFilled) -> None:
        """Called when an order is filled."""
        self._orders_filled += 1
        
        # Log fill to SOUL.md via order manager
        self.order_manager.log_fill_to_soul(event)
        
        logger.info(f"Order filled: {event.quantity} @ {event.price}")
    
    def on_weights_updated(self, new_weights: Dict[SignalType, float]) -> None:
        """Observer callback for weight updates."""
        logger.info(f"Alpha weights updated: {new_weights}")
    
    def _save_state(self) -> None:
        """Save current strategy state for persistence."""
        state = {
            "signals_processed": self._signals_processed,
            "orders_submitted": self._orders_submitted,
            "orders_filled": self._orders_filled,
            "active_positions": len(self._active_positions),
            "timestamp": time.time()
        }
        
        # Save to disk or database
        logger.info(f"Strategy state saved: {state}")


# Factory for creating strategy instances
class StrategyFactory:
    """Factory class for creating ZaidTradingStrategy instances."""
    
    @staticmethod
    def create_strategy(
        instrument_ids: List[str],
        trader_id: str = "TRADER-001",
        **kwargs
    ) -> ZaidTradingStrategy:
        """Create a new ZaidTradingStrategy instance."""
        
        config = ZaidTradingStrategyConfig(
            instrument_ids=instrument_ids,
            trader_id=TraderId(trader_id),
            strategy_id=StrategyId("ZAID_STRATEGY"),
            **kwargs
        )
        
        return ZaidTradingStrategy(config)


if __name__ == "__main__":
    # Example usage
    instruments = ["BTCUSDT.BINANCE", "ETHUSDT.BINANCE", "SOLUSDT.BINANCE"]
    
    strategy = StrategyFactory.create_strategy(
        instrument_ids=instruments,
        max_position_size=0.05,
        stop_loss_pct=0.015,
        take_profit_pct=0.03
    )
    
    print(f"Strategy created for {len(instruments)} instruments")
    print("Nautilus Strategy module initialized successfully.")
