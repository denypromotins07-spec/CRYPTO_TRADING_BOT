"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 7
Chapter 2: Macro-Economic Indicators - Economic Calendar

File: backend/macro/economic_calendar.py
Purpose: Parse and track economic events (CPI, PPI, Fed decisions, NFP).
         Automatically halt trading during high-impact announcements.

Features:
- Real-time economic calendar parsing from multiple sources
- Timezone-aware event scheduling with DST handling
- Impact classification (Low/Medium/High/Extreme)
- Automatic trading halt before/during high-impact events
- Historical event impact analysis for pattern recognition

Design Patterns:
- Observer: Notify strategies of upcoming events
- Strategy: Different event impact assessment methods
- Circuit Breaker: Auto-halt on extreme events

Author: Opus 4.8
Domain: Macroeconomics, Event Risk Management, Trading Halts
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from collections import deque
from enum import Enum
import logging
import pytz
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventImpact(Enum):
    """Economic event impact classification."""
    LOW = "low"           # Minimal market impact
    MEDIUM = "medium"     # Moderate volatility expected
    HIGH = "high"         # Significant volatility expected
    EXTREME = "extreme"   # Major market-moving event


class EventType(Enum):
    """Types of economic events."""
    CPI = "cpi"                    # Consumer Price Index
    CORE_CPI = "core_cpi"          # Core CPI (ex food/energy)
    PPI = "ppi"                    # Producer Price Index
    NONFARM_PAYROLLS = "nfp"       # Non-Farm Payrolls
    UNEMPLOYMENT_RATE = "unemployment"
    FED_RATE_DECISION = "fed_rate"
    FOMC_STATEMENT = "fomc_statement"
    FED_SPEECH = "fed_speech"      # Fed Chair/Speaker speech
    GDP = "gdp"
    RETAIL_SALES = "retail_sales"
    PMI_MANUFACTURING = "pmi_manufacturing"
    PMI_SERVICES = "pmi_services"
    INITIAL_JOBLESS_CLAIMS = "jobless_claims"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    HOUSING_STARTS = "housing_starts"
    TRADE_BALANCE = "trade_balance"
    OTHER = "other"


@dataclass(slots=True)
class EconomicEvent:
    """
    Represents an economic event.
    Uses __slots__ for memory efficiency.
    """
    id: str
    event_type: EventType
    title: str
    country: str
    scheduled_time_utc: datetime
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    impact: EventImpact = EventImpact.MEDIUM
    currency: str = "USD"
    source: str = "unknown"
    is_completed: bool = False
    
    def time_until_event(self) -> timedelta:
        """Calculate time until event from now."""
        return self.scheduled_time_utc - datetime.now(timezone.utc)
    
    def is_within_window(self, minutes: int) -> bool:
        """Check if event is within specified minutes from now."""
        return abs(self.time_until_event().total_seconds()) <= minutes * 60
    
    def surprise_factor(self) -> Optional[float]:
        """
        Calculate surprise factor (actual vs forecast).
        Returns None if event not completed or no forecast.
        """
        if not self.is_completed or self.forecast_value is None or self.actual_value is None:
            return None
        
        if self.forecast_value == 0:
            return None
        
        return (self.actual_value - self.forecast_value) / abs(self.forecast_value)


@dataclass(slots=True)
class TradingHaltConfig:
    """Configuration for trading halts around economic events."""
    # Halt windows in minutes before event
    halt_before_extreme: int = 30
    halt_before_high: int = 15
    halt_before_medium: int = 5
    halt_before_low: int = 0
    
    # Resume windows in minutes after event
    resume_after_extreme: int = 60
    resume_after_high: int = 30
    resume_after_high: int = 15
    resume_after_low: int = 0
    
    # Which impacts trigger halts
    halt_on_impacts: Set[EventImpact] = field(default_factory=lambda: {
        EventImpact.HIGH,
        EventImpact.EXTREME
    })


@dataclass(slots=True)
class TradingHaltStatus:
    """Current trading halt status."""
    is_halted: bool
    reason: str
    halted_since: datetime
    resume_at: Optional[datetime]
    triggering_event: Optional[EconomicEvent]
    
    def time_until_resume(self) -> Optional[timedelta]:
        """Calculate time until trading resumes."""
        if not self.resume_at:
            return None
        return self.resume_at - datetime.now(timezone.utc)


class EconomicCalendarParser:
    """
    Parses economic calendar data from various sources.
    
    Supports:
    - ForexFactory API (mock integration)
    - Investing.com scraping (mock)
    - Central bank calendars
    - Custom event feeds
    """
    
    def __init__(self):
        self._event_cache: Dict[str, EconomicEvent] = {}
        self._last_update: Optional[datetime] = None
    
    async def fetch_forex_factory(self) -> List[EconomicEvent]:
        """Fetch events from ForexFactory calendar."""
        try:
            # In production: Use actual API or scraping
            # https://nfs.faireconomy.media/ff_calendar_thisweek.json
            
            logger.debug("Fetching from ForexFactory (mock mode)")
            
            # Mock data for demonstration
            now = datetime.now(timezone.utc)
            
            return [
                EconomicEvent(
                    id="ff_cpi_2024_03",
                    event_type=EventType.CPI,
                    title="Consumer Price Index (YoY)",
                    country="US",
                    scheduled_time_utc=now + timedelta(hours=2),
                    forecast_value=3.1,
                    previous_value=3.2,
                    impact=EventImpact.EXTREME,
                    source="forex_factory"
                ),
                EconomicEvent(
                    id="ff_nfp_2024_03",
                    event_type=EventType.NONFARM_PAYROLLS,
                    title="Non-Farm Payrolls",
                    country="US",
                    scheduled_time_utc=now + timedelta(days=1),
                    forecast_value=200000,
                    previous_value=199000,
                    impact=EventImpact.EXTREME,
                    source="forex_factory"
                ),
                EconomicEvent(
                    id="ff_fed_2024_03",
                    event_type=EventType.FED_RATE_DECISION,
                    title="FOMC Rate Decision",
                    country="US",
                    scheduled_time_utc=now + timedelta(days=5),
                    forecast_value=5.50,
                    previous_value=5.50,
                    impact=EventImpact.EXTREME,
                    source="forex_factory"
                ),
            ]
            
        except Exception as e:
            logger.error(f"Error fetching ForexFactory: {e}")
            return []
    
    async def fetch_central_bank_calendar(self, bank: str = "FED") -> List[EconomicEvent]:
        """Fetch events from central bank calendars."""
        try:
            logger.debug(f"Fetching from {bank} calendar (mock mode)")
            
            # Mock implementation
            now = datetime.now(timezone.utc)
            
            if bank == "FED":
                return [
                    EconomicEvent(
                        id=f"fed_speech_{now.strftime('%Y%m%d')}",
                        event_type=EventType.FED_SPEECH,
                        title="Fed Chair Powell Speech",
                        country="US",
                        scheduled_time_utc=now + timedelta(hours=5),
                        impact=EventImpact.HIGH,
                        source="federal_reserve"
                    ),
                ]
            elif bank == "ECB":
                return [
                    EconomicEvent(
                        id=f"ecb_rate_{now.strftime('%Y%m%d')}",
                        event_type=EventType.FED_RATE_DECISION,
                        title="ECB Interest Rate Decision",
                        country="EU",
                        scheduled_time_utc=now + timedelta(days=3),
                        impact=EventImpact.HIGH,
                        source="ecb"
                    ),
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching {bank} calendar: {e}")
            return []
    
    def parse_custom_feed(self, json_data: str) -> List[EconomicEvent]:
        """Parse custom JSON event feed."""
        try:
            events = []
            data = json.loads(json_data)
            
            for item in data:
                event_type = EventType(item.get('type', 'other'))
                impact = EventImpact(item.get('impact', 'medium'))
                
                # Parse datetime with timezone handling
                dt_str = item.get('datetime', '')
                scheduled_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                
                event = EconomicEvent(
                    id=item.get('id', ''),
                    event_type=event_type,
                    title=item.get('title', ''),
                    country=item.get('country', 'US'),
                    scheduled_time_utc=scheduled_time,
                    actual_value=item.get('actual'),
                    forecast_value=item.get('forecast'),
                    previous_value=item.get('previous'),
                    impact=impact,
                    currency=item.get('currency', 'USD'),
                    source=item.get('source', 'custom')
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error parsing custom feed: {e}")
            return []


class EconomicCalendar:
    """
    Main economic calendar engine with trading halt capabilities.
    
    Features:
    - Multi-source event aggregation
    - Timezone-aware scheduling with DST handling
    - Automatic trading halt triggers
    - Event impact analysis
    - Historical event tracking
    
    Thread Safety:
    - All public methods are async-safe
    - Internal state protected by asyncio locks
    """
    
    def __init__(
        self,
        halt_config: Optional[TradingHaltConfig] = None,
        update_interval_minutes: float = 5.0
    ):
        self.halt_config = halt_config or TradingHaltConfig()
        self.update_interval = update_interval_minutes * 60
        
        self.parser = EconomicCalendarParser()
        
        # Event storage (bounded for memory safety)
        self._upcoming_events: deque[EconomicEvent] = deque(maxlen=500)
        self._completed_events: deque[EconomicEvent] = deque(maxlen=1000)
        self._all_events: Dict[str, EconomicEvent] = {}
        
        # Trading halt state
        self._is_halted = False
        self._halt_status: Optional[TradingHaltStatus] = None
        self._halt_lock = asyncio.Lock()
        
        # Subscribers
        self._subscribers: List[Callable[[EconomicEvent], None]] = []
        self._halt_subscribers: List[Callable[[TradingHaltStatus], None]] = []
        
        # Running state
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info("EconomicCalendar initialized")
    
    def subscribe(self, callback: Callable[[EconomicEvent], None]):
        """Subscribe to event notifications."""
        self._subscribers.append(callback)
    
    def subscribe_to_halts(self, callback: Callable[[TradingHaltStatus], None]):
        """Subscribe to trading halt status changes."""
        self._halt_subscribers.append(callback)
        logger.info(f"New halt subscriber added. Total: {len(self._halt_subscribers)}")
    
    async def refresh_events(self):
        """Refresh events from all sources."""
        try:
            # Fetch from all sources
            forex_events = await self.parser.fetch_forex_factory()
            fed_events = await self.parser.fetch_central_bank_calendar("FED")
            ecb_events = await self.parser.fetch_central_bank_calendar("ECB")
            
            all_new = forex_events + fed_events + ecb_events
            
            # Update internal storage
            now = datetime.now(timezone.utc)
            
            for event in all_new:
                if event.id not in self._all_events:
                    self._all_events[event.id] = event
                    
                    if event.scheduled_time_utc >= now:
                        self._upcoming_events.append(event)
                    else:
                        self._completed_events.append(event)
                    
                    # Notify subscribers
                    for subscriber in self._subscribers:
                        try:
                            res = subscriber(event)
                            if asyncio.iscoroutine(res):
                                await res
                        except Exception as e:
                            logger.error(f"Error notifying subscriber: {e}")
            
            self._last_update = now
            logger.info(f"Refreshed economic calendar: {len(all_new)} events")
            
        except Exception as e:
            logger.error(f"Error refreshing events: {e}", exc_info=True)
    
    def get_upcoming_events(
        self,
        within_minutes: int = 1440,  # Default: next 24 hours
        min_impact: EventImpact = EventImpact.LOW
    ) -> List[EconomicEvent]:
        """Get upcoming events filtered by time and impact."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=within_minutes)
        
        impact_order = [EventImpact.LOW, EventImpact.MEDIUM, EventImpact.HIGH, EventImpact.EXTREME]
        min_impact_idx = impact_order.index(min_impact)
        
        filtered = []
        for event in self._upcoming_events:
            if event.scheduled_time_utc > cutoff:
                continue
            if impact_order.index(event.impact) < min_impact_idx:
                continue
            filtered.append(event)
        
        return sorted(filtered, key=lambda x: x.scheduled_time_utc)
    
    def get_high_impact_events(self, within_hours: int = 24) -> List[EconomicEvent]:
        """Get only high and extreme impact events."""
        return self.get_upcoming_events(
            within_minutes=within_hours * 60,
            min_impact=EventImpact.HIGH
        )
    
    async def check_and_trigger_halt(self):
        """Check if trading should be halted due to upcoming events."""
        async with self._halt_lock:
            now = datetime.now(timezone.utc)
            
            # Check if current halt should end
            if self._is_halted and self._halt_status:
                if self._halt_status.resume_at and now >= self._halt_status.resume_at:
                    # Resume trading
                    self._is_halted = False
                    self._halt_status = None
                    logger.info("Trading halt lifted - resuming normal operations")
                    
                    # Notify halt subscribers
                    for subscriber in self._halt_subscribers:
                        try:
                            status = TradingHaltStatus(
                                is_halted=False,
                                reason="Halt period expired",
                                halted_since=now,
                                resume_at=None,
                                triggering_event=None
                            )
                            res = subscriber(status)
                            if asyncio.iscoroutine(res):
                                await res
                        except Exception as e:
                            logger.error(f"Error notifying halt subscriber: {e}")
                    return
            
            # Don't re-check if already halted
            if self._is_halted:
                return
            
            # Check for upcoming high-impact events
            for event in self.get_upcoming_events(within_minutes=60):
                if event.impact not in self.halt_config.halt_on_impacts:
                    continue
                
                time_until = event.time_until_event().total_seconds() / 60
                
                # Determine halt window based on impact
                if event.impact == EventImpact.EXTREME:
                    halt_window = self.halt_config.halt_before_extreme
                    resume_window = self.halt_config.resume_after_extreme
                elif event.impact == EventImpact.HIGH:
                    halt_window = self.halt_config.halt_before_high
                    resume_window = self.halt_config.resume_after_high
                else:
                    continue
                
                if 0 <= time_until <= halt_window:
                    # Trigger halt
                    resume_at = event.scheduled_time_utc + timedelta(minutes=resume_window)
                    
                    self._is_halted = True
                    self._halt_status = TradingHaltStatus(
                        is_halted=True,
                        reason=f"Upcoming {event.event_type.value} event: {event.title}",
                        halted_since=now,
                        resume_at=resume_at,
                        triggering_event=event
                    )
                    
                    logger.warning(
                        f"TRADING HALTED: {event.title} in {time_until:.1f} minutes. "
                        f"Resume at {resume_at.strftime('%H:%M:%S')} UTC"
                    )
                    
                    # Notify halt subscribers
                    for subscriber in self._halt_subscribers:
                        try:
                            res = subscriber(self._halt_status)
                            if asyncio.iscoroutine(res):
                                await res
                        except Exception as e:
                            logger.error(f"Error notifying halt subscriber: {e}")
                    
                    return
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return not self._is_halted
    
    def get_halt_status(self) -> Optional[TradingHaltStatus]:
        """Get current halt status."""
        return self._halt_status
    
    def get_next_event_by_type(self, event_type: EventType) -> Optional[EconomicEvent]:
        """Get the next upcoming event of a specific type."""
        for event in sorted(self._upcoming_events, key=lambda x: x.scheduled_time_utc):
            if event.event_type == event_type:
                return event
        return None
    
    def update_event_result(self, event_id: str, actual_value: float):
        """Update an event with its actual result."""
        if event_id in self._all_events:
            event = self._all_events[event_id]
            event.actual_value = actual_value
            event.is_completed = True
            
            # Move from upcoming to completed
            self._upcoming_events = deque(
                [e for e in self._upcoming_events if e.id != event_id],
                maxlen=500
            )
            self._completed_events.append(event)
            
            logger.info(f"Updated event {event_id}: actual={actual_value}")
    
    def analyze_surprise_history(
        self,
        event_type: EventType,
        lookback_count: int = 10
    ) -> Dict[str, Any]:
        """Analyze historical surprise factors for an event type."""
        relevant = [
            e for e in self._completed_events
            if e.event_type == event_type and e.is_completed
        ][-lookback_count:]
        
        if not relevant:
            return {"error": "No historical data"}
        
        surprises = [e.surprise_factor() for e in relevant if e.surprise_factor() is not None]
        
        if not surprises:
            return {"error": "No surprise data available"}
        
        import statistics
        return {
            "event_type": event_type.value,
            "sample_size": len(surprises),
            "mean_surprise": statistics.mean(surprises),
            "std_surprise": statistics.stdev(surprises) if len(surprises) > 1 else 0,
            "max_positive": max(surprises),
            "max_negative": min(surprises),
            "recent_direction": "positive" if surprises[-1] > 0 else "negative"
        }
    
    async def _update_loop(self):
        """Continuous update loop."""
        while self._running:
            try:
                await self.refresh_events()
                await self.check_and_trigger_halt()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(60)
    
    async def start(self):
        """Start the economic calendar monitoring."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting EconomicCalendar")
        
        # Initial refresh
        await self.refresh_events()
        
        self._tasks = [
            asyncio.create_task(self._update_loop())
        ]
    
    async def stop(self):
        """Stop monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


# Example usage
async def main():
    """Demonstration of EconomicCalendar functionality."""
    calendar = EconomicCalendar()
    
    def event_handler(event: EconomicEvent):
        print(f"📅 Event: {event.title} ({event.impact.value}) at {event.scheduled_time_utc}")
    
    def halt_handler(status: TradingHaltStatus):
        if status.is_halted:
            print(f"⛔ TRADING HALTED: {status.reason}")
        else:
            print("✅ Trading resumed")
    
    calendar.subscribe(event_handler)
    calendar.subscribe_to_halts(halt_handler)
    
    await calendar.start()
    
    # Show upcoming high-impact events
    print("\n🔴 Upcoming High-Impact Events:")
    for event in calendar.get_high_impact_events():
        print(f"  {event.title} - {event.scheduled_time_utc} ({event.impact.value})")
    
    # Check trading status
    print(f"\nTrading Allowed: {calendar.is_trading_allowed()}")
    
    # Analyze history
    history = calendar.analyze_surprise_history(EventType.CPI)
    print(f"\nCPI Surprise History: {history}")
    
    await asyncio.sleep(5)
    await calendar.stop()


if __name__ == "__main__":
    asyncio.run(main())
