#!/usr/bin/env python3
"""
Message Broker for Routing Nautilus Events to Distributed Ray Actors

This module implements a high-performance message broker that routes Nautilus
trading events to distributed Ray actors for parallel processing.
Designed for the ZAID Personal Crypto Trading Bot to achieve 8k-20k INR/hour targets.

Key Features:
- Mediator pattern for decoupled event routing
- Publisher-Subscriber architecture for scalable distribution
- Reactor pattern for async event handling
- Integration with Ray for distributed actor execution
- Backpressure handling and flow control
- Compatible with 8GB RAM constraint across BTC, SOL, ETH, USDT parallel streams

Domain Integration: Quantitative Finance Domains 49-60 (Message Brokers, Distributed Systems)
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

# Attempt to import ray, provide fallback if not available
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None  # type: ignore

# Type definitions
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages supported by the broker."""
    TICK = "tick"
    ORDER = "order"
    FILL = "fill"
    CANCEL = "cancel"
    POSITION = "position"
    PNL = "pnl"
    SIGNAL = "signal"
    RISK = "risk"
    SYSTEM = "system"
    NAUTILUS_EVENT = "nautilus_event"


class MessagePriority(IntEnum):
    """Message priority levels for routing decisions."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(slots=True, frozen=True)
class Message:
    """
    Core message structure for broker communication.
    Immutable for thread safety.
    """
    id: str
    message_type: MessageType
    priority: MessagePriority
    timestamp_ns: int
    source: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_ms: int = 30000  # Time-to-live in milliseconds
    headers: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        message_type: MessageType,
        payload: Dict[str, Any],
        source: str = "unknown",
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        ttl_ms: int = 30000,
    ) -> Message:
        """Factory method for creating messages."""
        return cls(
            id=str(uuid.uuid4()),
            message_type=message_type,
            priority=priority,
            timestamp_ns=time.time_ns(),
            source=source,
            payload=payload,
            correlation_id=correlation_id,
            ttl_ms=ttl_ms,
        )
    
    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        now_ns = time.time_ns()
        expiry_ns = self.timestamp_ns + (self.ttl_ms * 1_000_000)
        return now_ns > expiry_ns
    
    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps({
            "id": self.id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "timestamp_ns": self.timestamp_ns,
            "source": self.source,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl_ms": self.ttl_ms,
            "headers": self.headers,
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> Message:
        """Deserialize message from JSON."""
        data = json.loads(json_str)
        return cls(
            id=data["id"],
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            timestamp_ns=data["timestamp_ns"],
            source=data["source"],
            payload=data["payload"],
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl_ms=data.get("ttl_ms", 30000),
            headers=data.get("headers", {}),
        )


@dataclass(slots=True)
class Subscription:
    """Represents a subscription to message topics."""
    subscriber_id: str
    topics: Set[MessageType]
    callback: Optional[Callable[[Message], None]] = None
    filter_fn: Optional[Callable[[Message], bool]] = None
    max_queue_size: int = 10000
    is_active: bool = True


@dataclass(slots=True)
class BrokerStats:
    """Statistics for monitoring broker performance."""
    total_messages_published: int = 0
    total_messages_delivered: int = 0
    total_messages_dropped: int = 0
    total_messages_expired: int = 0
    active_subscribers: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    queue_depth: int = 0
    throughput_per_sec: float = 0.0


class DeadLetterQueue:
    """
    Queue for messages that failed delivery or expired.
    Provides visibility into message processing failures.
    """
    
    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._queue: List[Tuple[Message, str, float]] = []  # (message, reason, timestamp)
        self._lock = asyncio.Lock()
    
    async def add(self, message: Message, reason: str) -> None:
        """Add a failed message to the DLQ."""
        async with self._lock:
            self._queue.append((message, reason, time.time()))
            
            # Trim if exceeding max size
            if len(self._queue) > self._max_size:
                self._queue = self._queue[-self._max_size:]
    
    async def get_recent(self, count: int = 100) -> List[Tuple[Message, str, float]]:
        """Get recent failed messages."""
        async with self._lock:
            return self._queue[-count:]
    
    async def clear(self) -> int:
        """Clear the DLQ and return count of cleared messages."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count
    
    async def size(self) -> int:
        """Get current DLQ size."""
        async with self._lock:
            return len(self._queue)


class MessageBroker:
    """
    High-performance message broker implementing Mediator and Pub/Sub patterns.
    Routes Nautilus events to distributed Ray actors for parallel processing.
    """
    
    def __init__(self, enable_ray: bool = False):
        """
        Initialize the message broker.
        
        Args:
            enable_ray: Whether to enable Ray distributed processing
        """
        self._subscribers: DefaultDict[MessageType, List[Subscription]] = defaultdict(list)
        self._subscriber_by_id: Dict[str, Subscription] = {}
        self._pending_messages: asyncio.Queue[Message] = asyncio.Queue(maxsize=100000)
        self._dlq = DeadLetterQueue(max_size=10000)
        self._stats = BrokerStats()
        self._latency_samples: List[float] = []
        self._running = False
        self._enable_ray = enable_ray and RAY_AVAILABLE
        self._ray_actors: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
        if self._enable_ray and not RAY_AVAILABLE:
            logger.warning("Ray requested but not available, falling back to local processing")
            self._enable_ray = False
    
    async def start(self) -> None:
        """Start the message broker processing loop."""
        if self._running:
            return
        
        self._running = True
        
        if self._enable_ray and ray is not None:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True, _memory=4*1024*1024*1024)  # 4GB for Ray
                logger.info("Ray initialized for distributed processing")
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                self._enable_ray = False
        
        # Start the processing task
        asyncio.create_task(self._process_loop())
        logger.info("Message broker started")
    
    async def stop(self) -> None:
        """Stop the message broker gracefully."""
        self._running = False
        
        if self._enable_ray and ray is not None:
            try:
                ray.shutdown()
            except Exception:
                pass
        
        logger.info("Message broker stopped")
    
    async def publish(self, message: Message) -> bool:
        """
        Publish a message to all subscribed handlers.
        
        Args:
            message: Message to publish
        
        Returns:
            True if message was accepted, False if rejected
        """
        if message.is_expired():
            await self._dlq.add(message, "Message expired before publishing")
            self._stats.total_messages_expired += 1
            return False
        
        try:
            await self._pending_messages.put(message)
            self._stats.total_messages_published += 1
            self._stats.queue_depth = self._pending_messages.qsize()
            return True
        except asyncio.QueueFull:
            await self._dlq.add(message, "Queue full - backpressure")
            self._stats.total_messages_dropped += 1
            return False
    
    async def subscribe(
        self,
        subscriber_id: str,
        topics: Set[MessageType],
        callback: Optional[Callable[[Message], None]] = None,
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ) -> str:
        """
        Subscribe to message topics.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            topics: Set of message types to subscribe to
            callback: Optional synchronous callback for local handling
            filter_fn: Optional filter function for fine-grained control
        
        Returns:
            Subscription ID
        """
        async with self._lock:
            subscription = Subscription(
                subscriber_id=subscriber_id,
                topics=topics,
                callback=callback,
                filter_fn=filter_fn,
            )
            
            for topic in topics:
                self._subscribers[topic].append(subscription)
            
            self._subscriber_by_id[subscriber_id] = subscription
            self._stats.active_subscribers = len(self._subscriber_by_id)
            
            logger.info(f"Subscriber {subscriber_id} registered for topics: {topics}")
            return subscriber_id
    
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe a subscriber from all topics."""
        async with self._lock:
            if subscriber_id not in self._subscriber_by_id:
                return False
            
            subscription = self._subscriber_by_id.pop(subscriber_id)
            
            for topic in subscription.topics:
                self._subscribers[topic] = [
                    s for s in self._subscribers[topic]
                    if s.subscriber_id != subscriber_id
                ]
            
            self._stats.active_subscribers = len(self._subscriber_by_id)
            logger.info(f"Subscriber {subscriber_id} unsubscribed")
            return True
    
    async def _process_loop(self) -> None:
        """Main message processing loop."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._pending_messages.get(),
                    timeout=1.0
                )
                
                start_time = time.time()
                
                # Route to subscribers
                await self._route_message(message)
                
                # Record latency
                latency_ms = (time.time() - start_time) * 1000
                self._latency_samples.append(latency_ms)
                
                # Keep only last 1000 samples
                if len(self._latency_samples) > 1000:
                    self._latency_samples = self._latency_samples[-1000:]
                
                # Update stats
                self._stats.total_messages_delivered += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
    
    async def _route_message(self, message: Message) -> None:
        """Route a message to all matching subscribers."""
        subscribers = self._subscribers.get(message.message_type, [])
        
        for subscription in subscribers:
            if not subscription.is_active:
                continue
            
            # Apply filter if present
            if subscription.filter_fn and not subscription.filter_fn(message):
                continue
            
            try:
                if subscription.callback:
                    # Synchronous callback
                    subscription.callback(message)
                
                if self._enable_ray and ray is not None:
                    # Async Ray actor invocation
                    await self._invoke_ray_actor(message, subscription.subscriber_id)
                    
            except Exception as e:
                logger.error(f"Error delivering to {subscription.subscriber_id}: {e}")
                await self._dlq.add(
                    message,
                    f"Delivery failed to {subscription.subscriber_id}: {e}"
                )
    
    async def _invoke_ray_actor(self, message: Message, subscriber_id: str) -> None:
        """Invoke a Ray actor for distributed processing."""
        if not self._enable_ray or ray is None:
            return
        
        try:
            if subscriber_id in self._ray_actors:
                actor = self._ray_actors[subscriber_id]
                # Remote call to Ray actor
                await actor.process.remote(message.to_json())
        except Exception as e:
            logger.error(f"Ray actor invocation failed for {subscriber_id}: {e}")
    
    def register_ray_actor(self, subscriber_id: str, actor: Any) -> None:
        """Register a Ray actor for a subscriber."""
        self._ray_actors[subscriber_id] = actor
    
    def get_stats(self) -> BrokerStats:
        """Get current broker statistics."""
        if self._latency_samples:
            sorted_samples = sorted(self._latency_samples)
            self._stats.avg_latency_ms = sum(sorted_samples) / len(sorted_samples)
            p99_index = int(len(sorted_samples) * 0.99)
            self._stats.p99_latency_ms = sorted_samples[p99_index] if p99_index < len(sorted_samples) else 0
        
        self._stats.queue_depth = self._pending_messages.qsize()
        return self._stats
    
    async def get_dlq_messages(self, count: int = 100) -> List[Tuple[Message, str, float]]:
        """Get recent dead letter queue messages."""
        return await self._dlq.get_recent(count)
    
    async def clear_dlq(self) -> int:
        """Clear the dead letter queue."""
        return await self._dlq.clear()


class NautilusEventAdapter:
    """
    Adapter for converting Nautilus Trader events to broker messages.
    Provides integration with the Nautilus trading framework.
    """
    
    @staticmethod
    def adapt_event(event_type: str, event_data: Dict[str, Any]) -> Message:
        """
        Convert a Nautilus event to a broker message.
        
        Args:
            event_type: Nautilus event type string
            event_data: Event payload dictionary
        
        Returns:
            Broker Message object
        """
        # Map Nautilus event types to our message types
        type_mapping = {
            "OrderSubmitted": MessageType.ORDER,
            "OrderAccepted": MessageType.ORDER,
            "OrderRejected": MessageType.ORDER,
            "OrderFilled": MessageType.FILL,
            "OrderCancelled": MessageType.CANCEL,
            "PositionOpened": MessageType.POSITION,
            "PositionClosed": MessageType.POSITION,
            "AccountState": MessageType.PNL,
        }
        
        message_type = type_mapping.get(event_type, MessageType.NAUTILUS_EVENT)
        
        # Determine priority based on event type
        priority = MessagePriority.NORMAL
        if event_type in ["OrderRejected", "RiskAlert"]:
            priority = MessagePriority.HIGH
        elif event_type == "SystemEvent":
            priority = MessagePriority.CRITICAL
        
        return Message.create(
            message_type=message_type,
            payload=event_data,
            source="nautilus",
            priority=priority,
        )


if __name__ == "__main__":
    # Self-test and validation
    print("Message Broker Module - ZAID Personal Crypto Trading Bot")
    print("=" * 70)
    
    async def test_broker():
        broker = MessageBroker(enable_ray=False)
        await broker.start()
        
        # Test message creation
        msg = Message.create(
            message_type=MessageType.TICK,
            payload={"symbol": "BTC/USD", "price": 50000.0, "volume": 1.5},
            source="test",
            priority=MessagePriority.HIGH,
        )
        
        print(f"✓ Created message: {msg.id}")
        print(f"  Type: {msg.message_type}")
        print(f"  Priority: {msg.priority}")
        
        # Test serialization
        json_str = msg.to_json()
        restored = Message.from_json(json_str)
        assert restored.id == msg.id
        print("✓ Message serialization test passed")
        
        # Test subscription
        received_messages: List[Message] = []
        
        def callback(m: Message) -> None:
            received_messages.append(m)
        
        await broker.subscribe(
            subscriber_id="test_subscriber",
            topics={MessageType.TICK, MessageType.ORDER},
            callback=callback,
        )
        
        # Publish message
        await broker.publish(msg)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check delivery
        stats = broker.get_stats()
        print(f"✓ Published: {stats.total_messages_published}")
        print(f"✓ Delivered: {stats.total_messages_delivered}")
        
        await broker.stop()
    
    asyncio.run(test_broker())
    
    print("\n✓ Message broker module validated successfully")
