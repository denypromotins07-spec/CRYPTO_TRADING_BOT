"""
Rate Limiter for ZAID Personal Crypto Trading Bot
==================================================
Chapter 1: Binance REST API, Authentication, and Strict Rate Limiting Infrastructure

Implements a strict token bucket algorithm to prevent API bans on Binance.
Optimized for high-frequency trading with microsecond precision.

Features:
- Token bucket rate limiting with refill logic
- Per-endpoint rate limit tracking
- Automatic backoff on rate limit hits
- Support for Binance weight-based limits
- Thread-safe for concurrent BTC, SOL, ETH, USDT trading

Author: Opus 4.8
Stage: 2 of 100
"""

import asyncio
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types of rate limits enforced by Binance."""
    REQUESTS_WEIGHT = "REQUESTS_WEIGHT"  # Weighted request limit per minute
    ORDERS = "ORDERS"  # Order count limit per second/day
    RAW_REQUESTS = "RAW_REQUESTS"  # Raw request count limit


@dataclass
class RateLimitConfig:
    """Configuration for a specific rate limit."""
    limit_type: RateLimitType
    max_value: int  # Maximum allowed value
    window_seconds: int  # Time window in seconds
    weight_per_request: int = 1  # Weight consumed per request


@dataclass
class TokenBucket:
    """Token bucket state for rate limiting."""
    tokens: float
    max_tokens: float
    refill_rate: float  # Tokens per second
    last_refill: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        if not isinstance(self.lock, threading.Lock):
            self.lock = threading.Lock()
    
    def consume(self, tokens: float) -> bool:
        """
        Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def time_until_tokens(self, tokens: float) -> float:
        """Calculate time until specified tokens are available."""
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / self.refill_rate


@dataclass
class RateLimitState:
    """Current state of a rate limit."""
    limit_type: RateLimitType
    current_value: int
    max_value: int
    window_seconds: int
    reset_time: float
    is_exceeded: bool = False


class BinanceRateLimiter:
    """
    Strict rate limiter for Binance API compliance.
    
    Implements:
    - Token bucket algorithm for smooth rate limiting
    - Weight-based request tracking (Binance uses weights)
    - Per-endpoint limit monitoring
    - Automatic exponential backoff
    
    Binance Spot API Limits:
    - 1200 request weight per minute (RAW_REQUESTS)
    - 10 orders per second per symbol (ORDERS)
    - 100,000 orders per 24 hours (ORDERS)
    
    We enforce stricter internal limits to stay safe.
    """
    
    # Default Binance limits (we use 80% for safety margin)
    DEFAULT_REQUEST_WEIGHT_PER_MIN = 1200
    DEFAULT_ORDERS_PER_SECOND = 10
    DEFAULT_ORDERS_PER_DAY = 100000
    
    def __init__(
        self,
        request_weight_limit: Optional[int] = None,
        orders_per_second_limit: Optional[int] = None,
        safety_factor: float = 0.8
    ):
        """
        Initialize rate limiter.
        
        Args:
            request_weight_limit: Max request weight per minute
            orders_per_second_limit: Max orders per second
            safety_factor: Factor to apply to limits (0.8 = 80%)
        """
        self.safety_factor = safety_factor
        
        # Calculate effective limits
        effective_weight = int((request_weight_limit or self.DEFAULT_REQUEST_WEIGHT_PER_MIN) * safety_factor)
        effective_ops = int((orders_per_second_limit or self.DEFAULT_ORDERS_PER_SECOND) * safety_factor)
        
        # Token buckets for different limit types
        self._buckets: Dict[RateLimitType, TokenBucket] = {
            RateLimitType.REQUESTS_WEIGHT: TokenBucket(
                tokens=float(effective_weight),
                max_tokens=float(effective_weight),
                refill_rate=effective_weight / 60.0,  # Per second
            ),
            RateLimitType.ORDERS: TokenBucket(
                tokens=float(effective_ops),
                max_tokens=float(effective_ops),
                refill_rate=float(effective_ops),  # Per second
            ),
        }
        
        # Per-endpoint weight tracking
        self._endpoint_weights: Dict[str, int] = {
            "/api/v3/order": 1,
            "/api/v3/order/test": 1,
            "/api/v3/openOrders": 3,
            "/api/v3/allOrders": 10,
            "/api/v3/account": 10,
            "/api/v3/myTrades": 10,
            "/api/v3/depth": 50,  # Depends on limit parameter
            "/api/v3/trades": 50,
            "/api/v3/klines": 50,
            "/api/v3/ticker/price": 2,
            "/api/v3/ticker/bookTicker": 2,
            "/api/v3/exchangeInfo": 10,
            "/api/v3/time": 1,
        }
        
        # Rate limit states
        self._limit_states: Dict[RateLimitType, RateLimitState] = {}
        
        # Backoff state
        self._backoff_count: int = 0
        self._max_backoff_count: int = 5
        self._base_backoff_seconds: float = 1.0
        
        # Async lock for awaitable operations
        self._async_lock = asyncio.Lock()
        
        logger.info(f"Rate limiter initialized with {safety_factor*100:.0f}% safety factor")
        logger.info(f"  Request weight limit: {effective_weight}/min")
        logger.info(f"  Orders per second limit: {effective_ops}")
    
    def set_endpoint_weight(self, endpoint: str, weight: int):
        """Set custom weight for an endpoint."""
        self._endpoint_weights[endpoint] = weight
        logger.debug(f"Endpoint {endpoint} weight set to {weight}")
    
    def get_endpoint_weight(self, endpoint: str) -> int:
        """Get weight for an endpoint."""
        return self._endpoint_weights.get(endpoint, 1)
    
    async def acquire(
        self,
        endpoint: Optional[str] = None,
        order: bool = False,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire rate limit tokens before making a request.
        
        Args:
            endpoint: API endpoint being called (for weight calculation)
            order: Whether this is an order submission
            timeout: Maximum time to wait for tokens (None = wait indefinitely)
        
        Returns:
            True if tokens acquired, False if timeout exceeded
        """
        start_time = time.time()
        
        # Determine tokens needed
        if order:
            tokens_needed = 1.0
            bucket_type = RateLimitType.ORDERS
        elif endpoint:
            tokens_needed = float(self.get_endpoint_weight(endpoint))
            bucket_type = RateLimitType.REQUESTS_WEIGHT
        else:
            tokens_needed = 1.0
            bucket_type = RateLimitType.REQUESTS_WEIGHT
        
        while True:
            # Check if we can acquire tokens
            bucket = self._buckets[bucket_type]
            if bucket.consume(tokens_needed):
                return True
            
            # Calculate wait time
            wait_time = bucket.time_until_tokens(tokens_needed)
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    logger.warning(f"Rate limit timeout after {elapsed:.2f}s")
                    return False
                wait_time = min(wait_time, remaining)
            
            # Wait for tokens
            if wait_time > 0:
                logger.debug(f"Rate limit waiting {wait_time:.3f}s for {bucket_type.value}")
                await asyncio.sleep(wait_time)
    
    def try_acquire(
        self,
        endpoint: Optional[str] = None,
        order: bool = False
    ) -> bool:
        """
        Try to acquire tokens without waiting.
        
        Args:
            endpoint: API endpoint being called
            order: Whether this is an order submission
        
        Returns:
            True if tokens acquired immediately, False otherwise
        """
        if order:
            tokens_needed = 1.0
            bucket_type = RateLimitType.ORDERS
        elif endpoint:
            tokens_needed = float(self.get_endpoint_weight(endpoint))
            bucket_type = RateLimitType.REQUESTS_WEIGHT
        else:
            tokens_needed = 1.0
            bucket_type = RateLimitType.REQUESTS_WEIGHT
        
        return self._buckets[bucket_type].consume(tokens_needed)
    
    async def acquire_with_backoff(
        self,
        endpoint: Optional[str] = None,
        order: bool = False,
        max_retries: int = 3
    ) -> bool:
        """
        Acquire tokens with exponential backoff on failure.
        
        Args:
            endpoint: API endpoint
            order: Whether this is an order
            max_retries: Maximum retry attempts
        
        Returns:
            True if acquired, False if all retries failed
        """
        for attempt in range(max_retries):
            if self.try_acquire(endpoint=endpoint, order=order):
                self._backoff_count = 0  # Reset on success
                return True
            
            # Exponential backoff
            backoff_time = self._base_backoff_seconds * (2 ** attempt)
            logger.warning(f"Rate limit hit, backing off for {backoff_time:.2f}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(backoff_time)
        
        self._backoff_count += 1
        if self._backoff_count >= self._max_backoff_count:
            logger.error(f"Max backoff count ({self._max_backoff_count}) reached")
        
        return False
    
    def get_rate_limit_state(self, limit_type: RateLimitType) -> RateLimitState:
        """Get current state of a rate limit."""
        bucket = self._buckets[limit_type]
        with bucket.lock:
            bucket._refill()
            return RateLimitState(
                limit_type=limit_type,
                current_value=int(bucket.tokens),
                max_value=int(bucket.max_tokens),
                window_seconds=60 if limit_type == RateLimitType.REQUESTS_WEIGHT else 1,
                reset_time=bucket.last_refill + (bucket.max_tokens - bucket.tokens) / bucket.refill_rate,
                is_exceeded=bucket.tokens < 1,
            )
    
    def get_all_states(self) -> Dict[RateLimitType, RateLimitState]:
        """Get states for all rate limits."""
        return {lt: self.get_rate_limit_state(lt) for lt in RateLimitType}
    
    def reset_backoff(self):
        """Reset backoff counter after successful operations."""
        self._backoff_count = 0
    
    @property
    def current_backoff_count(self) -> int:
        """Get current backoff count."""
        return self._backoff_count


class AdaptiveRateLimiter(BinanceRateLimiter):
    """
    Adaptive rate limiter that adjusts based on API response headers.
    
    Monitors Binance response headers to dynamically adjust limits
    and avoid hitting rate limits proactively.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_headers: Dict[str, Any] = {}
        self._adjustment_factor: float = 1.0
    
    def update_from_headers(self, headers: Dict[str, str]):
        """
        Update rate limit state from Binance response headers.
        
        Headers monitored:
        - X-MBX-USED-WEIGHT-1M: Current weight used in 1 minute
        - X-MBX-ORDER-COUNT-1S: Order count in 1 second
        - X-MBX-ORDER-COUNT-1D: Order count in 1 day
        - Retry-After: Seconds to wait if rate limited
        """
        # Parse weight header
        used_weight = headers.get("X-MBX-USED-WEIGHT-1M")
        if used_weight:
            used = int(used_weight)
            bucket = self._buckets[RateLimitType.REQUESTS_WEIGHT]
            max_weight = int(bucket.max_tokens)
            
            # Adjust if we're using more than expected
            usage_ratio = used / max_weight
            if usage_ratio > 0.8:
                self._adjustment_factor = max(0.5, self._adjustment_factor * 0.9)
                logger.info(f"Reducing rate limit factor to {self._adjustment_factor:.2f} due to high usage")
        
        # Parse order count headers
        order_count_1s = headers.get("X-MBX-ORDER-COUNT-1S")
        if order_count_1s:
            count = int(order_count_1s)
            if count >= 8:  # Approaching limit
                logger.warning(f"Order count approaching limit: {count}/10 per second")
        
        self._response_headers = headers
    
    def get_adjusted_limit(self, limit_type: RateLimitType) -> float:
        """Get rate limit adjusted by adaptive factor."""
        base_bucket = self._buckets[limit_type]
        return base_bucket.max_tokens * self._adjustment_factor


# Singleton instance for global access
_rate_limiter_instance: Optional[BinanceRateLimiter] = None
_instance_lock = threading.Lock()


def get_rate_limiter() -> BinanceRateLimiter:
    """Get or create global rate limiter singleton."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        with _instance_lock:
            if _rate_limiter_instance is None:
                _rate_limiter_instance = BinanceRateLimiter()
    return _rate_limiter_instance


async def main():
    """Example usage of rate limiter."""
    limiter = BinanceRateLimiter(safety_factor=0.8)
    
    # Simulate multiple requests
    for i in range(20):
        acquired = await limiter.acquire(endpoint="/api/v3/order", order=True)
        if acquired:
            print(f"Request {i+1}: Acquired")
        else:
            print(f"Request {i+1}: Rate limited")
        await asyncio.sleep(0.05)  # 50ms between requests
    
    # Print final state
    states = limiter.get_all_states()
    for lt, state in states.items():
        print(f"{lt.value}: {state.current_value}/{state.max_value}")


if __name__ == "__main__":
    asyncio.run(main())
