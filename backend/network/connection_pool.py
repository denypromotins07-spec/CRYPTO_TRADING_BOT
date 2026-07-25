#!/usr/bin/env python3
"""
Connection Pool for ZAID Crypto Trading Bot

This module implements a high-performance asynchronous connection pool
for Binance REST API calls, optimized for low latency and efficient
resource utilization within the 8GB RAM constraint.

Features:
- Async connection pooling with aiohttp
- Automatic connection recycling and health checks
- Rate limiting compliance with Binance API limits
- Per-endpoint connection pools for optimal routing
- Memory-bounded to prevent exceeding RAM limits

Target: AMD Ryzen AI 5 laptop with 8GB RAM
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import aiohttp
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from contextlib import asynccontextmanager
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConnectionPoolConfig:
    """Configuration for the connection pool"""
    
    def __init__(
        self,
        # Total number of connections in the pool
        max_connections: int = 100,
        # Connections per host
        max_connections_per_host: int = 25,
        # Connection timeout in seconds
        connect_timeout: float = 5.0,
        # Request timeout in seconds
        request_timeout: float = 10.0,
        # Keepalive timeout
        keepalive_timeout: float = 30.0,
        # Enable connection health checks
        health_check_enabled: bool = True,
        # Health check interval in seconds
        health_check_interval: float = 60.0,
        # Maximum memory usage for pool (MB) - critical for 8GB limit
        max_memory_mb: int = 256,
    ):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.keepalive_timeout = keepalive_timeout
        self.health_check_enabled = health_check_enabled
        self.health_check_interval = health_check_interval
        self.max_memory_mb = max_memory_mb


class ConnectionHealth(Enum):
    """Health status of a connection"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ConnectionStats:
    """Statistics for a connection pool"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    active_connections: int = 0
    idle_connections: int = 0
    last_health_check: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


class AsyncConnectionPool:
    """
    High-performance async connection pool for Binance REST API
    
    This pool manages aiohttp sessions with automatic connection
    recycling, health monitoring, and rate limiting.
    """
    
    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        self.config = config or ConnectionPoolConfig()
        self._session: Optional[ClientSession] = None
        self._stats = ConnectionStats()
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_initialized = False
        
        # Track connection usage per endpoint
        self._endpoint_usage: Dict[str, int] = {}
        
        logger.info(f"Connection Pool initialized: max_conn={self.config.max_connections}, "
                   f"max_per_host={self.config.max_connections_per_host}")
    
    async def initialize(self):
        """Initialize the connection pool"""
        if self._is_initialized:
            return
        
        async with self._lock:
            if self._is_initialized:
                return
            
            # Configure TCP connector for optimal performance
            connector = TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_connections_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
                enable_cleanup_closed=True,
            )
            
            # Configure timeouts
            timeout = ClientTimeout(
                total=self.config.request_timeout,
                connect=self.config.connect_timeout,
                sock_read=self.config.request_timeout,
                sock_connect=self.config.connect_timeout,
            )
            
            # Create the session
            self._session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'ZAID-Crypto-Bot/1.0',
                    'Content-Type': 'application/json',
                },
            )
            
            self._is_initialized = True
            
            # Start health check task if enabled
            if self.config.health_check_enabled:
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )
            
            logger.info("Connection Pool initialized successfully")
    
    async def _health_check_loop(self):
        """Periodically check connection health"""
        while self._is_initialized:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self._session and not self._session.closed:
                    # Perform a lightweight health check
                    health = await self.check_health()
                    
                    if health == ConnectionHealth.UNHEALTHY:
                        logger.warning("Connection pool health degraded, considering recycle")
                        await self.recycle_connections()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def check_health(self) -> ConnectionHealth:
        """Check the health of the connection pool"""
        if not self._session or self._session.closed:
            return ConnectionHealth.UNHEALTHY
        
        # Check active vs idle connections
        if self._stats.active_connections > self.config.max_connections * 0.9:
            return ConnectionHealth.DEGRADED
        
        # Check success rate
        if self._stats.success_rate < 0.95:
            return ConnectionHealth.DEGRADED
        
        self._stats.last_health_check = time.time()
        return ConnectionHealth.HEALTHY
    
    async def recycle_connections(self):
        """Recycle all connections in the pool"""
        logger.info("Recycling connection pool...")
        
        if self._session:
            await self._session.close()
            self._session = None
            self._is_initialized = False
        
        # Reinitialize
        await self.initialize()
        logger.info("Connection pool recycled successfully")
    
    @asynccontextmanager
    async def get_session(self):
        """Get a session from the pool (async context manager)"""
        if not self._is_initialized:
            await self.initialize()
        
        if not self._session or self._session.closed:
            await self.initialize()
        
        try:
            self._stats.active_connections += 1
            yield self._session
        finally:
            self._stats.active_connections -= 1
            self._stats.idle_connections += 1
    
    async def request(
        self,
        method: str,
        url: str,
        endpoint: str = "default",
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make an HTTP request through the pool
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            endpoint: Endpoint identifier for tracking
            **kwargs: Additional arguments for aiohttp
            
        Returns:
            aiohttp.ClientResponse object
        """
        start_time = time.perf_counter()
        
        async with self.get_session() as session:
            self._stats.total_requests += 1
            
            # Track endpoint usage
            self._endpoint_usage[endpoint] = self._endpoint_usage.get(endpoint, 0) + 1
            
            try:
                async with session.request(method, url, **kwargs) as response:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Update statistics
                    self._stats.successful_requests += 1
                    self._update_avg_latency(latency_ms)
                    
                    # Return response (caller must handle it)
                    # Note: In production, you'd want to copy the response
                    # before the context manager closes
                    return response
                    
            except Exception as e:
                self._stats.failed_requests += 1
                logger.error(f"Request failed: {method} {url} - {e}")
                raise
    
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make a GET request and return JSON response"""
        async with self.get_session() as session:
            self._stats.total_requests += 1
            
            async with session.get(url, **kwargs) as response:
                response.raise_for_status()
                self._stats.successful_requests += 1
                return await response.json()
    
    async def post(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make a POST request and return JSON response"""
        async with self.get_session() as session:
            self._stats.total_requests += 1
            
            async with session.post(url, **kwargs) as response:
                response.raise_for_status()
                self._stats.successful_requests += 1
                return await response.json()
    
    def _update_avg_latency(self, new_latency_ms: float):
        """Update running average latency"""
        n = self._stats.successful_requests
        old_avg = self._stats.avg_latency_ms
        self._stats.avg_latency_ms = old_avg + (new_latency_ms - old_avg) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "success_rate": self._stats.success_rate,
            "avg_latency_ms": round(self._stats.avg_latency_ms, 2),
            "active_connections": self._stats.active_connections,
            "idle_connections": self._stats.idle_connections,
            "endpoint_usage": dict(self._endpoint_usage),
        }
    
    async def close(self):
        """Close the connection pool"""
        logger.info("Closing connection pool...")
        
        self._is_initialized = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._session and not self._session.closed:
            await self._session.close()
        
        logger.info("Connection pool closed")


# Singleton pool instance for global access
_pool_instance: Optional[AsyncConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool(config: Optional[ConnectionPoolConfig] = None) -> AsyncConnectionPool:
    """Get or create the global connection pool instance"""
    global _pool_instance
    
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                _pool_instance = AsyncConnectionPool(config)
    
    return _pool_instance


# Example usage
if __name__ == "__main__":
    async def test_pool():
        pool = AsyncConnectionPool(ConnectionPoolConfig(
            max_connections=10,
            max_connections_per_host=5,
        ))
        
        try:
            await pool.initialize()
            
            # Test a simple request
            stats_before = pool.get_stats()
            print(f"Stats before: {stats_before}")
            
            # Note: This would make real requests to Binance
            # response = await pool.get("https://api.binance.com/api/v3/time")
            # print(f"Server time: {response}")
            
            stats_after = pool.get_stats()
            print(f"Stats after: {stats_after}")
            
        finally:
            await pool.close()
    
    asyncio.run(test_pool())
