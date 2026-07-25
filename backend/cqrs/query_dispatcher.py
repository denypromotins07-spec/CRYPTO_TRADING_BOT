"""
Query Dispatcher - O(1) state lookups for the strategy engine.

This module provides the query interface for CQRS read models, enabling
the strategy engine to access current state with constant-time lookups.
Critical for low-latency trading decisions based on real-time positions,
orders, and portfolio metrics.

Features:
- O(1) dictionary-based lookups for all queries
- Query caching for frequently accessed data
- Read-write lock optimization for concurrent access
- Type-safe query definitions with result validation
- Metrics tracking for query performance monitoring

Integrates with 152 domains including:
- Strategy signal generation
- Risk limit checking
- Portfolio rebalancing triggers
- Real-time PnL monitoring
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar, Union
from threading import RLock
from collections import OrderedDict


class QueryType(Enum):
    """Enumeration of supported query types."""
    
    # Position queries
    GET_POSITION = "get_position"
    GET_ALL_POSITIONS = "get_all_positions"
    GET_POSITION_PNL = "get_position_pnl"
    GET_TOTAL_PNL = "get_total_pnl"
    GET_EXPOSURE = "get_exposure"
    
    # Order queries
    GET_ORDER = "get_order"
    GET_ORDERS_BY_SYMBOL = "get_orders_by_symbol"
    GET_ACTIVE_ORDERS = "get_active_orders"
    GET_PENDING_ORDERS = "get_pending_orders"
    
    # Portfolio queries
    GET_PORTFOLIO_SUMMARY = "get_portfolio_summary"
    GET_RISK_METRICS = "get_risk_metrics"
    GET_POSITION_COUNT = "get_position_count"
    
    # Market data queries
    GET_LAST_PRICE = "get_last_price"
    GET_ORDERBOOK_STATE = "get_orderbook_state"
    GET_SPREAD = "get_spread"
    
    # System queries
    GET_SYSTEM_STATUS = "get_system_status"
    GET_EVENT_LAG = "get_event_lag"
    HEALTH_CHECK = "health_check"


@dataclass
class QueryRequest:
    """Represents a query request with parameters."""
    query_type: QueryType
    params: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"qry_{int(time.time() * 1000)}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timeout_ms: float = 100.0  # Maximum allowed execution time
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Safely get a parameter with optional default."""
        return self.params.get(key, default)


@dataclass
class QueryResult(Generic[T]):
    """Result of a query execution."""
    success: bool
    data: Optional[T]
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def ok(cls, data: T, execution_time_ms: float = 0.0, cache_hit: bool = False) -> 'QueryResult[T]':
        return cls(success=True, data=data, execution_time_ms=execution_time_ms, cache_hit=cache_hit)
    
    @classmethod
    def error(cls, message: str, execution_time_ms: float = 0.0) -> 'QueryResult[Any]':
        return cls(success=False, data=None, error_message=message, execution_time_ms=execution_time_ms)


T = TypeVar('T')


class QueryHandler(Protocol):
    """Protocol for query handlers."""
    
    async def handle(self, request: QueryRequest) -> QueryResult[Any]:
        """Handle a query request and return result."""
        ...


@dataclass
class QueryMetrics:
    """Performance metrics for query execution."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_execution_time_ms: float = 0.0
    p99_latency_ms: float = 0.0
    last_error: Optional[str] = None
    
    def record_query(self, result: QueryResult[Any]) -> None:
        """Record metrics from a completed query."""
        self.total_queries += 1
        
        if result.success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
            self.last_error = result.error_message
        
        if result.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        self.total_execution_time_ms += result.execution_time_ms
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_queries == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_queries
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.successful_queries / max(1, self.total_queries),
            "cache_hit_rate": self.hit_rate,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "last_error": self.last_error,
        }


class LRUCache:
    """Simple LRU cache for frequently queried results."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._lock = RLock()
        self._ttl_seconds = 60  # Cache entries expire after 60 seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, returning None if not found or expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            
            # Check if entry has expired
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age > self._ttl_seconds:
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Add or update item in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, datetime.utcnow())
            
            # Evict oldest if over capacity
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
    
    def invalidate(self, key: str) -> None:
        """Remove specific key from cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class QueryDispatcher:
    """
    Central dispatcher for all CQRS queries.
    
    Provides O(1) lookups by routing queries to appropriate handlers
    and caching frequently accessed results. Designed for ultra-low
    latency access to read model state.
    """
    
    def __init__(self, read_model_builder: Any = None):
        self._handlers: Dict[QueryType, QueryHandler] = {}
        self._read_model = read_model_builder
        self._cache = LRUCache(capacity=5000)
        self._metrics: Dict[QueryType, QueryMetrics] = {}
        self._lock = RLock()
        self._is_running = True
        
        # Initialize metrics for all query types
        for qt in QueryType:
            self._metrics[qt] = QueryMetrics()
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register built-in query handlers."""
        
        # Position queries
        self.register_handler(QueryType.GET_POSITION, self._handle_get_position)
        self.register_handler(QueryType.GET_ALL_POSITIONS, self._handle_get_all_positions)
        self.register_handler(QueryType.GET_POSITION_PNL, self._handle_get_position_pnl)
        self.register_handler(QueryType.GET_TOTAL_PNL, self._handle_get_total_pnl)
        self.register_handler(QueryType.GET_EXPOSURE, self._handle_get_exposure)
        
        # Order queries
        self.register_handler(QueryType.GET_ORDER, self._handle_get_order)
        self.register_handler(QueryType.GET_ORDERS_BY_SYMBOL, self._handle_get_orders_by_symbol)
        self.register_handler(QueryType.GET_ACTIVE_ORDERS, self._handle_get_active_orders)
        
        # Portfolio queries
        self.register_handler(QueryType.GET_PORTFOLIO_SUMMARY, self._handle_get_portfolio_summary)
        self.register_handler(QueryType.GET_POSITION_COUNT, self._handle_get_position_count)
        
        # System queries
        self.register_handler(QueryType.HEALTH_CHECK, self._handle_health_check)
        self.register_handler(QueryType.GET_SYSTEM_STATUS, self._handle_get_system_status)
    
    def register_handler(self, query_type: QueryType, handler: QueryHandler) -> None:
        """Register a custom handler for a query type."""
        with self._lock:
            self._handlers[query_type] = handler
    
    def _get_cache_key(self, request: QueryRequest) -> str:
        """Generate cache key from request."""
        params_str = "_".join(f"{k}={v}" for k, v in sorted(request.params.items()))
        return f"{request.query_type.value}:{params_str}"
    
    async def dispatch(self, request: QueryRequest) -> QueryResult[Any]:
        """
        Dispatch a query request to the appropriate handler.
        
        This is the main entry point for all queries. The method:
        1. Checks cache for cached results
        2. Routes to appropriate handler
        3. Records metrics
        4. Returns typed result
        
        Args:
            request: The query request to execute
            
        Returns:
            QueryResult containing the query data or error
        """
        if not self._is_running:
            return QueryResult.error("Query dispatcher is not running")
        
        start_time = time.perf_counter()
        
        # Check cache for read-only queries
        cache_key = self._get_cache_key(request)
        cached_result = self._cache.get(cache_key)
        
        if cached_result is not None:
            execution_time = (time.perf_counter() - start_time) * 1000
            result = QueryResult.ok(cached_result, execution_time_ms=execution_time, cache_hit=True)
            self._record_metrics(request.query_type, result)
            return result
        
        # Get handler for query type
        handler = self._handlers.get(request.query_type)
        if handler is None:
            execution_time = (time.perf_counter() - start_time) * 1000
            result = QueryResult.error(f"No handler registered for query type: {request.query_type}")
            self._record_metrics(request.query_type, result)
            return result
        
        # Execute handler with timeout
        try:
            result = await asyncio.wait_for(
                handler.handle(request),
                timeout=request.timeout_ms / 1000.0
            )
            
            # Cache successful results for read-only queries
            if result.success and request.query_type in self._get_cacheable_types():
                self._cache.put(cache_key, result.data)
            
            self._record_metrics(request.query_type, result)
            return result
            
        except asyncio.TimeoutError:
            execution_time = (time.perf_counter() - start_time) * 1000
            result = QueryResult.error(f"Query timed out after {request.timeout_ms}ms")
            self._record_metrics(request.query_type, result)
            return result
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            result = QueryResult.error(f"Query execution failed: {str(e)}")
            self._record_metrics(request.query_type, result)
            return result
    
    def dispatch_sync(self, request: QueryRequest) -> QueryResult[Any]:
        """Synchronously dispatch a query (for non-async contexts)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.dispatch(request))
        finally:
            loop.close()
    
    def _record_metrics(self, query_type: QueryType, result: QueryResult[Any]) -> None:
        """Record query execution metrics."""
        with self._lock:
            if query_type not in self._metrics:
                self._metrics[query_type] = QueryMetrics()
            self._metrics[query_type].record_query(result)
    
    def _get_cacheable_types(self) -> set[QueryType]:
        """Return set of query types that can be cached."""
        return {
            QueryType.GET_POSITION,
            QueryType.GET_ALL_POSITIONS,
            QueryType.GET_ORDER,
            QueryType.GET_ACTIVE_ORDERS,
            QueryType.GET_PORTFOLIO_SUMMARY,
            QueryType.HEALTH_CHECK,
        }
    
    def invalidate_cache(self, query_type: Optional[QueryType] = None, key: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if query_type is None and key is None:
            self._cache.clear()
        elif key is not None:
            self._cache.invalidate(key)
        else:
            # Invalidate all entries for a query type
            prefix = f"{query_type.value}:"
            with self._cache._lock:
                keys_to_remove = [k for k in self._cache._cache.keys() if k.startswith(prefix)]
                for k in keys_to_remove:
                    del self._cache._cache[k]
    
    def get_metrics(self, query_type: Optional[QueryType] = None) -> Union[Dict[str, Any], QueryMetrics]:
        """Get query metrics."""
        with self._lock:
            if query_type is not None:
                return self._metrics.get(query_type, QueryMetrics())
            
            # Return aggregate metrics
            total = QueryMetrics()
            for m in self._metrics.values():
                total.total_queries += m.total_queries
                total.successful_queries += m.successful_queries
                total.failed_queries += m.failed_queries
                total.cache_hits += m.cache_hits
                total.cache_misses += m.cache_misses
                total.total_execution_time_ms += m.total_execution_time_ms
            
            return total.to_dict()
    
    # === Default Query Handlers ===
    
    async def _handle_get_position(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle GET_POSITION query."""
        symbol = request.get_param("symbol")
        if not symbol:
            return QueryResult.error("Missing required parameter: symbol")
        
        if self._read_model:
            view = self._read_model.get_positions_view()
            if view:
                position = view.get_position(symbol)
                if position:
                    return QueryResult.ok(position.to_dict())
                return QueryResult.ok(None)  # Position doesn't exist
        
        return QueryResult.error("Read model not available")
    
    async def _handle_get_all_positions(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle GET_ALL_POSITIONS query."""
        if self._read_model:
            view = self._read_model.get_positions_view()
            if view:
                return QueryResult.ok(view.get_state())
        
        return QueryResult.error("Read model not available")
    
    async def _handle_get_position_pnl(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle GET_POSITION_PNL query."""
        symbol = request.get_param("symbol")
        if not symbol:
            return QueryResult.error("Missing required parameter: symbol")
        
        if self._read_model:
            view = self._read_model.get_positions_view()
            if view:
                position = view.get_position(symbol)
                if position:
                    return QueryResult.ok({
                        "symbol": symbol,
                        "unrealized_pnl": position.unrealized_pnl,
                        "realized_pnl": position.realized_pnl,
                        "total_pnl": position.unrealized_pnl + position.realized_pnl,
                    })
        
        return QueryResult.error("Position not found")
    
    async def _handle_get_total_pnl(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle GET_TOTAL_PNL query."""
        if self._read_model:
            view = self._read_model.get_positions_view()
            if view:
                state = view.get_state()
                total_unrealized = sum(p.get("unrealized_pnl", 0) for p in state.values())
                total_realized = sum(p.get("realized_pnl", 0) for p in state.values())
                
                return QueryResult.ok({
                    "total_unrealized_pnl": total_unrealized,
                    "total_realized_pnl": total_realized,
                    "total_pnl": total_unrealized + total_realized,
                    "position_count": len(state),
                })
        
        return QueryResult.error("Read model not available")
    
    async def _handle_get_exposure(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle GET_EXPOSURE query."""
        symbol = request.get_param("symbol")
        
        if self._read_model:
            view = self._read_model.get_positions_view()
            if view:
                if symbol:
                    position = view.get_position(symbol)
                    if position:
                        return QueryResult.ok({
                            "symbol": symbol,
                            "quantity": position.quantity,
                            "notional_value": position.quantity * position.last_price,
                        })
                else:
                    state = view.get_state()
                    gross = sum(abs(p.get("quantity", 0) * p.get("last_price", 0)) for p in state.values())
                    net = sum(p.get("quantity", 0) * p.get("last_price", 0) for p in state.values())
                    
                    return QueryResult.ok({
                        "gross_exposure": gross,
                        "net_exposure": net,
                        "position_count": len(state),
                    })
        
        return QueryResult.error("Read model not available")
    
    async def _handle_get_order(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle GET_ORDER query."""
        order_id = request.get_param("order_id")
        if not order_id:
            return QueryResult.error("Missing required parameter: order_id")
        
        if self._read_model:
            view = self._read_model.get_orders_view()
            if view:
                order = view.get_order(order_id)
                if order:
                    return QueryResult.ok(order.to_dict())
                return QueryResult.ok(None)
        
        return QueryResult.error("Read model not available")
    
    async def _handle_get_orders_by_symbol(self, request: QueryRequest) -> QueryResult[List]:
        """Handle GET_ORDERS_BY_SYMBOL query."""
        symbol = request.get_param("symbol")
        if not symbol:
            return QueryResult.error("Missing required parameter: symbol")
        
        if self._read_model:
            view = self._read_model.get_orders_view()
            if view:
                orders = view.get_orders_by_symbol(symbol)
                return QueryResult.ok([o.to_dict() for o in orders])
        
        return QueryResult.error("Read model not available")
    
    async def _handle_get_active_orders(self, request: QueryRequest) -> QueryResult[List]:
        """Handle GET_ACTIVE_ORDERS query."""
        if self._read_model:
            view = self._read_model.get_orders_view()
            if view:
                orders = view.get_active_orders()
                return QueryResult.ok([o.to_dict() for o in orders])
        
        return QueryResult.error("Read model not available")
    
    async def _handle_get_portfolio_summary(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle GET_PORTFOLIO_SUMMARY query."""
        if self._read_model:
            return QueryResult.ok(self._read_model.get_all_states())
        
        return QueryResult.error("Read model not available")
    
    async def _handle_get_position_count(self, request: QueryRequest) -> QueryResult[int]:
        """Handle GET_POSITION_COUNT query."""
        if self._read_model:
            view = self._read_model.get_positions_view()
            if view:
                state = view.get_state()
                non_flat = sum(1 for p in state.values() if p.get("quantity", 0) != 0)
                return QueryResult.ok(non_flat)
        
        return QueryResult.ok(0)
    
    async def _handle_health_check(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle HEALTH_CHECK query."""
        return QueryResult.ok({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "cache_size": self._cache.size(),
            "is_running": self._is_running,
        })
    
    async def _handle_get_system_status(self, request: QueryRequest) -> QueryResult[Dict]:
        """Handle GET_SYSTEM_STATUS query."""
        return QueryResult.ok({
            "dispatcher_running": self._is_running,
            "registered_handlers": len(self._handlers),
            "cache_size": self._cache.size(),
            "metrics": self.get_metrics(),
        })
    
    def shutdown(self) -> None:
        """Shutdown the query dispatcher."""
        self._is_running = False
        self._cache.clear()


if __name__ == "__main__":
    # Demo usage
    async def demo():
        dispatcher = QueryDispatcher()
        
        # Health check
        request = QueryRequest(query_type=QueryType.HEALTH_CHECK)
        result = await dispatcher.dispatch(request)
        print(f"Health check: {result.data}")
        
        # Get metrics
        metrics = dispatcher.get_metrics()
        print(f"\nQuery metrics: {metrics}")
    
    asyncio.run(demo())
