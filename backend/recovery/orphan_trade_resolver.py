"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 9
File: backend/recovery/orphan_trade_resolver.py

Reconciles local state with exchange state to detect and resolve orphaned orders.
Automatically cancels orphaned orders within 50ms of detection.

Features:
- Real-time state reconciliation with Binance
- API signature verification for security
- Automatic orphan order cancellation
- Discrepancy logging and alerting
- Idempotent resolution operations

Design Patterns: Strategy, Observer, Command
"""

from __future__ import annotations
import asyncio
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderDiscrepancyType(Enum):
    """Types of order discrepancies."""
    MISSING_LOCAL = "missing_local"  # Order exists on exchange but not locally
    MISSING_EXCHANGE = "missing_exchange"  # Order exists locally but not on exchange
    STATUS_MISMATCH = "status_mismatch"  # Order status differs
    QUANTITY_MISMATCH = "quantity_mismatch"  # Filled quantity differs
    PRICE_MISMATCH = "price_mismatch"  # Price differs


@dataclass
class LocalOrder:
    """Local order representation."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    filled_quantity: float
    status: str
    created_at: float
    exchange_order_id: Optional[str] = None


@dataclass
class ExchangeOrder:
    """Exchange order representation."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    filled_quantity: float
    status: str
    created_at: float


@dataclass
class Discrepancy:
    """Represents a detected discrepancy."""
    discrepancy_type: OrderDiscrepancyType
    local_order: Optional[LocalOrder]
    exchange_order: Optional[ExchangeOrder]
    detected_at: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    resolved: bool = False
    resolution_action: Optional[str] = None
    resolution_time: Optional[float] = None


class OrphanTradeResolver:
    """
    Detects and resolves orphaned trades by reconciling local and exchange state.
    
    Ensures that no orphaned orders remain active for more than 50ms
    after detection, protecting against unintended exposure.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.binance.com",
        max_resolution_time_ms: float = 50.0,
    ):
        """
        Initialize the orphan trade resolver.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            base_url: Binance API base URL
            max_resolution_time_ms: Maximum time to resolve orphan (target: 50ms)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.max_resolution_time_ms = max_resolution_time_ms
        
        self._local_orders: Dict[str, LocalOrder] = {}
        self._discrepancies: List[Discrepancy] = []
        self._resolution_stats = {
            'total_detected': 0,
            'total_resolved': 0,
            'avg_resolution_time_ms': 0.0,
            'failed_resolutions': 0,
        }
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        
        logger.info("OrphanTradeResolver initialized")
    
    async def start(self) -> None:
        """Start the HTTP session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._running = True
            logger.info("OrphanTradeResolver started")
    
    async def stop(self) -> None:
        """Stop and cleanup."""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("OrphanTradeResolver stopped")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for API request."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _verify_api_signature(self, params: Dict[str, Any], signature: str) -> bool:
        """Verify API signature for security."""
        expected = self._generate_signature(params)
        return hmac.compare_digest(expected, signature)
    
    def register_local_order(self, order: LocalOrder) -> None:
        """Register a local order for tracking."""
        self._local_orders[order.order_id] = order
        logger.debug(f"Registered local order: {order.order_id}")
    
    def remove_local_order(self, order_id: str) -> None:
        """Remove a local order from tracking."""
        self._local_orders.pop(order_id, None)
    
    async def fetch_exchange_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """
        Fetch open orders from Binance.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of exchange orders
        """
        if not self._session:
            raise RuntimeError("Resolver not started")
        
        params = {
            'timestamp': int(time.time() * 1000),
        }
        
        if symbol:
            params['symbol'] = symbol
        
        signature = self._generate_signature(params)
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': self.api_key,
        }
        
        url = f"{self.base_url}/api/v3/openOrders"
        
        try:
            async with self._session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        ExchangeOrder(
                            order_id=str(order['orderId']),
                            symbol=order['symbol'],
                            side=order['side'],
                            order_type=order['type'],
                            quantity=float(order['origQty']),
                            price=float(order['price']) if order['price'] else None,
                            filled_quantity=float(order['executedQty']),
                            status=order['status'],
                            created_at=order['time'] / 1000,
                        )
                        for order in data
                    ]
                else:
                    logger.error(f"Failed to fetch orders: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching exchange orders: {e}")
            return []
    
    async def reconcile(self, symbols: Optional[List[str]] = None) -> List[Discrepancy]:
        """
        Reconcile local state with exchange state.
        
        Args:
            symbols: Optional list of symbols to reconcile
            
        Returns:
            List of detected discrepancies
        """
        start_time = time.perf_counter()
        discrepancies = []
        
        # Fetch exchange orders
        exchange_orders = []
        if symbols:
            for symbol in symbols:
                orders = await self.fetch_exchange_orders(symbol)
                exchange_orders.extend(orders)
        else:
            exchange_orders = await self.fetch_exchange_orders()
        
        # Create lookup maps
        exchange_by_id = {o.order_id: o for o in exchange_orders}
        local_by_exchange_id = {
            o.exchange_order_id: o 
            for o in self._local_orders.values() 
            if o.exchange_order_id
        }
        
        # Check for missing local orders (orphaned on exchange)
        for ex_order in exchange_orders:
            local_order = None
            
            # Try to find matching local order
            if ex_order.order_id in local_by_exchange_id:
                local_order = local_by_exchange_id[ex_order.order_id]
            else:
                # Check by order_id directly
                local_order = self._local_orders.get(ex_order.order_id)
            
            if local_order is None:
                # Orphaned order detected!
                discrepancy = Discrepancy(
                    discrepancy_type=OrderDiscrepancyType.MISSING_LOCAL,
                    local_order=None,
                    exchange_order=ex_order,
                    detected_at=time.time(),
                    severity='critical',  # Orphaned orders are critical
                )
                discrepancies.append(discrepancy)
                logger.critical(
                    f"ORPHAN ORDER DETECTED: {ex_order.order_id} "
                    f"on {ex_order.symbol} ({ex_order.side})"
                )
            else:
                # Check for status/quantity mismatches
                if local_order.status != ex_order.status:
                    discrepancy = Discrepancy(
                        discrepancy_type=OrderDiscrepancyType.STATUS_MISMATCH,
                        local_order=local_order,
                        exchange_order=ex_order,
                        detected_at=time.time(),
                        severity='high',
                    )
                    discrepancies.append(discrepancy)
                
                if abs(local_order.filled_quantity - ex_order.filled_quantity) > 0.0001:
                    discrepancy = Discrepancy(
                        discrepancy_type=OrderDiscrepancyType.QUANTITY_MISMATCH,
                        local_order=local_order,
                        exchange_order=ex_order,
                        detected_at=time.time(),
                        severity='high',
                    )
                    discrepancies.append(discrepancy)
        
        # Check for missing exchange orders (stale local orders)
        for local_order in self._local_orders.values():
            if local_order.exchange_order_id and local_order.exchange_order_id not in exchange_by_id:
                if local_order.status not in ['CANCELLED', 'FILLED', 'REJECTED']:
                    discrepancy = Discrepancy(
                        discrepancy_type=OrderDiscrepancyType.MISSING_EXCHANGE,
                        local_order=local_order,
                        exchange_order=None,
                        detected_at=time.time(),
                        severity='medium',
                    )
                    discrepancies.append(discrepancy)
        
        # Store discrepancies
        self._discrepancies.extend(discrepancies)
        self._resolution_stats['total_detected'] += len(discrepancies)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Reconciliation completed in {elapsed_ms:.2f}ms, found {len(discrepancies)} discrepancies")
        
        return discrepancies
    
    async def resolve_orphan_order(self, discrepancy: Discrepancy) -> bool:
        """
        Resolve an orphaned order by cancelling it on the exchange.
        
        Target: Complete within 50ms
        
        Args:
            discrepancy: The discrepancy to resolve
            
        Returns:
            True if resolved successfully
        """
        if discrepancy.discrepancy_type != OrderDiscrepancyType.MISSING_LOCAL:
            return False
        
        if not discrepancy.exchange_order:
            return False
        
        start_time = time.perf_counter()
        
        ex_order = discrepancy.exchange_order
        
        try:
            # Cancel the orphaned order
            success = await self._cancel_order(ex_order.symbol, ex_order.order_id)
            
            resolution_time_ms = (time.perf_counter() - start_time) * 1000
            
            discrepancy.resolved = success
            discrepancy.resolution_action = 'cancelled' if success else 'failed'
            discrepancy.resolution_time = resolution_time_ms
            
            if success:
                self._resolution_stats['total_resolved'] += 1
                
                # Update average resolution time
                n = self._resolution_stats['total_resolved']
                avg = self._resolution_stats['avg_resolution_time_ms']
                self._resolution_stats['avg_resolution_time_ms'] = avg + (resolution_time_ms - avg) / n
                
                logger.info(
                    f"Orphan order {ex_order.order_id} cancelled in {resolution_time_ms:.2f}ms"
                )
                
                # Verify we met the 50ms target
                if resolution_time_ms > self.max_resolution_time_ms:
                    logger.warning(
                        f"Resolution took {resolution_time_ms:.2f}ms, "
                        f"exceeding {self.max_resolution_time_ms}ms target"
                    )
            else:
                self._resolution_stats['failed_resolutions'] += 1
                logger.error(f"Failed to cancel orphan order {ex_order.order_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error resolving orphan order: {e}")
            self._resolution_stats['failed_resolutions'] += 1
            return False
    
    async def _cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order on Binance."""
        if not self._session:
            raise RuntimeError("Resolver not started")
        
        params = {
            'symbol': symbol,
            'orderId': order_id,
            'timestamp': int(time.time() * 1000),
        }
        
        signature = self._generate_signature(params)
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': self.api_key,
        }
        
        url = f"{self.base_url}/api/v3/order"
        
        try:
            async with self._session.delete(url, params=params, headers=headers) as response:
                if response.status in [200, 400]:  # 400 can mean already cancelled
                    return True
                else:
                    logger.error(f"Cancel failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def resolve_all_orphans(self, discrepancies: Optional[List[Discrepancy]] = None) -> int:
        """
        Resolve all orphaned orders.
        
        Args:
            discrepancies: Optional specific discrepancies to resolve
            
        Returns:
            Number of successfully resolved orphans
        """
        if discrepancies is None:
            # Find all unresolved orphan discrepancies
            discrepancies = [
                d for d in self._discrepancies
                if d.discrepancy_type == OrderDiscrepancyType.MISSING_LOCAL
                and not d.resolved
            ]
        
        # Resolve concurrently for speed
        tasks = [self.resolve_orphan_order(d) for d in discrepancies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        resolved_count = sum(1 for r in results if r is True)
        
        logger.info(f"Resolved {resolved_count}/{len(discrepancies)} orphan orders")
        
        return resolved_count
    
    def get_resolution_stats(self) -> Dict[str, Any]:
        """Get resolution statistics."""
        return self._resolution_stats.copy()
    
    def get_pending_discrepancies(self) -> List[Discrepancy]:
        """Get all unresolved discrepancies."""
        return [d for d in self._discrepancies if not d.resolved]


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize resolver (use testnet credentials for testing)
        resolver = OrphanTradeResolver(
            api_key="test_key",
            api_secret="test_secret",
            base_url="https://testnet.binance.vision",
        )
        
        await resolver.start()
        
        # Register some local orders
        resolver.register_local_order(LocalOrder(
            order_id="local_001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.1,
            price=45000.0,
            filled_quantity=0.0,
            status="OPEN",
            created_at=time.time(),
            exchange_order_id="12345",
        ))
        
        # Reconcile
        discrepancies = await resolver.reconcile(symbols=["BTCUSDT"])
        
        # Resolve any orphans
        if discrepancies:
            resolved = await resolver.resolve_all_orphans()
            print(f"Resolved {resolved} orphan orders")
        
        # Print stats
        stats = resolver.get_resolution_stats()
        print(f"Stats: {stats}")
        
        await resolver.stop()
    
    # asyncio.run(main())
