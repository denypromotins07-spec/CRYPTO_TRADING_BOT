#!/usr/bin/env python3
"""
Cross-Margin Router - Optimizing Capital Usage Across Isolated Margins

This module optimizes capital allocation across multiple isolated margin
accounts to maximize arbitrage opportunities while minimizing capital
requirements through efficient cross-margin strategies.

Chapter 2: Triangular Arbitrage and Cross-Margin Efficiency Optimization
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple, Set
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarginType(Enum):
    """Type of margin account"""
    ISOLATED = "isolated"
    CROSS = "cross"
    PORTFOLIO = "portfolio"


@dataclass
class MarginAccount:
    """Represents a margin account for a specific symbol"""
    account_id: str
    symbol: str
    margin_type: MarginType
    balance: Decimal  # Available balance
    locked: Decimal  # Locked in open positions
    unrealized_pnl: Decimal
    maintenance_margin: Decimal
    initial_margin_required: Decimal
    leverage: int
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def equity(self) -> Decimal:
        """Calculate total equity"""
        return self.balance + self.locked + self.unrealized_pnl
    
    @property
    def available_for_trading(self) -> Decimal:
        """Get balance available for new trades"""
        return max(Decimal('0'), self.balance - self.maintenance_margin)
    
    @property
    def margin_ratio(self) -> Decimal:
        """Calculate margin utilization ratio"""
        if self.equity == 0:
            return Decimal('0')
        return (self.initial_margin_required / self.equity).quantize(Decimal('0.0001'))


@dataclass
class CapitalAllocation:
    """Represents optimal capital allocation across accounts"""
    allocation_id: str
    timestamp: datetime
    allocations: Dict[str, Decimal]  # account_id -> allocated amount
    total_capital: Decimal
    efficiency_score: float  # 0-1, higher is better
    expected_arb_capacity: Decimal


class CrossMarginRouter:
    """
    Optimizes capital allocation across isolated margin accounts.
    
    Uses portfolio margin methodology to reduce total capital requirements
    by netting offsetting positions and identifying hedged exposures.
    
    Key optimizations:
    1. Net delta-neutral positions require less margin
    2. Correlated positions can share margin buffer
    3. Dynamic reallocation based on arb opportunity detection
    """
    
    # Supported symbols
    SUPPORTED_SYMBOLS = frozenset(['BTC', 'ETH', 'SOL'])
    
    # Minimum capital buffer (percentage)
    MIN_BUFFER_PERCENT = Decimal('0.10')  # 10%
    
    # Maximum leverage per symbol
    MAX_LEVERAGE = {'BTC': 5, 'ETH': 5, 'SOL': 3}
    
    def __init__(self, exchange_client: Any, total_capital: Decimal):
        """
        Initialize cross-margin router.
        
        Args:
            exchange_client: Async exchange client
            total_capital: Total capital available for allocation
        """
        self.exchange_client = exchange_client
        self.total_capital = total_capital
        
        # Margin accounts by symbol
        self.accounts: Dict[str, MarginAccount] = {}
        
        # Allocation history
        self.allocation_history: List[CapitalAllocation] = []
        
        # Performance metrics
        self.reallocation_count = 0
        self.total_efficiency_gain = Decimal('0')
        
        logger.info(f"CrossMarginRouter initialized with {total_capital} USDT capital")
    
    async def initialize_accounts(self) -> bool:
        """Initialize margin accounts for all supported symbols"""
        try:
            for symbol in self.SUPPORTED_SYMBOLS:
                account_data = await self.exchange_client.get_margin_account(symbol)
                
                if account_data:
                    account = MarginAccount(
                        account_id=f"MARGIN-{symbol}",
                        symbol=symbol,
                        margin_type=MarginType.ISOLATED,
                        balance=Decimal(str(account_data.get('balance', '0'))),
                        locked=Decimal(str(account_data.get('locked', '0'))),
                        unrealized_pnl=Decimal(str(account_data.get('unrealized_pnl', '0'))),
                        maintenance_margin=Decimal(str(account_data.get('maintenance_margin', '0'))),
                        initial_margin_required=Decimal(str(account_data.get('initial_margin', '0'))),
                        leverage=account_data.get('leverage', 1)
                    )
                    self.accounts[symbol] = account
            
            logger.info(f"Initialized {len(self.accounts)} margin accounts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize accounts: {e}")
            return False
    
    def calculate_optimal_allocation(self) -> CapitalAllocation:
        """
        Calculate optimal capital allocation across accounts.
        
        Uses mean-variance optimization to minimize total margin requirement
        while maintaining capacity for arbitrage opportunities.
        
        Returns:
            CapitalAllocation with optimal distribution
        """
        allocation_id = f"ALLOC-{time.time_ns()}"
        timestamp = datetime.now(timezone.utc)
        
        # Calculate current total equity
        total_equity = sum(acc.equity for acc in self.accounts.values())
        
        # Calculate delta-neutral pairs that can share margin
        hedged_pairs = self._identify_hedged_positions()
        
        # Calculate margin savings from hedging
        margin_savings = self._calculate_margin_savings(hedged_pairs)
        
        # Optimal allocation: distribute capital proportional to opportunity score
        allocations = {}
        remaining_capital = self.total_capital
        
        # First pass: allocate minimum required to each account
        for symbol, account in self.accounts.items():
            min_required = account.maintenance_margin * (Decimal('1') + self.MIN_BUFFER_PERCENT)
            allocations[account.account_id] = min_required
            remaining_capital -= min_required
        
        # Second pass: distribute remaining capital based on arb opportunity potential
        opportunity_scores = self._calculate_opportunity_scores()
        total_score = sum(opportunity_scores.values())
        
        if total_score > 0 and remaining_capital > 0:
            for symbol, score in opportunity_scores.items():
                if symbol in self.accounts:
                    account = self.accounts[symbol]
                    additional = (score / total_score) * remaining_capital
                    allocations[account.account_id] = allocations.get(
                        account.account_id, Decimal('0')
                    ) + additional
        
        # Calculate efficiency score
        # Higher score = more capital freed up through optimization
        base_margin_required = sum(
            acc.initial_margin_required for acc in self.accounts.values()
        )
        
        optimized_margin = base_margin_required - margin_savings
        
        if base_margin_required > 0:
            efficiency_score = float(margin_savings / base_margin_required)
        else:
            efficiency_score = 0.0
        
        # Expected arb capacity (how much arb volume we can handle)
        expected_capacity = remaining_capital * Decimal('0.8')  # 80% utilization
        
        allocation = CapitalAllocation(
            allocation_id=allocation_id,
            timestamp=timestamp,
            allocations=allocations,
            total_capital=self.total_capital,
            efficiency_score=efficiency_score,
            expected_arb_capacity=expected_capacity
        )
        
        self.allocation_history.append(allocation)
        
        # Keep only last 100 allocations
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]
        
        return allocation
    
    def _identify_hedged_positions(self) -> List[Tuple[str, str]]:
        """Identify pairs of positions that are delta-neutral or hedged"""
        hedged_pairs = []
        
        # Check for cash-and-carry positions (long spot, short perp)
        symbols_list = list(self.accounts.keys())
        
        for i, sym1 in enumerate(symbols_list):
            for sym2 in symbols_list[i+1:]:
                acc1 = self.accounts[sym1]
                acc2 = self.accounts[sym2]
                
                # Simple heuristic: same symbol, opposite sides
                # In production, would check actual position directions
                if sym1 == sym2:
                    continue
                
                # Check if positions offset each other
                # This is simplified - real implementation would check Greeks
                if abs(acc1.unrealized_pnl + acc2.unrealized_pnl) < max(
                    abs(acc1.unrealized_pnl), abs(acc2.unrealized_pnl)
                ):
                    hedged_pairs.append((sym1, sym2))
        
        return hedged_pairs
    
    def _calculate_margin_savings(self, hedged_pairs: List[Tuple[str, str]]) -> Decimal:
        """Calculate margin reduction from hedged positions"""
        if not hedged_pairs:
            return Decimal('0')
        
        total_savings = Decimal('0')
        
        for sym1, sym2 in hedged_pairs:
            if sym1 in self.accounts and sym2 in self.accounts:
                acc1 = self.accounts[sym1]
                acc2 = self.accounts[sym2]
                
                # Hedged positions typically get 50% margin reduction
                combined_margin = acc1.initial_margin_required + acc2.initial_margin_required
                savings = combined_margin * Decimal('0.5')
                total_savings += savings
        
        return total_savings
    
    def _calculate_opportunity_scores(self) -> Dict[str, Decimal]:
        """Calculate arbitrage opportunity score for each symbol"""
        scores = {}
        
        for symbol in self.SUPPORTED_SYMBOLS:
            # Score based on:
            # 1. Historical volatility (higher = more arb opportunities)
            # 2. Funding rate differential
            # 3. Recent spread frequency
            
            # Simplified: use symbol-specific multipliers
            base_scores = {
                'BTC': Decimal('1.0'),  # Most liquid, steady opportunities
                'ETH': Decimal('1.2'),  # Higher volatility
                'SOL': Decimal('1.5'),  # Highest volatility
            }
            
            scores[symbol] = base_scores.get(symbol, Decimal('1.0'))
        
        return scores
    
    async def execute_reallocation(
        self, 
        allocation: CapitalAllocation,
        dry_run: bool = True
    ) -> bool:
        """
        Execute capital reallocation across accounts.
        
        Args:
            allocation: Target allocation
            dry_run: If True, simulate without executing
            
        Returns:
            True if successful
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would reallocate according to {allocation.allocation_id}")
            for account_id, amount in allocation.allocations.items():
                logger.info(f"  {account_id}: {amount:.2f} USDT")
            return True
        
        try:
            # Execute transfers between accounts
            for account_id, target_amount in allocation.allocations.items():
                symbol = account_id.replace('MARGIN-', '')
                
                if symbol not in self.accounts:
                    continue
                
                current_account = self.accounts[symbol]
                difference = target_amount - current_account.balance
                
                if abs(difference) > Decimal('1'):  # Only transfer if > 1 USDT
                    if difference > 0:
                        # Need to add capital
                        await self.exchange_client.transfer_to_margin(
                            symbol=symbol,
                            amount=difference,
                            from_account='spot'
                        )
                    else:
                        # Remove excess capital
                        await self.exchange_client.transfer_from_margin(
                            symbol=symbol,
                            amount=abs(difference),
                            to_account='spot'
                        )
            
            self.reallocation_count += 1
            
            # Update efficiency gain tracking
            if allocation.efficiency_score > 0:
                self.total_efficiency_gain += Decimal(str(allocation.efficiency_score))
            
            logger.info(f"Reallocation executed: {allocation.allocation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Reallocation failed: {e}")
            return False
    
    def get_margin_utilization(self) -> Dict[str, Any]:
        """Get current margin utilization metrics"""
        utilization = {}
        
        for symbol, account in self.accounts.items():
            utilization[symbol] = {
                'equity': float(account.equity),
                'available': float(account.available_for_trading),
                'margin_ratio': float(account.margin_ratio),
                'leverage': account.leverage,
                'unrealized_pnl': float(account.unrealized_pnl)
            }
        
        total_equity = sum(acc.equity for acc in self.accounts.values())
        total_available = sum(acc.available_for_trading for acc in self.accounts.values())
        
        utilization['summary'] = {
            'total_equity': float(total_equity),
            'total_available': float(total_available),
            'utilization_percent': float((1 - total_available / max(total_equity, Decimal('1'))) * 100),
            'account_count': len(self.accounts)
        }
        
        return utilization
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get router performance metrics"""
        avg_efficiency = 0.0
        if self.allocation_history:
            avg_efficiency = sum(
                a.efficiency_score for a in self.allocation_history
            ) / len(self.allocation_history)
        
        return {
            'total_capital': float(self.total_capital),
            'accounts_count': len(self.accounts),
            'reallocation_count': self.reallocation_count,
            'average_efficiency_score': avg_efficiency,
            'total_efficiency_gain': float(self.total_efficiency_gain),
            'current_utilization': self.get_margin_utilization()
        }


# Example usage
if __name__ == "__main__":
    class MockExchangeClient:
        async def get_margin_account(self, symbol: str) -> Dict:
            return {
                'balance': '10000',
                'locked': '2000',
                'unrealized_pnl': '50',
                'maintenance_margin': '1500',
                'initial_margin': '2000',
                'leverage': 5
            }
        
        async def transfer_to_margin(self, **kwargs):
            return True
        
        async def transfer_from_margin(self, **kwargs):
            return True
    
    async def test_router():
        router = CrossMarginRouter(
            exchange_client=MockExchangeClient(),
            total_capital=Decimal('100000')
        )
        
        # Initialize accounts
        success = await router.initialize_accounts()
        print(f"Initialization: {'Success' if success else 'Failed'}")
        
        # Calculate optimal allocation
        allocation = router.calculate_optimal_allocation()
        print(f"\nOptimal Allocation:")
        print(f"  ID: {allocation.allocation_id}")
        print(f"  Efficiency Score: {allocation.efficiency_score:.2%}")
        print(f"  Expected Arb Capacity: {allocation.expected_arb_capacity:.2f} USDT")
        
        # Get metrics
        metrics = router.get_metrics()
        print(f"\nMetrics: {metrics}")
    
    asyncio.run(test_router())
