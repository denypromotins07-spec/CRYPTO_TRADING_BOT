#!/usr/bin/env python3
"""
Bellman-Ford Solver - Detecting Negative Weight Cycles for Instant Arbitrage

This module implements the Bellman-Ford algorithm to detect negative weight
cycles in the triangular arbitrage graph, which represent profitable
arbitrage opportunities. Optimized for real-time execution on every tick.

Chapter 2: Triangular Arbitrage and Cross-Margin Efficiency Optimization
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import defaultdict
import time
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """Represents a directed edge in the arbitrage graph"""
    from_symbol: str
    to_symbol: str
    rate: Decimal
    fee_bps: Decimal
    log_weight: float  # -ln(rate_after_fee)
    
    @classmethod
    def create(cls, from_symbol: str, to_symbol: str, rate: Decimal, fee_bps: Decimal) -> 'Edge':
        """Create edge with calculated log weight"""
        rate_after_fee = rate * (Decimal('1') - fee_bps / Decimal('10000'))
        log_weight = -math.log(float(rate_after_fee))
        return cls(
            from_symbol=from_symbol,
            to_symbol=to_symbol,
            rate=rate,
            fee_bps=fee_bps,
            log_weight=log_weight
        )


@dataclass
class ArbCycle:
    """Represents a detected arbitrage cycle"""
    cycle_symbols: List[str]
    total_log_weight: float
    profit_bps: Decimal
    edges: List[Edge]
    detection_time_ns: int
    confidence_score: float


class BellmanFordSolver:
    """
    High-performance Bellman-Ford solver for arbitrage detection.
    
    Detects negative weight cycles in the currency exchange graph,
    where negative cycles represent profitable arbitrage opportunities.
    
    The algorithm transforms exchange rates to logarithmic weights:
    weight = -ln(rate)
    
    A negative cycle in this transformed graph means:
    sum(-ln(rate_i)) < 0
    => -sum(ln(rate_i)) < 0
    => sum(ln(rate_i)) > 0
    => ln(product(rate_i)) > 0
    => product(rate_i) > 1
    
    This indicates a profitable cycle!
    """
    
    # Minimum profit threshold in bps to report
    MIN_PROFIT_BPS = Decimal('1.0')
    
    # Maximum iterations before timeout
    MAX_ITERATIONS = 100
    
    def __init__(self):
        """Initialize the Bellman-Ford solver"""
        # Graph represented as adjacency list
        self.edges: List[Edge] = []
        self.symbols: Set[str] = set()
        self.symbol_to_idx: Dict[str, int] = {}
        self.idx_to_symbol: Dict[int, str] = {}
        
        # Distance and predecessor arrays
        self._dist: List[float] = []
        self._predecessor: List[int] = []
        
        # Performance metrics
        self.detection_count = 0
        self.last_detection_time_ns = 0
        self.total_scans = 0
        
        logger.info("BellmanFordSolver initialized")
    
    def update_graph(self, edges: List[Tuple[str, str, Decimal, Decimal]]) -> None:
        """
        Update the graph with new edge data.
        
        Args:
            edges: List of (from_symbol, to_symbol, rate, fee_bps) tuples
        """
        self.edges = []
        self.symbols = set()
        
        for from_sym, to_sym, rate, fee_bps in edges:
            self.symbols.add(from_sym)
            self.symbols.add(to_sym)
            
            edge = Edge.create(from_sym, to_sym, rate, fee_bps)
            self.edges.append(edge)
        
        # Build symbol index mapping
        self.symbol_to_idx = {sym: idx for idx, sym in enumerate(sorted(self.symbols))}
        self.idx_to_symbol = {idx: sym for sym, idx in self.symbol_to_idx.items()}
        
        # Initialize arrays
        n = len(self.symbols)
        self._dist = [float('inf')] * n
        self._predecessor = [-1] * n
    
    def _relax_edge(self, u: int, v: int, weight: float) -> bool:
        """
        Relax an edge (u, v) with given weight.
        
        Returns True if relaxation occurred.
        """
        if self._dist[v] > self._dist[u] + weight + 1e-10:  # Small epsilon for float precision
            self._dist[v] = self._dist[u] + weight
            self._predecessor[v] = u
            return True
        return False
    
    def find_arbitrage_cycles(self, start_symbol: Optional[str] = None) -> List[ArbCycle]:
        """
        Find all arbitrage cycles using Bellman-Ford algorithm.
        
        Args:
            start_symbol: Optional starting symbol (uses first symbol if None)
            
        Returns:
            List of detected ArbCycle objects
        """
        self.total_scans += 1
        start_time_ns = time.time_ns()
        
        if not self.symbols or not self.edges:
            return []
        
        n = len(self.symbols)
        
        # Determine start vertex
        if start_symbol and start_symbol in self.symbol_to_idx:
            start_idx = self.symbol_to_idx[start_symbol]
        else:
            start_idx = 0
        
        # Initialize distances
        self._dist = [float('inf')] * n
        self._dist[start_idx] = 0.0
        self._predecessor = [-1] * n
        
        # Run Bellman-Ford: n-1 iterations
        for i in range(n - 1):
            updated = False
            
            for edge in self.edges:
                u = self.symbol_to_idx[edge.from_symbol]
                v = self.symbol_to_idx[edge.to_symbol]
                
                if self._relax_edge(u, v, edge.log_weight):
                    updated = True
            
            # Early termination if no updates
            if not updated:
                break
        
        # Check for negative cycles by trying to relax one more time
        cycles = []
        vertices_in_cycle: Set[int] = set()
        
        for edge in self.edges:
            u = self.symbol_to_idx[edge.from_symbol]
            v = self.symbol_to_idx[edge.to_symbol]
            
            if self._dist[v] > self._dist[u] + edge.log_weight + 1e-10:
                # Found a vertex that can still be relaxed - part of negative cycle
                vertices_in_cycle.add(v)
        
        # Extract cycles from vertices that can still be relaxed
        for vertex in vertices_in_cycle:
            cycle = self._extract_cycle(vertex, n)
            if cycle:
                cycles.append(cycle)
        
        # Deduplicate cycles (same cycle may be detected from different vertices)
        unique_cycles = self._deduplicate_cycles(cycles)
        
        # Filter by minimum profit threshold
        profitable_cycles = [
            c for c in unique_cycles 
            if c.profit_bps >= self.MIN_PROFIT_BPS
        ]
        
        if profitable_cycles:
            self.detection_count += len(profitable_cycles)
            self.last_detection_time_ns = time.time_ns()
            logger.info(f"Found {len(profitable_cycles)} arbitrage cycles")
        
        return profitable_cycles
    
    def _extract_cycle(self, start_vertex: int, n: int) -> Optional[ArbCycle]:
        """
        Extract a cycle from a vertex that's part of a negative cycle.
        
        Uses the predecessor array to trace back and find the cycle.
        """
        # First, go back n steps to ensure we're in the cycle
        current = start_vertex
        for _ in range(n):
            if self._predecessor[current] == -1:
                return None
            current = self._predecessor[current]
        
        # Now trace the cycle
        cycle_vertices = []
        visited = set()
        
        while current not in visited:
            visited.add(current)
            cycle_vertices.append(current)
            
            if self._predecessor[current] == -1:
                return None
            
            current = self._predecessor[current]
        
        # Find where the cycle starts
        cycle_start = current
        cycle_idx = cycle_vertices.index(cycle_start)
        cycle_vertices = cycle_vertices[cycle_idx:]
        
        # Convert to symbols
        cycle_symbols = [self.idx_to_symbol[v] for v in cycle_vertices]
        
        # Add the starting symbol at the end to complete the cycle
        cycle_symbols.append(cycle_symbols[0])
        
        # Calculate total log weight and profit
        total_log_weight = 0.0
        edges_in_cycle = []
        
        for i in range(len(cycle_symbols) - 1):
            from_sym = cycle_symbols[i]
            to_sym = cycle_symbols[i + 1]
            
            # Find the edge
            for edge in self.edges:
                if edge.from_symbol == from_sym and edge.to_symbol == to_sym:
                    edges_in_cycle.append(edge)
                    total_log_weight += edge.log_weight
                    break
        
        if not edges_in_cycle:
            return None
        
        # Profit calculation: exp(-total_log_weight) - 1
        profit_factor = math.exp(-total_log_weight)
        profit_bps = Decimal(str((profit_factor - 1) * 10000))
        
        # Confidence score based on profit magnitude and cycle length
        confidence = min(1.0, float(profit_bps) / 100.0) * (3.0 / len(cycle_symbols))
        
        return ArbCycle(
            cycle_symbols=cycle_symbols,
            total_log_weight=total_log_weight,
            profit_bps=profit_bps.quantize(Decimal('0.01'), rounding=ROUND_DOWN),
            edges=edges_in_cycle,
            detection_time_ns=time.time_ns(),
            confidence_score=confidence
        )
    
    def _deduplicate_cycles(self, cycles: List[ArbCycle]) -> List[ArbCycle]:
        """Remove duplicate cycles (same symbols, different starting points)"""
        seen: Set[frozenset] = set()
        unique = []
        
        for cycle in cycles:
            # Create a canonical representation of the cycle
            cycle_set = frozenset(cycle.cycle_symbols[:-1])  # Exclude repeated start
            
            if cycle_set not in seen:
                seen.add(cycle_set)
                unique.append(cycle)
        
        return unique
    
    def find_best_arbitrage(self) -> Optional[ArbCycle]:
        """Find the single best arbitrage opportunity"""
        cycles = self.find_arbitrage_cycles()
        
        if not cycles:
            return None
        
        # Return cycle with highest profit
        return max(cycles, key=lambda c: c.profit_bps)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get solver performance metrics"""
        return {
            'symbol_count': len(self.symbols),
            'edge_count': len(self.edges),
            'detection_count': self.detection_count,
            'total_scans': self.total_scans,
            'detection_rate': self.detection_count / max(1, self.total_scans),
            'last_detection_time_ns': self.last_detection_time_ns
        }


class RealTimeArbDetector:
    """
    Real-time arbitrage detector that processes ticks and triggers alerts.
    
    Wraps BellmanFordSolver with async capabilities for production use.
    """
    
    def __init__(self, min_profit_bps: Decimal = Decimal('5.0')):
        """Initialize real-time detector"""
        self.solver = BellmanFordSolver()
        self.min_profit_bps = min_profit_bps
        self.solver.MIN_PROFIT_BPS = min_profit_bps
        
        # Callbacks for arb notifications
        self._arb_callbacks: List[callable] = []
        
        # Running flag
        self._running = False
        
        logger.info(f"RealTimeArbDetector initialized (min_profit={min_profit_bps} bps)")
    
    def register_callback(self, callback: callable) -> None:
        """Register a callback for arbitrage notifications"""
        self._arb_callbacks.append(callback)
    
    def update_rates(self, rates: Dict[Tuple[str, str], Tuple[Decimal, Decimal]]) -> None:
        """
        Update exchange rates.
        
        Args:
            rates: Dict mapping (from, to) -> (rate, fee_bps)
        """
        edges = [
            (from_sym, to_sym, rate, fee_bps)
            for (from_sym, to_sym), (rate, fee_bps) in rates.items()
        ]
        self.solver.update_graph(edges)
    
    async def scan_for_arbitrage(self) -> List[ArbCycle]:
        """Scan for arbitrage opportunities"""
        cycles = self.solver.find_arbitrage_cycles()
        
        # Notify callbacks
        for cycle in cycles:
            for callback in self._arb_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(cycle)
                    else:
                        callback(cycle)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        return cycles
    
    async def run_continuous(self, scan_interval_ms: int = 10) -> None:
        """
        Run continuous arbitrage scanning.
        
        Args:
            scan_interval_ms: Time between scans in milliseconds
        """
        self._running = True
        interval_seconds = scan_interval_ms / 1000.0
        
        logger.info(f"Starting continuous scan (interval={scan_interval_ms}ms)")
        
        while self._running:
            try:
                await self.scan_for_arbitrage()
            except Exception as e:
                logger.error(f"Scan error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def stop(self) -> None:
        """Stop continuous scanning"""
        self._running = False
        logger.info("Stopped continuous scanning")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics"""
        solver_metrics = self.solver.get_metrics()
        solver_metrics['min_profit_bps'] = float(self.min_profit_bps)
        solver_metrics['running'] = self._running
        return solver_metrics


# Example usage
if __name__ == "__main__":
    async def test_detector():
        detector = RealTimeArbDetector(min_profit_bps=Decimal('1.0'))
        
        # Define callback
        def on_arb_detected(cycle: ArbCycle):
            print(f"\n🎯 ARBITRAGE DETECTED!")
            print(f"   Cycle: {' -> '.join(cycle.cycle_symbols)}")
            print(f"   Profit: {cycle.profit_bps} bps")
            print(f"   Confidence: {cycle.confidence_score:.2f}")
        
        detector.register_callback(on_arb_detected)
        
        # Set up test rates that create a profitable triangle
        # BTC -> ETH -> USDT -> BTC
        # Using rates that should create ~10 bps profit after fees
        
        rates = {
            ('BTC', 'ETH'): (Decimal('16.67'), Decimal('10')),
            ('ETH', 'BTC'): (Decimal('0.05998'), Decimal('10')),
            ('BTC', 'USDT'): (Decimal('50000'), Decimal('10')),
            ('USDT', 'BTC'): (Decimal('0.00002'), Decimal('10')),
            ('ETH', 'USDT'): (Decimal('3000'), Decimal('10')),
            ('USDT', 'ETH'): (Decimal('0.000333'), Decimal('10')),
        }
        
        detector.update_rates(rates)
        
        # Scan for arbitrage
        print("Scanning for arbitrage opportunities...")
        cycles = await detector.scan_for_arbitrage()
        
        if cycles:
            print(f"\nFound {len(cycles)} profitable cycles:")
            for cycle in cycles:
                print(f"  {' -> '.join(cycle.cycle_symbols)}: {cycle.profit_bps} bps")
        else:
            print("No profitable arbitrage found")
        
        print(f"\nMetrics: {detector.get_metrics()}")
    
    asyncio.run(test_detector())
