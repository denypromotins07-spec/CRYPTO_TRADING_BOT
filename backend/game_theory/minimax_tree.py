#!/usr/bin/env python3
"""
Minimax Tree with Alpha-Beta Pruning for Adversarial Market Analysis

This module implements a minimax decision tree with alpha-beta pruning
to evaluate adversarial market moves and optimize trading decisions
under worst-case scenarios.

Features:
- Depth-limited search to prevent CPU starvation
- Alpha-beta pruning for efficient exploration
- Transposition table for memoization
- Market-specific evaluation functions
- Strict memory bounds (<8GB RAM constraint)

Integrates quantitative finance domains:
- Game theory and adversarial modeling
- Decision theory under uncertainty
- Risk management frameworks
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
import numpy as np
from collections import OrderedDict


class PlayerType(Enum):
    """Player types in the adversarial game."""
    MAXIMIZER = auto()  # Our bot
    MINIMIZER = auto()  # Adversarial market/HFT


class MarketAction(Enum):
    """Available market actions."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    HEDGE = auto()
    SCALE_IN = auto()
    SCALE_OUT = auto()


@dataclass
class GameState:
    """Represents the current state of the market game."""
    position_size: float
    unrealized_pnl: float
    market_trend: float  # -1 to 1
    volatility: float
    liquidity_score: float  # 0 to 1
    opponent_pressure: float  # -1 to 1
    timestamp: int
    depth: int = 0
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return GameState(
            position_size=self.position_size,
            unrealized_pnl=self.unrealized_pnl,
            market_trend=self.market_trend,
            volatility=self.volatility,
            liquidity_score=self.liquidity_score,
            opponent_pressure=self.opponent_pressure,
            timestamp=self.timestamp,
            depth=self.depth
        )
    
    def is_terminal(self, max_depth: int) -> bool:
        """Check if state is terminal (depth limit or extreme conditions)."""
        if self.depth >= max_depth:
            return True
        # Terminal if extreme PnL (crash or windfall)
        if abs(self.unrealized_pnl) > 0.5:  # 50% move
            return True
        return False


@dataclass
class Move:
    """Represents a possible move in the game tree."""
    action: MarketAction
    size: float
    expected_impact: float
    confidence: float


@dataclass
class SearchResult:
    """Result from minimax search."""
    best_move: Optional[Move]
    value: float
    nodes_evaluated: int
    depth_reached: int
    pruning_count: int


class TranspositionTable:
    """Memoization table for game states using LRU cache."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.table: OrderedDict[str, Tuple[float, int, Optional[Move]]] = OrderedDict()
    
    def _hash_state(self, state: GameState) -> str:
        """Create hash key for game state."""
        return f"{state.position_size:.4f}_{state.unrealized_pnl:.4f}_{state.market_trend:.4f}_{state.volatility:.4f}"
    
    def get(self, state: GameState) -> Optional[Tuple[float, int, Optional[Move]]]:
        """Retrieve cached result for state."""
        key = self._hash_state(state)
        if key in self.table:
            # Move to end (most recently used)
            self.table.move_to_end(key)
            return self.table[key]
        return None
    
    def store(self, state: GameState, value: float, depth: int, move: Optional[Move]) -> None:
        """Store result in cache."""
        key = self._hash_state(state)
        
        # Evict oldest if at capacity
        while len(self.table) >= self.max_size:
            self.table.popitem(last=False)
        
        self.table[key] = (value, depth, move)
    
    def clear(self) -> None:
        """Clear the transposition table."""
        self.table.clear()


class MinimaxTree:
    """
    Minimax decision tree with alpha-beta pruning for adversarial market analysis.
    
    This class evaluates potential market moves considering worst-case
    adversarial responses from HFT algorithms or market makers.
    
    Memory footprint: ~5MB for default configuration
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        branching_factor: int = 5,
        eval_cache_size: int = 10000
    ):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.transposition_table = TranspositionTable(eval_cache_size)
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        # Evaluation function weights (can be learned)
        self.weights = {
            'pnl': 1.0,
            'trend': 0.5,
            'volatility': -0.3,
            'liquidity': 0.2,
            'pressure': -0.4,
        }
    
    def search(self, state: GameState, time_limit_ms: int = 100) -> SearchResult:
        """
        Perform minimax search with alpha-beta pruning.
        
        Args:
            state: Current game state
            time_limit_ms: Maximum search time in milliseconds
            
        Returns:
            SearchResult with best move and statistics
        """
        import time
        start_time = time.time()
        self.nodes_evaluated = 0
        self.pruning_count = 0
        
        alpha = float('-inf')
        beta = float('inf')
        
        best_value, best_move = self._maximize(state, alpha, beta, start_time, time_limit_ms)
        
        return SearchResult(
            best_move=best_move,
            value=best_value,
            nodes_evaluated=self.nodes_evaluated,
            depth_reached=min(self.max_depth, state.depth + self._get_actual_depth()),
            pruning_count=self.pruning_count
        )
    
    def _maximize(
        self,
        state: GameState,
        alpha: float,
        beta: float,
        start_time: float,
        time_limit_ms: int
    ) -> Tuple[float, Optional[Move]]:
        """Maximizing player (our bot) turn."""
        self.nodes_evaluated += 1
        
        # Check time limit periodically
        if self.nodes_evaluated % 100 == 0:
            if (time.time() - start_time) * 1000 > time_limit_ms:
                return self._evaluate(state), None
        
        # Check terminal state
        if state.is_terminal(self.max_depth):
            return self._evaluate(state), None
        
        # Check transposition table
        cached = self.transposition_table.get(state)
        if cached is not None and cached[1] >= state.depth:
            return cached[0], cached[2]
        
        best_value = float('-inf')
        best_move: Optional[Move] = None
        
        # Generate possible moves
        moves = self._generate_moves(state)
        
        for move in moves:
            next_state = self._apply_move(state, move, PlayerType.MAXIMIZER)
            value, _ = self._minimize(next_state, alpha, beta, start_time, time_limit_ms)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, best_value)
            
            # Beta cutoff (pruning)
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        # Store in transposition table
        self.transposition_table.store(state, best_value, state.depth, best_move)
        
        return best_value, best_move
    
    def _minimize(
        self,
        state: GameState,
        alpha: float,
        beta: float,
        start_time: float,
        time_limit_ms: int
    ) -> Tuple[float, Optional[Move]]:
        """Minimizing player (adversary) turn."""
        self.nodes_evaluated += 1
        
        # Check time limit
        if self.nodes_evaluated % 100 == 0:
            if (time.time() - start_time) * 1000 > time_limit_ms:
                return self._evaluate(state), None
        
        # Check terminal state
        if state.is_terminal(self.max_depth):
            return self._evaluate(state), None
        
        # Check transposition table
        cached = self.transposition_table.get(state)
        if cached is not None and cached[1] >= state.depth:
            return cached[0], cached[2]
        
        best_value = float('inf')
        best_move: Optional[Move] = None
        
        # Generate adversarial moves
        adv_moves = self._generate_adversarial_moves(state)
        
        for move in adv_moves:
            next_state = self._apply_move(state, move, PlayerType.MINIMIZER)
            value, _ = self._maximize(next_state, alpha, beta, start_time, time_limit_ms)
            
            if value < best_value:
                best_value = value
                best_move = move
            
            beta = min(beta, best_value)
            
            # Alpha cutoff (pruning)
            if beta <= alpha:
                self.pruning_count += 1
                break
        
        # Store in transposition table
        self.transposition_table.store(state, best_value, state.depth, best_move)
        
        return best_value, best_move
    
    def _generate_moves(self, state: GameState) -> List[Move]:
        """Generate legal moves for our bot."""
        moves = []
        
        # Base move sizes relative to position
        base_size = abs(state.position_size) * 0.25 if state.position_size != 0 else 1000.0
        
        # Generate moves based on market conditions
        if state.market_trend > 0.3:
            # Bullish: consider buying or scaling in
            moves.append(Move(MarketAction.BUY, base_size, 0.02, 0.7))
            moves.append(Move(MarketAction.SCALE_IN, base_size * 0.5, 0.015, 0.6))
        elif state.market_trend < -0.3:
            # Bearish: consider selling or scaling out
            moves.append(Move(MarketAction.SELL, base_size, -0.02, 0.7))
            moves.append(Move(MarketAction.SCALE_OUT, base_size * 0.5, -0.015, 0.6))
        
        # Always include hold and hedge options
        moves.append(Move(MarketAction.HOLD, 0.0, 0.0, 0.5))
        
        if abs(state.position_size) > 0:
            moves.append(Move(MarketAction.HEDGE, abs(state.position_size) * 0.5, 0.0, 0.8))
        
        # Limit branching factor
        return moves[:self.branching_factor]
    
    def _generate_adversarial_moves(self, state: GameState) -> List[Move]:
        """Generate adversarial moves (market/HFT responses)."""
        moves = []
        
        # Adversary tries to maximize our losses
        base_impact = state.volatility * 0.5
        
        if state.opponent_pressure > 0.5:
            # Strong selling pressure from adversary
            moves.append(Move(MarketAction.SELL, 0.0, -base_impact, 0.8))
        elif state.opponent_pressure < -0.5:
            # Strong buying pressure (trap)
            moves.append(Move(MarketAction.BUY, 0.0, base_impact * 0.5, 0.6))
        
        # Liquidity drain scenario
        if state.liquidity_score < 0.3:
            moves.append(Move(MarketAction.HOLD, 0.0, -base_impact * 0.3, 0.9))
        
        # Normal market movement
        moves.append(Move(MarketAction.HOLD, 0.0, 0.0, 0.5))
        
        return moves[:self.branching_factor]
    
    def _apply_move(self, state: GameState, move: Move, player: PlayerType) -> GameState:
        """Apply a move and return new state."""
        new_state = state.copy()
        new_state.depth = state.depth + 1
        
        if player == PlayerType.MAXIMIZER:
            # Our move affects position and PnL
            if move.action == MarketAction.BUY:
                new_state.position_size += move.size
                new_state.unrealized_pnl -= move.size * move.expected_impact * 0.5  # Slippage
            elif move.action == MarketAction.SELL:
                new_state.position_size -= move.size
                new_state.unrealized_pnl += move.size * move.expected_impact * 0.5
            elif move.action == MarketAction.HEDGE:
                # Hedge reduces effective exposure
                new_state.unrealized_pnl *= (1.0 - move.size / max(abs(state.position_size), 1))
        else:
            # Adversary move affects market conditions
            new_state.market_trend += move.expected_impact * 0.1
            new_state.opponent_pressure += move.expected_impact * 0.2
            new_state.unrealized_pnl += move.expected_impact * abs(state.position_size)
        
        # Clamp values to valid ranges
        new_state.market_trend = max(-1.0, min(1.0, new_state.market_trend))
        new_state.opponent_pressure = max(-1.0, min(1.0, new_state.opponent_pressure))
        
        return new_state
    
    def _evaluate(self, state: GameState) -> float:
        """Evaluate terminal or leaf state."""
        score = 0.0
        
        # PnL contribution (most important)
        score += self.weights['pnl'] * state.unrealized_pnl
        
        # Trend alignment
        trend_alignment = np.sign(state.position_size) * state.market_trend
        score += self.weights['trend'] * trend_alignment
        
        # Volatility penalty (risk)
        score += self.weights['volatility'] * state.volatility
        
        # Liquidity bonus
        score += self.weights['liquidity'] * state.liquidity_score
        
        # Opponent pressure penalty
        pressure_risk = np.sign(state.position_size) * state.opponent_pressure
        score += self.weights['pressure'] * pressure_risk
        
        # Depth penalty (prefer quicker resolutions)
        score -= 0.01 * state.depth
        
        return score
    
    def _get_actual_depth(self) -> int:
        """Get actual depth reached in last search."""
        return self.max_depth  # Simplified
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update evaluation function weights."""
        self.weights.update(weights)
    
    def clear_cache(self) -> None:
        """Clear transposition table."""
        self.transposition_table.clear()


def evaluate_market_scenario(
    position_size: float,
    pnl: float,
    trend: float,
    volatility: float,
    max_search_time_ms: int = 50
) -> SearchResult:
    """
    Convenience function to evaluate current market scenario.
    
    Returns minimax search result with recommended action.
    """
    state = GameState(
        position_size=position_size,
        unrealized_pnl=pnl,
        market_trend=trend,
        volatility=volatility,
        liquidity_score=0.7,
        opponent_pressure=0.0,
        timestamp=0
    )
    
    tree = MinimaxTree(max_depth=6, branching_factor=5)
    return tree.search(state, time_limit_ms=max_search_time_ms)


if __name__ == "__main__":
    # Example usage
    state = GameState(
        position_size=1000.0,
        unrealized_pnl=0.02,
        market_trend=0.3,
        volatility=0.15,
        liquidity_score=0.8,
        opponent_pressure=-0.2,
        timestamp=0
    )
    
    tree = MinimaxTree(max_depth=6)
    result = tree.search(state, time_limit_ms=50)
    
    print(f"Nodes evaluated: {result.nodes_evaluated}")
    print(f"Pruning count: {result.pruning_count}")
    print(f"Best value: {result.value:.4f}")
    
    if result.best_move:
        print(f"Recommended action: {result.best_move.action.name}")
        print(f"Size: {result.best_move.size:.2f}")
        print(f"Confidence: {result.best_move.confidence:.2f}")
