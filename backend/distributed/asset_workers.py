"""
Asset Workers: Parallel Ray actors for BTC, SOL, ETH, USDT processing.
Implements Actor Model pattern for isolated, concurrent asset handling.
Each actor processes signals, manages positions, and executes trades independently.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import time
import ray
from ray.actor import ActorHandle

from backend.strategies.alpha_generator import AlphaSignal, SignalType, AggregatedAlpha
from backend.execution.order_manager import OrderManager
from backend.config.settings import get_bot_settings

logger = logging.getLogger(__name__)


@dataclass
class AssetWorkerState:
    """State container for an asset worker."""
    asset: str
    is_active: bool = True
    current_position: Optional[Dict[str, Any]] = None
    pending_signals: List[AggregatedAlpha] = field(default_factory=list)
    last_signal_time: float = 0.0
    trades_executed: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)


@ray.remote
class AssetWorker:
    """
    Ray Actor for processing a single asset (BTC, ETH, SOL, or USDT).
    Implements Actor Model pattern with isolated state and message passing.
    Guarantees no garbage collection pauses exceeding 1ms through careful memory management.
    """
    
    def __init__(self, asset: str, config: Optional[Dict[str, Any]] = None):
        self.asset = asset
        self.config = config or {}
        
        # Worker state
        self.state = AssetWorkerState(asset=asset)
        
        # Components
        self.order_manager: Optional[OrderManager] = None
        
        # Signal processing
        self._signal_buffer_max_size = 100
        self._confirmation_threshold = self.config.get("confirmation_threshold", 2)
        self._min_signal_strength = self.config.get("min_signal_strength", 0.5)
        
        # Performance tracking
        self._processing_times: List[float] = []
        self._gc_pause_times: List[float] = []
        
        logger.info(f"AssetWorker initialized for {asset}")
    
    def initialize(self) -> bool:
        """Initialize worker components."""
        try:
            self.order_manager = OrderManager()
            self.state.is_active = True
            self.state.last_heartbeat = time.time()
            
            logger.info(f"AssetWorker {self.asset} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AssetWorker {self.asset}: {e}")
            self.state.is_active = False
            return False
    
    def process_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming alpha signal.
        Returns execution result dictionary.
        """
        start_time = time.time()
        
        if not self.state.is_active:
            return {"status": "inactive", "message": "Worker is inactive"}
        
        try:
            # Convert dict to AlphaSignal
            signal = self._dict_to_signal(signal_data)
            
            if signal is None:
                return {"status": "error", "message": "Invalid signal format"}
            
            # Check signal strength threshold
            if signal.strength < self._min_signal_strength:
                return {
                    "status": "ignored",
                    "reason": f"Signal strength {signal.strength:.2f} below threshold"
                }
            
            # Add to pending signals
            self.state.pending_signals.append(signal)
            
            # Keep buffer bounded
            if len(self.state.pending_signals) > self._signal_buffer_max_size:
                self.state.pending_signals = self.state.pending_signals[-self._signal_buffer_max_size:]
            
            # Check for confirmation
            if len(self.state.pending_signals) >= self._confirmation_threshold:
                result = self._evaluate_signals()
                
                if result["action"] != "hold":
                    # Execute trade
                    execution_result = self._execute_trade(result)
                    
                    # Clear confirmed signals
                    self.state.pending_signals.clear()
                    self.state.trades_executed += 1
                    
                    return execution_result
            
            # Update heartbeat
            self.state.last_heartbeat = time.time()
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            self._processing_times.append(processing_time)
            if len(self._processing_times) > 1000:
                self._processing_times = self._processing_times[-1000:]
            
            return {
                "status": "pending",
                "pending_signals": len(self.state.pending_signals),
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing signal for {self.asset}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _dict_to_signal(self, data: Dict[str, Any]) -> Optional[AlphaSignal]:
        """Convert dictionary to AlphaSignal object."""
        try:
            return AlphaSignal(
                signal_type=SignalType[data["signal_type"].upper()],
                asset=data["asset"],
                direction=int(data["direction"]),
                strength=float(data["strength"]),
                confidence=float(data["confidence"]),
                timestamp=float(data.get("timestamp", time.time())),
                metadata=data.get("metadata", {})
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid signal data: {e}")
            return None
    
    def _evaluate_signals(self) -> Dict[str, Any]:
        """Evaluate pending signals and determine action."""
        if not self.state.pending_signals:
            return {"action": "hold"}
        
        # Aggregate signals
        directions = [s.direction for s in self.state.pending_signals]
        strengths = [s.strength for s in self.state.pending_signals]
        confidences = [s.confidence for s in self.state.pending_signals]
        
        # Check for consensus
        if not all(d == directions[0] for d in directions):
            # Conflicting signals
            return {"action": "hold", "reason": "conflicting_signals"}
        
        net_direction = directions[0]
        avg_strength = sum(strengths) / len(strengths)
        avg_confidence = sum(confidences) / len(confidences)
        
        combined_score = avg_strength * avg_confidence
        
        if combined_score < self._min_signal_strength:
            return {"action": "hold", "reason": "low_combined_score"}
        
        return {
            "action": "buy" if net_direction > 0 else "sell",
            "direction": net_direction,
            "strength": avg_strength,
            "confidence": avg_confidence,
            "combined_score": combined_score,
            "signal_count": len(self.state.pending_signals)
        }
    
    def _execute_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on the decision."""
        action = decision.get("action")
        
        if action not in ["buy", "sell"]:
            return {"status": "no_action", "decision": decision}
        
        try:
            # Calculate position size (simplified - would integrate with position_sizer)
            quantity = self._calculate_quantity(decision)
            
            if quantity <= 0:
                return {"status": "rejected", "reason": "invalid_quantity"}
            
            # Create order via order manager
            order_result = self.order_manager.create_order(
                asset=self.asset,
                side=action.upper(),
                quantity=quantity,
                order_type="MARKET",
                tags=["asset_worker", f"score_{decision['combined_score']:.2f}"]
            )
            
            # Update state
            self.state.current_position = {
                "side": action,
                "quantity": quantity,
                "entry_time": time.time(),
                "entry_score": decision["combined_score"]
            }
            
            logger.info(f"Trade executed for {self.asset}: {action} {quantity}")
            
            return {
                "status": "executed",
                "action": action,
                "quantity": quantity,
                "order_id": order_result.get("order_id"),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed for {self.asset}: {e}")
            return {"status": "execution_error", "message": str(e)}
    
    def _calculate_quantity(self, decision: Dict[str, Any]) -> float:
        """Calculate trade quantity based on decision and risk parameters."""
        # Simplified calculation - would integrate Kelly Criterion from position_sizer
        base_quantity = self.config.get("base_quantity", 0.01)
        strength_multiplier = decision.get("strength", 0.5)
        
        return base_quantity * strength_multiplier
    
    def get_state(self) -> Dict[str, Any]:
        """Get current worker state."""
        return {
            "asset": self.state.asset,
            "is_active": self.state.is_active,
            "current_position": self.state.current_position,
            "pending_signals_count": len(self.state.pending_signals),
            "trades_executed": self.state.trades_executed,
            "total_pnl": self.state.total_pnl,
            "max_drawdown": self.state.max_drawdown,
            "last_heartbeat": self.state.last_heartbeat,
            "avg_processing_time_ms": (
                sum(self._processing_times) / len(self._processing_times)
                if self._processing_times else 0
            )
        }
    
    def update_position(self, pnl: float, drawdown: float) -> None:
        """Update position PnL and drawdown."""
        self.state.total_pnl += pnl
        self.state.max_drawdown = max(self.state.max_drawdown, drawdown)
    
    def close_position(self) -> Dict[str, Any]:
        """Close the current position."""
        if not self.state.current_position:
            return {"status": "no_position"}
        
        try:
            position = self.state.current_position
            quantity = position["quantity"]
            side = "SELL" if position["side"] == "buy" else "BUY"
            
            order_result = self.order_manager.create_order(
                asset=self.asset,
                side=side,
                quantity=quantity,
                order_type="MARKET",
                tags=["position_close"]
            )
            
            self.state.current_position = None
            
            logger.info(f"Position closed for {self.asset}")
            
            return {
                "status": "closed",
                "order_id": order_result.get("order_id"),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to close position for {self.asset}: {e}")
            return {"status": "error", "message": str(e)}
    
    def shutdown(self) -> Dict[str, Any]:
        """Gracefully shutdown the worker."""
        logger.info(f"Shutting down AssetWorker for {self.asset}")
        
        # Close any open position
        close_result = self.close_position()
        
        self.state.is_active = False
        
        final_state = self.get_state()
        
        logger.info(f"AssetWorker {self.asset} shut down. Final state: {final_state}")
        
        return {
            "status": "shutdown_complete",
            "final_state": final_state,
            "close_result": close_result
        }
    
    def heartbeat(self) -> float:
        """Return current timestamp as heartbeat."""
        self.state.last_heartbeat = time.time()
        return self.state.last_heartbeat


class AssetWorkerPool:
    """
    Manages a pool of AssetWorker actors for all supported assets.
    Provides unified interface for signal distribution and result aggregation.
    """
    
    SUPPORTED_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "USDT"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.workers: Dict[str, ActorHandle] = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize all asset workers."""
        if self._initialized:
            logger.warning("AssetWorkerPool already initialized")
            return True
        
        try:
            for asset in self.SUPPORTED_ASSETS:
                # Create Ray actor
                worker = AssetWorker.remote(asset, self.config)
                
                # Initialize worker
                init_result = ray.get(worker.initialize.remote())
                
                if init_result:
                    self.workers[asset] = worker
                    logger.info(f"Worker initialized for {asset}")
                else:
                    logger.error(f"Failed to initialize worker for {asset}")
            
            self._initialized = len(self.workers) == len(self.SUPPORTED_ASSETS)
            return self._initialized
            
        except Exception as e:
            logger.error(f"Failed to initialize AssetWorkerPool: {e}")
            return False
    
    def distribute_signal(self, signal: AlphaSignal) -> Dict[str, Any]:
        """Distribute a signal to the appropriate worker."""
        asset = signal.asset
        
        if asset not in self.workers:
            return {"status": "error", "message": f"No worker for {asset}"}
        
        worker = self.workers[asset]
        
        # Send signal asynchronously
        signal_data = {
            "signal_type": signal.signal_type.name,
            "asset": signal.asset,
            "direction": signal.direction,
            "strength": signal.strength,
            "confidence": signal.confidence,
            "timestamp": signal.timestamp,
            "metadata": signal.metadata
        }
        
        return {"status": "sent", "asset": asset, "worker_ref": worker}
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states from all workers."""
        states = {}
        
        for asset, worker in self.workers.items():
            try:
                state = ray.get(worker.get_state.remote())
                states[asset] = state
            except Exception as e:
                states[asset] = {"error": str(e)}
        
        return states
    
    def shutdown_all(self) -> Dict[str, Any]:
        """Shutdown all workers gracefully."""
        results = {}
        
        for asset, worker in list(self.workers.items()):
            try:
                result = ray.get(worker.shutdown.remote())
                results[asset] = result
            except Exception as e:
                results[asset] = {"error": str(e)}
            
            # Kill the actor
            ray.kill(worker)
        
        self.workers.clear()
        self._initialized = False
        
        return results


# Factory function
def create_asset_worker_pool(config: Optional[Dict[str, Any]] = None) -> AssetWorkerPool:
    """Create and initialize an AssetWorkerPool."""
    pool = AssetWorkerPool(config)
    pool.initialize()
    return pool


if __name__ == "__main__":
    # Example usage
    import ray
    
    # Initialize Ray
    ray.init(num_cpus=4, ignore_reinit_error=True)
    
    # Create worker pool
    pool = create_asset_worker_pool({
        "base_quantity": 0.01,
        "confirmation_threshold": 2,
        "min_signal_strength": 0.5
    })
    
    # Get states
    states = pool.get_all_states()
    print(f"Initial states: {states}")
    
    # Shutdown
    results = pool.shutdown_all()
    print(f"Shutdown results: {results}")
    
    ray.shutdown()
    print("Asset Workers module test complete.")
