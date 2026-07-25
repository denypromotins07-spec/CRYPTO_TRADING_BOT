"""
Weight Manager for Hot-Swapping ML Models
Safely swaps model weights without stopping the bot.
Optimized for 8GB RAM with atomic operations.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import threading
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """Saved model state for rollback."""
    name: str
    weights_hash: str
    weights: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    timestamp: float
    performance_score: Optional[float] = None


@dataclass
class WeightManagerConfig:
    """Configuration for weight manager."""
    # Rollback settings
    max_checkpoints: int = 5
    auto_rollback_threshold: float = -0.1  # Performance drop threshold
    
    # Validation settings
    validate_weights: bool = True
    check_nan_inf: bool = True
    check_bounds: bool = True
    
    # Atomic swap settings
    use_locking: bool = True
    swap_timeout: float = 5.0  # Seconds
    
    # Monitoring
    track_performance: bool = True
    performance_window: int = 100


class WeightManager:
    """
    Hot-Swap Weight Manager for ML Models.
    
    Implements:
    - Atomic weight swapping without service interruption
    - Automatic rollback on performance degradation
    - Checkpoint management with bounded memory
    - Thread-safe concurrent access
    - Weight validation (NaN/Inf checks, bounds)
    
    All operations are designed for zero-downtime updates.
    """
    
    def __init__(self, config: WeightManagerConfig):
        self.config = config
        
        # Current weights per model
        self.current_weights: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Checkpoint history
        self.checkpoints: Dict[str, deque] = {
            'main': deque(maxlen=config.max_checkpoints)
        }
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = {
            'main': deque(maxlen=config.performance_window)
        }
        
        # Locking for atomic operations
        self._lock = threading.RLock() if config.use_locking else None
        
        # Swap state
        self._swap_in_progress: bool = False
        self._last_swap_time: float = 0.0
        
        # Callbacks
        self._on_swap_callbacks: List[Callable] = []
        self._on_rollback_callbacks: List[Callable] = []
        
        logger.info(f"WeightManager initialized: max_checkpoints={config.max_checkpoints}")
    
    def register_model(
        self,
        model_name: str,
        initial_weights: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new model with initial weights.
        
        Args:
            model_name: Unique identifier for the model
            initial_weights: Dictionary of weight arrays
            metadata: Optional metadata (architecture, training info, etc.)
        """
        with self._lock if self._lock else threading.nullcontext():
            # Validate weights
            if self.config.validate_weights:
                self._validate_weights(initial_weights, model_name)
            
            # Store current weights
            self.current_weights[model_name] = {
                k: v.copy() for k, v in initial_weights.items()
            }
            
            # Create initial checkpoint
            self._create_checkpoint(model_name, initial_weights, metadata or {})
            
            logger.info(f"Model '{model_name}' registered with "
                       f"{len(initial_weights)} weight tensors")
    
    def swap_weights(
        self,
        model_name: str,
        new_weights: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Atomically swap model weights.
        
        Args:
            model_name: Model to update
            new_weights: New weight dictionary
            metadata: Optional metadata for new weights
            force: Force swap even if validation fails
        
        Returns:
            Tuple of (success, message)
        """
        import time
        start_time = time.time()
        
        with self._lock if self._lock else threading.nullcontext():
            # Check if another swap is in progress
            if self._swap_in_progress and not force:
                return False, "Another swap is in progress"
            
            # Check timeout
            if time.time() - self._last_swap_time < self.config.swap_timeout:
                if not force:
                    return False, "Swap rate limit exceeded"
            
            self._swap_in_progress = True
            
            try:
                # Validate new weights
                if self.config.validate_weights and not force:
                    valid, error = self._validate_weights(new_weights, model_name)
                    if not valid:
                        return False, f"Validation failed: {error}"
                
                # Get current weights for checkpoint
                if model_name not in self.current_weights:
                    return False, f"Model '{model_name}' not registered"
                
                current = self.current_weights[model_name]
                
                # Create checkpoint before swap
                self._create_checkpoint(
                    model_name,
                    current,
                    {'reason': 'pre_swap', 'timestamp': start_time}
                )
                
                # Atomic swap (dictionary assignment is atomic in Python)
                self.current_weights[model_name] = {
                    k: v.copy() for k, v in new_weights.items()
                }
                
                self._last_swap_time = time.time()
                self._swap_in_progress = False
                
                # Notify callbacks
                self._notify_swap_callbacks(model_name, new_weights)
                
                logger.info(f"Weights swapped for '{model_name}' in "
                           f"{time.time() - start_time:.4f}s")
                
                return True, "Swap successful"
                
            except Exception as e:
                self._swap_in_progress = False
                logger.error(f"Swap failed: {e}")
                return False, f"Swap failed: {e}"
    
    def rollback(self, model_name: str, steps: int = 1) -> Tuple[bool, str]:
        """
        Rollback model weights to previous checkpoint.
        
        Args:
            model_name: Model to rollback
            steps: Number of checkpoints to go back
        
        Returns:
            Tuple of (success, message)
        """
        with self._lock if self._lock else threading.nullcontext():
            if model_name not in self.current_weights:
                return False, f"Model '{model_name}' not found"
            
            checkpoint_queue = self.checkpoints.get('main', deque())
            
            if len(checkpoint_queue) <= steps:
                return False, "Insufficient checkpoints for rollback"
            
            # Get target checkpoint
            target = list(checkpoint_queue)[-steps - 1]
            
            # Restore weights
            self.current_weights[model_name] = {
                k: v.copy() for k, v in target.weights.items()
            }
            
            logger.info(f"Rolled back '{model_name}' to checkpoint "
                       f"from {target.timestamp}")
            
            # Notify callbacks
            self._notify_rollback_callbacks(model_name, target)
            
            return True, f"Rolled back {steps} checkpoint(s)"
    
    def auto_rollback_if_needed(self, model_name: str) -> Tuple[bool, str]:
        """
        Automatically rollback if performance degraded.
        
        Returns:
            Tuple of (rolled_back, reason)
        """
        if not self.config.track_performance:
            return False, "Performance tracking disabled"
        
        perf_history = list(self.performance_history.get('main', []))
        
        if len(perf_history) < 10:
            return False, "Insufficient performance data"
        
        # Compare recent vs historical performance
        recent_avg = np.mean(perf_history[-5:])
        historical_avg = np.mean(perf_history[:-5])
        
        if recent_avg - historical_avg < self.config.auto_rollback_threshold:
            success, msg = self.rollback(model_name, steps=1)
            if success:
                logger.warning(f"Auto-rollback triggered: performance dropped "
                             f"from {historical_avg:.4f} to {recent_avg:.4f}")
            return success, msg
        
        return False, "Performance within acceptable range"
    
    def _validate_weights(
        self,
        weights: Dict[str, np.ndarray],
        model_name: str
    ) -> Tuple[bool, str]:
        """Validate weight tensors."""
        for name, arr in weights.items():
            if not isinstance(arr, np.ndarray):
                return False, f"'{name}' is not numpy array"
            
            if self.config.check_nan_inf:
                if np.any(np.isnan(arr)):
                    return False, f"'{name}' contains NaN values"
                if np.any(np.isinf(arr)):
                    return False, f"'{name}' contains Inf values"
            
            if self.config.check_bounds:
                if np.any(np.abs(arr) > 1e6):
                    return False, f"'{name}' has extreme values"
        
        return True, "Valid"
    
    def _create_checkpoint(
        self,
        model_name: str,
        weights: Dict[str, np.ndarray],
        metadata: Dict[str, Any]
    ) -> None:
        """Create a checkpoint for potential rollback."""
        import time
        
        # Compute hash for quick comparison
        weights_flat = np.concatenate([w.flatten() for w in weights.values()])
        weights_hash = hashlib.md5(weights_flat.tobytes()).hexdigest()[:12]
        
        checkpoint = ModelCheckpoint(
            name=model_name,
            weights_hash=weights_hash,
            weights={k: v.copy() for k, v in weights.items()},
            metadata=metadata,
            timestamp=time.time(),
        )
        
        queue = self.checkpoints.setdefault('main', deque(maxlen=self.config.max_checkpoints))
        queue.append(checkpoint)
    
    def record_performance(self, model_name: str, score: float) -> None:
        """Record performance metric for auto-rollback monitoring."""
        if not self.config.track_performance:
            return
        
        queue = self.performance_history.setdefault('main', 
                     deque(maxlen=self.config.performance_window))
        queue.append(score)
        
        # Update latest checkpoint with performance
        if 'main' in self.checkpoints and len(self.checkpoints['main']) > 0:
            self.checkpoints['main'][-1].performance_score = score
    
    def get_current_weights(self, model_name: str) -> Optional[Dict[str, np.ndarray]]:
        """Get current weights for a model (read-only copy)."""
        if model_name not in self.current_weights:
            return None
        
        return {k: v.copy() for k, v in self.current_weights[model_name].items()}
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints."""
        summary = {
            'n_checkpoints': len(self.checkpoints.get('main', [])),
            'checkpoints': [],
        }
        
        for i, cp in enumerate(self.checkpoints.get('main', [])):
            summary['checkpoints'].append({
                'index': i,
                'name': cp.name,
                'hash': cp.weights_hash,
                'timestamp': cp.timestamp,
                'performance_score': cp.performance_score,
                'metadata': cp.metadata,
            })
        
        return summary
    
    def register_swap_callback(self, callback: Callable) -> None:
        """Register callback to be called after successful swap."""
        self._on_swap_callbacks.append(callback)
    
    def register_rollback_callback(self, callback: Callable) -> None:
        """Register callback to be called after rollback."""
        self._on_rollback_callbacks.append(callback)
    
    def _notify_swap_callbacks(self, model_name: str, weights: Dict[str, np.ndarray]) -> None:
        """Notify all swap callbacks."""
        for callback in self._on_swap_callbacks:
            try:
                callback(model_name, weights)
            except Exception as e:
                logger.error(f"Swap callback error: {e}")
    
    def _notify_rollback_callbacks(self, model_name: str, checkpoint: ModelCheckpoint) -> None:
        """Notify all rollback callbacks."""
        for callback in self._on_rollback_callbacks:
            try:
                callback(model_name, checkpoint)
            except Exception as e:
                logger.error(f"Rollback callback error: {e}")


# Example usage
if __name__ == "__main__":
    config = WeightManagerConfig(
        max_checkpoints=5,
        auto_rollback_threshold=-0.1,
    )
    
    manager = WeightManager(config)
    
    # Register initial model
    initial_weights = {
        'layer1': np.random.randn(10, 5),
        'layer2': np.random.randn(5, 2),
    }
    
    manager.register_model('ppo_policy', initial_weights, {'architecture': '2layer'})
    
    # Simulate weight swaps
    for i in range(5):
        new_weights = {
            'layer1': np.random.randn(10, 5) * (1 + i * 0.1),
            'layer2': np.random.randn(5, 2) * (1 + i * 0.1),
        }
        
        success, msg = manager.swap_weights('ppo_policy', new_weights)
        print(f"Swap {i+1}: {success}, {msg}")
        
        # Record simulated performance
        manager.record_performance('ppo_policy', 1.0 - i * 0.15)
    
    # Check if auto-rollback triggers
    rolled_back, reason = manager.auto_rollback_if_needed('ppo_policy')
    print(f"\nAuto-rollback: {rolled_back}, {reason}")
    
    # Summary
    summary = manager.get_checkpoint_summary()
    print(f"\nCheckpoints: {summary['n_checkpoints']}")
