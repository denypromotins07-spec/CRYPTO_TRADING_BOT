"""
Online Learning Engine for ZAID Personal Crypto Trading Bot
Continuously updates model weights without LLM
Implements recursive least squares and adaptive gradient methods
Gracefully degrades when feature variance exceeds thresholds

Part of the 152 domains of quantitative finance implementation.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelUpdate:
    """Result of model update operation."""
    symbol: str
    success: bool
    weights_updated: int
    learning_rate: float
    degradation_factor: float
    timestamp: float
    message: str


class RecursiveLeastSquares:
    """
    Recursive Least Squares for online linear regression.
    Memory efficient O(n^2) update instead of O(n^3) matrix inversion.
    """
    
    def __init__(self, n_features: int, forgetting_factor: float = 0.99):
        self.n_features = n_features
        self.lambda_ = forgetting_factor  # Forgetting factor
        
        # Initialize covariance matrix inverse
        self.P = np.eye(n_features) * 1000  # Large initial uncertainty
        
        # Weight vector
        self.weights = np.zeros(n_features)
        
        # Tracking
        self.n_updates = 0
        self.cumulative_error = 0.0
        
    def update(self, x: np.ndarray, y: float) -> Tuple[float, float]:
        """
        Update model with new observation.
        Returns (prediction, error).
        """
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")
        
        # Prediction
        y_pred = np.dot(self.weights, x)
        
        # Error
        error = y - y_pred
        
        # Kalman gain
        Px = np.dot(self.P, x)
        denominator = self.lambda_ + np.dot(x, Px)
        
        if abs(denominator) < 1e-10:
            # Numerical stability - skip update
            return y_pred, error
        
        K = Px / denominator
        
        # Update weights
        self.weights = self.weights + K * error
        
        # Update covariance matrix
        self.P = (self.P - np.outer(K, np.dot(x, self.P))) / self.lambda_
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2
        
        self.n_updates += 1
        self.cumulative_error += error ** 2
        
        return y_pred, error
    
    def predict(self, x: np.ndarray) -> float:
        """Make prediction without updating."""
        return np.dot(self.weights, x)
    
    def get_weights(self) -> np.ndarray:
        """Get current weight vector."""
        return self.weights.copy()
    
    def reset(self):
        """Reset model state."""
        self.P = np.eye(self.n_features) * 1000
        self.weights = np.zeros(self.n_features)
        self.n_updates = 0
        self.cumulative_error = 0.0


class AdaptiveGradientLearner:
    """
    Online learner with adaptive learning rates (AdaGrad-style).
    More robust to non-stationary data.
    """
    
    def __init__(self, n_features: int, initial_lr: float = 0.01,
                 variance_threshold: float = 10.0):
        self.n_features = n_features
        self.initial_lr = initial_lr
        self.variance_threshold = variance_threshold
        
        # Weights
        self.weights = np.zeros(n_features)
        
        # Accumulated squared gradients
        self.G = np.zeros(n_features)
        
        # Feature statistics for variance monitoring
        self.feature_mean = np.zeros(n_features)
        self.feature_M2 = np.zeros(n_features)
        self.n_samples = 0
        
        # Degradation tracking
        self.degradation_factor = 1.0
        
    def update_feature_stats(self, x: np.ndarray):
        """Update running statistics for feature variance monitoring."""
        self.n_samples += 1
        
        delta = x - self.feature_mean
        self.feature_mean += delta / self.n_samples
        delta2 = x - self.feature_mean
        self.feature_M2 += delta * delta2
        
        # Check variance
        if self.n_samples > 10:
            variances = self.feature_M2 / self.n_samples
            
            # If any feature variance exceeds threshold, degrade learning
            max_var_ratio = np.max(variances) / self.variance_threshold
            if max_var_ratio > 1.0:
                self.degradation_factor = min(1.0, 1.0 / max_var_ratio)
            else:
                self.degradation_factor = 1.0
    
    def update(self, x: np.ndarray, y: float) -> Tuple[float, float]:
        """Update model with new observation."""
        # Update feature statistics first
        self.update_feature_stats(x)
        
        # Prediction
        y_pred = np.dot(self.weights, x)
        
        # Error
        error = y - y_pred
        
        # Gradient
        gradient = -error * x
        
        # Apply degradation factor
        effective_gradient = gradient * self.degradation_factor
        
        # Accumulate squared gradients
        self.G += effective_gradient ** 2
        
        # Adaptive learning rate
        eps = 1e-8
        adjusted_lr = self.initial_lr / (np.sqrt(self.G) + eps)
        
        # Update weights
        self.weights -= adjusted_lr * effective_gradient
        
        return y_pred, error
    
    def predict(self, x: np.ndarray) -> float:
        """Make prediction."""
        return np.dot(self.weights, x)
    
    def should_degrade(self) -> bool:
        """Check if model is in degraded mode."""
        return self.degradation_factor < 0.5
    
    def get_status(self) -> Dict[str, any]:
        """Get learner status."""
        return {
            'degradation_factor': self.degradation_factor,
            'is_degraded': self.should_degrade(),
            'n_samples': self.n_samples,
            'effective_lr_range': (
                float(self.initial_lr / (np.sqrt(np.max(self.G)) + 1e-8)),
                float(self.initial_lr / (np.sqrt(np.min(self.G) + 1e-8)))
            )
        }


class EnsembleOnlineLearner:
    """
    Ensemble of online learners for robustness.
    Uses weighted averaging based on recent performance.
    """
    
    def __init__(self, n_features: int, n_models: int = 3):
        self.n_features = n_features
        self.n_models = n_models
        
        # Create diverse models
        self.models = [
            RecursiveLeastSquares(n_features, forgetting_factor=0.99),
            RecursiveLeastSquares(n_features, forgetting_factor=0.95),
            AdaptiveGradientLearner(n_features, initial_lr=0.01)
        ][:n_models]
        
        # Model weights (performance-based)
        self.model_weights = np.ones(n_models) / n_models
        
        # Recent errors for weight updates
        self.recent_errors: deque = deque(maxlen=50)
        
    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Make ensemble prediction.
        Returns (prediction, uncertainty).
        """
        predictions = [model.predict(x) for model in self.models]
        
        # Weighted average
        weighted_pred = np.average(predictions, weights=self.model_weights)
        
        # Uncertainty from prediction variance
        uncertainty = np.std(predictions)
        
        return weighted_pred, uncertainty
    
    def update(self, x: np.ndarray, y: float) -> ModelUpdate:
        """Update all models and adjust ensemble weights."""
        predictions = []
        errors = []
        
        for i, model in enumerate(self.models):
            pred, err = model.update(x, y)
            predictions.append(pred)
            errors.append(err ** 2)
        
        # Update model weights based on recent performance
        self._update_model_weights(errors)
        
        # Find best performing model
        best_idx = np.argmin(errors)
        
        return ModelUpdate(
            symbol="ensemble",
            success=True,
            weights_updated=len(self.models),
            learning_rate=self.model_weights[best_idx],
            degradation_factor=min(self.model_weights),
            timestamp=0.0,
            message=f"Best model: {best_idx}, avg error: {np.mean(errors):.6f}"
        )
    
    def _update_model_weights(self, errors: List[float]):
        """Update ensemble weights based on inverse error."""
        # Inverse error weighting
        inv_errors = 1.0 / (np.array(errors) + 1e-6)
        
        # Exponential moving average update
        alpha = 0.1
        self.model_weights = (1 - alpha) * self.model_weights + alpha * inv_errors
        
        # Normalize
        self.model_weights /= np.sum(self.model_weights)


class OnlineLearningEngine:
    """
    Main engine for online learning across all symbols.
    Singleton pattern for global access.
    """
    
    _instance: Optional['OnlineLearningEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.models: Dict[str, EnsembleOnlineLearner] = {}
        self.target_buffers: Dict[str, deque] = {}
        self.feature_buffers: Dict[str, deque] = {}
        
        self.max_buffer = 100
        self.variance_threshold = 10.0
        
        self._initialized = True
        logger.info("OnlineLearningEngine initialized")
    
    def get_or_create_model(self, symbol: str, n_features: int):
        """Get or create model for a symbol."""
        if symbol not in self.models:
            self.models[symbol] = EnsembleOnlineLearner(n_features)
            self.target_buffers[symbol] = deque(maxlen=self.max_buffer)
            self.feature_buffers[symbol] = deque(maxlen=self.max_buffer)
    
    def train_step(self, symbol: str, features: np.ndarray, 
                   target: float) -> ModelUpdate:
        """Perform one training step."""
        n_features = len(features)
        self.get_or_create_model(symbol, n_features)
        
        # Store in buffer
        self.feature_buffers[symbol].append(features.copy())
        self.target_buffers[symbol].append(target)
        
        # Update model
        update = self.models[symbol].update(features, target)
        update.symbol = symbol
        
        # Check for degradation
        if self.models[symbol].models[1].should_degrade():
            update.message += " [DEGRADED MODE]"
            logger.warning(f"Model for {symbol} in degraded mode due to high variance")
        
        return update
    
    def predict(self, symbol: str, features: np.ndarray) -> Tuple[float, float]:
        """Get prediction for given features."""
        if symbol not in self.models:
            return 0.0, float('inf')
        
        return self.models[symbol].predict(features)
    
    def get_model_status(self, symbol: str) -> Optional[Dict[str, any]]:
        """Get detailed model status."""
        if symbol not in self.models:
            return None
        
        model = self.models[symbol]
        
        return {
            'symbol': symbol,
            'n_models': model.n_models,
            'ensemble_weights': model.model_weights.tolist(),
            'buffer_size': len(self.target_buffers[symbol]),
            'is_trained': len(self.target_buffers[symbol]) >= 20
        }
    
    def graceful_degradation(self, symbol: str, 
                            variance_exceeded: bool) -> Dict[str, any]:
        """
        Handle graceful degradation when variance exceeds thresholds.
        """
        if symbol not in self.models:
            return {'action': 'none', 'reason': 'no_model'}
        
        if variance_exceeded:
            # Reduce learning rates
            for mdl in self.models[symbol].models:
                if isinstance(mdl, AdaptiveGradientLearner):
                    mdl.initial_lr *= 0.5
                    mdl.degradation_factor = 0.3
            
            logger.info(f"Applied graceful degradation for {symbol}")
            
            return {
                'action': 'degraded',
                'learning_rate_reduced': True,
                'variance_threshold': self.variance_threshold
            }
        
        return {'action': 'normal'}
    
    def export_weights(self, symbol: str) -> Optional[np.ndarray]:
        """Export model weights for persistence."""
        if symbol not in self.models:
            return None
        
        # Export primary model weights
        return self.models[symbol].models[0].get_weights()
    
    def import_weights(self, symbol: str, weights: np.ndarray):
        """Import pre-trained weights."""
        n_features = len(weights)
        self.get_or_create_model(symbol, n_features)
        
        # Set weights in primary model
        self.models[symbol].models[0].weights = weights.copy()
        logger.info(f"Imported weights for {symbol}")


if __name__ == "__main__":
    # Example usage
    engine = OnlineLearningEngine()
    
    # Simulate training data
    true_weights = np.array([0.5, -0.3, 0.2, 0.1, -0.1])
    n_features = len(true_weights)
    
    for i in range(200):
        # Generate features
        features = np.random.randn(n_features)
        
        # Generate target with noise
        target = np.dot(true_weights, features) + np.random.randn() * 0.1
        
        # Train
        update = engine.train_step("BTCUSDT", features, target)
        
        if i % 40 == 0:
            status = engine.get_model_status("BTCUSDT")
            print(f"Step {i}:")
            print(f"  Buffer size: {status['buffer_size']}")
            print(f"  Ensemble weights: {[f'{w:.3f}' for w in status['ensemble_weights']]}")
            print(f"  Message: {update.message}")
    
    # Test prediction
    test_features = np.random.randn(n_features)
    pred, uncertainty = engine.predict("BTCUSDT", test_features)
    print(f"\nPrediction: {pred:.4f} ± {uncertainty:.4f}")
