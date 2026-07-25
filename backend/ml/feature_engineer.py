"""
Feature Engineering Engine for ZAID Personal Crypto Trading Bot
Ultra-fast, zero-copy feature extraction and scaling
Optimized with NumPy for vectorized operations
Memory-efficient pipeline implementation

Part of the 152 domains of quantitative finance implementation.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features for categorization."""
    PRICE = "PRICE"
    VOLUME = "VOLUME"
    MOMENTUM = "MOMENTUM"
    VOLATILITY = "VOLATILITY"
    ORDER_FLOW = "ORDER_FLOW"
    TECHNICAL = "TECHNICAL"


@dataclass
class FeatureMetadata:
    """Metadata for each feature."""
    name: str
    feature_type: FeatureType
    scale: str  # 'raw', 'normalized', 'standardized'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None


class FeatureScaler:
    """
    Online feature scaler using Welford's algorithm.
    Supports multiple scaling methods.
    """
    
    def __init__(self, method: str = 'zscore', clip_outliers: bool = True):
        self.method = method
        self.clip_outliers = clip_outliers
        
        # Running statistics
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        # For robust scaling
        self.values_buffer: deque = deque(maxlen=1000)
        
    def update(self, value: float) -> float:
        """Update scaler and return scaled value."""
        self.n += 1
        
        # Update running mean and variance (Welford's algorithm)
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Store for percentile calculations
        self.values_buffer.append(value)
        
        return self.transform(value)
    
    def transform(self, value: float) -> float:
        """Transform a value using current statistics."""
        if self.n < 2:
            return 0.0
        
        variance = self.M2 / self.n
        std = np.sqrt(variance) if variance > 0 else 1.0
        
        if self.method == 'zscore':
            scaled = (value - self.mean) / std if std > 0 else 0.0
            
        elif self.method == 'minmax':
            range_val = self.max_val - self.min_val
            scaled = (value - self.min_val) / range_val if range_val > 0 else 0.5
            
        elif self.method == 'robust':
            if len(self.values_buffer) >= 10:
                sorted_vals = sorted(self.values_buffer)
                q1 = sorted_vals[int(len(sorted_vals) * 0.25)]
                q3 = sorted_vals[int(len(sorted_vals) * 0.75)]
                median = sorted_vals[int(len(sorted_vals) * 0.5)]
                iqr = q3 - q1
                scaled = (value - median) / iqr if iqr > 0 else 0.0
            else:
                scaled = 0.0
        else:
            scaled = value
        
        # Clip outliers
        if self.clip_outliers:
            scaled = np.clip(scaled, -5.0, 5.0)
        
        return scaled
    
    def reset(self):
        """Reset scaler statistics."""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.values_buffer.clear()


class FeatureCalculator:
    """
    Base class for individual feature calculators.
    Implements flyweight pattern for memory efficiency.
    """
    
    def __init__(self, name: str, feature_type: FeatureType):
        self.name = name
        self.feature_type = feature_type
        self.scaler = FeatureScaler()
        
    def calculate(self, data: Dict[str, any]) -> float:
        """Calculate feature value from input data."""
        raise NotImplementedError
    
    def update_and_get(self, data: Dict[str, any]) -> Tuple[float, float]:
        """Calculate feature and update scaler, returns (raw, scaled)."""
        raw = self.calculate(data)
        scaled = self.scaler.update(raw)
        return raw, scaled


class ReturnFeature(FeatureCalculator):
    """Calculate price returns."""
    
    def __init__(self, lookback: int = 1):
        super().__init__(f"return_{lookback}", FeatureType.PRICE)
        self.lookback = lookback
        self.price_history: deque = deque(maxlen=lookback + 1)
        
    def calculate(self, data: Dict[str, any]) -> float:
        price = data.get('price', 0)
        self.price_history.append(price)
        
        if len(self.price_history) <= self.lookback:
            return 0.0
        
        old_price = list(self.price_history)[-self.lookback - 1]
        if old_price == 0:
            return 0.0
        
        return (price - old_price) / old_price


class VolatilityFeature(FeatureCalculator):
    """Calculate rolling volatility."""
    
    def __init__(self, window: int = 10):
        super().__init__(f"volatility_{window}", FeatureType.VOLATILITY)
        self.window = window
        self.returns: deque = deque(maxlen=window + 1)
        
    def calculate(self, data: Dict[str, any]) -> float:
        price = data.get('price', 0)
        
        if len(self.returns) > 0:
            last_price = self.returns[-1][1] if self.returns else None
            if last_price and last_price > 0:
                ret = np.log(price / last_price)
                self.returns.append((price, ret))
        else:
            self.returns.append((price, 0))
        
        if len(self.returns) < self.window:
            return 0.0
        
        returns_array = np.array([r[1] for r in list(self.returns)[-self.window:]])
        return float(np.std(returns_array) * np.sqrt(252))


class VolumeFeature(FeatureCalculator):
    """Calculate volume-based features."""
    
    def __init__(self, window: int = 20):
        super().__init__(f"volume_ratio_{window}", FeatureType.VOLUME)
        self.window = window
        self.volumes: deque = deque(maxlen=window)
        
    def calculate(self, data: Dict[str, any]) -> float:
        volume = data.get('volume', 0)
        self.volumes.append(volume)
        
        if len(self.volumes) < self.window:
            return 1.0
        
        avg_volume = np.mean(list(self.volumes)[-self.window:])
        if avg_volume == 0:
            return 1.0
        
        return volume / avg_volume


class MomentumFeature(FeatureCalculator):
    """Calculate momentum indicators."""
    
    def __init__(self, lookback: int = 14):
        super().__init__(f"momentum_{lookback}", FeatureType.MOMENTUM)
        self.lookback = lookback
        self.prices: deque = deque(maxlen=lookback + 1)
        
    def calculate(self, data: Dict[str, any]) -> float:
        price = data.get('price', 0)
        self.prices.append(price)
        
        if len(self.prices) <= self.lookback:
            return 0.0
        
        old_price = list(self.prices)[-self.lookback - 1]
        if old_price == 0:
            return 0.0
        
        # Rate of change
        return (price - old_price) / old_price * 100


class OrderFlowFeature(FeatureCalculator):
    """Calculate order flow imbalance features."""
    
    def __init__(self):
        super().__init__("order_flow_imbalance", FeatureType.ORDER_FLOW)
        
    def calculate(self, data: Dict[str, any]) -> float:
        buy_volume = data.get('buy_volume', 0)
        sell_volume = data.get('sell_volume', 0)
        
        total = buy_volume + sell_volume
        if total == 0:
            return 0.0
        
        return (buy_volume - sell_volume) / total


class FeaturePipeline:
    """
    Main feature engineering pipeline.
    Implements strategy pattern for flexible feature combinations.
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.features: Dict[str, List[FeatureCalculator]] = {}
        self.feature_metadata: Dict[str, List[FeatureMetadata]] = {}
        
        self._initialize_features()
        
    def _initialize_features(self):
        """Initialize all feature calculators for each symbol."""
        for symbol in self.symbols:
            self.features[symbol] = [
                ReturnFeature(1),
                ReturnFeature(5),
                ReturnFeature(10),
                VolatilityFeature(10),
                VolatilityFeature(20),
                VolumeFeature(20),
                MomentumFeature(7),
                MomentumFeature(14),
                OrderFlowFeature(),
            ]
            
            self.feature_metadata[symbol] = [
                FeatureMetadata(f.name, f.feature_type, 'zscore')
                for f in self.features[symbol]
            ]
    
    def process(self, symbol: str, market_data: Dict[str, any]) -> Dict[str, any]:
        """Process market data through feature pipeline."""
        if symbol not in self.features:
            self._initialize_features()
        
        result = {
            'symbol': symbol,
            'timestamp': market_data.get('timestamp', 0),
            'features': {},
            'feature_vector': None
        }
        
        feature_values = []
        
        for feature in self.features[symbol]:
            raw, scaled = feature.update_and_get(market_data)
            result['features'][feature.name] = {
                'raw': raw,
                'scaled': scaled
            }
            feature_values.append(scaled)
        
        # Create zero-copy numpy array for ML models
        result['feature_vector'] = np.array(feature_values, dtype=np.float32)
        
        return result
    
    def get_feature_names(self, symbol: str) -> List[str]:
        """Get list of feature names for a symbol."""
        if symbol not in self.features:
            return []
        return [f.name for f in self.features[symbol]]
    
    def get_feature_dimension(self, symbol: str) -> int:
        """Get feature vector dimension."""
        return len(self.features.get(symbol, []))
    
    def add_custom_feature(self, symbol: str, feature: FeatureCalculator):
        """Add custom feature to pipeline."""
        if symbol not in self.features:
            self._initialize_features()
        self.features[symbol].append(feature)


class NautilusAdapter:
    """
    Adapter for feeding features to Nautilus trading engine.
    Zero-latency feature delivery.
    """
    
    def __init__(self, pipeline: FeaturePipeline):
        self.pipeline = pipeline
        self.feature_buffers: Dict[str, deque] = {}
        self.max_buffer = 100
        
    def prepare_for_nautilus(self, symbol: str, market_data: Dict[str, any]) -> bytes:
        """Prepare features in Nautilus-compatible format."""
        result = self.pipeline.process(symbol, market_data)
        
        if symbol not in self.feature_buffers:
            self.feature_buffers[symbol] = deque(maxlen=self.max_buffer)
        
        # Store feature vector
        if result['feature_vector'] is not None:
            self.feature_buffers[symbol].append({
                'timestamp': result['timestamp'],
                'features': result['feature_vector'].tobytes()
            })
        
        # Return serialized features
        return result['feature_vector'].tobytes() if result['feature_vector'] else b''
    
    def get_recent_features(self, symbol: str, count: int = 10) -> Optional[np.ndarray]:
        """Get recent feature vectors for sequence models."""
        if symbol not in self.feature_buffers:
            return None
        
        buffer = self.feature_buffers[symbol]
        if len(buffer) < count:
            return None
        
        vectors = []
        for item in list(buffer)[-count:]:
            vectors.append(np.frombuffer(item['features'], dtype=np.float32))
        
        return np.vstack(vectors)


if __name__ == "__main__":
    # Example usage
    pipeline = FeaturePipeline(["BTCUSDT", "ETHUSDT"])
    
    # Simulate market data
    base_price = 45000
    for i in range(50):
        market_data = {
            'price': base_price * (1 + np.random.randn() * 0.001),
            'volume': 1000 + np.random.randn() * 200,
            'buy_volume': 500 + np.random.randn() * 100,
            'sell_volume': 500 + np.random.randn() * 100,
            'timestamp': i
        }
        
        result = pipeline.process("BTCUSDT", market_data)
        
        if i % 10 == 0:
            print(f"Step {i}:")
            print(f"  Feature dim: {pipeline.get_feature_dimension('BTCUSDT')}")
            print(f"  Vector shape: {result['feature_vector'].shape}")
            for name, vals in result['features'].items():
                print(f"  {name}: raw={vals['raw']:.4f}, scaled={vals['scaled']:.4f}")
        
        base_price = market_data['price']
