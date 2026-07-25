"""
Time Series Analysis Engine for ZAID Personal Crypto Trading Bot
Implements GARCH, Kalman Filters, and forecasting models
Optimized with NumPy/SciPy for efficient matrix operations
Memory-efficient recursive implementations

Part of the 152 domains of quantitative finance implementation.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Forecast result with confidence intervals."""
    symbol: str
    forecast: float
    lower_bound: float
    upper_bound: float
    confidence: float
    horizon: int
    timestamp: float
    model: str


class GARCHModel:
    """
    GARCH(1,1) volatility model implementation.
    Used for volatility forecasting and risk management.
    """
    
    def __init__(self, max_lag: int = 100):
        self.max_lag = max_lag
        self.returns: Deque[float] = deque(maxlen=max_lag)
        
        # GARCH parameters (will be estimated)
        self.omega: Optional[float] = None  # Constant term
        self.alpha: Optional[float] = None  # ARCH term coefficient
        self.beta: Optional[float] = None   # GARCH term coefficient
        
        # Conditional variance
        self.sigma_sq: Optional[float] = None
        
        # Long-run average variance
        self.unconditional_variance: Optional[float] = None
        
    def update(self, price: float) -> Optional[Dict[str, float]]:
        """Update model with new price and estimate parameters."""
        if len(self.returns) > 0:
            ret = np.log(price / self.returns[-1]) if self.returns[-1] > 0 else 0
            self.returns.append(ret)
        else:
            self.returns.append(0)
            return None
        
        if len(self.returns) < 30:
            return None
        
        # Estimate parameters using method of moments (simplified)
        returns_array = np.array(list(self.returns))
        
        # Unconditional variance
        self.unconditional_variance = np.var(returns_array)
        
        # Initialize GARCH parameters (standard values)
        if self.omega is None:
            self.omega = 0.000001
            self.alpha = 0.1
            self.beta = 0.85
        
        # Update conditional variance recursively
        if self.sigma_sq is None:
            self.sigma_sq = self.unconditional_variance
        else:
            last_ret = returns_array[-1]
            self.sigma_sq = (self.omega + 
                           self.alpha * (last_ret ** 2) + 
                           self.beta * self.sigma_sq)
        
        # Volatility forecast (annualized)
        daily_vol = np.sqrt(self.sigma_sq)
        annualized_vol = daily_vol * np.sqrt(252)
        
        return {
            'conditional_variance': self.sigma_sq,
            'daily_volatility': daily_vol,
            'annualized_volatility': annualized_vol,
            'unconditional_variance': self.unconditional_variance
        }
    
    def forecast_volatility(self, horizon: int = 10) -> np.ndarray:
        """Forecast volatility for next n periods."""
        if self.sigma_sq is None or self.unconditional_variance is None:
            return np.zeros(horizon)
        
        forecasts = np.zeros(horizon)
        current_var = self.sigma_sq
        
        for i in range(horizon):
            forecasts[i] = current_var
            # Mean reversion to unconditional variance
            current_var = (self.omega + 
                          (self.alpha + self.beta) * current_var +
                          self.alpha * (1 - self.alpha - self.beta) * self.unconditional_variance)
        
        return np.sqrt(forecasts)
    
    def get_var(self, position_value: float, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if self.sigma_sq is None:
            return 0.0
        
        daily_vol = np.sqrt(self.sigma_sq)
        
        # Z-score for confidence level
        z_scores = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
        z = z_scores.get(confidence, 1.645)
        
        return position_value * daily_vol * z


class KalmanFilter:
    """
    Kalman Filter for state estimation and trend tracking.
    Adaptive to changing market conditions.
    """
    
    def __init__(self, initial_state: float = 0.0, 
                 process_noise: float = 0.001,
                 measurement_noise: float = 0.1):
        # State vector [level, slope]
        self.x = np.array([initial_state, 0.0])
        
        # State covariance
        self.P = np.eye(2) * 1.0
        
        # Process noise covariance
        self.Q = np.array([[process_noise, 0], [0, process_noise * 0.1]])
        
        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])
        
        # State transition matrix
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        
        # Observation matrix
        self.H = np.array([[1.0, 0.0]])
        
        self.last_update_time: Optional[float] = None
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()
    
    def update(self, measurement: float, timestamp: float) -> Dict[str, float]:
        """Update filter with new measurement."""
        # Predict
        x_pred, P_pred = self.predict()
        
        # Innovation
        y = measurement - self.H @ x_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = x_pred + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        
        self.last_update_time = timestamp
        
        return {
            'level': float(self.x[0]),
            'slope': float(self.x[1]),
            'estimated_value': float(self.x[0]),
            'trend': 'UP' if self.x[1] > 0 else 'DOWN' if self.x[1] < 0 else 'FLAT'
        }
    
    def get_smoothed_value(self) -> float:
        """Get current smoothed/filtered value."""
        return float(self.x[0])
    
    def get_trend_strength(self) -> float:
        """Get trend strength indicator."""
        return float(abs(self.x[1]))


class ARIMAModel:
    """
    Simplified ARIMA-like forecasting using recursive least squares.
    Optimized for online learning.
    """
    
    def __init__(self, ar_order: int = 3, ma_order: int = 2):
        self.ar_order = ar_order
        self.ma_order = ma_order
        
        self.prices: Deque[float] = deque(maxlen=ar_order * 10)
        self.residuals: Deque[float] = deque(maxlen=ma_order * 10)
        
        # AR coefficients (initialized via Yule-Walker approximation)
        self.ar_coeffs: Optional[np.ndarray] = None
        
        # MA coefficients
        self.ma_coeffs: Optional[np.ndarray] = None
        
        # Constant term
        self.constant: float = 0.0
        
    def update(self, price: float) -> Optional[float]:
        """Update model and return one-step forecast."""
        self.prices.append(price)
        
        if len(self.prices) < self.ar_order + 5:
            return None
        
        # Convert to returns for stationarity
        prices_array = np.array(list(self.prices))
        returns = np.diff(np.log(prices_array))
        
        if len(returns) < self.ar_order:
            return None
        
        # Estimate AR coefficients using least squares (simplified)
        if self.ar_coeffs is None:
            self._estimate_coefficients(returns)
        
        # One-step ahead forecast
        forecast = self.constant
        for i in range(self.ar_order):
            if len(returns) > i:
                forecast += self.ar_coeffs[i] * returns[-(i+1)]
        
        # Add MA component
        if self.ma_coeffs is not None:
            for i in range(min(self.ma_order, len(self.residuals))):
                forecast += self.ma_coeffs[i] * list(self.residuals)[-(i+1)]
        
        # Calculate residual
        actual_return = np.log(prices_array[-1] / prices_array[-2])
        residual = actual_return - forecast
        self.residuals.append(residual)
        
        # Convert back to price forecast
        last_price = prices_array[-1]
        forecast_price = last_price * np.exp(forecast)
        
        return forecast_price
    
    def _estimate_coefficients(self, returns: np.ndarray):
        """Estimate AR coefficients using autocorrelation."""
        n = len(returns)
        
        # Sample mean
        self.constant = np.mean(returns)
        centered = returns - self.constant
        
        # Autocorrelation-based estimation (Yule-Walker simplified)
        self.ar_coeffs = np.zeros(self.ar_order)
        
        for k in range(self.ar_order):
            if k == 0:
                var = np.var(centered)
                if var > 0:
                    self.ar_coeffs[k] = np.correlate(centered[:-1], centered[1:])[0] / (n * var)
            else:
                if len(centered) > k + 1:
                    self.ar_coeffs[k] = np.corrcoef(centered[:-k-1], centered[k+1:])[0, 1] if len(centered) > k + 1 else 0
        
        # Ensure stationarity
        if np.sum(np.abs(self.ar_coeffs)) >= 1:
            self.ar_coeffs *= 0.9 / np.sum(np.abs(self.ar_coeffs))
        
        # Initialize MA coefficients
        self.ma_coeffs = np.zeros(self.ma_order) * 0.1


class TimeSeriesEngine:
    """
    Main engine coordinating all time series models.
    Singleton pattern for global access.
    """
    
    _instance: Optional['TimeSeriesEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.garch_models: Dict[str, GARCHModel] = {}
        self.kalman_filters: Dict[str, KalmanFilter] = {}
        self.arima_models: Dict[str, ARIMAModel] = {}
        
        self.price_history: Dict[str, Deque[float]] = {}
        self.max_history = 1000
        
        self._initialized = True
        logger.info("TimeSeriesEngine initialized")
    
    def get_or_create_models(self, symbol: str):
        """Get or create models for a symbol."""
        if symbol not in self.garch_models:
            self.garch_models[symbol] = GARCHModel()
            self.kalman_filters[symbol] = KalmanFilter()
            self.arima_models[symbol] = ARIMAModel()
            self.price_history[symbol] = deque(maxlen=self.max_history)
    
    def update(self, symbol: str, price: float, timestamp: float) -> Dict[str, any]:
        """Update all models with new price."""
        self.get_or_create_models(symbol)
        
        self.price_history[symbol].append(price)
        
        results = {
            'symbol': symbol,
            'timestamp': timestamp,
            'price': price,
            'models': {}
        }
        
        # GARCH volatility
        garch_result = self.garch_models[symbol].update(price)
        if garch_result:
            results['models']['garch'] = garch_result
        
        # Kalman filter
        kalman_result = self.kalman_filters[symbol].update(price, timestamp)
        results['models']['kalman'] = kalman_result
        
        # ARIMA forecast
        arima_forecast = self.arima_models[symbol].update(price)
        if arima_forecast:
            results['models']['arima'] = {
                'forecast': arima_forecast,
                'direction': 'UP' if arima_forecast > price else 'DOWN'
            }
        
        return results
    
    def get_volatility_forecast(self, symbol: str, horizon: int = 10) -> Optional[np.ndarray]:
        """Get volatility forecast for a symbol."""
        if symbol not in self.garch_models:
            return None
        return self.garch_models[symbol].forecast_volatility(horizon)
    
    def get_trend_estimate(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current trend estimate from Kalman filter."""
        if symbol not in self.kalman_filters:
            return None
        
        kf = self.kalman_filters[symbol]
        return {
            'level': kf.get_smoothed_value(),
            'slope': kf.x[1],
            'trend_strength': kf.get_trend_strength()
        }
    
    def get_var(self, symbol: str, position_value: float, 
                confidence: float = 0.95) -> float:
        """Get Value at Risk for a position."""
        if symbol not in self.garch_models:
            return 0.0
        return self.garch_models[symbol].get_var(position_value, confidence)
    
    def recalibrate(self, symbol: str):
        """Recalibrate models (use during downtime window)."""
        if symbol not in self.garch_models or len(self.price_history[symbol]) < 100:
            return
        
        prices = list(self.price_history[symbol])
        
        # Reset GARCH
        self.garch_models[symbol] = GARCHModel()
        for p in prices[-100:]:
            self.garch_models[symbol].update(p)
        
        # Reset Kalman with better initial state
        avg_price = np.mean(prices[-50:])
        self.kalman_filters[symbol] = KalmanFilter(initial_state=avg_price)
        
        # Reset ARIMA
        self.arima_models[symbol] = ARIMAModel()
        for p in prices[-50:]:
            self.arima_models[symbol].update(p)
        
        logger.info(f"Recalibrated models for {symbol}")


if __name__ == "__main__":
    import time
    
    engine = TimeSeriesEngine()
    
    # Simulate price updates
    base_price = 45000
    for i in range(100):
        price = base_price * (1 + np.random.randn() * 0.001)
        result = engine.update("BTCUSDT", price, time.time())
        
        if i % 20 == 0 and 'models' in result:
            print(f"Step {i}: Price={price:.2f}")
            if 'kalman' in result['models']:
                print(f"  Kalman Level: {result['models']['kalman']['level']:.2f}")
            if 'garch' in result['models']:
                print(f"  Volatility: {result['models']['garch']['annualized_volatility']*100:.2f}%")
