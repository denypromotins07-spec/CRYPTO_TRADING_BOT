#!/usr/bin/env python3
"""
Kalman Filter Module for ZAID Trading Bot
Ultra-lightweight Kalman filters for noise reduction in price data
Automatically adjusts process noise covariance during high volatility

This module provides adaptive filtering for tick data smoothing,
state estimation, and signal extraction under the 8GB RAM constraint.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class KalmanModel(Enum):
    """Supported Kalman filter models"""
    CONSTANT_VELOCITY = "cv"  # Position + velocity state
    CONSTANT_ACCELERATION = "ca"  # Position + velocity + acceleration
    ADAPTIVE = "adaptive"  # Auto-tuning based on innovation


@dataclass
class KalmanState:
    """Container for Kalman filter state"""
    x: np.ndarray  # State vector
    P: np.ndarray  # State covariance
    t: float  # Timestamp
    
    def copy(self) -> 'KalmanState':
        return KalmanState(
            x=self.x.copy(),
            P=self.P.copy(),
            t=self.t
        )


@dataclass
class KalmanResult:
    """Result of a Kalman filter update"""
    state: KalmanState
    measurement: float
    prediction: float
    innovation: float
    innovation_covariance: float
    kalman_gain: np.ndarray
    is_valid: bool
    timestamp_ns: int


class KalmanFilter1D:
    """
    One-dimensional Kalman filter for price smoothing.
    
    Implements constant velocity model with adaptive noise tuning.
    """
    
    def __init__(self,
                 initial_value: float = 0.0,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 dt: float = 1.0):
        """
        Initialize 1D Kalman filter.
        
        Args:
            initial_value: Initial state estimate
            process_noise: Process noise variance (Q)
            measurement_noise: Measurement noise variance (R)
            dt: Time step between measurements
        """
        self.dt = dt
        
        # State: [position, velocity]
        self.x = np.array([initial_value, 0.0])
        
        # State covariance
        self.P = np.eye(2) * 0.5
        
        # Process noise matrix
        self.Q_base = np.array([
            [process_noise * dt**4 / 4, process_noise * dt**3 / 2],
            [process_noise * dt**3 / 2, process_noise * dt**2]
        ])
        
        # Measurement noise
        self.R = measurement_noise
        
        # Transition matrix
        self.F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        
        # Observation matrix (we only observe position)
        self.H = np.array([[1.0, 0.0]])
        
        # Adaptive tuning parameters
        self._innovation_history: List[float] = []
        self._max_innovation_history = 50
        self._volatility_estimate = 0.0
        
        # Performance tracking
        self._update_count = 0
        self._last_update_ns = 0
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform prediction step.
        
        Returns:
            Tuple of (predicted state, predicted covariance)
        """
        # x_pred = F * x
        x_pred = self.F @ self.x
        
        # P_pred = F * P * F^T + Q
        P_pred = self.F @ self.P @ self.F.T + self.Q_base
        
        return x_pred, P_pred
    
    def update(self, 
               measurement: float,
               timestamp: Optional[float] = None) -> KalmanResult:
        """
        Perform full predict-update cycle with new measurement.
        
        Args:
            measurement: New observed value
            timestamp: Optional timestamp for the measurement
            
        Returns:
            KalmanResult with updated state and diagnostics
        """
        import time
        start_ns = time.perf_counter_ns()
        
        # Prediction step
        x_pred, P_pred = self.predict()
        
        # Innovation (measurement residual)
        z = np.array([measurement])
        innovation = z - self.H @ x_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        S_scalar = S[0, 0]
        
        # Kalman gain
        K = P_pred @ self.H.T / S_scalar
        
        # Update state
        self.x = x_pred + K.flatten() * innovation[0]
        
        # Update covariance: P = (I - K*H) * P_pred
        I = np.eye(2)
        self.P = (I - K @ self.H) @ P_pred
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2
        
        # Track innovation for adaptive tuning
        self._innovation_history.append(abs(innovation[0]))
        if len(self._innovation_history) > self._max_innovation_history:
            self._innovation_history.pop(0)
        
        # Update volatility estimate
        if len(self._innovation_history) >= 10:
            self._volatility_estimate = np.std(self._innovation_history)
            self._adapt_noise()
        
        self._update_count += 1
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._last_update_ns = elapsed_ns
        
        return KalmanResult(
            state=KalmanState(
                x=self.x.copy(),
                P=self.P.copy(),
                t=timestamp if timestamp else time.time()
            ),
            measurement=measurement,
            prediction=float(self.H @ x_pred),
            innovation=float(innovation[0]),
            innovation_covariance=S_scalar,
            kalman_gain=K.copy(),
            is_valid=True,
            timestamp_ns=elapsed_ns
        )
    
    def _adapt_noise(self) -> None:
        """Adaptively tune process noise based on innovation statistics"""
        if self._volatility_estimate < 1e-10:
            return
        
        # Scale process noise with observed volatility
        scale_factor = 1.0 + min(self._volatility_estimate * 10, 5.0)
        
        # Update Q matrix
        base_q = self.Q_base[1, 1] / self.dt**2  # Extract base process noise
        adapted_q = base_q * scale_factor
        
        self.Q_base = np.array([
            [adapted_q * self.dt**4 / 4, adapted_q * self.dt**3 / 2],
            [adapted_q * self.dt**3 / 2, adapted_q * self.dt**2]
        ])
    
    def get_position(self) -> float:
        """Get current position estimate"""
        return float(self.x[0])
    
    def get_velocity(self) -> float:
        """Get current velocity estimate"""
        return float(self.x[1])
    
    def get_uncertainty(self) -> float:
        """Get position uncertainty (standard deviation)"""
        return float(np.sqrt(self.P[0, 0]))
    
    def reset(self, initial_value: float = 0.0) -> None:
        """Reset filter to initial state"""
        self.x = np.array([initial_value, 0.0])
        self.P = np.eye(2) * 0.5
        self._innovation_history.clear()
        self._volatility_estimate = 0.0


class AdaptiveKalmanFilter(KalmanFilter1D):
    """
    Kalman filter with automatic parameter adaptation.
    
    Uses innovation-based adaptation to handle regime changes
    in cryptocurrency markets.
    """
    
    def __init__(self,
                 initial_value: float = 0.0,
                 base_process_noise: float = 0.01,
                 base_measurement_noise: float = 0.1,
                 dt: float = 1.0,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive Kalman filter.
        
        Args:
            initial_value: Initial state estimate
            base_process_noise: Base process noise
            base_measurement_noise: Base measurement noise
            dt: Time step
            adaptation_rate: How quickly to adapt (0-1)
        """
        super().__init__(
            initial_value=initial_value,
            process_noise=base_process_noise,
            measurement_noise=base_measurement_noise,
            dt=dt
        )
        
        self.adaptation_rate = adaptation_rate
        self.base_R = base_measurement_noise
        self.base_Q = base_process_noise
        
        # Regime detection
        self._high_volatility_mode = False
        self._volatility_threshold = 0.05
    
    def update(self,
               measurement: float,
               timestamp: Optional[float] = None) -> KalmanResult:
        """Update with adaptive noise adjustment"""
        result = super().update(measurement, timestamp)
        
        # Check for regime change
        normalized_innovation = abs(result.innovation) / max(result.innovation_covariance, 1e-10)
        
        if normalized_innovation > 3.0:  # More than 3 sigma
            self._high_volatility_mode = True
            # Temporarily increase measurement noise to reject outliers
            self.R = self.base_R * (1.0 + normalized_innovation)
        else:
            # Gradually return to base noise
            self.R = self.base_R + self.adaptation_rate * (self.R - self.base_R)
            if abs(self.R - self.base_R) < 1e-6:
                self._high_volatility_mode = False
        
        return result
    
    def is_high_volatility(self) -> bool:
        """Check if filter is in high volatility mode"""
        return self._high_volatility_mode


class KalmanFilterBank:
    """
    Bank of parallel Kalman filters with different parameters.
    
    Uses weighted combination based on likelihood for robust estimation.
    """
    
    def __init__(self,
                 initial_value: float = 0.0,
                 n_filters: int = 5):
        """
        Initialize filter bank.
        
        Args:
            initial_value: Initial state estimate
            n_filters: Number of parallel filters
        """
        self.filters: List[KalmanFilter1D] = []
        self.weights = np.ones(n_filters) / n_filters
        
        # Create filters with different noise parameters
        for i in range(n_filters):
            q = 0.001 * (10 ** (i / (n_filters - 1) * 3))  # Log-spaced Q
            r = 0.01 * (10 ** ((n_filters - 1 - i) / (n_filters - 1) * 2))  # Log-spaced R
            
            kf = KalmanFilter1D(
                initial_value=initial_value,
                process_noise=q,
                measurement_noise=r
            )
            self.filters.append(kf)
    
    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update all filters and compute weighted estimate.
        
        Returns:
            Tuple of (estimated value, uncertainty)
        """
        innovations = []
        innovation_vars = []
        
        # Update each filter
        for kf in self.filters:
            result = kf.update(measurement)
            innovations.append(result.innovation)
            innovation_vars.append(result.innovation_covariance)
        
        # Update weights based on likelihood
        log_likelihoods = []
        for inn, var in zip(innovations, innovation_vars):
            # Gaussian log-likelihood
            ll = -0.5 * (np.log(2 * np.pi * var) + inn**2 / var)
            log_likelihoods.append(ll)
        
        # Convert to weights via softmax
        log_likelihoods = np.array(log_likelihoods)
        log_likelihoods -= np.max(log_likelihoods)  # Numerical stability
        likelihoods = np.exp(log_likelihoods)
        self.weights = likelihoods / np.sum(likelihoods)
        
        # Weighted combination of estimates
        estimates = np.array([kf.get_position() for kf in self.filters])
        uncertainties = np.array([kf.get_uncertainty() for kf in self.filters])
        
        combined_estimate = float(np.sum(self.weights * estimates))
        combined_uncertainty = float(np.sqrt(np.sum(self.weights * uncertainties**2)))
        
        return combined_estimate, combined_uncertainty
    
    def get_best_filter_index(self) -> int:
        """Get index of filter with highest weight"""
        return int(np.argmax(self.weights))


def smooth_price_series(prices: np.ndarray,
                        process_noise: float = 0.01,
                        measurement_noise: float = 0.1) -> np.ndarray:
    """
    Smooth a price series using Kalman filtering.
    
    Args:
        prices: Array of raw prices
        process_noise: Process noise parameter
        measurement_noise: Measurement noise parameter
        
    Returns:
        Array of smoothed prices
    """
    kf = KalmanFilter1D(
        initial_value=float(prices[0]),
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )
    
    smoothed = [kf.get_position()]
    
    for price in prices[1:]:
        result = kf.update(float(price))
        smoothed.append(kf.get_position())
    
    return np.array(smoothed)


if __name__ == "__main__":
    # Self-test
    np.random.seed(42)
    
    # Generate noisy signal
    n_samples = 100
    true_signal = np.sin(np.linspace(0, 4 * np.pi, n_samples))
    noise = np.random.randn(n_samples) * 0.3
    noisy_signal = true_signal + noise
    
    # Test basic Kalman filter
    kf = KalmanFilter1D(
        initial_value=noisy_signal[0],
        process_noise=0.01,
        measurement_noise=0.1
    )
    
    smoothed = [kf.get_position()]
    for price in noisy_signal[1:]:
        result = kf.update(float(price))
        smoothed.append(kf.get_position())
    
    smoothed = np.array(smoothed)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((smoothed - true_signal)**2))
    print(f"Kalman Filter RMSE: {rmse:.4f}")
    
    # Test adaptive filter
    akf = AdaptiveKalmanFilter(initial_value=noisy_signal[0])
    
    for price in noisy_signal:
        result = akf.update(float(price))
    
    print(f"High volatility mode: {akf.is_high_volatility()}")
    
    # Test filter bank
    kfb = KalmanFilterBank(initial_value=noisy_signal[0], n_filters=5)
    
    for price in noisy_signal:
        estimate, uncertainty = kfb.update(float(price))
    
    best_idx = kfb.get_best_filter_index()
    print(f"Best filter index: {best_idx}")
    print(f"Filter weights: {kfb.weights}")
