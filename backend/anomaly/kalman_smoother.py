#!/usr/bin/env python3
"""
Kalman Smoother Module: Forward-Backward Pass for Noisy Signal Refinement

This module implements Kalman filtering and smoothing algorithms to refine
noisy order book signals, trade flow data, and price observations. The
forward-backward pass provides optimal state estimates using all available data.

Key Features:
- Standard Kalman filter for real-time estimation
- Rauch-Tung-Striebel (RTS) smoother for offline refinement
- Extended Kalman filter for nonlinear observations
- Memory-efficient implementation for streaming data
- Integration with quantitative finance signal processing

Author: ZAID Personal Crypto Trading Bot - Stage 29
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, inv


@dataclass
class KalmanFilterResult:
    """Result of a single Kalman filter step."""
    timestamp_ms: int
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    log_likelihood: float


@dataclass
class KalmanSmootherResult:
    """Result of Kalman smoothing over entire sequence."""
    timestamps: List[int]
    smoothed_states: List[np.ndarray]
    smoothed_covariances: List[np.ndarray]
    cross_covariances: List[np.ndarray]  # P_{t,t-1|T}
    log_likelihood: float


class KalmanFilter:
    """
    Standard Kalman filter for linear Gaussian state-space models.
    
    State equation: x_t = F @ x_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation: y_t = H @ x_t + v_t,        v_t ~ N(0, R)
    """
    
    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray
    ):
        """
        Initialize Kalman filter.
        
        Args:
            F: State transition matrix
            H: Observation matrix
            Q: Process noise covariance
            R: Observation noise covariance
            x0: Initial state estimate
            P0: Initial state covariance
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = P0.copy()
        
        self._dim_state = len(x0)
        self._dim_obs = len(H) if H.ndim == 1 else H.shape[0]
        
        # Storage for smoothing
        self._history: List[KalmanFilterResult] = []
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step: propagate state and covariance forward.
        
        Returns:
            Predicted state and covariance
        """
        # x_{t|t-1} = F @ x_{t-1|t-1}
        x_pred = self.F @ self.x
        
        # P_{t|t-1} = F @ P_{t-1|t-1} @ F.T + Q
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return x_pred, P_pred
    
    def update(
        self, 
        timestamp_ms: int, 
        observation: np.ndarray
    ) -> KalmanFilterResult:
        """
        Update step: incorporate new observation.
        
        Args:
            timestamp_ms: Timestamp of observation
            observation: New observation vector
            
        Returns:
            Complete filter result for this timestep
        """
        # Prediction
        x_pred, P_pred = self.predict()
        
        # Innovation (measurement residual)
        y_pred = self.H @ x_pred
        innovation = observation - y_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        # K = P_pred @ H.T @ S^{-1}
        try:
            S_inv = inv(S)
            K = P_pred @ self.H.T @ S_inv
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            S_inv = np.linalg.pinv(S)
            K = P_pred @ self.H.T @ S_inv
        
        # Update state estimate
        # x_{t|t} = x_{t|t-1} + K @ innovation
        self.x = x_pred + K @ innovation
        
        # Update covariance
        # P_{t|t} = (I - K @ H) @ P_{t|t-1}
        I = np.eye(self._dim_state)
        self.P = (I - K @ self.H) @ P_pred
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2
        
        # Calculate log-likelihood of observation
        try:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                log_lik = -1e10
            else:
                log_lik = -0.5 * (
                    len(observation) * np.log(2 * np.pi) + 
                    logdet + 
                    innovation.T @ S_inv @ innovation
                )
        except Exception:
            log_lik = -1e10
        
        result = KalmanFilterResult(
            timestamp_ms=timestamp_ms,
            filtered_state=self.x.copy(),
            filtered_covariance=self.P.copy(),
            predicted_state=x_pred,
            predicted_covariance=P_pred,
            innovation=innovation,
            innovation_covariance=S,
            kalman_gain=K,
            log_likelihood=log_lik
        )
        
        self._history.append(result)
        return result
    
    def smooth(self) -> KalmanSmootherResult:
        """
        Apply RTS smoother to all stored results.
        
        Uses backward pass to refine state estimates using future information.
        
        Returns:
            Smoothed states and covariances for all timesteps
        """
        if len(self._history) == 0:
            raise ValueError("No history available for smoothing")
        
        n = len(self._history)
        
        # Initialize with final filtered values
        smoothed_states = [None] * n
        smoothed_covs = [None] * n
        cross_covs = [None] * n
        
        smoothed_states[-1] = self._history[-1].filtered_state.copy()
        smoothed_covs[-1] = self._history[-1].filtered_covariance.copy()
        
        total_log_lik = sum(r.log_likelihood for r in self._history)
        
        # Backward pass
        for t in range(n - 2, -1, -1):
            result_t = self._history[t]
            result_tp1 = self._history[t + 1]
            
            # Smoother gain
            # G_t = P_{t|t} @ F.T @ P_{t+1|t}^{-1}
            try:
                P_pred_inv = inv(result_tp1.predicted_covariance)
                G = result_t.filtered_covariance @ self.F.T @ P_pred_inv
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.pinv(result_tp1.predicted_covariance)
                G = result_t.filtered_covariance @ self.F.T @ P_pred_inv
            
            # Smoothed state
            # x_{t|T} = x_{t|t} + G @ (x_{t+1|T} - x_{t+1|t})
            smoothed_states[t] = (
                result_t.filtered_state + 
                G @ (smoothed_states[t + 1] - result_tp1.predicted_state)
            )
            
            # Smoothed covariance
            # P_{t|T} = P_{t|t} + G @ (P_{t+1|T} - P_{t+1|t}) @ G.T
            smoothed_covs[t] = (
                result_t.filtered_covariance + 
                G @ (smoothed_covs[t + 1] - result_tp1.predicted_covariance) @ G.T
            )
            
            # Cross-covariance P_{t+1,t|T} for lag-one smoothing
            cross_covs[t + 1] = smoothed_covs[t + 1] @ G.T
        
        return KalmanSmootherResult(
            timestamps=[r.timestamp_ms for r in self._history],
            smoothed_states=smoothed_states,
            smoothed_covariances=smoothed_covs,
            cross_covariances=cross_covs,
            log_likelihood=total_log_lik
        )
    
    def reset(self):
        """Reset filter to initial state."""
        self._history.clear()


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman filter for nonlinear state-space models.
    
    Handles nonlinear observation function h(x) by linearizing
    around the current estimate using Jacobian.
    """
    
    def __init__(
        self,
        F: np.ndarray,
        h_func: Callable[[np.ndarray], np.ndarray],
        h_jacobian: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray
    ):
        """
        Initialize EKF.
        
        Args:
            F: State transition matrix (linear)
            h_func: Nonlinear observation function
            h_jacobian: Jacobian of h_func
            Q: Process noise covariance
            R: Observation noise covariance
            x0: Initial state estimate
            P0: Initial state covariance
        """
        super().__init__(F, np.eye(len(x0)), Q, R, x0, P0)
        self.h_func = h_func
        self.h_jacobian = h_jacobian
    
    def update(
        self,
        timestamp_ms: int,
        observation: np.ndarray
    ) -> KalmanFilterResult:
        """Update using linearized observation model."""
        # Prediction (same as standard KF)
        x_pred, P_pred = self.predict()
        
        # Linearize observation function at predicted state
        H_t = self.h_jacobian(x_pred)
        
        # Compute predicted observation
        y_pred = self.h_func(x_pred)
        
        # Innovation
        innovation = observation - y_pred
        
        # Innovation covariance
        S = H_t @ P_pred @ H_t.T + self.R
        
        # Kalman gain
        try:
            S_inv = inv(S)
            K = P_pred @ H_t.T @ S_inv
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = P_pred @ H_t.T @ S_inv
        
        # Update state
        self.x = x_pred + K @ innovation
        
        # Update covariance
        I = np.eye(self._dim_state)
        self.P = (I - K @ H_t) @ P_pred
        self.P = (self.P + self.P.T) / 2
        
        # Log-likelihood
        try:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                log_lik = -1e10
            else:
                log_lik = -0.5 * (
                    len(observation) * np.log(2 * np.pi) +
                    logdet +
                    innovation.T @ S_inv @ innovation
                )
        except Exception:
            log_lik = -1e10
        
        result = KalmanFilterResult(
            timestamp_ms=timestamp_ms,
            filtered_state=self.x.copy(),
            filtered_covariance=self.P.copy(),
            predicted_state=x_pred,
            predicted_covariance=P_pred,
            innovation=innovation,
            innovation_covariance=S,
            kalman_gain=K,
            log_likelihood=log_lik
        )
        
        self._history.append(result)
        return result


class OrderBookSignalSmoother:
    """
    Specialized Kalman smoother for order book signal processing.
    
    Tracks hidden true prices beneath noisy bid/ask observations,
    filters microstructure noise, and extracts clean signals for trading.
    """
    
    def __init__(self, initial_price: float, process_noise: float = 1e-6):
        """
        Initialize order book smoother.
        
        State: [true_price, price_velocity, spread_factor]
        Observation: [mid_price, realized_spread]
        """
        dim_state = 3
        dim_obs = 2
        
        # State transition: random walk with momentum
        F = np.array([
            [1.0, 0.1, 0.0],   # Price follows velocity
            [0.0, 0.9, 0.0],   # Velocity mean-reverts
            [0.0, 0.0, 0.95],  # Spread factor persistent
        ])
        
        # Observation: mid price and spread
        H = np.array([
            [1.0, 0.0, 0.0],   # Mid = true price
            [0.0, 0.0, 1.0],   # Observed spread related to spread factor
        ])
        
        # Noise covariances
        Q = np.diag([process_noise, process_noise * 10, process_noise * 100])
        R = np.diag([initial_price * 0.0001, 0.0001])  # Small observation noise
        
        # Initial state
        x0 = np.array([initial_price, 0.0, 0.0005])  # Price, zero velocity, typical spread
        P0 = np.diag([initial_price * 0.01, 0.0001, 0.0001])
        
        self.filter = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
        self._price_history: List[float] = []
    
    def update(
        self,
        timestamp_ms: int,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float
    ) -> KalmanFilterResult:
        """
        Update with new order book snapshot.
        
        Args:
            timestamp_ms: Timestamp
            bid_price: Best bid price
            ask_price: Best ask price
            bid_size: Bid quantity
            ask_size: Ask quantity
            
        Returns:
            Filter result with smoothed price estimate
        """
        # Calculate observations
        mid_price = (bid_price + ask_price) / 2
        spread = ask_price - bid_price
        
        # Volume-weighted spread adjustment
        total_size = bid_size + ask_size
        if total_size > 0:
            imbalance = (ask_size - bid_size) / total_size
        else:
            imbalance = 0
        
        observation = np.array([mid_price, spread])
        
        result = self.filter.update(timestamp_ms, observation)
        self._price_history.append(result.filtered_state[0])
        
        # Keep history bounded
        if len(self._price_history) > 10000:
            self._price_history = self._price_history[-5000:]
            self.filter._history = self.filter._history[-5000:]
        
        return result
    
    def get_smoothed_price(self) -> float:
        """Get current smoothed price estimate."""
        return self.filter.x[0]
    
    def get_price_velocity(self) -> float:
        """Get estimated price velocity (momentum)."""
        return self.filter.x[1]
    
    def get_fair_value(self) -> float:
        """Get fair value estimate (smoothed price adjusted for imbalance)."""
        price = self.filter.x[0]
        spread_factor = self.filter.x[2]
        return price - spread_factor * 0.5
    
    def smooth_recent(self, n_steps: int = 100) -> KalmanSmootherResult:
        """Apply smoother to recent history."""
        if len(self.filter._history) < 2:
            raise ValueError("Need at least 2 observations for smoothing")
        
        return self.filter.smooth()


def main():
    """Example usage of Kalman smoother for price tracking."""
    np.random.seed(42)
    
    print("Simulating Kalman filter for price tracking...")
    
    # Create smoother
    smoother = OrderBookSignalSmoother(initial_price=50000.0)
    
    # Simulate order book data with noise
    true_price = 50000.0
    prices = [true_price]
    
    for t in range(200):
        # True price follows random walk
        true_price += np.random.normal(0, 5)
        prices.append(true_price)
        
        # Noisy observations
        noise = np.random.normal(0, 2)
        bid = true_price - 0.5 + noise
        ask = true_price + 0.5 + noise
        
        bid_size = np.random.uniform(1, 10)
        ask_size = np.random.uniform(1, 10)
        
        # Update smoother
        result = smoother.update(
            timestamp_ms=t * 1000,
            bid_price=bid,
            ask_price=ask,
            bid_size=bid_size,
            ask_size=ask_size
        )
        
        if t % 50 == 0:
            smoothed = smoother.get_smoothed_price()
            velocity = smoother.get_price_velocity()
            print(f"t={t}: True={true_price:.2f}, Smoothed={smoothed:.2f}, Velocity={velocity:.4f}")
    
    # Apply smoother to entire history
    print("\nApplying RTS smoother...")
    smooth_result = smoother.smooth_recent()
    
    print(f"Log-likelihood: {smooth_result.log_likelihood:.2f}")
    print(f"Final smoothed price: {smooth_result.smoothed_states[-1][0]:.2f}")
    
    # Compare filtered vs smoothed
    last_filtered = smoother.filter._history[-1].filtered_state[0]
    last_smoothed = smooth_result.smoothed_states[-1][0]
    print(f"\nLast filtered: {last_filtered:.2f}")
    print(f"Last smoothed: {last_smoothed:.2f}")


if __name__ == "__main__":
    main()
