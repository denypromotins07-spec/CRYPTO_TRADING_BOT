#!/usr/bin/env python3
"""
Treatment Effect Estimation for News Events

This module estimates Conditional Average Treatment Effects (CATE) of news
events on crypto markets in the ZAID PERSONAL CRYPTO TRADING BOT. Uses
causal ML techniques to estimate heterogeneous treatment effects.

Features:
- CATE estimation via causal forests and meta-learners
- Propensity score matching for observational data
- Doubly robust estimation combining outcome modeling and propensity scores
- Real-time news impact quantification
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreatmentEffectResult:
    """Result of treatment effect estimation."""
    ate: float  # Average Treatment Effect
    ate_ci: Tuple[float, float]  # Confidence interval
    cate_features: Dict[str, float]  # Feature-specific CATEs
    standard_error: float
    n_treated: int
    n_control: int
    method: str


class BaseMetaLearner(ABC):
    """Base class for meta-learner CATE estimators."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'BaseMetaLearner':
        """Fit the meta-learner."""
        pass
    
    @abstractmethod
    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate conditional average treatment effects."""
        pass
    
    @abstractmethod
    def estimate_ate(self, X: np.ndarray) -> float:
        """Estimate average treatment effect."""
        pass


class SLLearner(BaseMetaLearner):
    """
    S-Learner: Single model approach.
    
    Fits a single model mu(X, T) and estimates CATE as:
    CATE(x) = mu(x, T=1) - mu(x, T=0)
    """
    
    def __init__(self, base_model=None):
        self.base_model = base_model or RandomForestRegressor(n_estimators=100)
        self.fitted_model = None
        
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'SLLearner':
        # Concatenate features and treatment
        XT = np.column_stack([X, T])
        self.fitted_model = self.base_model.fit(XT, Y)
        return self
    
    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Model not fitted")
        
        # Predict under treatment and control
        T1 = np.ones(len(X))
        T0 = np.zeros(len(X))
        
        XT1 = np.column_stack([X, T1])
        XT0 = np.column_stack([X, T0])
        
        mu1 = self.fitted_model.predict(XT1)
        mu0 = self.fitted_model.predict(XT0)
        
        return mu1 - mu0
    
    def estimate_ate(self, X: np.ndarray) -> float:
        cate = self.estimate_cate(X)
        return np.mean(cate)


class TLearner(BaseMetaLearner):
    """
    T-Learner: Two-model approach.
    
    Fits separate models for treated and control:
    mu_1(X) for treated, mu_0(X) for control
    CATE(x) = mu_1(x) - mu_0(x)
    """
    
    def __init__(self, treatment_model=None, control_model=None):
        self.treatment_model = treatment_model or RandomForestRegressor(n_estimators=100)
        self.control_model = control_model or RandomForestRegressor(n_estimators=100)
        self.fitted_treatment = None
        self.fitted_control = None
        
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'TLearner':
        # Split by treatment status
        treated_mask = T == 1
        control_mask = T == 0
        
        X_treated = X[treated_mask]
        Y_treated = Y[treated_mask]
        X_control = X[control_mask]
        Y_control = Y[control_mask]
        
        # Fit separate models
        if len(X_treated) > 0:
            self.fitted_treatment = self.treatment_model.fit(X_treated, Y_treated)
        if len(X_control) > 0:
            self.fitted_control = self.control_model.fit(X_control, Y_control)
        
        return self
    
    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        if self.fitted_treatment is None or self.fitted_control is None:
            raise ValueError("Model not fitted")
        
        mu1 = self.fitted_treatment.predict(X)
        mu0 = self.fitted_control.predict(X)
        
        return mu1 - mu0
    
    def estimate_ate(self, X: np.ndarray) -> float:
        cate = self.estimate_cate(X)
        return np.mean(cate)


class XLearner(BaseMetaLearner):
    """
    X-Learner: Cross-fitting approach for better performance with imbalanced treatment.
    
    Steps:
    1. Fit mu_1(X) and mu_0(X) like T-learner
    2. Impute individual treatment effects
    3. Fit CATE models on imputed effects
    4. Combine using propensity scores
    """
    
    def __init__(
        self,
        treatment_model=None,
        control_model=None,
        cate_model=None,
        propensity_model=None
    ):
        self.treatment_model = treatment_model or RandomForestRegressor(n_estimators=100)
        self.control_model = control_model or RandomForestRegressor(n_estimators=100)
        self.cate_model_t = cate_model or RandomForestRegressor(n_estimators=50)
        self.cate_model_c = cate_model or RandomForestRegressor(n_estimators=50)
        self.propensity_model = propensity_model or LogisticRegression()
        
        self.fitted_treatment = None
        self.fitted_control = None
        self.fitted_cate_t = None
        self.fitted_cate_c = None
        self.fitted_propensity = None
        
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'XLearner':
        treated_mask = T == 1
        control_mask = T == 0
        
        X_treated = X[treated_mask]
        Y_treated = Y[treated_mask]
        X_control = X[control_mask]
        Y_control = Y[control_mask]
        
        # Step 1: Fit outcome models
        if len(X_treated) > 0:
            self.fitted_treatment = self.treatment_model.fit(X_treated, Y_treated)
        if len(X_control) > 0:
            self.fitted_control = self.control_model.fit(X_control, Y_control)
        
        # Step 2: Impute individual treatment effects
        # D_t = Y_t - mu_0(X_t)  (effect for treated units)
        # D_c = mu_1(X_c) - Y_c  (effect for control units)
        if self.fitted_control is not None and len(X_treated) > 0:
            D_treated = Y_treated - self.fitted_control.predict(X_treated)
            self.fitted_cate_t = self.cate_model_t.fit(X_treated, D_treated)
        
        if self.fitted_treatment is not None and len(X_control) > 0:
            D_control = self.fitted_treatment.predict(X_control) - Y_control
            self.fitted_cate_c = self.cate_model_c.fit(X_control, D_control)
        
        # Step 3: Fit propensity model
        self.fitted_propensity = self.propensity_model.fit(T.reshape(-1, 1), T)
        
        return self
    
    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        if self.fitted_cate_t is None or self.fitted_cate_c is None:
            raise ValueError("Model not fitted")
        
        # Get propensity scores
        try:
            e_X = self.fitted_propensity.predict_proba(X[:, :1])[:, 1]
        except:
            e_X = np.full(len(X), 0.5)  # Default if propensity fails
        
        # Get CATE estimates from both models
        tau_t = self.fitted_cate_t.predict(X)
        tau_c = self.fitted_cate_c.predict(X)
        
        # Combine using propensity weights
        return e_X * tau_t + (1 - e_X) * tau_c
    
    def estimate_ate(self, X: np.ndarray) -> float:
        cate = self.estimate_cate(X)
        return np.mean(cate)


class DoublyRobustEstimator:
    """
    Doubly Robust estimator combining outcome modeling and inverse propensity weighting.
    
    Provides consistent estimates if EITHER the outcome model OR the propensity model
    is correctly specified.
    """
    
    def __init__(
        self,
        outcome_model=None,
        propensity_model=None
    ):
        self.outcome_model = outcome_model or GradientBoostingRegressor(n_estimators=100)
        self.propensity_model = propensity_model or LogisticRegression()
        
        self.fitted_outcome_t = None
        self.fitted_outcome_c = None
        self.fitted_propensity = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'DoublyRobustEstimator':
        treated_mask = T == 1
        control_mask = T == 0
        
        # Fit outcome models
        if np.sum(treated_mask) > 0:
            self.fitted_outcome_t = self.outcome_model.fit(X[treated_mask], Y[treated_mask])
        if np.sum(control_mask) > 0:
            self.fitted_outcome_c = self.outcome_model.fit(X[control_mask], Y[control_mask])
        
        # Fit propensity model
        self.fitted_propensity = self.propensity_model.fit(X, T)
        
        return self
    
    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> TreatmentEffectResult:
        """Estimate ATE with doubly robust estimator."""
        n = len(X)
        
        # Get propensity scores
        e_X = self.fitted_propensity.predict_proba(X)[:, 1]
        e_X = np.clip(e_X, 0.01, 0.99)  # Clip for stability
        
        # Get outcome predictions
        mu1 = self.fitted_outcome_t.predict(X) if self.fitted_outcome_t is not None else np.zeros(n)
        mu0 = self.fitted_outcome_c.predict(X) if self.fitted_outcome_c is not None else np.zeros(n)
        
        # Doubly robust estimator
        # DR = mu1 - mu0 + (T/e(X))*(Y - mu1) - ((1-T)/(1-e(X)))*(Y - mu0)
        dr_terms = (
            mu1 - mu0 +
            (T / e_X) * (Y - mu1) -
            ((1 - T) / (1 - e_X)) * (Y - mu0)
        )
        
        ate = np.mean(dr_terms)
        
        # Bootstrap for confidence interval
        bootstrap_ates = []
        for _ in range(500):
            idx = np.random.choice(n, n, replace=True)
            bootstrap_ates.append(np.mean(dr_terms[idx]))
        
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        se = np.std(bootstrap_ates)
        
        # Estimate CATE by feature
        cate_features = {}
        for i in range(min(X.shape[1], 10)):  # Limit to first 10 features
            feature_name = f"feature_{i}"
            # Simple approximation: correlate feature with individual TE
            individual_te = mu1 - mu0
            cate_features[feature_name] = np.corrcoef(X[:, i], individual_te)[0, 1]
        
        return TreatmentEffectResult(
            ate=ate,
            ate_ci=(ci_lower, ci_upper),
            cate_features=cate_features,
            standard_error=se,
            n_treated=int(np.sum(T)),
            n_control=int(np.sum(1 - T)),
            method="DoublyRobust"
        )


def estimate_news_impact(
    news_features: np.ndarray,
    treatment_indicator: np.ndarray,
    price_returns: np.ndarray,
    volume_changes: np.ndarray
) -> Dict[str, TreatmentEffectResult]:
    """
    Estimate the causal impact of news events on crypto prices.
    
    Args:
        news_features: Features extracted from news (sentiment, topic, etc.)
        treatment_indicator: Binary indicator of news occurrence
        price_returns: Log returns of the asset
        volume_changes: Changes in trading volume
        
    Returns:
        Dictionary with treatment effects on different outcomes
    """
    results = {}
    
    # Estimate effect on price returns
    dr_price = DoublyRobustEstimator()
    dr_price.fit(news_features, treatment_indicator, price_returns)
    results['price_effect'] = dr_price.estimate_ate(
        news_features, treatment_indicator, price_returns
    )
    
    # Estimate effect on volume
    dr_volume = DoublyRobustEstimator()
    dr_volume.fit(news_features, treatment_indicator, volume_changes)
    results['volume_effect'] = dr_volume.estimate_ate(
        news_features, treatment_indicator, volume_changes
    )
    
    return results


# Example usage
if __name__ == "__main__":
    # Simulate data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: sentiment score, news category, time of day, market state
    X = np.random.randn(n_samples, 4)
    
    # Treatment: positive news event
    propensity = 1 / (1 + np.exp(-X[:, 0]))  # Higher sentiment -> more likely treatment
    T = (np.random.rand(n_samples) < propensity).astype(int)
    
    # Outcome: price returns with heterogeneous treatment effect
    true_cate = 0.02 * X[:, 0] + 0.01 * X[:, 1]  # Effect varies by sentiment and category
    noise = np.random.randn(n_samples) * 0.01
    Y = 0.001 + true_cate * T + noise
    
    # Fit X-Learner
    xlearner = XLearner()
    xlearner.fit(X, T, Y)
    
    cate_estimates = xlearner.estimate_cate(X)
    ate_estimate = xlearner.estimate_ate(X)
    
    print(f"\nX-Learner Results:")
    print(f"  Estimated ATE: {ate_estimate:.6f}")
    print(f"  True ATE (approx): {np.mean(true_cate):.6f}")
    print(f"  CATE range: [{cate_estimates.min():.6f}, {cate_estimates.max():.6f}]")
    
    # Fit Doubly Robust estimator
    dr = DoublyRobustEstimator()
    dr.fit(X, T, Y)
    result = dr.estimate_ate(X, T, Y)
    
    print(f"\nDoubly Robust Results:")
    print(f"  ATE: {result.ate:.6f}")
    print(f"  95% CI: ({result.ate_ci[0]:.6f}, {result.ate_ci[1]:.6f})")
    print(f"  N treated: {result.n_treated}, N control: {result.n_control}")
    print(f"  Method: {result.method}")
