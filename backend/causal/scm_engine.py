#!/usr/bin/env python3
"""
Structural Causal Model (SCM) Engine

This module defines Structural Causal Models for order flow shocks
in the ZAID PERSONAL CRYPTO TRADING BOT. SCMs provide the mathematical
framework for counterfactual reasoning and causal effect estimation.

Features:
- Definition of structural equations for crypto market variables
- Support for exogenous noise modeling
- Counterfactual computation via abduction-action-prediction
- Delayed treatment effect modeling for Fed announcements
"""

from __future__ import annotations
from typing import Dict, Set, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class StructuralEquation:
    """
    Represents a structural equation in the SCM.
    
    X_i = f_i(PA_i, U_i, epsilon_i)
    
    where:
    - PA_i are the parent variables (direct causes)
    - U_i are unobserved exogenous variables
    - epsilon_i is independent noise
    """
    variable: str
    parents: List[str]
    function: Callable[..., float]
    exogenous_vars: List[str] = field(default_factory=list)
    noise_distribution: str = "normal"
    noise_params: Tuple[float, float] = (0.0, 1.0)  # mean, std
    
    def evaluate(
        self,
        parent_values: Dict[str, float],
        exogenous_values: Optional[Dict[str, float]] = None,
        noise_value: Optional[float] = None
    ) -> float:
        """Evaluate the structural equation given parent values."""
        # Generate noise if not provided
        if noise_value is None:
            if self.noise_distribution == "normal":
                noise_value = np.random.normal(*self.noise_params)
            elif self.noise_distribution == "uniform":
                noise_value = np.random.uniform(*self.noise_params)
            elif self.noise_distribution == "exponential":
                noise_value = np.random.exponential(self.noise_params[1])
        
        # Call the structural function
        kwargs = {**parent_values}
        if exogenous_values:
            kwargs.update(exogenous_values)
        kwargs['epsilon'] = noise_value
        
        return self.function(**kwargs)


class SCM:
    """
    Structural Causal Model representing a system of causal relationships.
    
    An SCM is defined as a tuple <U, V, F, P(U)> where:
    - U: Exogenous variables (outside the model)
    - V: Endogenous variables (determined by the model)
    - F: Structural equations mapping parents to children
    - P(U): Distribution over exogenous variables
    """
    
    def __init__(self, name: str = "CryptoMarketSCM"):
        self.name = name
        self.equations: Dict[str, StructuralEquation] = {}
        self.exogenous_distributions: Dict[str, Tuple[str, Tuple]] = {}
        self.causal_graph: Dict[str, Set[str]] = {}  # child -> parents
        self.reverse_graph: Dict[str, Set[str]] = {}  # parent -> children
        
    def add_equation(self, equation: StructuralEquation) -> None:
        """Add a structural equation to the model."""
        var = equation.variable
        self.equations[var] = equation
        
        # Update causal graph
        self.causal_graph[var] = set(equation.parents)
        for parent in equation.parents:
            if parent not in self.reverse_graph:
                self.reverse_graph[parent] = set()
            self.reverse_graph[parent].add(var)
        
        # Register exogenous distributions
        for exo_var in equation.exogenous_vars:
            if exo_var not in self.exogenous_distributions:
                self.exogenous_distributions[exo_var] = (
                    equation.noise_distribution,
                    equation.noise_params
                )
    
    def get_topological_order(self) -> List[str]:
        """Get variables in topological order for sequential evaluation."""
        visited = set()
        order = []
        
        def visit(var: str):
            if var in visited:
                return
            visited.add(var)
            
            # Visit parents first
            for parent in self.causal_graph.get(var, set()):
                visit(parent)
            
            order.append(var)
        
        for var in self.equations:
            visit(var)
        
        return order
    
    def sample(
        self,
        n_samples: int = 1,
        interventions: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample from the SCM (possibly under intervention).
        
        Args:
            n_samples: Number of samples to draw
            interventions: Dictionary of do() interventions
            
        Returns:
            Dictionary mapping variable names to sampled values
        """
        interventions = interventions or {}
        results: Dict[str, List[float]] = {var: [] for var in self.equations}
        
        for _ in range(n_samples):
            sample = {}
            
            # Evaluate in topological order
            for var in self.get_topological_order():
                if var in interventions:
                    # Apply do() intervention
                    sample[var] = interventions[var]
                else:
                    eq = self.equations[var]
                    parent_vals = {p: sample[p] for p in eq.parents if p in sample}
                    exo_vals = {}
                    
                    sample[var] = eq.evaluate(parent_vals, exo_vals)
                
                results[var].append(sample[var])
        
        return {k: np.array(v) for k, v in results.items()}
    
    def compute_counterfactual(
        self,
        observed: Dict[str, float],
        intervention: Dict[str, float],
        query_var: str
    ) -> float:
        """
        Compute a counterfactual: What would Y have been if X had been x?
        
        Uses the three-step process:
        1. Abduction: Infer exogenous variables from observations
        2. Action: Apply intervention to structural equations
        3. Prediction: Compute outcome under modified model
        
        Args:
            observed: Observed values of variables
            intervention: do() interventions to apply
            query_var: Variable to compute counterfactual for
            
        Returns:
            Counterfactual value of query variable
        """
        # Step 1: Abduction - infer noise terms
        inferred_noise: Dict[str, float] = {}
        
        for var in self.get_topological_order():
            if var not in observed:
                continue
                
            eq = self.equations[var]
            parent_vals = {p: observed[p] for p in eq.parents if p in observed}
            
            # Solve for noise: observed = f(parents, noise)
            # This requires invertibility; using numerical inversion
            actual_value = observed[var]
            predicted_without_noise = eq.evaluate(parent_vals, {}, noise_value=0.0)
            inferred_noise[var] = actual_value - predicted_without_noise
        
        # Step 2 & 3: Action + Prediction
        counterfactual_sample = {}
        
        for var in self.get_topological_order():
            if var in intervention:
                # Apply intervention
                counterfactual_sample[var] = intervention[var]
            else:
                eq = self.equations[var]
                parent_vals = {
                    p: counterfactual_sample[p] 
                    for p in eq.parents 
                    if p in counterfactual_sample
                }
                
                # Use inferred noise
                noise = inferred_noise.get(var, 0.0)
                counterfactual_sample[var] = eq.evaluate(
                    parent_vals, {}, noise_value=noise
                )
        
        return counterfactual_sample.get(query_var, float('nan'))
    
    def estimate_causal_effect(
        self,
        treatment: str,
        outcome: str,
        n_samples: int = 10000,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> Dict[str, float]:
        """
        Estimate the average causal effect (ACE) of treatment on outcome.
        
        ACE = E[Y | do(X=1)] - E[Y | do(X=0)]
        
        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name
            n_samples: Number of Monte Carlo samples
            treatment_values: Values for treated and control conditions
            
        Returns:
            Dictionary with ACE estimate and confidence interval
        """
        # Sample under do(X = treatment_values[1])
        treated_samples = self.sample(n_samples, interventions={treatment: treatment_values[1]})
        y_treated = treated_samples[outcome].mean()
        
        # Sample under do(X = treatment_values[0])
        control_samples = self.sample(n_samples, interventions={treatment: treatment_values[0]})
        y_control = control_samples[outcome].mean()
        
        ace = y_treated - y_control
        
        # Bootstrap for confidence interval
        bootstrap_aces = []
        for _ in range(1000):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            ace_boot = (
                treated_samples[outcome][idx].mean() - 
                control_samples[outcome][idx].mean()
            )
            bootstrap_aces.append(ace_boot)
        
        ci_lower = np.percentile(bootstrap_aces, 2.5)
        ci_upper = np.percentile(bootstrap_aces, 97.5)
        
        return {
            'ace': ace,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'y_treated': y_treated,
            'y_control': y_control
        }


class DelayedTreatmentSCM(SCM):
    """
    SCM extension for modeling delayed treatment effects.
    
    Particularly useful for modeling Federal Reserve announcements
    where the effect on crypto markets unfolds over time.
    """
    
    def __init__(self, base_scm: SCM, delay_steps: int = 5):
        super().__init__(f"{base_scm.name}_Delayed")
        self.base_scm = base_scm
        self.delay_steps = delay_steps
        self._setup_delayed_equations()
    
    def _setup_delayed_equations(self):
        """Create time-unrolled equations for delayed effects."""
        # Copy base equations with lag structure
        for var, eq in self.base_scm.equations.items():
            for t in range(self.delay_steps + 1):
                delayed_var = f"{var}_t{t}"
                delayed_parents = [
                    f"{p}_t{max(0, t-1)}" for p in eq.parents
                ]
                
                # Modify function to include decay factor
                def make_delayed_func(base_func, decay=0.8):
                    def delayed_func(**kwargs):
                        # Extract current-time values
                        current_kwargs = {
                            k.replace(f'_t{t}', ''): v 
                            for k, v in kwargs.items()
                        }
                        base_result = base_func(**current_kwargs)
                        
                        # Add persistence from previous timestep
                        prev_var = f"{eq.variable}_t{t-1}"
                        if prev_var in kwargs and t > 0:
                            base_result = decay * kwargs[prev_var] + (1-decay) * base_result
                        
                        return base_result
                    return delayed_func
                
                delayed_eq = StructuralEquation(
                    variable=delayed_var,
                    parents=delayed_parents,
                    function=make_delayed_func(eq.function),
                    exogenous_vars=eq.exogenous_vars,
                    noise_distribution=eq.noise_distribution,
                    noise_params=eq.noise_params
                )
                
                self.add_equation(delayed_eq)
    
    def simulate_announcement_effect(
        self,
        announcement_magnitude: float,
        announcement_time: int = 0
    ) -> Dict[str, List[float]]:
        """
        Simulate the effect of an announcement over time.
        
        Args:
            announcement_magnitude: Size of the shock
            announcement_time: Time step when announcement occurs
            
        Returns:
            Time series of variable values post-announcement
        """
        results: Dict[str, List[float]] = {
            f"{var}_t{t}": [] 
            for var in self.base_scm.equations 
            for t in range(self.delay_steps + 1)
        }
        
        # Run simulation with intervention at announcement time
        interventions = {}
        for var in self.base_scm.equations:
            interventions[f"{var}_t{announcement_time}"] = announcement_magnitude
        
        final_state = self.sample(1, interventions=interventions)
        
        for key, value in final_state.items():
            results[key].append(value[0])
        
        return results


def create_crypto_order_flow_scm() -> SCM:
    """
    Create a standard SCM for crypto order flow analysis.
    
    Variables:
    - OrderImbalance: Net buying pressure
    - PriceChange: Log returns
    - Volatility: Realized volatility
    - Volume: Trading volume
    - Sentiment: Market sentiment index
    """
    scm = SCM("CryptoOrderFlow")
    
    # Order Imbalance equation
    scm.add_equation(StructuralEquation(
        variable="OrderImbalance",
        parents=["Sentiment"],
        function=lambda Sentiment, epsilon: 0.5 * Sentiment + epsilon,
        noise_params=(0.0, 0.1)
    ))
    
    # Price Change equation
    scm.add_equation(StructuralEquation(
        variable="PriceChange",
        parents=["OrderImbalance", "Volatility"],
        function=lambda OrderImbalance, Volatility, epsilon: 
            0.3 * OrderImbalance - 0.1 * Volatility + epsilon,
        noise_params=(0.0, 0.02)
    ))
    
    # Volatility equation
    scm.add_equation(StructuralEquation(
        variable="Volatility",
        parents=["Volume"],
        function=lambda Volume, epsilon: 0.2 * np.log(Volume + 1) + epsilon,
        noise_params=(0.0, 0.05)
    ))
    
    # Volume equation
    scm.add_equation(StructuralEquation(
        variable="Volume",
        parents=["Sentiment"],
        function=lambda Sentiment, epsilon: 1000 + 50 * Sentiment + epsilon,
        noise_params=(0.0, 100)
    ))
    
    # Sentiment equation (exogenous)
    scm.add_equation(StructuralEquation(
        variable="Sentiment",
        parents=[],
        function=lambda epsilon: epsilon,
        noise_params=(0.0, 1.0)
    ))
    
    return scm


if __name__ == "__main__":
    # Example usage
    scm = create_crypto_order_flow_scm()
    
    # Sample from the model
    samples = scm.sample(n_samples=1000)
    print("Sample statistics:")
    for var, values in samples.items():
        print(f"  {var}: mean={values.mean():.4f}, std={values.std():.4f}")
    
    # Estimate causal effect of Sentiment on PriceChange
    effect = scm.estimate_causal_effect(
        treatment="Sentiment",
        outcome="PriceChange",
        treatment_values=(-1.0, 1.0)
    )
    print(f"\nCausal Effect of Sentiment on PriceChange:")
    print(f"  ACE: {effect['ace']:.6f}")
    print(f"  95% CI: [{effect['ci_lower']:.6f}, {effect['ci_upper']:.6f}]")
    
    # Compute counterfactual
    observed = {
        "Sentiment": 0.5,
        "OrderImbalance": 0.3,
        "Volume": 1100,
        "Volatility": 0.15,
        "PriceChange": 0.02
    }
    
    counterfactual = scm.compute_counterfactual(
        observed=observed,
        intervention={"Sentiment": -0.5},
        query_var="PriceChange"
    )
    print(f"\nCounterfactual: If Sentiment had been -0.5 instead of 0.5,")
    print(f"  PriceChange would have been: {counterfactual:.6f}")
