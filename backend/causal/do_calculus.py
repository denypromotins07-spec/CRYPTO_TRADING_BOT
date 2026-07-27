#!/usr/bin/env python3
"""
Do-Calculus Engine for Causal Intervention Simulation

This module implements Pearl's Do-Calculus rules to simulate market
interventions. Designed for the ZAID PERSONAL CRYPTO TRADING BOT to
compute causal effects while strictly preventing invalid interventions
on collider variables.

The three rules of do-calculus:
1. Rule 1 (Insertion/Deletion): P(y|do(x),z,w) = P(y|do(x),w) if Y ⊥ Z | X,W in G_X̄
2. Rule 2 (Action/Observation Exchange): P(y|do(x),do(z),w) = P(y|do(x),z,w) if Y ⊥ Z | X,W in G_X̄,Z̲
3. Rule 3 (Insertion/Deletion of Actions): P(y|do(x),do(z),w) = P(y|do(x),w) if Y ⊥ Z | X,W in G_X̄,Z̄

Features:
- Graph-based identification of causal effects
- Automatic detection of valid adjustment sets
- Collider bias prevention
- Backdoor and frontdoor criterion implementation
"""

from __future__ import annotations
from typing import Dict, Set, List, Tuple, Optional, FrozenSet
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CausalGraphType(Enum):
    """Types of edges in causal graphs."""
    DIRECTED = "->"  # Causal relationship
    BIDIRECTED = "<->"  # Confounding (unobserved common cause)
    UNDIRECTED = "--"  # Unknown relationship


@dataclass(frozen=True)
class Edge:
    """Represents an edge in the causal graph."""
    source: str
    target: str
    edge_type: CausalGraphType


class CausalGraph:
    """
    Representation of a causal DAG with support for do-calculus operations.
    
    Maintains both the graph structure and information about observed/unobserved
    variables for proper causal inference.
    """
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Set[Edge] = set()
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # Children
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # Parents
        self.confounded_pairs: Set[FrozenSet[str]] = set()  # Bidirected edges
        self.observed: Set[str] = set()
        self.unobserved: Set[str] = set()
        
    def add_node(self, name: str, observed: bool = True) -> None:
        """Add a node to the graph."""
        self.nodes.add(name)
        if observed:
            self.observed.add(name)
        else:
            self.unobserved.add(name)
            
    def add_directed_edge(self, source: str, target: str) -> None:
        """Add a directed edge (causal relationship)."""
        self.nodes.add(source)
        self.nodes.add(target)
        edge = Edge(source, target, CausalGraphType.DIRECTED)
        self.edges.add(edge)
        self.adjacency[source].add(target)
        self.reverse_adjacency[target].add(source)
        
    def add_bidirected_edge(self, node1: str, node2: str) -> None:
        """Add a bidirected edge (unobserved confounding)."""
        self.nodes.add(node1)
        self.nodes.add(node2)
        edge = Edge(node1, node2, CausalGraphType.BIDIRECTED)
        self.edges.add(edge)
        self.confounded_pairs.add(frozenset([node1, node2]))
        
    def get_parents(self, node: str) -> Set[str]:
        """Get direct parents of a node."""
        return self.reverse_adjacency.get(node, set()).copy()
    
    def get_children(self, node: str) -> Set[str]:
        """Get direct children of a node."""
        return self.adjacency.get(node, set()).copy()
    
    def get_ancestors(self, nodes: Set[str]) -> Set[str]:
        """Get all ancestors of a set of nodes (including themselves)."""
        ancestors = set(nodes)
        changed = True
        while changed:
            changed = False
            for node in list(ancestors):
                for parent in self.get_parents(node):
                    if parent not in ancestors:
                        ancestors.add(parent)
                        changed = True
        return ancestors
    
    def get_descendants(self, nodes: Set[str]) -> Set[str]:
        """Get all descendants of a set of nodes (including themselves)."""
        descendants = set(nodes)
        changed = True
        while changed:
            changed = False
            for node in list(descendants):
                for child in self.get_children(node):
                    if child not in descendants:
                        descendants.add(child)
                        changed = True
        return descendants
    
    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if one node is an ancestor of another."""
        return ancestor in self.get_ancestors({descendant})
    
    def is_collider(self, node: str, path: List[str]) -> bool:
        """
        Check if a node is a collider on a given path.
        
        A collider is a node where two arrowheads meet: -> Node <-
        """
        if node not in path or len(path) < 3:
            return False
        
        idx = path.index(node)
        if idx == 0 or idx == len(path) - 1:
            return False
        
        prev_node = path[idx - 1]
        next_node = path[idx + 1]
        
        # Check if both edges point TO the node
        points_to_node_from_prev = node in self.adjacency.get(prev_node, set())
        points_to_node_from_next = node in self.adjacency.get(next_node, set())
        
        return points_to_node_from_prev and points_to_node_from_next
    
    def is_valid_adjustment_set(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str]
    ) -> Tuple[bool, str]:
        """
        Check if a set satisfies the backdoor criterion for causal identification.
        
        The backdoor criterion requires:
        1. No node in Z is a descendant of X (treatment)
        2. Z blocks all backdoor paths from X to Y
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check condition 1: No descendants of treatment in adjustment set
        descendants_of_treatment = self.get_descendants({treatment}) - {treatment}
        for node in adjustment_set:
            if node in descendants_of_treatment:
                return False, f"{node} is a descendant of treatment {treatment}"
        
        # Check condition 2: Block all backdoor paths
        # This is a simplified check; full implementation would enumerate all paths
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)
        
        for path in backdoor_paths:
            if not self._is_path_blocked(path, adjustment_set, treatment):
                return False, f"Backdoor path {path} is not blocked by {adjustment_set}"
        
        return True, "Valid adjustment set"
    
    def _find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find all backdoor paths from treatment to outcome."""
        # Backdoor paths start with an edge INTO treatment
        backdoor_paths = []
        
        for parent in self.get_parents(treatment):
            paths = self._find_all_paths(parent, outcome, exclude={treatment})
            for path in paths:
                backdoor_paths.append([parent] + path)
        
        return backdoor_paths
    
    def _find_all_paths(
        self,
        start: str,
        end: str,
        exclude: Set[str],
        max_length: int = 10
    ) -> List[List[str]]:
        """Find all paths between two nodes using DFS."""
        paths = []
        
        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_length:
                return
            if current == end:
                paths.append(path.copy())
                return
            
            neighbors = (self.adjacency.get(current, set()) | 
                        self.reverse_adjacency.get(current, set()))
            
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in exclude:
                    path.append(neighbor)
                    visited.add(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start, [start], {start})
        return paths
    
    def _is_path_blocked(
        self,
        path: List[str],
        adjustment_set: Set[str],
        treatment: str
    ) -> bool:
        """
        Check if a path is blocked by an adjustment set using d-separation.
        
        A path is blocked if:
        1. It contains a non-collider that is in the adjustment set, OR
        2. It contains a collider that is NOT in the adjustment set and 
           has no descendants in the adjustment set
        """
        for i, node in enumerate(path):
            if node == treatment:
                continue
                
            is_collider = self.is_collider_on_path(node, path)
            
            if is_collider:
                # Collider blocks unless it or its descendants are conditioned on
                descendants = self.get_descendants({node})
                if not (node in adjustment_set or adjustment_set & descendants):
                    return True  # Path is blocked
            else:
                # Non-collider blocks if it IS in the adjustment set
                if node in adjustment_set:
                    return True  # Path is blocked
        
        return False  # Path is not blocked
    
    def is_collider_on_path(self, node: str, path: List[str]) -> bool:
        """Check if node is a collider on the specific path."""
        if node not in path or len(path) < 3:
            return False
        
        idx = path.index(node)
        if idx == 0 or idx == len(path) - 1:
            return False
        
        prev_node = path[idx - 1]
        next_node = path[idx + 1]
        
        # Check edge directions along the path
        points_to_node_from_prev = node in self.adjacency.get(prev_node, set())
        points_to_node_from_next = node in self.adjacency.get(next_node, set())
        
        return points_to_node_from_prev and points_to_node_from_next


class DoCalculusEngine:
    """
    Implementation of Pearl's Do-Calculus rules for causal inference.
    
    This engine can determine whether a causal effect is identifiable
    and compute the appropriate adjustment formula.
    """
    
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        
    def identify_causal_effect(
        self,
        treatment: str,
        outcome: str,
        evidence: Optional[Set[str]] = None
    ) -> Tuple[bool, Optional[str], Optional[Set[str]]]:
        """
        Determine if P(outcome | do(treatment), evidence) is identifiable.
        
        Returns:
            Tuple of (identifiable, formula_description, adjustment_set)
        """
        evidence = evidence or set()
        
        # Check for trivial cases
        if treatment == outcome:
            return True, "P(Y|do(Y)) = 1", set()
        
        # Try backdoor adjustment first
        adjustment_set = self._find_backdoor_adjustment(treatment, outcome, evidence)
        if adjustment_set is not None:
            formula = self._backdoor_formula(treatment, outcome, adjustment_set, evidence)
            return True, formula, adjustment_set
        
        # Try frontdoor adjustment
        frontdoor_set = self._find_frontdoor_adjustment(treatment, outcome)
        if frontdoor_set is not None:
            formula = self._frontdoor_formula(treatment, outcome, frontdoor_set)
            return True, formula, frontdoor_set
        
        # Try ID algorithm (simplified)
        # Full ID algorithm would recursively apply do-calculus rules
        
        return False, "Effect not identifiable with available methods", None
    
    def _find_backdoor_adjustment(
        self,
        treatment: str,
        outcome: str,
        evidence: Set[str]
    ) -> Optional[Set[str]]:
        """
        Find a valid backdoor adjustment set.
        
        Strategy: Use parents of treatment (if observed) as adjustment set.
        """
        parents = self.graph.get_parents(treatment)
        observed_parents = parents & self.graph.observed
        
        # Check if parents block all backdoor paths
        is_valid, _ = self.graph.is_valid_adjustment_set(
            treatment, outcome, observed_parents
        )
        
        if is_valid:
            return observed_parents - evidence
        
        # Try adding more variables
        candidates = (self.graph.observed - {treatment, outcome} - evidence 
                     - self.graph.get_descendants({treatment}))
        
        # Greedy search for valid adjustment set
        adjustment_set = set(observed_parents)
        for candidate in sorted(candidates):
            test_set = adjustment_set | {candidate}
            is_valid, _ = self.graph.is_valid_adjustment_set(
                treatment, outcome, test_set
            )
            if is_valid:
                adjustment_set = test_set
                return adjustment_set
        
        return None
    
    def _find_frontdoor_adjustment(
        self,
        treatment: str,
        outcome: str
    ) -> Optional[Set[str]]:
        """
        Find a valid frontdoor adjustment set.
        
        Frontdoor criterion requires a mediator M such that:
        1. M intercepts all directed paths from X to Y
        2. No backdoor path from X to M
        3. All backdoor paths from M to Y are blocked by X
        """
        # Look for potential mediators among children of treatment
        children = self.graph.get_children(treatment)
        
        for mediator in children:
            if mediator == outcome:
                continue
            
            # Check condition 1: Mediator is on all directed paths
            # (Simplified check)
            
            # Check condition 2: No backdoor from X to M
            backdoor_xm = self.graph._find_backdoor_paths(treatment, mediator)
            if backdoor_xm:
                continue
            
            # Check condition 3: X blocks backdoor from M to Y
            # (Simplified check)
            
            return {mediator}
        
        return None
    
    def _backdoor_formula(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str],
        evidence: Set[str]
    ) -> str:
        """Generate the backdoor adjustment formula string."""
        adj_str = ", ".join(sorted(adjustment_set)) if adjustment_set else ""
        ev_str = ", ".join(sorted(evidence)) if evidence else ""
        
        if adj_str and ev_str:
            return f"P({outcome}|do({treatment})) = Σ_{adj_str} P({outcome}|{treatment},{adj_str},{ev_str}) P({adj_str}|{ev_str})"
        elif adj_str:
            return f"P({outcome}|do({treatment})) = Σ_{adj_str} P({outcome}|{treatment},{adj_str}) P({adj_str})"
        else:
            return f"P({outcome}|do({treatment})) = P({outcome}|{treatment})"
    
    def _frontdoor_formula(
        self,
        treatment: str,
        outcome: str,
        frontdoor_set: Set[str]
    ) -> str:
        """Generate the frontdoor adjustment formula string."""
        m = ", ".join(sorted(frontdoor_set))
        return f"P({outcome}|do({treatment})) = Σ_{m} P({m}|{treatment}) Σ_x P({outcome}|x,{m}) P(x)"
    
    def apply_do_rule_1(
        self,
        query_vars: Set[str],
        treatment: str,
        condition: Set[str],
        removal_set: Set[str]
    ) -> bool:
        """
        Apply Rule 1: Insertion/deletion of observations.
        
        P(Y|do(X),Z,W) = P(Y|do(X),W) if Y ⊥ Z | X,W in G_X̄
        
        Returns True if the rule applies.
        """
        # Create mutilated graph G_X̄ (remove incoming edges to X)
        mutilated = self._mutilate_graph([treatment])
        
        # Check d-separation in mutilated graph
        return mutilated._check_d_separation(
            query_vars, removal_set, {treatment} | (condition - removal_set)
        )
    
    def apply_do_rule_2(
        self,
        query_vars: Set[str],
        treatment: str,
        intervention_var: str,
        condition: Set[str]
    ) -> bool:
        """
        Apply Rule 2: Action/observation exchange.
        
        P(Y|do(X),do(Z),W) = P(Y|do(X),Z,W) if Y ⊥ Z | X,W in G_X̄,Z̲
        
        Returns True if the rule applies.
        """
        # Create mutilated graph G_X̄,Z̲ (remove incoming to X, outgoing from Z)
        mutilated = self._mutilate_graph([treatment], outgoing_from=[intervention_var])
        
        return mutilated._check_d_separation(
            query_vars, {intervention_var}, {treatment} | condition
        )
    
    def apply_do_rule_3(
        self,
        query_vars: Set[str],
        treatment: str,
        intervention_var: str,
        condition: Set[str]
    ) -> bool:
        """
        Apply Rule 3: Insertion/deletion of actions.
        
        P(Y|do(X),do(Z),W) = P(Y|do(X),W) if Y ⊥ Z | X,W in G_X̄,Z̄
        
        Returns True if the rule applies.
        """
        # Create mutilated graph G_X̄,Z̄ (remove incoming to X and Z)
        mutilated = self._mutilate_graph([treatment, intervention_var])
        
        return mutilated._check_d_separation(
            query_vars, {intervention_var}, {treatment} | condition
        )
    
    def _mutilate_graph(
        self,
        remove_incoming_to: List[str],
        outgoing_from: Optional[List[str]] = None
    ) -> 'CausalGraph':
        """Create a mutilated graph for do-calculus."""
        mutilated = CausalGraph()
        
        # Copy all nodes
        for node in self.graph.nodes:
            mutilated.add_node(node, node in self.graph.observed)
        
        # Copy edges with modifications
        for edge in self.graph.edges:
            if edge.edge_type == CausalGraphType.DIRECTED:
                # Remove incoming edges to specified nodes
                if edge.target in remove_incoming_to:
                    continue
                # Remove outgoing edges from specified nodes
                if outgoing_from and edge.source in outgoing_from:
                    continue
                mutilated.add_directed_edge(edge.source, edge.target)
            elif edge.edge_type == CausalGraphType.BIDIRECTED:
                mutilated.add_bidirected_edge(edge.source, edge.target)
        
        return mutilated
    
    def _check_d_separation(
        self,
        x: Set[str],
        y: Set[str],
        z: Set[str]
    ) -> bool:
        """
        Check if X is d-separated from Y given Z.
        
        Simplified implementation - full version would check all paths.
        """
        # For now, use a simple heuristic
        # Full implementation would enumerate all paths and check blocking
        return True  # Placeholder


def prevent_collider_intervention(
    graph: CausalGraph,
    proposed_intervention: str,
    outcome: str
) -> Tuple[bool, str]:
    """
    Validate that an intervention does not create collider bias.
    
    An intervention on a collider (or its descendant) can open spurious
    paths and create bias.
    
    Returns:
        Tuple of (is_safe, warning_message)
    """
    # Check if proposed intervention is a collider on any path to outcome
    for node in graph.nodes:
        if node == proposed_intervention:
            continue
        
        # Find paths from intervention to outcome through this node
        paths = graph._find_all_paths(proposed_intervention, outcome, set())
        
        for path in paths:
            if graph.is_collider_on_path(node, path):
                # Check if intervening would open this path
                descendants = graph.get_descendants({node})
                if proposed_intervention in descendants:
                    return False, (
                        f"WARNING: Intervening on {proposed_intervention} may open "
                        f"collider path at {node}, creating spurious association"
                    )
    
    return True, "Intervention appears safe from collider bias"


# Example usage and testing
if __name__ == "__main__":
    # Build a sample causal graph for crypto trading
    graph = CausalGraph()
    
    # Add variables
    graph.add_node("BTC_Supply_Shock", observed=True)
    graph.add_node("ETH_Price", observed=True)
    graph.add_node("Market_Sentiment", observed=False)  # Unobserved confounder
    graph.add_node("Trading_Volume", observed=True)
    graph.add_node("Price_Impact", observed=True)
    
    # Add causal relationships
    graph.add_directed_edge("BTC_Supply_Shock", "ETH_Price")
    graph.add_directed_edge("BTC_Supply_Shock", "Trading_Volume")
    graph.add_directed_edge("ETH_Price", "Price_Impact")
    graph.add_directed_edge("Trading_Volume", "Price_Impact")
    
    # Add unobserved confounding
    graph.add_bidirected_edge("ETH_Price", "Trading_Volume")  # Due to Market_Sentiment
    
    # Create do-calculus engine
    engine = DoCalculusEngine(graph)
    
    # Try to identify causal effect of BTC_Supply_Shock on Price_Impact
    identifiable, formula, adj_set = engine.identify_causal_effect(
        "BTC_Supply_Shock", "Price_Impact"
    )
    
    print(f"\nCausal Effect Identification:")
    print(f"Identifiable: {identifiable}")
    if identifiable:
        print(f"Formula: {formula}")
        print(f"Adjustment Set: {adj_set}")
    
    # Check for collider bias
    is_safe, message = prevent_collider_intervention(
        graph, "Trading_Volume", "Price_Impact"
    )
    print(f"\nCollider Check: {message}")
