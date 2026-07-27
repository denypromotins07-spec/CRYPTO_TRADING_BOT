//! Confounder Adjustment for Hidden Market Variables
//!
//! This module identifies and adjusts for hidden market confounders
//! in causal analysis. Designed for the ZAID PERSONAL CRYPTO TRADING BOT
//! to ensure valid causal inference despite unobserved variables.
//!
//! Features:
//! - Backdoor path detection and blocking
//! - Instrumental variable identification
//! - Frontdoor criterion implementation
//! - Sensitivity analysis for unobserved confounding

use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::Instant;

/// Represents a potential confounder in the causal graph
#[derive(Debug, Clone)]
pub struct Confounder {
    pub name: String,
    pub is_observed: bool,
    pub affects_treatment: bool,
    pub affects_outcome: bool,
    pub strength: f64, // Estimated confounding strength
}

/// Result of confounder adjustment analysis
#[derive(Debug, Clone)]
pub struct AdjustmentResult {
    pub is_identifiable: bool,
    pub adjustment_set: Vec<String>,
    pub method: AdjustmentMethod,
    pub bias_estimate: f64,
    pub confidence_interval: (f64, f64),
}

/// Method used for confounder adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjustmentMethod {
    BackdoorAdjustment,
    FrontdoorAdjustment,
    InstrumentalVariable,
    InverseProbabilityWeighting,
    PropensityScoreMatching,
    DifferenceInDifferences,
    RegressionDiscontinuity,
}

/// Graph structure for causal analysis
pub struct CausalGraph {
    nodes: HashSet<String>,
    directed_edges: HashMap<String, HashSet<String>>, // parent -> children
    reverse_edges: HashMap<String, HashSet<String>>,  // child -> parents
    bidirected_edges: HashSet<(String, String)>,      // Unobserved confounding
    observed: HashSet<String>,
    unobserved: HashSet<String>,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            directed_edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            bidirected_edges: HashSet::new(),
            observed: HashSet::new(),
            unobserved: HashSet::new(),
        }
    }

    pub fn add_node(&mut self, name: &str, observed: bool) {
        self.nodes.insert(name.to_string());
        if observed {
            self.observed.insert(name.to_string());
        } else {
            self.unobserved.insert(name.to_string());
        }
        self.directed_edges.entry(name.to_string()).or_default();
        self.reverse_edges.entry(name.to_string()).or_default();
    }

    pub fn add_directed_edge(&mut self, from: &str, to: &str) {
        self.directed_edges
            .entry(from.to_string())
            .or_default()
            .insert(to.to_string());
        self.reverse_edges
            .entry(to.to_string())
            .or_default()
            .insert(from.to_string());
    }

    pub fn add_bidirected_edge(&mut self, node1: &str, node2: &str) {
        let pair = if node1 < node2 {
            (node1.to_string(), node2.to_string())
        } else {
            (node2.to_string(), node1.to_string())
        };
        self.bidirected_edges.insert(pair);
    }

    /// Get all backdoor paths from treatment to outcome
    pub fn find_backdoor_paths(&self, treatment: &str, outcome: &str) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        
        // Backdoor paths start with an edge INTO treatment
        if let Some(parents) = self.reverse_edges.get(treatment) {
            for parent in parents {
                // Find paths from parent to outcome that don't go through treatment
                let mut current_path = vec![parent.clone()];
                self._dfs_paths(
                    parent,
                    outcome,
                    &mut current_path,
                    &mut paths,
                    &HashSet::from([treatment.to_string()]),
                );
            }
        }

        paths
    }

    fn _dfs_paths(
        &self,
        current: &str,
        target: &str,
        path: &mut Vec<String>,
        all_paths: &mut Vec<Vec<String>>,
        excluded: &HashSet<String>,
    ) {
        if path.len() > 15 {
            return; // Prevent infinite loops
        }

        if current == target {
            all_paths.push(path.clone());
            return;
        }

        // Get neighbors (both directions for undirected path traversal)
        let mut neighbors = HashSet::new();
        if let Some(children) = self.directed_edges.get(current) {
            neighbors.extend(children);
        }
        if let Some(parents) = self.reverse_edges.get(current) {
            neighbors.extend(parents);
        }

        for neighbor in neighbors {
            if !excluded.contains(neighbor) && !path.contains(neighbor) {
                path.push(neighbor.clone());
                self._dfs_paths(neighbor, target, path, all_paths, excluded);
                path.pop();
            }
        }
    }

    /// Check if a set blocks all backdoor paths
    pub fn blocks_all_backdoors(
        &self,
        treatment: &str,
        outcome: &str,
        adjustment_set: &HashSet<String>,
    ) -> bool {
        let backdoor_paths = self.find_backdoor_paths(treatment, outcome);

        for path in &backdoor_paths {
            if !self.is_path_blocked(path, adjustment_set, treatment) {
                return false;
            }
        }

        true
    }

    /// Check if a path is blocked by the adjustment set (d-separation)
    fn is_path_blocked(
        &self,
        path: &[String],
        adjustment_set: &HashSet<String>,
        treatment: &str,
    ) -> bool {
        for (i, node) in path.iter().enumerate() {
            if node == treatment {
                continue;
            }

            let is_collider = self.is_collider_on_path(path, i);

            if is_collider {
                // Collider blocks unless conditioned on (or its descendant)
                if !adjustment_set.contains(node) {
                    // Check descendants
                    let descendants = self.get_descendants(node);
                    let has_descendant_in_set = descendants
                        .iter()
                        .any(|d| adjustment_set.contains(d));
                    if !has_descendant_in_set {
                        return true; // Path is blocked
                    }
                }
            } else {
                // Non-collider blocks if IN the adjustment set
                if adjustment_set.contains(node) {
                    return true; // Path is blocked
                }
            }
        }

        false // Path is not blocked
    }

    fn is_collider_on_path(&self, path: &[String], idx: usize) -> bool {
        if idx == 0 || idx >= path.len() - 1 {
            return false;
        }

        let prev = &path[idx - 1];
        let node = &path[idx];
        let next = &path[idx + 1];

        // Check if both edges point TO the node
        let points_from_prev = self
            .directed_edges
            .get(prev)
            .map(|c| c.contains(node))
            .unwrap_or(false);
        let points_from_next = self
            .directed_edges
            .get(next)
            .map(|c| c.contains(node))
            .unwrap_or(false);

        points_from_prev && points_from_next
    }

    fn get_descendants(&self, node: &str) -> HashSet<String> {
        let mut descendants = HashSet::new();
        descendants.insert(node.to_string());

        let mut changed = true;
        while changed {
            changed = false;
            let current: Vec<String> = descendants.iter().cloned().collect();
            for n in current {
                if let Some(children) = self.directed_edges.get(&n) {
                    for child in children {
                        if !descendants.contains(child) {
                            descendants.insert(child.clone());
                            changed = true;
                        }
                    }
                }
            }
        }

        descendants
    }

    /// Find a valid adjustment set using the backdoor criterion
    pub fn find_adjustment_set(&self, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        // Start with parents of treatment (excluding descendants of treatment)
        let parents = self.reverse_edges.get(treatment).cloned().unwrap_or_default();
        let descendants = self.get_descendants(treatment);
        
        let candidate_set: HashSet<String> = parents
            .into_iter()
            .filter(|p| !descendants.contains(p))
            .filter(|p| self.observed.contains(p))
            .collect();

        if self.blocks_all_backdoors(treatment, outcome, &candidate_set) {
            return Some(candidate_set);
        }

        // Try adding more observed variables
        let additional_candidates: Vec<String> = self
            .observed
            .iter()
            .filter(|n| *n != treatment && *n != outcome)
            .filter(|n| !descendants.contains(*n))
            .filter(|n| !candidate_set.contains(*n))
            .cloned()
            .collect();

        // Greedy search
        let mut adjustment_set = candidate_set;
        for candidate in &additional_candidates {
            let test_set: HashSet<String> = adjustment_set
                .iter()
                .cloned()
                .chain(Some(candidate.clone()))
                .collect();

            if self.blocks_all_backdoors(treatment, outcome, &test_set) {
                adjustment_set = test_set;
                return Some(adjustment_set);
            }
        }

        None
    }

    /// Check if frontdoor criterion is satisfied
    pub fn check_frontdoor_criterion(
        &self,
        treatment: &str,
        outcome: &str,
        mediator: &str,
    ) -> bool {
        // Condition 1: Mediator intercepts all directed paths from treatment to outcome
        // (Simplified check)

        // Condition 2: No backdoor path from treatment to mediator
        let backdoor_tm = self.find_backdoor_paths(treatment, mediator);
        if !backdoor_tm.is_empty() {
            return false;
        }

        // Condition 3: All backdoor paths from mediator to outcome are blocked by treatment
        let backdoor_mo = self.find_backdoor_paths(mediator, outcome);
        for path in &backdoor_mo {
            if !path.contains(&treatment.to_string()) {
                return false;
            }
        }

        true
    }

    /// Find instrumental variables
    pub fn find_instrumental_variables(
        &self,
        treatment: &str,
        outcome: &str,
    ) -> Vec<String> {
        let mut instruments = Vec::new();

        for node in &self.observed {
            if node == treatment || node == outcome {
                continue;
            }

            // IV conditions:
            // 1. Relevance: IV affects treatment
            let affects_treatment = self
                .directed_edges
                .get(node)
                .map(|c| c.contains(treatment))
                .unwrap_or(false);

            if !affects_treatment {
                continue;
            }

            // 2. Exclusion: IV does not directly affect outcome (only through treatment)
            let directly_affects_outcome = self
                .directed_edges
                .get(node)
                .map(|c| c.contains(outcome))
                .unwrap_or(false);

            if directly_affects_outcome {
                continue;
            }

            // 3. Exchangeability: No common causes of IV and outcome
            let has_common_cause = self.has_unobserved_confounding(node, outcome);
            if has_common_cause {
                continue;
            }

            instruments.push(node.clone());
        }

        instruments
    }

    fn has_unobserved_confounding(&self, node1: &str, node2: &str) -> bool {
        let pair = if node1 < node2 {
            (node1.to_string(), node2.to_string())
        } else {
            (node2.to_string(), node1.to_string())
        };
        self.bidirected_edges.contains(&pair)
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Main confounder adjustment engine
pub struct ConfounderAdjuster {
    graph: CausalGraph,
    data_cache: HashMap<String, Vec<f64>>,
}

impl ConfounderAdjuster {
    pub fn new(graph: CausalGraph) -> Self {
        Self {
            graph,
            data_cache: HashMap::new(),
        }
    }

    pub fn add_data(&mut self, variable: &str, values: Vec<f64>) {
        self.data_cache.insert(variable.to_string(), values);
    }

    /// Perform confounder adjustment and estimate causal effect
    pub fn adjust(
        &self,
        treatment: &str,
        outcome: &str,
    ) -> AdjustmentResult {
        let start = Instant::now();

        // Try backdoor adjustment first
        if let Some(adj_set) = self.graph.find_adjustment_set(treatment, outcome) {
            let effect = self.estimate_via_backdoor(treatment, outcome, &adj_set);
            
            log::info!(
                "Backdoor adjustment completed in {:?}, effect: {:.4}",
                start.elapsed(),
                effect.point_estimate
            );

            return AdjustmentResult {
                is_identifiable: true,
                adjustment_set: adj_set.into_iter().collect(),
                method: AdjustmentMethod::BackdoorAdjustment,
                bias_estimate: effect.bias,
                confidence_interval: effect.ci,
            };
        }

        // Try frontdoor adjustment
        for mediator in &self.graph.observed {
            if mediator != treatment && mediator != outcome {
                if self.graph.check_frontdoor_criterion(treatment, outcome, mediator) {
                    let effect = self.estimate_via_frontdoor(treatment, outcome, mediator);
                    
                    return AdjustmentResult {
                        is_identifiable: true,
                        adjustment_set: vec![mediator.clone()],
                        method: AdjustmentMethod::FrontdoorAdjustment,
                        bias_estimate: effect.bias,
                        confidence_interval: effect.ci,
                    };
                }
            }
        }

        // Try instrumental variables
        let instruments = self.graph.find_instrumental_variables(treatment, outcome);
        if !instruments.is_empty() {
            let effect = self.estimate_via_iv(treatment, outcome, &instruments[0]);
            
            return AdjustmentResult {
                is_identifiable: true,
                adjustment_set: vec![instruments[0].clone()],
                method: AdjustmentMethod::InstrumentalVariable,
                bias_estimate: effect.bias,
                confidence_interval: effect.ci,
            };
        }

        // Not identifiable
        AdjustmentResult {
            is_identifiable: false,
            adjustment_set: Vec::new(),
            method: AdjustmentMethod::BackdoorAdjustment,
            bias_estimate: f64::NAN,
            confidence_interval: (f64::NAN, f64::NAN),
        }
    }

    fn estimate_via_backdoor(
        &self,
        _treatment: &str,
        _outcome: &str,
        _adj_set: &HashSet<String>,
    ) -> EffectEstimate {
        // Placeholder - would implement regression/matching
        EffectEstimate {
            point_estimate: 0.0,
            bias: 0.0,
            ci: (0.0, 0.0),
        }
    }

    fn estimate_via_frontdoor(
        &self,
        _treatment: &str,
        _outcome: &str,
        _mediator: &str,
    ) -> EffectEstimate {
        // Placeholder - would implement frontdoor formula
        EffectEstimate {
            point_estimate: 0.0,
            bias: 0.0,
            ci: (0.0, 0.0),
        }
    }

    fn estimate_via_iv(
        &self,
        _treatment: &str,
        _outcome: &str,
        _instrument: &str,
    ) -> EffectEstimate {
        // Placeholder - would implement 2SLS
        EffectEstimate {
            point_estimate: 0.0,
            bias: 0.0,
            ci: (0.0, 0.0),
        }
    }

    /// Sensitivity analysis: how strong must unobserved confounding be to explain the effect?
    pub fn sensitivity_analysis(&self, observed_effect: f64) -> SensitivityResult {
        // E-value calculation (VanderWeele & Ding)
        // Minimum confounding strength needed to explain away the effect
        
        let e_value = if observed_effect >= 1.0 {
            observed_effect + (observed_effect * (observed_effect - 1.0)).sqrt()
        } else {
            1.0
        };

        SensitivityResult {
            e_value,
            interpretation: format!(
                "An unobserved confounder would need RR={:.2} with both treatment and outcome \
                 to explain away this effect",
                e_value
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct EffectEstimate {
    point_estimate: f64,
    bias: f64,
    ci: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct SensitivityResult {
    pub e_value: f64,
    pub interpretation: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backdoor_adjustment() {
        let mut graph = CausalGraph::new();

        // Classic confounding structure: Z -> X, Z -> Y, X -> Y
        graph.add_node("Z", true);   // Observed confounder
        graph.add_node("X", true);   // Treatment
        graph.add_node("Y", true);   // Outcome

        graph.add_directed_edge("Z", "X");
        graph.add_directed_edge("Z", "Y");
        graph.add_directed_edge("X", "Y");

        let adj_set = graph.find_adjustment_set("X", "Y");
        
        assert!(adj_set.is_some());
        assert!(adj_set.as_ref().unwrap().contains("Z"));
    }

    #[test]
    fn test_unobserved_confounder() {
        let mut graph = CausalGraph::new();

        graph.add_node("U", false);  // Unobserved confounder
        graph.add_node("X", true);
        graph.add_node("Y", true);

        graph.add_directed_edge("U", "X");
        graph.add_directed_edge("U", "Y");
        graph.add_directed_edge("X", "Y");

        // Mark U as unobserved confounding between X and Y
        graph.add_bidirected_edge("X", "Y");

        let adj_set = graph.find_adjustment_set("X", "Y");
        
        // Should not find a valid adjustment set (not identifiable via backdoor)
        assert!(adj_set.is_none());
    }

    #[test]
    fn test_instrumental_variable() {
        let mut graph = CausalGraph::new();

        graph.add_node("Z", true);   // Instrument
        graph.add_node("X", true);   // Treatment
        graph.add_node("Y", true);   // Outcome
        graph.add_node("U", false);  // Unobserved confounder

        graph.add_directed_edge("Z", "X");
        graph.add_directed_edge("X", "Y");
        graph.add_directed_edge("U", "X");
        graph.add_directed_edge("U", "Y");

        let instruments = graph.find_instrumental_variables("X", "Y");
        
        assert!(instruments.contains(&"Z".to_string()));
    }
}
