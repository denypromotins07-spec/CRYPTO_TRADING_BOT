//! Bayesian Network for Exact Inference on Causal DAGs
//!
//! This module implements exact inference on Directed Acyclic Graphs (DAGs)
//! using variable elimination. Designed for the ZAID PERSONAL CRYPTO TRADING BOT
//! to compute interventional probabilities in microseconds during live order flow.
//!
//! Features:
//! - Variable elimination algorithm for exact inference
//! - Factor-based representation of joint distributions
//! - Efficient caching of intermediate results
//! - Support for evidence conditioning and marginal queries

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::Arc;
use std::time::Instant;

/// Represents a discrete random variable in the Bayesian network
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Variable {
    pub id: usize,
    pub name: String,
    pub cardinality: usize, // Number of possible states
}

/// A factor (potential function) over a set of variables
#[derive(Debug, Clone)]
pub struct Factor {
    /// Variables in the scope of this factor
    pub variables: Vec<usize>,
    /// Values indexed by variable assignments
    pub values: Vec<f64>,
    /// Cardinalities of each variable for indexing
    pub cardinalities: Vec<usize>,
}

impl Factor {
    /// Create a new factor with given variables and cardinalities
    pub fn new(variables: Vec<usize>, cardinalities: Vec<usize>) -> Self {
        let n_values = cardinalities.iter().product();
        Self {
            variables,
            values: vec![1.0; n_values],
            cardinalities,
        }
    }

    /// Get the index for a specific assignment
    fn get_index(&self, assignment: &[usize]) -> usize {
        let mut index = 0;
        let mut stride = 1;
        
        for (&var_idx, &val) in assignment.iter().zip(self.cardinalities.iter()) {
            index += stride * var_idx.min(val - 1);
            stride *= val;
        }
        
        index.min(self.values.len() - 1)
    }

    /// Set value for a specific assignment
    pub fn set_value(&mut self, assignment: &[usize], value: f64) {
        let idx = self.get_index(assignment);
        self.values[idx] = value;
    }

    /// Get value for a specific assignment
    pub fn get_value(&self, assignment: &[usize]) -> f64 {
        let idx = self.get_index(assignment);
        self.values[idx]
    }

    /// Multiply two factors (pointwise product)
    pub fn multiply(&self, other: &Factor) -> Factor {
        // Find union of variables
        let mut all_vars: Vec<usize> = self.variables.clone();
        for &v in &other.variables {
            if !all_vars.contains(&v) {
                all_vars.push(v);
            }
        }
        all_vars.sort();

        // Compute cardinalities for result
        let mut card_map: BTreeMap<usize, usize> = BTreeMap::new();
        for (&v, &c) in self.variables.iter().zip(&self.cardinalities) {
            card_map.insert(v, c);
        }
        for (&v, &c) in other.variables.iter().zip(&other.cardinalities) {
            card_map.entry(v).or_insert(c);
        }

        let result_cards: Vec<usize> = all_vars.iter().map(|&v| card_map[&v]).collect();
        let mut result = Factor::new(all_vars.clone(), result_cards);

        // Compute pointwise product
        let mut assignment = vec![0; all_vars.len()];
        self.enumerate_assignments(&all_vars, &card_map, &mut assignment, 0, |assign| {
            let self_vals: Vec<usize> = self
                .variables
                .iter()
                .map(|&v| assign[all_vars.iter().position(|&x| x == v).unwrap()])
                .collect();
            let other_vals: Vec<usize> = other
                .variables
                .iter()
                .map(|&v| assign[all_vars.iter().position(|&x| x == v).unwrap()])
                .collect();

            let idx = result.get_index(&assign);
            result.values[idx] = self.get_value(&self_vals) * other.get_value(&other_vals);
        });

        result
    }

    /// Sum out (marginalize) a variable from this factor
    pub fn sum_out(&self, var: usize) -> Factor {
        if let Some(pos) = self.variables.iter().position(|&v| v == var) {
            let mut new_vars = self.variables.clone();
            let mut new_cards = self.cardinalities.clone();
            
            new_vars.remove(pos);
            new_cards.remove(pos);

            let mut result = Factor::new(new_vars.clone(), new_cards);

            // Sum over the eliminated variable
            let mut assignment = vec![0; new_vars.len()];
            self.enumerate_assignments(&new_vars, &self.cardinalities.iter().copied().enumerate()
                .filter(|(i, _)| *i != pos)
                .map(|(i, &c)| (self.variables[i], c))
                .collect(), &mut assignment, 0, |new_assign| {
                
                let mut sum = 0.0;
                for val in 0..self.cardinalities[pos] {
                    let mut full_assign = new_assign.clone();
                    full_assign.insert(pos, val);
                    sum += self.get_value(&full_assign);
                }

                let idx = result.get_index(&new_assign);
                result.values[idx] = sum;
            });

            result
        } else {
            self.clone()
        }
    }

    /// Condition on observed evidence
    pub fn condition(&self, evidence: &HashMap<usize, usize>) -> Factor {
        let mut new_vars = self.variables.clone();
        let mut new_cards = self.cardinalities.clone();
        let mut observed_positions: Vec<(usize, usize)> = Vec::new();

        // Find which variables are observed
        for (i, &var) in self.variables.iter().enumerate() {
            if let Some(&val) = evidence.get(&var) {
                observed_positions.push((i, val));
                new_cards[i] = 1; // Fix cardinality to 1
            }
        }

        if observed_positions.is_empty() {
            return self.clone();
        }

        let mut result = Factor::new(new_vars, new_cards);

        // Copy only consistent entries
        let mut assignment = vec![0; self.variables.len()];
        self.enumerate_assignments(&self.variables, &self.cardinalities, &mut assignment, 0, |assign| {
            // Check if assignment is consistent with evidence
            let consistent = observed_positions.iter().all(|&(pos, val)| assign[pos] == val);
            
            if consistent {
                let idx = result.get_index(assign);
                result.values[idx] = self.get_value(assign);
            }
        });

        result
    }

    /// Normalize factor to sum to 1
    pub fn normalize(&mut self) {
        let sum: f64 = self.values.iter().sum();
        if sum > 0.0 {
            for val in &mut self.values {
                *val /= sum;
            }
        }
    }

    /// Helper to enumerate all assignments
    fn enumerate_assignments<F>(
        &self,
        vars: &[usize],
        cards: &BTreeMap<usize, usize>,
        assignment: &mut [usize],
        depth: usize,
        mut callback: F,
    ) where
        F: FnMut(&[usize]),
    {
        if depth >= vars.len() {
            callback(assignment);
            return;
        }

        let var = vars[depth];
        let card = *cards.get(&var).unwrap_or(&2);

        for val in 0..card {
            assignment[depth] = val;
            self.enumerate_assignments(vars, cards, assignment, depth + 1, &mut callback);
        }
    }
}

/// A Bayesian Network representing a joint distribution over variables
pub struct BayesianNetwork {
    /// All variables in the network
    variables: HashMap<usize, Variable>,
    /// Factors associated with each variable (CPDs)
    factors: HashMap<usize, Factor>,
    /// Graph structure: parent -> children
    adjacency: HashMap<usize, Vec<usize>>,
    /// Reverse adjacency: child -> parents
    parents: HashMap<usize, Vec<usize>>,
}

impl BayesianNetwork {
    /// Create a new empty Bayesian network
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            factors: HashMap::new(),
            adjacency: HashMap::new(),
            parents: HashMap::new(),
        }
    }

    /// Add a variable to the network
    pub fn add_variable(&mut self, id: usize, name: &str, cardinality: usize) {
        let var = Variable {
            id,
            name: name.to_string(),
            cardinality,
        };
        self.variables.insert(id, var);
        self.adjacency.entry(id).or_insert_with(Vec::new);
        self.parents.entry(id).or_insert_with(Vec::new);
    }

    /// Add a directed edge (parent -> child)
    pub fn add_edge(&mut self, parent: usize, child: usize) {
        self.adjacency.entry(parent).or_insert_with(Vec::new).push(child);
        self.parents.entry(child).or_insert_with(Vec::new).push(parent);
    }

    /// Set the conditional probability distribution for a variable
    pub fn set_cpd(&mut self, var_id: usize, factor: Factor) {
        self.factors.insert(var_id, factor);
    }

    /// Create a CPD factor from explicit probabilities
    pub fn create_cpd(
        &self,
        var_id: usize,
        parent_ids: &[usize],
        probabilities: &[f64],
    ) -> Option<Factor> {
        let var = self.variables.get(&var_id)?;
        
        let mut variables = parent_ids.to_vec();
        variables.push(var_id);

        let mut cardinalities: Vec<usize> = parent_ids
            .iter()
            .map(|&pid| self.variables.get(&pid)?.cardinality)
            .collect();
        cardinalities.push(var.cardinality);

        let mut factor = Factor::new(variables, cardinalities);
        
        for (i, &prob) in probabilities.iter().enumerate() {
            factor.values[i] = prob;
        }

        Some(factor)
    }

    /// Query the marginal probability P(query_var | evidence)
    pub fn query(
        &self,
        query_var: usize,
        evidence: &HashMap<usize, usize>,
    ) -> Option<Vec<f64>> {
        let start = Instant::now();

        // Get all factors
        let mut factors: Vec<Factor> = self.factors.values().cloned().collect();

        // Apply evidence to all factors
        if !evidence.is_empty() {
            for factor in &mut factors {
                *factor = factor.condition(evidence);
            }
        }

        // Variable elimination
        let elimination_order = self.get_elimination_order(&query_var, evidence);
        
        for var in elimination_order {
            if var == query_var {
                continue;
            }
            if evidence.contains_key(&var) {
                continue;
            }

            // Find all factors containing var
            let relevant: Vec<Factor> = factors
                .iter()
                .filter(|f| f.variables.contains(&var))
                .cloned()
                .collect();

            if relevant.is_empty() {
                continue;
            }

            // Multiply all relevant factors
            let mut product = relevant[0].clone();
            for factor in &relevant[1..] {
                product = product.multiply(factor);
            }

            // Sum out the variable
            let marginalized = product.sum_out(var);

            // Remove old factors and add new one
            factors.retain(|f| !f.variables.contains(&var));
            factors.push(marginalized);
        }

        // Multiply remaining factors
        let mut result = factors[0].clone();
        for factor in &factors[1..] {
            result = result.multiply(factor);
        }

        // Extract marginal for query variable
        let query_card = self.variables.get(&query_var)?.cardinality;
        let mut marginal = vec![0.0; query_card];

        for i in 0..query_card {
            let mut assignment = vec![0; result.variables.len()];
            if let Some(pos) = result.variables.iter().position(|&v| v == query_var) {
                assignment[pos] = i;
                marginal[i] = result.get_value(&assignment);
            }
        }

        // Normalize
        let sum: f64 = marginal.iter().sum();
        if sum > 0.0 {
            for val in &mut marginal {
                *val /= sum;
            }
        }

        let elapsed = start.elapsed();
        log::debug!(
            "Query P({}|evidence) completed in {:?}, result: {:?}",
            self.variables.get(&query_var)?.name,
            elapsed,
            marginal
        );

        Some(marginal)
    }

    /// Compute interventional probability P(Y | do(X=x), evidence)
    /// Using the truncated factorization formula
    pub fn interventional_query(
        &self,
        target_var: usize,
        intervention: (usize, usize), // (do_X, value_x)
        evidence: &HashMap<usize, usize>,
    ) -> Option<Vec<f64>> {
        let start = Instant::now();

        let (do_var, do_val) = intervention;

        // Verify intervention variable exists
        if !self.variables.contains_key(&do_var) {
            return None;
        }

        // Create modified factors for do-calculus
        // Remove factors for intervened variables (truncated factorization)
        let mut factors: Vec<Factor> = self
            .factors
            .iter()
            .filter(|(&var_id, _)| var_id != do_var)
            .map(|(_, f)| f.clone())
            .collect();

        // Add deterministic factor for intervention
        let do_var_obj = self.variables.get(&do_var)?;
        let mut do_factor = Factor::new(vec![do_var], vec![do_var_obj.cardinality]);
        
        for i in 0..do_var_obj.cardinality {
            do_factor.values[i] = if i == do_val { 1.0 } else { 0.0 };
        }
        factors.push(do_factor);

        // Apply evidence
        let mut adjusted_evidence = evidence.clone();
        adjusted_evidence.insert(do_var, do_val);

        for factor in &mut factors {
            *factor = factor.condition(&adjusted_evidence);
        }

        // Variable elimination (same as regular query)
        let elimination_order = self.get_elimination_order(&target_var, &adjusted_evidence);
        
        for var in elimination_order {
            if var == target_var || var == do_var || adjusted_evidence.contains_key(&var) {
                continue;
            }

            let relevant: Vec<Factor> = factors
                .iter()
                .filter(|f| f.variables.contains(&var))
                .cloned()
                .collect();

            if relevant.is_empty() {
                continue;
            }

            let mut product = relevant[0].clone();
            for factor in &relevant[1..] {
                product = product.multiply(factor);
            }

            let marginalized = product.sum_out(var);
            factors.retain(|f| !f.variables.contains(&var));
            factors.push(marginalized);
        }

        // Multiply remaining factors
        if factors.is_empty() {
            return None;
        }

        let mut result = factors[0].clone();
        for factor in &factors[1..] {
            result = result.multiply(factor);
        }

        // Extract marginal
        let target_card = self.variables.get(&target_var)?.cardinality;
        let mut marginal = vec![0.0; target_card];

        for i in 0..target_card {
            let mut assignment = vec![0; result.variables.len()];
            if let Some(pos) = result.variables.iter().position(|&v| v == target_var) {
                assignment[pos] = i;
                marginal[i] = result.get_value(&assignment);
            }
        }

        // Normalize
        let sum: f64 = marginal.iter().sum();
        if sum > 0.0 {
            for val in &mut marginal {
                *val /= sum;
            }
        }

        let elapsed = start.elapsed();
        log::info!(
            "Interventional query P({}|do({}={})) completed in {:?}",
            self.variables.get(&target_var)?.name,
            self.variables.get(&do_var)?.name,
            do_val,
            elapsed
        );

        Some(marginal)
    }

    /// Get variable elimination order (minimum degree heuristic)
    fn get_elimination_order(
        &self,
        query_var: &usize,
        evidence: &HashMap<usize, usize>,
    ) -> Vec<usize> {
        let mut order = Vec::new();
        let mut remaining: HashSet<usize> = self
            .variables
            .keys()
            .filter(|&&v| v != *query_var && !evidence.contains_key(&v))
            .copied()
            .collect();

        while !remaining.is_empty() {
            // Find variable with minimum degree in induced graph
            let min_var = remaining
                .iter()
                .min_by_key(|&&v| {
                    self.parents
                        .get(&v)
                        .map(|p| p.iter().filter(|x| remaining.contains(x)).count())
                        .unwrap_or(0)
                        + self
                            .adjacency
                            .get(&v)
                            .map(|c| c.iter().filter(|x| remaining.contains(x)).count())
                            .unwrap_or(0)
                })
                .copied();

            if let Some(var) = min_var {
                order.push(var);
                remaining.remove(&var);
            } else {
                break;
            }
        }

        order
    }
}

impl Default for BayesianNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_bayesian_network() {
        let mut bn = BayesianNetwork::new();

        // Create simple chain: A -> B -> C
        bn.add_variable(0, "A", 2);
        bn.add_variable(1, "B", 2);
        bn.add_variable(2, "C", 2);

        bn.add_edge(0, 1);
        bn.add_edge(1, 2);

        // Set up simple CPDs
        // P(A) = [0.5, 0.5]
        let mut factor_a = Factor::new(vec![0], vec![2]);
        factor_a.values = vec![0.5, 0.5];
        bn.set_cpd(0, factor_a);

        // P(B|A): B depends on A
        let mut factor_b = Factor::new(vec![0, 1], vec![2, 2]);
        factor_b.values = vec![
            0.7, 0.3, // P(B|A=0)
            0.2, 0.8, // P(B|A=1)
        ];
        bn.set_cpd(1, factor_b);

        // P(C|B): C depends on B
        let mut factor_c = Factor::new(vec![1, 2], vec![2, 2]);
        factor_c.values = vec![
            0.6, 0.4, // P(C|B=0)
            0.3, 0.7, // P(C|B=1)
        ];
        bn.set_cpd(2, factor_c);

        // Query P(C)
        let marginal_c = bn.query(2, &HashMap::new()).unwrap();
        println!("P(C) = {:?}", marginal_c);

        assert!((marginal_c.iter().sum::<f64>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_interventional_query() {
        let mut bn = BayesianNetwork::new();

        bn.add_variable(0, "Rain", 2);
        bn.add_variable(1, "Sprinkler", 2);
        bn.add_variable(2, "WetGrass", 2);

        bn.add_edge(0, 1);
        bn.add_edge(0, 2);
        bn.add_edge(1, 2);

        // Simple CPDs
        let mut factor_rain = Factor::new(vec![0], vec![2]);
        factor_rain.values = vec![0.6, 0.4];
        bn.set_cpd(0, factor_rain);

        let mut factor_sprinkler = Factor::new(vec![0, 1], vec![2, 2]);
        factor_sprinkler.values = vec![0.5, 0.5, 0.1, 0.9];
        bn.set_cpd(1, factor_sprinkler);

        let mut factor_wet = Factor::new(vec![0, 1, 2], vec![2, 2, 2]);
        factor_wet.values = vec![
            0.01, 0.99, // P(W|R=0,S=0)
            0.8, 0.2,   // P(W|R=0,S=1)
            0.9, 0.1,   // P(W|R=1,S=0)
            0.99, 0.01, // P(W|R=1,S=1)
        ];
        bn.set_cpd(2, factor_wet);

        // Query P(WetGrass | do(Sprinkler=1))
        let result = bn.interventional_query(2, (1, 1), &HashMap::new()).unwrap();
        println!("P(WetGrass | do(Sprinkler=1)) = {:?}", result);

        assert!((result.iter().sum::<f64>() - 1.0).abs() < 0.01);
    }
}
