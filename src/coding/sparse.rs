//! Sparse coefficient generation for RLNC with configurable sparsity levels

use crate::utils::CodingRng;
use binius_field::{underlier::WithUnderlier, Field as BiniusField};

/// Configuration for sparse coefficient generation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseConfig {
    /// Target sparsity level (0.0 = all zeros, 1.0 = all non-zeros)
    pub sparsity: f64,
    /// Maximum number of non-zero coefficients
    pub max_non_zeros: Option<usize>,
    /// Minimum number of non-zero coefficients
    pub min_non_zeros: usize,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparsity: 1.0, // Full density by default
            max_non_zeros: None,
            min_non_zeros: 1,
        }
    }
}

impl SparseConfig {
    /// Create a new sparse configuration with specified sparsity
    pub fn new(sparsity: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&sparsity),
            "Sparsity must be between 0.0 and 1.0"
        );
        Self {
            sparsity,
            ..Default::default()
        }
    }

    /// Set maximum non-zero coefficients
    pub fn with_max_non_zeros(mut self, max: usize) -> Self {
        self.max_non_zeros = Some(max);
        self
    }

    /// Set minimum non-zero coefficients
    pub fn with_min_non_zeros(mut self, min: usize) -> Self {
        self.min_non_zeros = min;
        self
    }

    /// Calculate actual number of non-zero coefficients for given symbols
    pub fn calculate_non_zeros(&self, symbols: usize) -> usize {
        let target = (symbols as f64 * self.sparsity).round() as usize;
        let actual = target.max(self.min_non_zeros);

        if let Some(max) = self.max_non_zeros {
            actual.min(max).min(symbols)
        } else {
            actual.min(symbols)
        }
    }
}

/// Sparse coefficient generator for RLNC
#[derive(Debug, Clone)]
pub struct SparseCoeffGenerator<F: BiniusField> {
    config: SparseConfig,
    rng: CodingRng,
    _marker: std::marker::PhantomData<F>,
}

impl<F: BiniusField> SparseCoeffGenerator<F> {
    /// Create a new sparse coefficient generator
    pub fn new(config: SparseConfig) -> Self {
        Self {
            config,
            rng: CodingRng::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new sparse coefficient generator with seed
    pub fn with_seed(config: SparseConfig, seed: [u8; 32]) -> Self {
        Self {
            config,
            rng: CodingRng::from_seed(seed),
            _marker: std::marker::PhantomData,
        }
    }

    /// Generate sparse coefficients for RLNC
    pub fn generate_coefficients(&mut self, symbols: usize) -> Vec<F>
    where
        F: WithUnderlier<Underlier = u8>,
    {
        let mut coeffs = vec![F::ZERO; symbols];
        let non_zeros = self.config.calculate_non_zeros(symbols);

        if non_zeros == 0 {
            return coeffs;
        }

        // Generate positions for non-zero coefficients
        let mut positions: Vec<usize> = (0..symbols).collect();
        self.rng.shuffle(&mut positions);

        // Fill non-zero positions with random coefficients
        for &pos in positions.iter().take(non_zeros) {
            let coeff = self.rng.generate_coefficient();
            coeffs[pos] = coeff;
        }

        // Ensure at least one non-zero coefficient if min_non_zeros > 0
        if coeffs.iter().all(|c| c.is_zero()) && non_zeros > 0 {
            coeffs[0] = F::from_underlier(1u8);
        }

        coeffs
    }

    /// Get the current configuration
    pub fn config(&self) -> &SparseConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: SparseConfig) {
        self.config = config;
    }

    /// Set the seed for deterministic coefficient generation
    pub fn set_seed(&mut self, seed: [u8; 32]) {
        self.rng = CodingRng::from_seed(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_sparse_config() {
        let config = SparseConfig::new(0.5);
        assert_eq!(config.sparsity, 0.5);
        assert_eq!(config.calculate_non_zeros(10), 5);
        assert_eq!(config.calculate_non_zeros(3), 2); // min_non_zeros = 1
    }

    #[test]
    fn test_sparse_config_bounds() {
        let config = SparseConfig::new(0.3)
            .with_min_non_zeros(2)
            .with_max_non_zeros(5);

        assert_eq!(config.calculate_non_zeros(10), 3); // 30% of 10 = 3
        assert_eq!(config.calculate_non_zeros(5), 2); // min constraint
        assert_eq!(config.calculate_non_zeros(20), 5); // max constraint
    }

    #[test]
    fn test_sparse_generator_full_density() {
        let config = SparseConfig::new(1.0);
        let mut generator = SparseCoeffGenerator::<GF256>::new(config);

        let coeffs = generator.generate_coefficients(5);
        assert_eq!(coeffs.len(), 5);
        assert!(coeffs.iter().all(|c| !c.is_zero()));
    }

    #[test]
    fn test_sparse_generator_zero_density() {
        let config = SparseConfig::new(0.0).with_min_non_zeros(0);
        let mut generator = SparseCoeffGenerator::<GF256>::new(config);

        let coeffs = generator.generate_coefficients(5);
        assert_eq!(coeffs.len(), 5);
        assert!(coeffs.iter().all(|c| c.is_zero()));
    }

    #[test]
    fn test_sparse_generator_partial_density() {
        let config = SparseConfig::new(0.4);
        let mut generator = SparseCoeffGenerator::<GF256>::new(config);

        let coeffs = generator.generate_coefficients(10);
        let non_zeros = coeffs.iter().filter(|c| !c.is_zero()).count();
        assert_eq!(non_zeros, 4); // 0.4 * 10 = 4
    }

    #[test]
    fn test_sparse_generator_deterministic() {
        let config = SparseConfig::new(0.5);
        let mut gen1 = SparseCoeffGenerator::<GF256>::with_seed(config, [42; 32]);
        let mut gen2 = SparseCoeffGenerator::<GF256>::with_seed(config, [42; 32]);

        let coeffs1 = gen1.generate_coefficients(5);
        let coeffs2 = gen2.generate_coefficients(5);

        assert_eq!(coeffs1, coeffs2);
    }

    #[test]
    fn test_sparse_generator_edge_cases() {
        let config = SparseConfig::new(0.0).with_min_non_zeros(1);
        let mut generator = SparseCoeffGenerator::<GF256>::new(config);

        // Should ensure at least one non-zero
        let coeffs = generator.generate_coefficients(3);
        assert!(coeffs.iter().any(|c| !c.is_zero()));
    }
}
