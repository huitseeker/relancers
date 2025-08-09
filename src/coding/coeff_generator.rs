use crate::coding::sparse::SparseCoeffGenerator;
use crate::utils::CodingRng;
use binius_field::Field as BiniusField;
use enum_dispatch::enum_dispatch;

/// Trait for coefficient generation in RLNC
#[enum_dispatch]
pub trait CoeffGenerator<F: BiniusField> {
    /// Generate coefficients for the given number of symbols
    fn generate_coefficients(&mut self, symbols: usize) -> Vec<F>
    where
        F: From<u8> + Into<u8>;

    /// Set the seed for deterministic coefficient generation
    fn set_seed(&mut self, seed: [u8; 32]);
}

/// Implement CoeffGenerator for CodingRng
impl<F: BiniusField> CoeffGenerator<F> for CodingRng {
    fn generate_coefficients(&mut self, symbols: usize) -> Vec<F>
    where
        F: From<u8> + Into<u8>,
    {
        self.generate_coefficients(symbols)
    }

    fn set_seed(&mut self, seed: [u8; 32]) {
        *self = CodingRng::from_seed(seed);
    }
}

/// Implement CoeffGenerator for SparseCoeffGenerator
impl<F: BiniusField> CoeffGenerator<F> for SparseCoeffGenerator<F> {
    fn generate_coefficients(&mut self, symbols: usize) -> Vec<F>
    where
        F: From<u8> + Into<u8>,
    {
        self.generate_coefficients(symbols)
    }

    fn set_seed(&mut self, seed: [u8; 32]) {
        self.set_seed(seed);
    }
}

/// Enum for configured coefficient generators.
///
/// This enum allows switching between different coefficient generator implementations.
#[enum_dispatch(CoeffGenerator<F>)]
#[derive(Debug, Clone)]
pub enum ConfiguredCoeffGenerator<F: BiniusField> {
    /// Coefficient generator using a random number generator.
    CodingRng(CodingRng),
    /// Coefficient generator that produces sparse coefficients.
    SparseCoeffGenerator(SparseCoeffGenerator<F>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::sparse::SparseConfig;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_configured_coeff_generator_from_coding_rng() {
        let rng = CodingRng::new();
        let mut generator = ConfiguredCoeffGenerator::<GF256>::from(rng);

        let coeffs = CoeffGenerator::generate_coefficients(&mut generator, 5);
        assert_eq!(coeffs.len(), 5);
    }

    #[test]
    fn test_configured_coeff_generator_from_sparse() {
        let config = SparseConfig::new(0.5);
        let sparse = SparseCoeffGenerator::<GF256>::new(config);
        let mut generator = ConfiguredCoeffGenerator::<GF256>::from(sparse);

        let coeffs = generator.generate_coefficients(5);
        assert_eq!(coeffs.len(), 5);
    }

    #[test]
    fn test_configured_coeff_generator_seeding() {
        let rng = CodingRng::new();
        let mut generator = ConfiguredCoeffGenerator::<GF256>::from(rng);

        generator.set_seed([42; 32]);
    }

    #[test]
    fn test_configured_coeff_generator_sparse_seeding() {
        let config = SparseConfig::new(0.5);
        let sparse = SparseCoeffGenerator::<GF256>::new(config);
        let mut generator = ConfiguredCoeffGenerator::<GF256>::from(sparse);

        generator.set_seed([42; 32]);
    }

    #[test]
    fn test_configured_coeff_generator_deterministic() {
        let rng1 = CodingRng::from_seed([42; 32]);
        let rng2 = CodingRng::from_seed([42; 32]);

        let mut gen1 = ConfiguredCoeffGenerator::<GF256>::from(rng1);
        let mut gen2 = ConfiguredCoeffGenerator::<GF256>::from(rng2);

        let coeffs1 = gen1.generate_coefficients(3);
        let coeffs2 = gen2.generate_coefficients(3);

        assert_eq!(coeffs1, coeffs2);
    }

    #[test]
    fn test_coding_rng_coeff_generator() {
        let mut rng = CodingRng::new();
        let coeffs = rng.generate_coefficients::<GF256>(5);
        assert_eq!(coeffs.len(), 5);
    }

    #[test]
    fn test_coding_rng_seeding() {
        let mut rng1 = CodingRng::new();
        let mut rng2 = CodingRng::new();

        <CodingRng as CoeffGenerator<GF256>>::set_seed(&mut rng1, [42; 32]);
        <CodingRng as CoeffGenerator<GF256>>::set_seed(&mut rng2, [42; 32]);

        let coeffs1 = rng1.generate_coefficients::<GF256>(3);
        let coeffs2 = rng2.generate_coefficients::<GF256>(3);

        assert_eq!(coeffs1, coeffs2);
    }

    #[test]
    fn test_sparse_coeff_generator() {
        let config = SparseConfig::new(0.5);
        let mut generator = SparseCoeffGenerator::<GF256>::new(config);
        let coeffs = generator.generate_coefficients(10);
        assert_eq!(coeffs.len(), 10);
    }

    #[test]
    fn test_sparse_coeff_generator_seeding() {
        let config = SparseConfig::new(0.5);
        let mut gen1 = SparseCoeffGenerator::<GF256>::new(config);
        let mut gen2 = SparseCoeffGenerator::<GF256>::new(config);

        gen1.set_seed([42; 32]);
        gen2.set_seed([42; 32]);

        let coeffs1 = gen1.generate_coefficients(5);
        let coeffs2 = gen2.generate_coefficients(5);

        assert_eq!(coeffs1, coeffs2);
    }
}
