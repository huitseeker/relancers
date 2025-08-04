use binius_field::Field as BiniusField;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Random number generator wrapper for network coding
pub struct CodingRng {
    rng: ChaCha8Rng,
}

impl CodingRng {
    /// Create a new RNG with a random seed
    pub fn new() -> Self {
        Self {
            rng: ChaCha8Rng::from_os_rng(),
        }
    }

    /// Create a new RNG with a specific seed
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self {
            rng: ChaCha8Rng::from_seed(seed),
        }
    }

    /// Generate random coefficients for network coding
    pub fn generate_coefficients<F>(&mut self, count: usize) -> Vec<F>
    where
        F: BiniusField,
    {
        let mut res = Vec::with_capacity(count);
        for _ in 0..count {
            res.push(F::random(&mut self.rng));
        }
        res
    }

    /// Generate a single random coefficient
    pub fn generate_coefficient<F>(&mut self) -> F
    where
        F: BiniusField,
    {
        F::random(&mut self.rng)
    }

    /// Shuffle a slice in place using Fisher-Yates algorithm
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        use rand::seq::SliceRandom;
        slice.shuffle(&mut self.rng);
    }
}

impl Default for CodingRng {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_rng_generation() {
        let mut rng = CodingRng::from_seed([0; 32]);
        let coeffs = rng.generate_coefficients::<GF256>(10);

        assert_eq!(coeffs.len(), 10);

        // With the same seed, should produce same results
        let mut rng2 = CodingRng::from_seed([0; 32]);
        let coeffs2 = rng2.generate_coefficients::<GF256>(10);

        assert_eq!(coeffs, coeffs2);
    }
}
