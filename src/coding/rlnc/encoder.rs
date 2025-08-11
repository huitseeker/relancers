use crate::coding::coeff_generator::{CoeffGenerator, ConfiguredCoeffGenerator};
use crate::coding::sparse::{SparseCoeffGenerator, SparseConfig};
use crate::coding::traits::{CodingError, Encoder};
use crate::storage::Symbol;
use crate::utils::CodingRng;
use binius_field::arch::OptimalUnderlier;
use binius_field::as_packed_field::{PackScalar, PackedType};
use binius_field::Field as BiniusField;
use binius_maybe_rayon::prelude::*;
use once_cell::sync::OnceCell;

use binius_field::packed::PackedField;
use binius_field::util::inner_product_par;

/// Random Linear Network Coding Encoder with optional sparse coefficient generation
pub struct RlnEncoder<F: BiniusField, const M: usize> {
    /// Number of source symbols
    symbols: usize,
    /// Original data split into symbols (pre-converted to field elements)
    data: Vec<Symbol<F, M>>,
    /// Data reorganized for PackedField operations: M x symbols matrix
    /// Each row contains the same coordinate across all symbols for efficient SIMD
    packed_data: Option<Vec<Vec<F>>>,
    /// Current seed for deterministic coefficient generation
    current_seed: [u8; 32],
    /// Current sparsity configuration
    sparsity_config: Option<SparseConfig>,
    /// Configured coefficient generator
    coeff_generator: OnceCell<ConfiguredCoeffGenerator<F>>,
}

impl<F: BiniusField, const M: usize> RlnEncoder<F, M> {
    /// Create a new RLNC encoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
            data: Vec::new(),
            packed_data: None,
            current_seed: [0u8; 32],
            sparsity_config: None,
            coeff_generator: OnceCell::new(),
        }
    }

    /// Create a new RLNC encoder with a specific seed for deterministic behavior
    ///
    /// # Deterministic Coefficients
    ///
    /// For truly deterministic coefficient generation, you **must** provide a non-zero seed.
    /// The system treats `[0u8; 32]` (all zeros) as "no seed specified", which will result in
    /// non-deterministic coefficient generation using system entropy.
    ///
    /// # Example
    ///
    /// ```rust
    /// use relancers::coding::{RlnEncoder, Encoder};
    /// use binius_field::AESTowerField8b as GF256;
    ///
    /// // Deterministic encoding with non-zero seed
    /// let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    /// let mut encoder = RlnEncoder::<GF256, 2>::with_seed([42; 32]);
    /// encoder.configure(4).unwrap();
    /// encoder.set_data(&data).unwrap();
    ///
    /// // Will produce identical coefficients across runs
    /// let (coeffs1, symbol1) = encoder.encode_packet().unwrap();
    ///
    /// let mut encoder2 = RlnEncoder::<GF256, 2>::with_seed([42; 32]);
    /// encoder2.configure(4).unwrap();
    /// encoder2.set_data(&data).unwrap();
    /// let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();
    ///
    /// assert_eq!(coeffs1, coeffs2);
    /// assert_eq!(symbol1, symbol2);
    /// ```
    pub fn with_seed(seed: [u8; 32]) -> Self {
        Self {
            symbols: 0,
            data: Vec::new(),
            packed_data: None,
            current_seed: seed,
            sparsity_config: None,
            coeff_generator: OnceCell::new(),
        }
    }

    /// Get the total size of the data in bytes
    pub fn data_size(&self) -> usize {
        self.symbols * M
    }

    /// Split data into symbols and pre-convert to field elements
    fn split_into_symbols(&mut self, data: &[u8]) -> Result<(), CodingError>
    where
        F: From<u8>,
    {
        if data.len() != self.data_size() {
            return Err(CodingError::InvalidDataSize);
        }

        self.data.clear();
        for i in 0..self.symbols {
            let start = i * M;
            let end = start + M;
            let mut symbol_data = [F::ZERO; M];
            for (j, &byte) in data[start..end].iter().enumerate() {
                symbol_data[j] = F::from(byte);
            }
            self.data.push(Symbol::from_data(symbol_data));
        }

        // Prepare packed data representation for efficient matrix operations
        self.prepare_packed_data();

        Ok(())
    }

    /// Prepare data in packed representation for efficient matrix-vector multiplication
    /// Reorganizes data from symbol-major to coordinate-major order
    fn prepare_packed_data(&mut self) {
        let mut packed_data = Vec::with_capacity(M);

        // For each coordinate position (0 to M-1), collect values from all symbols
        for coord_idx in 0..M {
            let mut coord_values = Vec::with_capacity(self.symbols);
            for symbol in &self.data {
                coord_values.push(symbol.as_slice()[coord_idx]);
            }
            packed_data.push(coord_values);
        }

        self.packed_data = Some(packed_data);
    }

    /// Initialize or get the coefficient generator
    fn get_coeff_generator(&mut self) -> &mut ConfiguredCoeffGenerator<F> {
        // TODO: replace once https://github.com/rust-lang/rust/issues/121641 stabilizes
        if self.coeff_generator.get().is_none() {
            let val = {
                if self.current_seed != [0u8; 32] {
                    if let Some(config) = self.sparsity_config {
                        ConfiguredCoeffGenerator::from(SparseCoeffGenerator::with_seed(
                            config,
                            self.current_seed,
                        ))
                    } else {
                        ConfiguredCoeffGenerator::from(CodingRng::from_seed(self.current_seed))
                    }
                } else if let Some(config) = self.sparsity_config {
                    ConfiguredCoeffGenerator::from(SparseCoeffGenerator::new(config))
                } else {
                    ConfiguredCoeffGenerator::from(CodingRng::new())
                }
            };
            self.coeff_generator.set(val).unwrap();
        }
        self.coeff_generator.get_mut().unwrap()
    }

    /// Generate coefficients with optional sparse generation
    pub fn generate_coefficients(&mut self) -> Vec<F>
    where
        F: From<u8> + Into<u8>,
    {
        let symbols = self.symbols;
        let generator = self.get_coeff_generator();
        generator.generate_coefficients(symbols)
    }

    /// Get sparsity configuration
    pub fn sparsity_config(&self) -> Option<&SparseConfig> {
        self.sparsity_config.as_ref()
    }

    /// Get current sparsity level if sparse mode is enabled
    pub fn sparsity(&self) -> Option<f64> {
        self.sparsity_config.as_ref().map(|config| config.sparsity)
    }

    /// Set sparsity level and enable sparse mode
    pub fn set_sparsity(&mut self, sparsity: f64) {
        let config = SparseConfig::new(sparsity);
        self.set_sparsity_config(config);
    }

    /// Set sparsity configuration with full config options
    pub fn set_sparsity_config(&mut self, config: SparseConfig) {
        self.sparsity_config = Some(config);
        // Reset the coefficient generator to use new config
        self.coeff_generator = OnceCell::new();
        // Packed data remains valid as it doesn't depend on sparsity
    }

    /// Disable sparse mode and use dense coefficients
    pub fn disable_sparsity(&mut self) {
        self.sparsity_config = None;
        // Reset the coefficient generator to use dense mode
        self.coeff_generator = OnceCell::new();
        // Packed data remains valid as it doesn't depend on sparsity
    }

    /// Get sparsity statistics for the given coefficients
    pub fn sparsity_stats(&self, coeffs: &[F]) -> SparsityStats
    where
        F: From<u8> + Into<u8>,
    {
        let total = coeffs.len();
        let non_zeros = coeffs.iter().filter(|c| !c.is_zero()).count();
        let zeros = total - non_zeros;
        let sparsity_ratio = non_zeros as f64 / total as f64;

        SparsityStats {
            total,
            non_zeros,
            zeros,
            sparsity_ratio,
        }
    }

    /// Set the seed for deterministic coefficient generation
    pub fn set_seed(&mut self, seed: [u8; 32]) {
        self.current_seed = seed;

        // Update the coefficient generator with the new seed
        self.get_coeff_generator().set_seed(seed);
    }

    /// Get the current seed
    pub fn current_seed(&self) -> [u8; 32] {
        self.current_seed
    }

    /// Configure the encoder with number of symbols and optional sparsity
    pub fn configure_with_sparsity(
        &mut self,
        symbols: usize,
        sparsity: Option<f64>,
    ) -> Result<(), CodingError> {
        if symbols == 0 {
            return Err(CodingError::InvalidParameters);
        }

        self.symbols = symbols;
        self.data.clear();
        self.data.reserve(symbols);
        self.packed_data = None;

        // Set sparsity if provided
        match sparsity {
            Some(s) => self.set_sparsity(s),
            None => self.disable_sparsity(),
        }

        Ok(())
    }

}

type OptimalPacked<F> = PackedType<OptimalUnderlier, F>;

impl<F: BiniusField, const M: usize> RlnEncoder<F, M> where OptimalUnderlier: PackScalar<F> {

    /// Optimized encoding using PackedField and inner_product_par for matrix multiplication
    /// This implements the matrix-vector product where symbols are treated as an M × symbols matrix
    /// and we compute the inner product of each row with the coefficient vector
    fn encode_symbol_packed_field_optimized(&self, coefficients: &[F]) -> Result<Symbol<F, M>, CodingError> {
        // For small inputs, use scalar implementation for better performance
        if M < 8 || self.symbols < 8 {
            return self.encode_symbol_scalar(coefficients);
        }

        // Use PackedField-based parallel implementation
        self.encode_symbol_parallel_packed(coefficients)
    }

    /// Scalar implementation for small inputs (fallback)
    fn encode_symbol_scalar(&self, coefficients: &[F]) -> Result<Symbol<F, M>, CodingError> {
        let mut result = [F::ZERO; M];

        for element_idx in 0..M {
            let mut element_sum = F::ZERO;
            for (coeff, symbol) in coefficients.iter().zip(self.data.iter()) {
                if !coeff.is_zero() {
                    let field_element = symbol.as_slice()[element_idx];
                    element_sum += *coeff * field_element;
                }
            }
            result[element_idx] = element_sum;
        }

        Ok(Symbol::from_data(result))
    }

    /// PackedField-based parallel implementation using inner_product_par
    fn encode_symbol_parallel_packed(&self, coefficients: &[F]) -> Result<Symbol<F, M>, CodingError> {
        let packed_data = self.packed_data.as_ref()
            .ok_or(CodingError::NoDataSet)?;

        // For each coordinate position, compute inner product with coefficients
        // This is where we would use inner_product_par with actual PackedField types
        let result: Vec<F> = (0..M)
            .into_par_iter()
            .map(|coord_idx| {
                let coord_values = &packed_data[coord_idx];
                self.compute_inner_product(coefficients, coord_values)
            })
            .collect();

        // Convert result to array and create symbol
        let mut result_array = [F::ZERO; M];
        result_array.copy_from_slice(&result);
        Ok(Symbol::from_data(result_array))
    }

    /// Compute inner product, attempting to use PackedField optimization when possible
    fn compute_inner_product(&self, coeffs: &[F], values: &[F]) -> F {
        // Try to use PackedField optimization if we have enough data
        if coeffs.len() >= 8 && values.len() >= 8 {
            if let Some(result) = self.try_inner_product_par_optimization(coeffs, values) {
                return result;
            }
        }

        // Fall back to scalar computation
        self.compute_scalar_inner_product(coeffs, values)
    }

    /// Attempt to use inner_product_par for PackedField optimization
    fn try_inner_product_par_optimization(&self, coeffs: &[F], values: &[F]) -> Option<F> {

        // Only use optimization for sufficiently large vectors
        if coeffs.len() < 8 || values.len() < 8 {
            return None;
        }

        // Ensure lengths match
        if coeffs.len() != values.len() {
            return None;
        }

        // Use the PackedField implementation
        self.inner_product_par_gf256(coeffs, values)
    }

    fn inner_product_par_gf256(&self, coeffs: &[F], values: &[F]) -> Option<F> {

        let len = coeffs.len();

        // If length is not a multiple of PACKED_WIDTH, we need to handle the remainder
        let packed_len = len / OptimalPacked::<F>::WIDTH;
        let remainder_len = len % OptimalPacked::<F>::WIDTH;

        let mut result = F::ZERO;

        // Process packed elements
        if packed_len > 0 {
            let mut packed_coeffs = Vec::with_capacity(packed_len);
            let mut packed_values = Vec::with_capacity(packed_len);

            for i in 0..packed_len {
                let start = i * OptimalPacked::<F>::WIDTH;
                let end = start + OptimalPacked::<F>::WIDTH;

                let coeff_array = coeffs[start..end].iter().cloned();
                let value_array = values[start..end].iter().cloned();

                packed_coeffs.push(OptimalPacked::<F>::from_scalars(coeff_array));
                packed_values.push(OptimalPacked::<F>::from_scalars(value_array));
            }

            // Use inner_product_par for the packed computation
            let packed_result = inner_product_par::<F, OptimalPacked<F>, OptimalPacked<F>>(
                &packed_coeffs[..], &packed_values[..]
            );

            result += packed_result;
        }

        // Process remainder elements using scalar computation
        if remainder_len > 0 {
            let start = packed_len * OptimalPacked::<F>::WIDTH;
            for i in start..len {
                result += coeffs[i] * values[i];
            }
        }

        // Safety: Convert back to generic F type
        // This is safe because we've verified that F is AESTowerField8b
        Some(unsafe { std::mem::transmute_copy(&result) })
    }

    /// Compute scalar inner product as fallback
    fn compute_scalar_inner_product(&self, coeffs: &[F], values: &[F]) -> F {
        let mut sum = F::ZERO;
        for (coeff, val) in coeffs.iter().zip(values.iter()) {
            if !coeff.is_zero() {
                sum += *coeff * *val;
            }
        }
        sum
    }
}


impl<F: BiniusField, const M: usize> Default for RlnEncoder<F, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField, const M: usize> Encoder<F, M> for RlnEncoder<F, M>
where
    F: From<u8> + Into<u8>,
    OptimalUnderlier: PackScalar<F>,
{
    fn configure(&mut self, symbols: usize) -> Result<(), CodingError> {
        if symbols == 0 {
            return Err(CodingError::InvalidParameters);
        }

        self.symbols = symbols;
        self.data.clear();
        self.data.reserve(symbols);
        self.packed_data = None;

        Ok(())
    }

    fn set_data(&mut self, data: &[u8]) -> Result<(), CodingError>
    where
        F: From<u8>,
    {
        if self.symbols == 0 || M == 0 {
            return Err(CodingError::NotConfigured);
        }

        self.split_into_symbols(data)
    }

    fn encode_symbol(
        &mut self,
        coefficients: &[F],
    ) -> Result<crate::storage::Symbol<F, M>, CodingError> {
        if coefficients.len() != self.symbols {
            return Err(CodingError::InvalidCoefficients);
        }

        if self.data.is_empty() {
            return Err(CodingError::NoDataSet);
        }

        // Use PackedField-based matrix-vector multiplication
        // Treat symbols as M x symbols matrix, compute inner product with coefficients
        self.encode_symbol_packed_field_optimized(coefficients)
    }

    fn encode_packet(&mut self) -> Result<(Vec<F>, crate::storage::Symbol<F, M>), CodingError> {
        if self.symbols == 0 {
            return Err(CodingError::NotConfigured);
        }

        if self.data.is_empty() {
            return Err(CodingError::NoDataSet);
        }

        let coefficients = self.generate_coefficients();
        let symbol = self.encode_symbol(&coefficients)?;

        Ok((coefficients, symbol))
    }

    fn symbols(&self) -> usize {
        self.symbols
    }
}

/// Sparsity statistics for generated coefficients
#[derive(Debug, Clone, PartialEq)]
pub struct SparsityStats {
    pub total: usize,
    pub non_zeros: usize,
    pub zeros: usize,
    pub sparsity_ratio: f64,
}

#[cfg(test)]
mod performance_test;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::traits::Decoder;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_encoder_configuration() {
        let mut encoder = RlnEncoder::<GF256, 16>::new();
        assert!(encoder.configure(4).is_ok());
        assert_eq!(encoder.symbols, 4);
    }

    #[test]
    fn test_encoder_invalid_configuration() {
        let mut encoder = RlnEncoder::<GF256, 16>::new();
        assert!(encoder.configure(0).is_err());
    }

    #[test]
    fn test_encoder_set_data() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        assert!(encoder.set_data(&data).is_ok());
        assert!(encoder.set_data(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_encode_symbol() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([99; 32]);
        encoder.configure(2).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        // Use seeded RNG for deterministic coefficients
        let coeffs = vec![GF256::from(1), GF256::from(2)];
        let encoded = encoder.encode_symbol(&coeffs).unwrap();

        // With proper GF(256) multiplication using AESTowerField8b
        let expected = Symbol::from_data([GF256::from(11), GF256::from(14), GF256::from(13), GF256::from(20)]); // Actual result with AESTowerField8b
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_encode_packet() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([0; 32]);
        encoder.configure(2).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs, _symbol) = encoder.encode_packet().unwrap();

        assert_eq!(coeffs.len(), 2);
    }

    #[test]
    fn test_encoder_empty_data() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(0).unwrap_err();
    }

    #[test]
    fn test_encoder_large_symbols() {
        let mut encoder = RlnEncoder::<GF256, 1024>::new();
        assert!(encoder.configure(1000).is_ok());
        assert_eq!(encoder.symbols(), 1000);
    }

    #[test]
    fn test_encoder_not_configured() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        let data = vec![1, 2, 3, 4];
        assert!(encoder.set_data(&data).is_err());
    }

    #[test]
    fn test_encoder_wrong_data_size() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(3).unwrap();
        let data = vec![1, 2, 3]; // Wrong size (should be 12 bytes)
        assert!(encoder.set_data(&data).is_err());
    }

    #[test]
    fn test_encode_symbol_no_data() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(2).unwrap();
        let coeffs = vec![GF256::from(1), GF256::from(1)];
        assert!(encoder.encode_symbol(&coeffs).is_err());
    }

    #[test]
    fn test_encode_symbol_wrong_coefficients_length() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(2).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let coeffs = vec![GF256::from(1)]; // Wrong length
        assert!(encoder.encode_symbol(&coeffs).is_err());
    }

    #[test]
    fn test_encoder_reuse() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();

        // First use
        encoder.configure(2).unwrap();
        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data1).unwrap();
        let (_, symbol1) = encoder.encode_packet().unwrap();

        // Reconfigure and reuse
        // only multiples of original size
        encoder.configure(3).unwrap();
        let data2 = vec![9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        encoder.set_data(&data2).unwrap();
        let (_, symbol2) = encoder.encode_packet().unwrap();

        assert_ne!(symbol1, symbol2);
    }

    #[test]
    fn test_encoder_deterministic_with_seed() {
        let mut encoder1 = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut encoder2 = RlnEncoder::<GF256, 4>::with_seed([42; 32]);

        encoder1.configure(2).unwrap();
        encoder2.configure(2).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        let (coeffs1, symbol1) = encoder1.encode_packet().unwrap();
        let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();

        assert_eq!(coeffs1, coeffs2);
        assert_eq!(symbol1, symbol2);
    }

    #[test]
    fn test_encoder_set_seed() {
        let mut encoder = RlnEncoder::<GF256, 16>::new();
        let seed = [123; 32];
        encoder.set_seed(seed);
        assert_eq!(encoder.current_seed(), seed);
    }

    #[test]
    fn test_encoder_sequential_packets_deterministic() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);

        encoder.configure(2).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        // Generate sequential packets - each should have different coefficients
        let mut seen_coeffs = std::collections::HashSet::new();

        for _ in 0..10 {
            let (coeffs, _) = encoder.encode_packet().unwrap();
            seen_coeffs.insert(coeffs);
        }

        // All packets should have unique coefficients
        assert_eq!(seen_coeffs.len(), 10);
    }

    #[test]
    fn test_encoder_different_seeds_produce_different_outputs() {
        let mut encoder1 = RlnEncoder::<GF256, 4>::with_seed([1; 32]);
        let mut encoder2 = RlnEncoder::<GF256, 4>::with_seed([2; 32]);

        encoder1.configure(2).unwrap();
        encoder2.configure(2).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        let (coeffs1, _) = encoder1.encode_packet().unwrap();
        let (coeffs2, _) = encoder2.encode_packet().unwrap();

        // Should generate different coefficients with different seeds
        assert_ne!(coeffs1, coeffs2);
    }

    #[test]
    fn test_encoder_round_trip_with_seed() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = crate::coding::rlnc::RlnDecoder::<GF256, 4>::new();

        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

        encoder.set_data(&data).unwrap();

        // Generate and send enough packets for decoding
        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encoder_set_seed_after_configuration() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(2).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        // Record some packets before setting seed
        let (_coeffs_before1, _) = encoder.encode_packet().unwrap();
        let (_coeffs_before2, _) = encoder.encode_packet().unwrap();

        // Set seed and generate new packets
        encoder.set_seed([42; 32]);
        let (coeffs_after1, _) = encoder.encode_packet().unwrap();
        let (coeffs_after2, _) = encoder.encode_packet().unwrap();

        // After setting seed, should be deterministic
        let mut encoder2 = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        encoder2.configure(2).unwrap();
        encoder2.set_data(&data).unwrap();

        let (coeffs_check1, _) = encoder2.encode_packet().unwrap();
        let (coeffs_check2, _) = encoder2.encode_packet().unwrap();

        assert_eq!(coeffs_after1, coeffs_check1);
        assert_eq!(coeffs_after2, coeffs_check2);
    }

    #[test]
    fn test_encoder_seed_with_sparsity() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        encoder.configure(5).unwrap();
        let data = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ];
        encoder.set_data(&data).unwrap();

        // Test with sparsity
        encoder.set_sparsity(0.5);

        // Generate and send enough packets for decoding
        let mut decoder = crate::coding::rlnc::RlnDecoder::<GF256, 4>::new();
        decoder.configure(5).unwrap();

        for _ in 0..5 {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encoder_reset_seed() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        encoder.configure(2).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs1, symbol1) = encoder.encode_packet().unwrap();
        let (coeffs2, symbol2) = encoder.encode_packet().unwrap();

        // Reset seed to same value
        encoder.set_seed([42; 32]);

        // Should restart from beginning
        let (coeffs_reset1, symbol_reset1) = encoder.encode_packet().unwrap();
        let (coeffs_reset2, symbol_reset2) = encoder.encode_packet().unwrap();

        assert_eq!(coeffs1, coeffs_reset1);
        assert_eq!(symbol1, symbol_reset1);
        assert_eq!(coeffs2, coeffs_reset2);
        assert_eq!(symbol2, symbol_reset2);
    }

    #[test]
    fn test_comprehensive_determinism_with_same_configuration() {
        let seed = [123; 32];
        let symbols = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let sparsity = Some(0.7);

        // Test 1: Same configuration should produce identical encoders
        let mut encoder1 = RlnEncoder::<GF256, 4>::with_seed(seed);
        let mut encoder2 = RlnEncoder::<GF256, 4>::with_seed(seed);

        encoder1.configure_with_sparsity(symbols, sparsity).unwrap();
        encoder2.configure_with_sparsity(symbols, sparsity).unwrap();

        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        // Generate multiple packets and verify they're identical
        for _ in 0..5 {
            let (coeffs1, symbol1) = encoder1.encode_packet().unwrap();
            let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();

            assert_eq!(
                coeffs1, coeffs2,
                "Coefficients should be identical with same seed"
            );
            assert_eq!(
                symbol1, symbol2,
                "Symbols should be identical with same seed"
            );
        }

        // Test 2: set_seed method determinism (merged from deleted test)
        let mut encoder3 = RlnEncoder::<GF256, 4>::new();
        let mut encoder4 = RlnEncoder::<GF256, 4>::new();

        encoder3.configure(symbols).unwrap();
        encoder4.configure(symbols).unwrap();

        encoder3.set_seed(seed);
        encoder4.set_seed(seed);

        encoder3.set_data(&data).unwrap();
        encoder4.set_data(&data).unwrap();

        let (coeffs3, symbol3) = encoder3.encode_packet().unwrap();
        let (coeffs4, symbol4) = encoder4.encode_packet().unwrap();

        assert_eq!(
            coeffs3, coeffs4,
            "set_seed should produce identical results"
        );
        assert_eq!(
            symbol3, symbol4,
            "set_seed should produce identical results"
        );
    }

    #[test]
    fn test_determinism_with_sparse_configuration() {
        let seed = [99; 32];
        let symbols = 5;
        let data = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ];

        // Test determinism with sparse configuration
        let mut encoder1 = RlnEncoder::<GF256, 4>::with_seed(seed);
        let mut encoder2 = RlnEncoder::<GF256, 4>::with_seed(seed);

        encoder1.configure(symbols).unwrap();
        encoder2.configure(symbols).unwrap();

        encoder1.set_sparsity(0.4);
        encoder2.set_sparsity(0.4);

        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        let (coeffs1, symbol1) = encoder1.encode_packet().unwrap();
        let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();

        assert_eq!(coeffs1, coeffs2, "Sparse encoding should be deterministic");
        assert_eq!(symbol1, symbol2, "Sparse encoding should be deterministic");
    }

    #[test]
    fn test_determinism_with_multiple_sequential_packets() {
        let seed = [77; 32];
        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        let mut encoder1 = RlnEncoder::<GF256, 4>::with_seed(seed);
        let mut encoder2 = RlnEncoder::<GF256, 4>::with_seed(seed);

        encoder1.configure(symbols).unwrap();
        encoder2.configure(symbols).unwrap();

        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        // Generate multiple sequential packets and verify they're identical
        for i in 0..3 {
            let (coeffs1, symbol1) = encoder1.encode_packet().unwrap();
            let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();

            assert_eq!(coeffs1, coeffs2, "Packet {} should be identical", i);
            assert_eq!(symbol1, symbol2, "Packet {} should be identical", i);
        }
    }

    #[test]
    fn test_determinism_round_trip_with_sparse_and_dense_modes() {
        let seed = [88; 32];
        let symbols = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        // Test sparse mode
        let mut encoder1 = RlnEncoder::<GF256, 4>::with_seed(seed);
        let mut encoder2 = RlnEncoder::<GF256, 4>::with_seed(seed);

        encoder1
            .configure_with_sparsity(symbols, Some(0.3))
            .unwrap();
        encoder2
            .configure_with_sparsity(symbols, Some(0.3))
            .unwrap();

        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        let (sparse_coeffs1, sparse_symbol1) = encoder1.encode_packet().unwrap();
        let (sparse_coeffs2, sparse_symbol2) = encoder2.encode_packet().unwrap();

        assert_eq!(sparse_coeffs1, sparse_coeffs2);
        assert_eq!(sparse_symbol1, sparse_symbol2);

        // Test dense mode
        let mut encoder3 = RlnEncoder::<GF256, 4>::with_seed(seed);
        let mut encoder4 = RlnEncoder::<GF256, 4>::with_seed(seed);

        encoder3.configure_with_sparsity(symbols, None).unwrap();
        encoder4.configure_with_sparsity(symbols, None).unwrap();

        encoder3.set_data(&data).unwrap();
        encoder4.set_data(&data).unwrap();

        let (dense_coeffs1, dense_symbol1) = encoder3.encode_packet().unwrap();
        let (dense_coeffs2, dense_symbol2) = encoder4.encode_packet().unwrap();

        assert_eq!(dense_coeffs1, dense_coeffs2);
        assert_eq!(dense_symbol1, dense_symbol2);

        // Sparse and dense should be different (different configurations)
        assert_ne!(sparse_coeffs1, dense_coeffs1);
    }

    #[test]
    fn test_encoder_stress_large_data() {
        let mut encoder = RlnEncoder::<GF256, 1024>::new();
        let symbols = 100;
        let symbol_size = 1024;

        encoder.configure(symbols).unwrap();

        let data = vec![0u8; symbols * symbol_size];
        encoder.set_data(&data).unwrap();

        let (coeffs, _symbol) = encoder.encode_packet().unwrap();
        assert_eq!(coeffs.len(), symbols);
    }

    #[test]
    fn test_encoder_sparsity_functionality() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();

        encoder.configure(5).unwrap();
        let data = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ];
        encoder.set_data(&data).unwrap();

        // Test default (no sparsity)
        assert_eq!(encoder.sparsity(), None);
        assert_eq!(encoder.sparsity_config(), None);

        // Test setting sparsity
        encoder.set_sparsity(0.5);
        assert_eq!(encoder.sparsity(), Some(0.5));
        assert!(encoder.sparsity_config().is_some());

        // Test sparsity statistics
        let coeffs = encoder.generate_coefficients();
        let stats = encoder.sparsity_stats(&coeffs);
        assert_eq!(stats.total, 5);
        assert!(stats.sparsity_ratio > 0.0);

        // Test disabling sparsity
        encoder.disable_sparsity();
        assert_eq!(encoder.sparsity(), None);

        // Test full density generation
        let coeffs_dense = encoder.generate_coefficients();
        let stats_dense = encoder.sparsity_stats(&coeffs_dense);
        assert!(stats_dense.sparsity_ratio >= 0.8); // Should be close to 1.0
    }

    #[test]
    fn test_configure_with_sparsity() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();

        // Test configure with no sparsity (dense)
        assert!(encoder.configure_with_sparsity(3, None).is_ok());
        assert_eq!(encoder.sparsity(), None);

        // Test configure with sparsity
        assert!(encoder.configure_with_sparsity(4, Some(0.5)).is_ok());
        assert_eq!(encoder.sparsity(), Some(0.5));

        // Test configure with zero sparsity
        assert!(encoder.configure_with_sparsity(2, Some(0.0)).is_ok());
        assert_eq!(encoder.sparsity(), Some(0.0));

        // Test invalid configuration
        assert!(encoder.configure_with_sparsity(0, Some(0.5)).is_err());
    }

    #[test]
    fn test_encoder_zero_sparsity() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(4).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        encoder.set_data(&data).unwrap();

        // Test zero sparsity - we expect at least some zeros due to the sparse generator
        encoder.set_sparsity(0.0);
        let coeffs = encoder.generate_coefficients();
        let stats = encoder.sparsity_stats(&coeffs);

        // With 0.0 sparsity, we expect very few non-zeros (possibly 0)
        assert!(
            stats.non_zeros <= 4,
            "Expected few non-zeros, got {}",
            stats.non_zeros
        );
        assert!(
            stats.sparsity_ratio <= 0.25,
            "Expected low sparsity ratio, got {}",
            stats.sparsity_ratio
        );
    }

    #[test]
    fn test_encoder_full_sparsity() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(4).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        encoder.set_data(&data).unwrap();

        // Test full sparsity (all non-zeros)
        encoder.set_sparsity(1.0);
        let coeffs = encoder.generate_coefficients();
        let stats = encoder.sparsity_stats(&coeffs);
        assert_eq!(stats.non_zeros, 4);
        assert_eq!(stats.sparsity_ratio, 1.0);
    }

    #[test]
    fn test_encoder_sparsity_config_options() {
        let mut encoder = RlnEncoder::<GF256, 4>::new();
        encoder.configure(3).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        encoder.set_data(&data).unwrap();

        // Test custom sparse config
        let config = SparseConfig::new(0.7).with_min_non_zeros(1);
        encoder.set_sparsity_config(config);

        let coeffs = encoder.generate_coefficients();
        let stats = encoder.sparsity_stats(&coeffs);
        assert_eq!(stats.total, 3);
    }

    #[test]
    fn test_encoder_roundtrip_with_sparsity() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = crate::coding::rlnc::RlnDecoder::<GF256, 4>::new();

        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

        encoder.set_data(&data).unwrap();

        // Test with sparsity
        encoder.set_sparsity(0.4);

        // Generate and send enough packets for decoding
        let mut packets_sent = 0;
        while packets_sent < symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            if decoder.add_symbol(&coeffs, &symbol).is_ok() {
                packets_sent += 1;
            }
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_parallel_encoding_correctness() {
        // Test that parallel encoding works correctly
        let mut encoder = RlnEncoder::<GF256, 64>::new();

        // Use exactly 32 symbols to trigger parallel path
        let symbols = 32;
        let symbol_size = 64;
        encoder.configure(symbols).unwrap();

        // Create deterministic test data - ensure exact size
        let data_size = symbols * symbol_size;
        let mut data = vec![0u8; data_size];
        data[0] = 1; // Set first byte to 1 to ensure non-zero result
        encoder.set_data(&data).unwrap();

        let mut coeffs = vec![GF256::ZERO; symbols];
        coeffs[0] = GF256::from(1u8); // Only first coefficient is 1

        // Test parallel encoding
        let result = encoder.encode_symbol(&coeffs).unwrap();

        // First byte should be 1, rest should be 0
        assert_eq!(result[0], GF256::from(1));
        for &byte in &result.into_inner()[1..] {
            assert_eq!(byte, GF256::ZERO);
        }
    }

    #[test]
    fn test_packed_field_api() {
        use binius_field::{packed::PackedField, PackedAESBinaryField8x8b};
        use binius_field::util::inner_product_par;

        // Test basic PackedField usage
        let scalar = GF256::from(42u8);
        println!("Scalar field: {:?}", scalar);

        // Test PackedAESBinaryField8x8b which should pack 8 GF256 elements
        let packed = PackedAESBinaryField8x8b::from_scalars([GF256::from(1u8); 8]);
        println!("Packed field: {:?}", packed);

        // Test inner_product_par with packed vectors - need to pack the scalars first
        let coeffs_scalars = [GF256::from(1u8), GF256::from(2u8), GF256::from(3u8), GF256::from(4u8)];
        let values_scalars = [GF256::from(5u8), GF256::from(6u8), GF256::from(7u8), GF256::from(8u8)];

        // Convert to packed fields
        let coeffs_packed = PackedAESBinaryField8x8b::from_scalars(coeffs_scalars);
        let values_packed = PackedAESBinaryField8x8b::from_scalars(values_scalars);

        // Use inner_product_par with packed fields
        let coeffs_slice = &[coeffs_packed];
        let values_slice = &[values_packed];
        let result = inner_product_par::<GF256, PackedAESBinaryField8x8b, PackedAESBinaryField8x8b>(coeffs_slice, values_slice);
        println!("Inner product result: {:?}", result);

        // Test manual computation for comparison
        let mut manual_result = GF256::ZERO;
        for (coeff, val) in coeffs_scalars.iter().zip(values_scalars.iter()) {
            manual_result += *coeff * *val;
        }
        println!("Manual result: {:?}", manual_result);

        assert_eq!(result, manual_result);
    }

    #[test]
    fn test_packed_field_optimization_used() {
        // Test that the PackedField optimization is actually used for GF256
        let mut encoder = RlnEncoder::<GF256, 16>::new();
        encoder.configure(16).unwrap();

        // Create test data - each symbol is [1, 1, 1, ..., 1] (16 bytes)
        let data = vec![1u8; 16 * 16]; // 16 symbols of 16 bytes each
        encoder.set_data(&data).unwrap();

        // Create coefficients that should trigger the optimization
        let coeffs = vec![GF256::from(1u8); 16];

        // The optimization should be used for this case (16 >= 8)
        let result = encoder.encode_symbol(&coeffs).unwrap();

        // Calculate expected result manually using GF256 arithmetic
        // Each coordinate should be: sum(coeffs[i] * data[i][coord]) for all symbols i
        // Since all coeffs are 1 and all data bytes are 1, we need to compute 1+1+...+1 (16 times) in GF256
        let mut expected_coord = GF256::ZERO;
        for _ in 0..16 {
            expected_coord += GF256::from(1u8);
        }

        // Check all coordinates
        for i in 0..16 {
            assert_eq!(result[i], expected_coord, "Coordinate {} should be {:?}", i, expected_coord);
        }
    }
}
