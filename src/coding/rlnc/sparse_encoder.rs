//! Sparse RLNC encoder with configurable sparsity levels

use crate::coding::sparse::{SparseCoeffGenerator, SparseConfig};
use crate::coding::traits::{CodingError, Encoder};
use crate::storage::Symbol;
use crate::utils::CodingRng;
use binius_field::Field as BiniusField;
use std::marker::PhantomData;

/// Sparse RLNC encoder with configurable sparsity levels
pub struct SparseRlnEncoder<F: BiniusField> {
    /// Number of source symbols
    symbols: usize,
    /// Size of each symbol in bytes
    symbol_size: usize,
    /// Original data split into symbols
    data: Vec<Symbol>,
    /// Sparse coefficient generator
    sparse_generator: SparseCoeffGenerator<F>,
    /// Regular RNG for non-sparse generation
    rng: CodingRng,
    /// Whether to use sparse generation
    use_sparse: bool,
    _marker: PhantomData<F>,
}

impl<F: BiniusField> SparseRlnEncoder<F> {
    /// Create a new sparse RLNC encoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
            symbol_size: 0,
            data: Vec::new(),
            sparse_generator: SparseCoeffGenerator::new(SparseConfig::default()),
            rng: CodingRng::new(),
            use_sparse: false,
            _marker: PhantomData,
        }
    }

    /// Create a new sparse RLNC encoder with sparse configuration
    pub fn with_sparse_config(config: SparseConfig) -> Self {
        Self {
            symbols: 0,
            symbol_size: 0,
            data: Vec::new(),
            sparse_generator: SparseCoeffGenerator::new(config),
            rng: CodingRng::new(),
            use_sparse: true,
            _marker: PhantomData,
        }
    }

    /// Create a new sparse RLNC encoder with seed
    pub fn with_seed(seed: [u8; 32]) -> Self {
        let mut seed_bytes = [0u8; 32];
        seed_bytes.copy_from_slice(&seed);
        Self {
            symbols: 0,
            symbol_size: 0,
            data: Vec::new(),
            sparse_generator: SparseCoeffGenerator::with_seed(SparseConfig::default(), seed_bytes),
            rng: CodingRng::from_seed(seed_bytes),
            use_sparse: false,
            _marker: PhantomData,
        }
    }

    /// Enable or disable sparse coefficient generation
    pub fn set_sparse_mode(&mut self, enabled: bool) {
        self.use_sparse = enabled;
    }

    /// Set sparse configuration
    pub fn set_sparse_config(&mut self, config: SparseConfig) {
        self.sparse_generator.set_config(config);
        self.use_sparse = true;
    }

    /// Get current sparse configuration
    pub fn sparse_config(&self) -> &SparseConfig {
        self.sparse_generator.config()
    }

    /// Get sparsity level (ratio of non-zero coefficients)
    pub fn sparsity(&self) -> f64 {
        self.sparse_config().sparsity
    }

    /// Set sparsity level
    pub fn set_sparsity(&mut self, sparsity: f64) {
        let mut config = *self.sparse_config();
        config.sparsity = sparsity;
        self.set_sparse_config(config);
    }

    /// Get total size of the data in bytes
    pub fn data_size(&self) -> usize {
        self.symbols * self.symbol_size
    }

    /// Split data into symbols
    fn split_into_symbols(&mut self, data: &[u8]) -> Result<(), CodingError> {
        if data.len() != self.data_size() {
            return Err(CodingError::InvalidDataSize);
        }

        self.data.clear();
        for i in 0..self.symbols {
            let start = i * self.symbol_size;
            let end = start + self.symbol_size;
            let symbol_data = data[start..end].to_vec();
            self.data.push(Symbol::from_data(symbol_data));
        }

        Ok(())
    }

    /// Generate coefficients based on current mode
    pub fn generate_coefficients(&mut self) -> Vec<F>
    where
        F: From<u8>,
    {
        if self.use_sparse {
            self.sparse_generator.generate_coefficients(self.symbols)
        } else {
            self.rng.generate_coefficients(self.symbols)
        }
    }

    /// Get sparsity statistics for the last generated coefficients
    pub fn sparsity_stats(&self, coeffs: &[F]) -> SparsityStats {
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
}

impl<F: BiniusField> Default for SparseRlnEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField> Encoder<F> for SparseRlnEncoder<F>
where
    F: From<u8> + Into<u8>,
{
    fn configure(&mut self, symbols: usize, symbol_size: usize) -> Result<(), CodingError> {
        if symbols == 0 || symbol_size == 0 {
            return Err(CodingError::InvalidParameters);
        }

        self.symbols = symbols;
        self.symbol_size = symbol_size;
        self.data.clear();
        self.data.reserve(symbols);

        Ok(())
    }

    fn set_data(&mut self, data: &[u8]) -> Result<(), CodingError> {
        if self.symbols == 0 || self.symbol_size == 0 {
            return Err(CodingError::NotConfigured);
        }

        self.split_into_symbols(data)
    }

    fn encode_symbol(&mut self, coefficients: &[F]) -> Result<Vec<u8>, CodingError> {
        if coefficients.len() != self.symbols {
            return Err(CodingError::InvalidCoefficients);
        }

        if self.data.is_empty() {
            return Err(CodingError::NoDataSet);
        }

        let mut encoded = Symbol::zero(self.symbol_size);

        for (coeff, symbol) in coefficients.iter().zip(self.data.iter()) {
            if !coeff.is_zero() {
                let scaled = symbol.scaled(*coeff);
                encoded.add_assign(&scaled);
            }
        }

        Ok(encoded.into_inner())
    }

    fn encode_packet(&mut self) -> Result<(Vec<F>, Vec<u8>), CodingError> {
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

    fn symbol_size(&self) -> usize {
        self.symbol_size
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
mod tests {
    use super::*;
    use crate::coding::traits::Decoder;
    use binius_field::BinaryField8b as GF256;

    #[test]
    fn test_sparse_encoder_configuration() {
        let mut encoder = SparseRlnEncoder::<GF256>::new();
        assert!(encoder.configure(4, 16).is_ok());
        assert_eq!(encoder.symbols(), 4);
        assert_eq!(encoder.symbol_size(), 16);
    }

    #[test]
    fn test_sparse_encoder_sparse_mode() {
        let config = SparseConfig::new(0.5);
        let mut encoder = SparseRlnEncoder::<GF256>::with_sparse_config(config);

        encoder.configure(4, 4).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        encoder.set_data(&data).unwrap();

        // Generate coefficients and check sparsity
        let coeffs = encoder.generate_coefficients();
        assert_eq!(coeffs.len(), 4);

        let stats = encoder.sparsity_stats(&coeffs);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.non_zeros, 2); // 50% sparsity
        assert_eq!(stats.sparsity_ratio, 0.5);
    }

    #[test]
    fn test_sparse_encoder_full_density() {
        let mut encoder = SparseRlnEncoder::<GF256>::new();
        encoder.set_sparse_mode(false);

        encoder.configure(5, 4).unwrap();
        let data = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ];
        encoder.set_data(&data).unwrap();

        let coeffs = encoder.generate_coefficients();
        let stats = encoder.sparsity_stats(&coeffs);
        assert_eq!(stats.non_zeros, 5); // Full density
        assert_eq!(stats.sparsity_ratio, 1.0);
    }

    #[test]
    fn test_sparse_encoder_zero_sparsity() {
        let config = SparseConfig::new(0.0).with_min_non_zeros(0);
        let mut encoder = SparseRlnEncoder::<GF256>::with_sparse_config(config);

        encoder.configure(4, 4).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        encoder.set_data(&data).unwrap();

        let coeffs = encoder.generate_coefficients();
        let stats = encoder.sparsity_stats(&coeffs);
        assert_eq!(stats.non_zeros, 0);
        assert_eq!(stats.sparsity_ratio, 0.0);
    }

    #[test]
    fn test_sparse_encoder_round_trip() {
        let mut encoder = SparseRlnEncoder::<GF256>::with_sparse_config(SparseConfig::new(0.6));
        let mut decoder = crate::coding::rlnc::RlnDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

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
    fn test_sparse_encoder_sparsity_settings() {
        let mut encoder = SparseRlnEncoder::<GF256>::new();

        assert_eq!(encoder.sparsity(), 1.0); // Default is full density

        encoder.set_sparsity(0.25);
        assert_eq!(encoder.sparsity(), 0.25);

        encoder.set_sparse_mode(false);
        assert_eq!(encoder.sparsity(), 0.25); // Config preserved
    }

    #[test]
    fn test_sparse_encoder_with_seed() {
        let mut encoder1 = SparseRlnEncoder::<GF256>::with_seed([42; 32]);
        let mut encoder2 = SparseRlnEncoder::<GF256>::with_seed([42; 32]);

        encoder1.configure(3, 4).unwrap();
        encoder2.configure(3, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        encoder1.set_sparse_mode(true);
        encoder2.set_sparse_mode(true);

        let (coeffs1, symbol1) = encoder1.encode_packet().unwrap();
        let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();

        assert_eq!(coeffs1, coeffs2);
        assert_eq!(symbol1, symbol2);
    }

    #[test]
    fn test_sparse_encoder_performance_comparison() {
        let symbols = 100;
        let symbol_size = 1024;
        let data = vec![0u8; symbols * symbol_size];

        // Test full density
        let mut full_encoder = SparseRlnEncoder::<GF256>::new();
        full_encoder.configure(symbols, symbol_size).unwrap();
        full_encoder.set_data(&data).unwrap();

        let coeffs = full_encoder.generate_coefficients();
        let stats = full_encoder.sparsity_stats(&coeffs);
        assert_eq!(stats.non_zeros, symbols);

        // Test 10% sparsity
        let mut sparse_encoder =
            SparseRlnEncoder::<GF256>::with_sparse_config(SparseConfig::new(0.1));
        sparse_encoder.configure(symbols, symbol_size).unwrap();
        sparse_encoder.set_data(&data).unwrap();

        let coeffs = sparse_encoder.generate_coefficients();
        let stats = sparse_encoder.sparsity_stats(&coeffs);
        assert_eq!(stats.non_zeros, 10); // 10% of 100
    }
}
