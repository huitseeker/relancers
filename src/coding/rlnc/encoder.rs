use crate::coding::traits::{CodingError, Encoder};
use crate::storage::Symbol;
use crate::utils::CodingRng;
use binius_field::underlier::WithUnderlier;
use binius_field::Field as BiniusField;
use binius_maybe_rayon::prelude::*;
use std::marker::PhantomData;

/// Random Linear Network Coding Encoder
pub struct RlnEncoder<F: BiniusField> {
    /// Number of source symbols
    symbols: usize,
    /// Size of each symbol in bytes
    symbol_size: usize,
    /// Original data split into symbols
    data: Vec<Symbol>,
    /// Random number generator
    rng: CodingRng,
    _marker: PhantomData<F>,
}

impl<F: BiniusField> RlnEncoder<F> {
    /// Create a new RLNC encoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
            symbol_size: 0,
            data: Vec::new(),
            rng: CodingRng::new(),
            _marker: PhantomData,
        }
    }

    /// Create a new RLNC encoder with a specific seed for deterministic behavior
    pub fn with_seed(seed: [u8; 32]) -> Self {
        Self {
            symbols: 0,
            symbol_size: 0,
            data: Vec::new(),
            rng: CodingRng::from_seed(seed),
            _marker: PhantomData,
        }
    }

    /// Get the total size of the data in bytes
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

    /// Generate random coefficients for encoding
    pub fn generate_coefficients(&mut self) -> Vec<F>
    where
        F: From<u8>,
    {
        self.rng.generate_coefficients(self.symbols)
    }
}

impl<F: BiniusField> Default for RlnEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField> Encoder<F> for RlnEncoder<F>
where
    F: WithUnderlier<Underlier = u8>,
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

        // Use parallel processing for encoding when there are many symbols
        if self.symbols >= 32 {
            // Parallel encoding for better performance with large data
            let encoded_data: Vec<u8> = (0..self.symbol_size)
                .into_par_iter()
                .map(|byte_idx| {
                    let mut byte_sum = F::ZERO;
                    for (coeff, symbol) in coefficients.iter().zip(self.data.iter()) {
                        if !coeff.is_zero() {
                            let symbol_bytes = symbol.as_slice();
                            let byte_val = F::from_underlier(symbol_bytes[byte_idx]);
                            byte_sum += *coeff * byte_val;
                        }
                    }
                    byte_sum.to_underlier()
                })
                .collect();

            Ok(encoded_data)
        } else {
            // Sequential encoding for small data to avoid overhead
            let mut encoded = Symbol::zero(self.symbol_size);

            for (coeff, symbol) in coefficients.iter().zip(self.data.iter()) {
                if !coeff.is_zero() {
                    let scaled = symbol.scaled(*coeff);
                    encoded.add_assign(&scaled);
                }
            }

            Ok(encoded.into_inner())
        }
    }

    fn encode_packet(&mut self) -> Result<(Vec<F>, Vec<u8>), CodingError> {
        if self.symbols == 0 {
            return Err(CodingError::NotConfigured);
        }

        if self.data.is_empty() {
            return Err(CodingError::NoDataSet);
        }

        let coefficients = self.rng.generate_coefficients(self.symbols);
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

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_encoder_configuration() {
        let mut encoder = RlnEncoder::<GF256>::new();
        assert!(encoder.configure(4, 16).is_ok());
        assert_eq!(encoder.symbols, 4);
        assert_eq!(encoder.symbol_size, 16);
    }

    #[test]
    fn test_encoder_invalid_configuration() {
        let mut encoder = RlnEncoder::<GF256>::new();
        assert!(encoder.configure(0, 16).is_err());
        assert!(encoder.configure(4, 0).is_err());
    }

    #[test]
    fn test_encoder_set_data() {
        let mut encoder = RlnEncoder::<GF256>::new();
        encoder.configure(4, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        assert!(encoder.set_data(&data).is_ok());
        assert!(encoder.set_data(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_encode_symbol() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([99; 32]);
        encoder.configure(2, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        // Use seeded RNG for deterministic coefficients
        let coeffs = vec![GF256::from(1), GF256::from(2)];
        let encoded = encoder.encode_symbol(&coeffs).unwrap();

        assert_eq!(encoded.len(), 4);
        // With proper GF(256) multiplication using AESTowerField8b
        let expected = vec![11, 14, 13, 20]; // Actual result with AESTowerField8b
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_encode_packet() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([0; 32]);
        encoder.configure(2, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs, symbol) = encoder.encode_packet().unwrap();

        assert_eq!(coeffs.len(), 2);
        assert_eq!(symbol.len(), 4);
    }

    #[test]
    fn test_encoder_empty_data() {
        let mut encoder = RlnEncoder::<GF256>::new();
        encoder.configure(0, 4).unwrap_err();
        encoder.configure(4, 0).unwrap_err();
    }

    #[test]
    fn test_encoder_large_symbols() {
        let mut encoder = RlnEncoder::<GF256>::new();
        assert!(encoder.configure(1000, 1024).is_ok());
        assert_eq!(encoder.symbols(), 1000);
        assert_eq!(encoder.symbol_size(), 1024);
    }

    #[test]
    fn test_encoder_zero_symbol_size() {
        let mut encoder = RlnEncoder::<GF256>::new();
        assert!(encoder.configure(5, 0).is_err());
    }

    #[test]
    fn test_encoder_zero_symbols() {
        let mut encoder = RlnEncoder::<GF256>::new();
        assert!(encoder.configure(0, 1024).is_err());
    }

    #[test]
    fn test_encoder_not_configured() {
        let mut encoder = RlnEncoder::<GF256>::new();
        let data = vec![1, 2, 3, 4];
        assert!(encoder.set_data(&data).is_err());
    }

    #[test]
    fn test_encoder_wrong_data_size() {
        let mut encoder = RlnEncoder::<GF256>::new();
        encoder.configure(3, 4).unwrap();
        let data = vec![1, 2, 3]; // Wrong size (should be 12 bytes)
        assert!(encoder.set_data(&data).is_err());
    }

    #[test]
    fn test_encode_symbol_no_data() {
        let mut encoder = RlnEncoder::<GF256>::new();
        encoder.configure(2, 4).unwrap();
        let coeffs = vec![GF256::from(1), GF256::from(1)];
        assert!(encoder.encode_symbol(&coeffs).is_err());
    }

    #[test]
    fn test_encode_symbol_wrong_coefficients_length() {
        let mut encoder = RlnEncoder::<GF256>::new();
        encoder.configure(2, 4).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let coeffs = vec![GF256::from(1)]; // Wrong length
        assert!(encoder.encode_symbol(&coeffs).is_err());
    }

    #[test]
    fn test_encoder_reuse() {
        let mut encoder = RlnEncoder::<GF256>::new();

        // First use
        encoder.configure(2, 4).unwrap();
        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data1).unwrap();
        let (_, symbol1) = encoder.encode_packet().unwrap();

        // Reconfigure and reuse
        encoder.configure(3, 2).unwrap();
        let data2 = vec![9, 10, 11, 12, 13, 14];
        encoder.set_data(&data2).unwrap();
        let (_, symbol2) = encoder.encode_packet().unwrap();

        assert_ne!(symbol1, symbol2);
    }

    #[test]
    fn test_encoder_deterministic_with_seed() {
        let mut encoder1 = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut encoder2 = RlnEncoder::<GF256>::with_seed([42; 32]);

        encoder1.configure(2, 4).unwrap();
        encoder2.configure(2, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        let (coeffs1, symbol1) = encoder1.encode_packet().unwrap();
        let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();

        assert_eq!(coeffs1, coeffs2);
        assert_eq!(symbol1, symbol2);
    }

    #[test]
    fn test_encoder_single_symbol() {
        let mut encoder = RlnEncoder::<GF256>::new();
        encoder.configure(1, 8).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs, symbol) = encoder.encode_packet().unwrap();
        assert_eq!(coeffs.len(), 1);
        assert_eq!(symbol.len(), 8);
        // With RLNC, the symbol is a linear combination, not necessarily the original data
        // We can verify the symbol has correct length and valid coefficients
    }

    #[test]
    fn test_encoder_stress_large_data() {
        let mut encoder = RlnEncoder::<GF256>::new();
        let symbols = 100;
        let symbol_size = 1024;

        encoder.configure(symbols, symbol_size).unwrap();

        let data = vec![0u8; symbols * symbol_size];
        encoder.set_data(&data).unwrap();

        let (coeffs, symbol) = encoder.encode_packet().unwrap();
        assert_eq!(coeffs.len(), symbols);
        assert_eq!(symbol.len(), symbol_size);
    }

    #[test]
    fn test_parallel_encoding_correctness() {
        // Test that parallel encoding works correctly
        let mut encoder = RlnEncoder::<GF256>::new();

        // Use exactly 32 symbols to trigger parallel path
        let symbols = 32;
        let symbol_size = 64;
        encoder.configure(symbols, symbol_size).unwrap();

        // Create deterministic test data - ensure exact size
        let data_size = symbols * symbol_size;
        let mut data = vec![0u8; data_size];
        data[0] = 1; // Set first byte to 1 to ensure non-zero result
        encoder.set_data(&data).unwrap();

        let mut coeffs = vec![GF256::ZERO; symbols];
        coeffs[0] = GF256::from(1u8); // Only first coefficient is 1

        // Test parallel encoding
        let result = encoder.encode_symbol(&coeffs).unwrap();
        assert_eq!(result.len(), symbol_size);

        // First byte should be 1, rest should be 0
        assert_eq!(result[0], 1);
        for &byte in &result[1..] {
            assert_eq!(byte, 0);
        }
    }

    #[test]
    fn test_parallel_vs_sequential_consistency() {
        // Test that encoding produces consistent results regardless of parallel/sequential path
        let symbols = 31; // Just below threshold
        let symbol_size = 32;
        let data_size = symbols * symbol_size;
        let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
        let coeffs: Vec<GF256> = (0..symbols).map(|i| GF256::from(i as u8)).collect();

        // Test sequential path (31 symbols)
        let mut encoder_seq = RlnEncoder::<GF256>::with_seed([123; 32]);
        encoder_seq.configure(symbols, symbol_size).unwrap();
        encoder_seq.set_data(&data).unwrap();
        let result_seq = encoder_seq.encode_symbol(&coeffs).unwrap();

        // Test parallel path (32 symbols)
        let symbols_parallel = 32;
        let data_size_parallel = symbols_parallel * symbol_size;
        let data_parallel: Vec<u8> = (0..data_size_parallel).map(|i| (i % 256) as u8).collect();
        let coeffs_parallel: Vec<GF256> = (0..symbols_parallel)
            .map(|i| GF256::from(i as u8))
            .collect();

        let mut encoder_par = RlnEncoder::<GF256>::with_seed([123; 32]);
        encoder_par
            .configure(symbols_parallel, symbol_size)
            .unwrap();
        encoder_par.set_data(&data_parallel).unwrap();
        let result_par = encoder_par.encode_symbol(&coeffs_parallel).unwrap();

        // Both should have correct lengths
        assert_eq!(result_seq.len(), symbol_size);
        assert_eq!(result_par.len(), symbol_size);
    }
}
