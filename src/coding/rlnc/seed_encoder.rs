//! Seed-based RLNC encoder for bandwidth-efficient deterministic coefficients

use crate::coding::traits::{CodingError, Encoder};
use crate::storage::Symbol;
use crate::utils::CodingRng;
use binius_field::underlier::WithUnderlier;
use binius_field::Field as BiniusField;
use std::marker::PhantomData;

/// Seed-based RLNC encoder that uses small seeds instead of full coefficient vectors
pub struct SeedRlnEncoder<F: BiniusField> {
    /// Number of source symbols
    symbols: usize,
    /// Size of each symbol in bytes
    symbol_size: usize,
    /// Original data split into symbols
    data: Vec<Symbol>,
    /// Current seed value for deterministic coefficient generation
    current_seed: u32,
    /// Random number generator used for deterministic generation
    rng: CodingRng,
    /// Counter for packet generation
    packet_counter: u64,
    _marker: PhantomData<F>,
}

impl<F: BiniusField> SeedRlnEncoder<F> {
    /// Create a new seed-based RLNC encoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
            symbol_size: 0,
            data: Vec::new(),
            current_seed: 0,
            rng: CodingRng::new(),
            packet_counter: 0,
            _marker: PhantomData,
        }
    }

    /// Create a new seed-based RLNC encoder with a specific seed
    pub fn with_seed(seed: u32) -> Self {
        let mut seed_bytes = [0u8; 32];
        seed_bytes[..4].copy_from_slice(&seed.to_le_bytes());
        Self {
            symbols: 0,
            symbol_size: 0,
            data: Vec::new(),
            current_seed: seed,
            rng: CodingRng::from_seed(seed_bytes),
            packet_counter: 0,
            _marker: PhantomData,
        }
    }

    /// Set the seed for deterministic coefficient generation
    pub fn set_seed(&mut self, seed: u32) {
        self.current_seed = seed;
        let mut seed_bytes = [0u8; 32];
        seed_bytes[..4].copy_from_slice(&seed.to_le_bytes());
        self.rng = CodingRng::from_seed(seed_bytes);
        self.packet_counter = 0;
    }

    /// Get the current seed
    pub fn current_seed(&self) -> u32 {
        self.current_seed
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

    /// Generate coefficients deterministically from seed and packet index
    fn generate_coefficients(&mut self, packet_index: u64) -> Vec<F> {
        // Create deterministic seed based on master seed and packet index
        let mut combined_seed = [0u8; 32];
        combined_seed[..4].copy_from_slice(&self.current_seed.to_le_bytes());
        combined_seed[4..12].copy_from_slice(&packet_index.to_le_bytes());

        let mut rng = CodingRng::from_seed(combined_seed);
        rng.generate_coefficients(self.symbols)
    }
}

impl<F: BiniusField> Default for SeedRlnEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField> Encoder<F> for SeedRlnEncoder<F>
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
        self.packet_counter = 0;

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

        let coefficients = self.generate_coefficients(self.packet_counter);
        let symbol = self.encode_symbol(&coefficients)?;

        self.packet_counter += 1;

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
    use crate::coding::traits::Decoder;
    use crate::coding::RlnDecoder;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_seed_encoder_configuration() {
        let mut encoder = SeedRlnEncoder::<GF256>::new();
        assert!(encoder.configure(4, 16).is_ok());
        assert_eq!(encoder.symbols(), 4);
        assert_eq!(encoder.symbol_size(), 16);
    }

    #[test]
    fn test_seed_encoder_set_seed() {
        let mut encoder = SeedRlnEncoder::<GF256>::new();
        encoder.set_seed(12345);
        assert_eq!(encoder.current_seed(), 12345);
    }

    #[test]
    fn test_seed_encoder_deterministic() {
        let mut encoder1 = SeedRlnEncoder::<GF256>::with_seed(42);
        let mut encoder2 = SeedRlnEncoder::<GF256>::with_seed(42);

        encoder1.configure(3, 4).unwrap();
        encoder2.configure(3, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        // Should generate identical packets with same seed
        for _ in 0..5 {
            let (coeffs1, symbol1) = encoder1.encode_packet().unwrap();
            let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();

            assert_eq!(coeffs1, coeffs2);
            assert_eq!(symbol1, symbol2);
        }
    }

    #[test]
    fn test_seed_encoder_different_seeds() {
        let mut encoder1 = SeedRlnEncoder::<GF256>::with_seed(1);
        let mut encoder2 = SeedRlnEncoder::<GF256>::with_seed(2);

        encoder1.configure(2, 4).unwrap();
        encoder2.configure(2, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        let (coeffs1, _) = encoder1.encode_packet().unwrap();
        let (coeffs2, _) = encoder2.encode_packet().unwrap();

        // Should generate different coefficients with different seeds
        assert_ne!(coeffs1, coeffs2);
    }

    #[test]
    fn test_seed_encoder_round_trip() {
        let mut encoder = SeedRlnEncoder::<GF256>::with_seed(12345);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

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
    fn test_seed_encoder_sequential_packets() {
        let mut encoder = SeedRlnEncoder::<GF256>::with_seed(777);

        encoder.configure(2, 4).unwrap();
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
}
