//! Reed-Solomon encoder implementation using Binius

use crate::coding::traits::{CodingError, Encoder};
use crate::storage::Symbol;
use binius_field::{underlier::WithUnderlier, Field as BiniusField};
// Note: Using our own Reed-Solomon implementation as Binius RS functions aren't available directly
// The implementation uses systematic Vandermonde matrices for compatibility
use std::marker::PhantomData;

/// Reed-Solomon encoder using Binius implementation
pub struct RsEncoder<F: BiniusField> {
    /// Number of source symbols (k)
    symbols: usize,
    /// Size of each symbol in bytes
    symbol_size: usize,
    /// Original data split into symbols
    data: Vec<Symbol>,
    /// Total number of symbols (n)
    total_symbols: usize,
    _marker: PhantomData<F>,
}

impl<F: BiniusField> RsEncoder<F> {
    /// Create a new Reed-Solomon encoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
            symbol_size: 0,
            data: Vec::new(),
            total_symbols: 0,
            _marker: PhantomData,
        }
    }

    /// Create a new Reed-Solomon encoder with a specific seed
    pub fn with_seed(_seed: [u8; 32]) -> Self {
        Self::new()
    }

    /// Get the total size of the data in bytes
    pub fn data_size(&self) -> usize {
        self.symbols * self.symbol_size
    }

    /// Generate a Vandermonde matrix row
    fn vandermonde_row(&self, x: F, k: usize) -> Vec<F> {
        let mut row = vec![F::ONE; k];
        let mut power = F::ONE;

        for i in 1..k {
            power *= x;
            row[i] = power;
        }

        row
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
}

impl<F: BiniusField> Default for RsEncoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField> Encoder<F> for RsEncoder<F>
where
    F: WithUnderlier<Underlier = u8>,
{
    fn configure(&mut self, symbols: usize, symbol_size: usize) -> Result<(), CodingError> {
        if symbols == 0 || symbol_size == 0 {
            return Err(CodingError::InvalidParameters);
        }

        self.symbols = symbols;
        self.symbol_size = symbol_size;
        self.total_symbols = symbols * 2; // Example: n = 2k
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

        // Generate evaluation point (symbol index)
        let index = (self.data.len() % self.total_symbols) as u8;
        let point = F::from_underlier(index);
        let coefficients = self.vandermonde_row(point, self.symbols);
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
    fn test_rs_encoder_configuration() {
        let mut encoder = RsEncoder::<GF256>::new();
        assert!(encoder.configure(4, 16).is_ok());
        assert_eq!(encoder.symbols(), 4);
        assert_eq!(encoder.symbol_size(), 16);
    }

    #[test]
    fn test_rs_encoder_set_data() {
        let mut encoder = RsEncoder::<GF256>::new();
        encoder.configure(4, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        assert!(encoder.set_data(&data).is_ok());
        assert!(encoder.set_data(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_vandermonde_row() {
        let encoder = RsEncoder::<GF256>::new();
        let row = encoder.vandermonde_row(GF256::from(2), 3);
        assert_eq!(row.len(), 3);
    }

    #[test]
    fn test_encode_packet() {
        let mut encoder = RsEncoder::<GF256>::with_seed([0; 32]);
        encoder.configure(2, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs, symbol) = encoder.encode_packet().unwrap();

        assert_eq!(coeffs.len(), 2);
        assert_eq!(symbol.len(), 4);
    }

    #[test]
    fn test_rs_encoder_empty_data() {
        let mut encoder = RsEncoder::<GF256>::new();
        assert!(encoder.configure(0, 4).is_err());
        assert!(encoder.configure(4, 0).is_err());
    }

    #[test]
    fn test_rs_encoder_large_symbols() {
        let mut encoder = RsEncoder::<GF256>::new();
        assert!(encoder.configure(255, 1600).is_ok()); // Max RS symbols
        assert_eq!(encoder.symbols(), 255);
        assert_eq!(encoder.symbol_size(), 1600);
    }

    #[test]
    fn test_rs_encoder_zero_symbol_size() {
        let mut encoder = RsEncoder::<GF256>::new();
        assert!(encoder.configure(5, 0).is_err());
    }

    #[test]
    fn test_rs_encoder_zero_symbols() {
        let mut encoder = RsEncoder::<GF256>::new();
        assert!(encoder.configure(0, 1024).is_err());
    }

    #[test]
    fn test_rs_encoder_not_configured() {
        let mut encoder = RsEncoder::<GF256>::new();
        let data = vec![1, 2, 3, 4];
        assert!(encoder.set_data(&data).is_err());
    }

    #[test]
    fn test_rs_encoder_wrong_data_size() {
        let mut encoder = RsEncoder::<GF256>::new();
        encoder.configure(3, 4).unwrap();
        let data = vec![1, 2, 3]; // Wrong size (should be 12 bytes)
        assert!(encoder.set_data(&data).is_err());
    }

    #[test]
    fn test_rs_encode_symbol_no_data() {
        let mut encoder = RsEncoder::<GF256>::new();
        encoder.configure(2, 4).unwrap();
        let coeffs = vec![GF256::from(1), GF256::from(1)];
        assert!(encoder.encode_symbol(&coeffs).is_err());
    }

    #[test]
    fn test_rs_encode_symbol_wrong_coefficients_length() {
        let mut encoder = RsEncoder::<GF256>::new();
        encoder.configure(2, 4).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let coeffs = vec![GF256::from(1)]; // Wrong length
        assert!(encoder.encode_symbol(&coeffs).is_err());
    }

    #[test]
    fn test_rs_encoder_reuse() {
        let mut encoder = RsEncoder::<GF256>::new();

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
    fn test_rs_encoder_single_symbol() {
        let mut encoder = RsEncoder::<GF256>::new();
        encoder.configure(1, 8).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs, symbol) = encoder.encode_packet().unwrap();
        assert_eq!(coeffs.len(), 1);
        assert_eq!(symbol.len(), 8);
        assert_eq!(symbol, data);
    }

    #[test]
    fn test_rs_encoder_stress_large_data() {
        let mut encoder = RsEncoder::<GF256>::new();
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
    fn test_rs_vandermonde_edge_cases() {
        let encoder = RsEncoder::<GF256>::new();

        // Test with zero
        let row = encoder.vandermonde_row(GF256::from(0), 3);
        assert_eq!(row.len(), 3);
        assert_eq!(row[0], GF256::ONE);

        // Test with one
        let row = encoder.vandermonde_row(GF256::from(1), 3);
        assert_eq!(row.len(), 3);
        assert_eq!(row[0], GF256::ONE);
        assert_eq!(row[1], GF256::ONE);
        assert_eq!(row[2], GF256::ONE);

        // Test with max field elements
        let row = encoder.vandermonde_row(GF256::from(255), 3);
        assert_eq!(row.len(), 3);
    }
}
