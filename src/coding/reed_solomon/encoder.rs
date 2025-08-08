//! Reed-Solomon encoder implementation using Binius

use crate::coding::traits::{CodingError, Encoder};
use crate::storage::Symbol;
use binius_field::{underlier::WithUnderlier, Field as BiniusField};
// Note: Using our own Reed-Solomon implementation as Binius RS functions aren't available directly
// The implementation uses systematic Vandermonde matrices for compatibility
use std::marker::PhantomData;

/// Reed-Solomon encoder using Binius implementation
pub struct RsEncoder<F: BiniusField, const M: usize> {
    /// Number of source symbols (k)
    symbols: usize,
    /// Original data split into symbols
    data: Vec<Symbol<M>>,
    /// Total number of symbols (n)
    total_symbols: usize,
    _marker: PhantomData<F>,
}

impl<F: BiniusField, const M: usize> RsEncoder<F, M> {
    /// Create a new Reed-Solomon encoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
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
        self.symbols * M
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
            let start = i * M;
            let end = start + M;
            let symbol_data = data[start..end].try_into().unwrap();
            self.data.push(Symbol::from_data(symbol_data));
        }

        Ok(())
    }
}

impl<F: BiniusField, const M: usize> Default for RsEncoder<F, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField, const M: usize> Encoder<F, M> for RsEncoder<F, M>
where
    F: WithUnderlier<Underlier = u8>,
{
    fn configure(&mut self, symbols: usize) -> Result<(), CodingError> {
        if symbols == 0 || M == 0 {
            return Err(CodingError::InvalidParameters);
        }

        self.symbols = symbols;
        self.total_symbols = symbols * 2; // Example: n = 2k
        self.data.clear();
        self.data.reserve(symbols);

        Ok(())
    }

    fn set_data(&mut self, data: &[u8]) -> Result<(), CodingError> {
        if self.symbols == 0 {
            return Err(CodingError::NotConfigured);
        }

        self.split_into_symbols(data)
    }

    fn encode_symbol(&mut self, coefficients: &[F]) -> Result<Symbol<M>, CodingError> {
        if coefficients.len() != self.symbols {
            return Err(CodingError::InvalidCoefficients);
        }

        if self.data.is_empty() {
            return Err(CodingError::NoDataSet);
        }

        #[inline(always)]
        fn encode_byte<F, const M: usize>(
            coefficients: &[F],
            symbols: &[Symbol<M>],
            byte_idx: usize,
        ) -> u8
        where
            F: BiniusField + WithUnderlier<Underlier = u8>,
        {
            let mut byte_sum = F::ZERO;
            for (coeff, symbol) in coefficients.iter().zip(symbols.iter()) {
                if !coeff.is_zero() {
                    let byte = symbol.as_slice()[byte_idx];
                    if byte != 0 {
                        let field_byte = F::from_underlier(byte);
                        byte_sum += *coeff * field_byte;
                    }
                }
            }
            byte_sum.to_underlier()
        }

        let mut result = [0u8; M];
        for byte_idx in 0..M {
            result[byte_idx] = encode_byte(coefficients, &self.data, byte_idx);
        }
        Ok(Symbol::from_data(result))
    }

    fn encode_packet(&mut self) -> Result<(Vec<F>, Symbol<M>), CodingError> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_rs_encoder_configuration() {
        let mut encoder = RsEncoder::<GF256, 16>::new();
        assert!(encoder.configure(4).is_ok());
        assert_eq!(encoder.symbols(), 4);
    }

    #[test]
    fn test_rs_encoder_set_data() {
        let mut encoder = RsEncoder::<GF256, 4>::new();
        encoder.configure(4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        assert!(encoder.set_data(&data).is_ok());
        assert!(encoder.set_data(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_vandermonde_row() {
        let encoder = RsEncoder::<GF256, 3>::new();
        let row = encoder.vandermonde_row(GF256::from(2), 3);
        assert_eq!(row.len(), 3);
    }

    #[test]
    fn test_encode_packet() {
        let mut encoder = RsEncoder::<GF256, 4>::with_seed([0; 32]);
        encoder.configure(2).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs, _symbol) = encoder.encode_packet().unwrap();

        assert_eq!(coeffs.len(), 2);
    }

    #[test]
    fn test_rs_encoder_empty_data() {
        let mut encoder = RsEncoder::<GF256, 4>::new();
        assert!(encoder.configure(0).is_err());
    }

    #[test]
    fn test_rs_encoder_large_symbols() {
        let mut encoder: RsEncoder<GF256, 1600> = RsEncoder::<GF256, 1600>::new();
        assert!(encoder.configure(255).is_ok()); // Max RS symbols
        assert_eq!(encoder.symbols(), 255);
    }

    #[test]
    fn test_rs_encoder_zero_symbol_size() {
        let mut encoder = RsEncoder::<GF256, 0>::new();
        assert!(encoder.configure(5).is_err());
    }

    #[test]
    fn test_rs_encoder_zero_symbols() {
        let mut encoder = RsEncoder::<GF256, 1024>::new();
        assert!(encoder.configure(0).is_err());
    }

    #[test]
    fn test_rs_encoder_not_configured() {
        let mut encoder = RsEncoder::<GF256, 4>::new();
        let data = vec![1, 2, 3, 4];
        assert!(encoder.set_data(&data).is_err());
    }

    #[test]
    fn test_rs_encoder_wrong_data_size() {
        let mut encoder = RsEncoder::<GF256, 4>::new();
        encoder.configure(3).unwrap();
        let data = vec![1, 2, 3]; // Wrong size (should be 12 bytes)
        assert!(encoder.set_data(&data).is_err());
    }

    #[test]
    fn test_rs_encode_symbol_no_data() {
        let mut encoder = RsEncoder::<GF256, 4>::new();
        encoder.configure(2).unwrap();
        let coeffs = vec![GF256::from(1), GF256::from(1)];
        assert!(encoder.encode_symbol(&coeffs).is_err());
    }

    #[test]
    fn test_rs_encode_symbol_wrong_coefficients_length() {
        let mut encoder = RsEncoder::<GF256, 4>::new();
        encoder.configure(2).unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let coeffs = vec![GF256::from(1)]; // Wrong length
        assert!(encoder.encode_symbol(&coeffs).is_err());
    }

    #[test]
    fn test_rs_encoder_reuse() {
        let mut encoder = RsEncoder::<GF256, 4>::new();

        // First use
        encoder.configure(2).unwrap();
        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data1).unwrap();
        let (_, symbol1) = encoder.encode_packet().unwrap();

        // Reconfigure and reuse: only multiples or original
        encoder.configure(3).unwrap();
        let data2 = vec![9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        encoder.set_data(&data2).unwrap();
        let (_, symbol2) = encoder.encode_packet().unwrap();

        assert_ne!(symbol1, symbol2);
    }

    #[test]
    fn test_rs_encoder_single_symbol() {
        let mut encoder = RsEncoder::<GF256, 8>::new();
        encoder.configure(1).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs, symbol) = encoder.encode_packet().unwrap();
        assert_eq!(coeffs.len(), 1);
        assert_eq!(symbol.into_inner()[..], data[..]);
    }

    #[test]
    fn test_rs_encoder_stress_large_data() {
        const SSIZE: usize = 1024;
        let mut encoder = RsEncoder::<GF256, SSIZE>::new();
        let symbols = 100;

        encoder.configure(symbols).unwrap();

        let data = vec![0u8; symbols * SSIZE];
        encoder.set_data(&data).unwrap();

        let (coeffs, _symbol) = encoder.encode_packet().unwrap();
        assert_eq!(coeffs.len(), symbols);
    }

    #[test]
    fn test_rs_vandermonde_edge_cases() {
        let encoder = RsEncoder::<GF256, 3>::new();

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
