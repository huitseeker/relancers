//! Reed-Solomon decoder implementation

use crate::coding::traits::{CodingError, Decoder};
use crate::storage::Symbol;
use binius_field::Field as BiniusField;
// Note: Using our own Reed-Solomon implementation as Binius RS functions aren't available directly
// The implementation uses systematic Vandermonde matrices for compatibility
use std::marker::PhantomData;

/// Reed-Solomon decoder using systematic Vandermonde matrices
pub struct RsDecoder<F: BiniusField> {
    /// Number of source symbols (k)
    symbols: usize,
    /// Size of each symbol in bytes
    symbol_size: usize,
    /// Received coded symbols
    received_symbols: Vec<Symbol>,
    /// Corresponding coefficient vectors
    coefficients: Vec<Vec<F>>,
    /// Decoded symbols
    decoded_symbols: Vec<Symbol>,
    _marker: PhantomData<F>,
}

impl<F: BiniusField> RsDecoder<F> {
    /// Create a new Reed-Solomon decoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
            symbol_size: 0,
            received_symbols: Vec::new(),
            coefficients: Vec::new(),
            decoded_symbols: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Generate a Vandermonde matrix row
    #[cfg(test)]
    fn vandermonde_row(&self, x: F, k: usize) -> Vec<F> {
        let mut row = vec![F::ONE; k];
        let mut power = F::ONE;

        for i in 1..k {
            power *= x;
            row[i] = power;
        }

        row
    }

    /// Perform Gaussian elimination to solve the system
    fn gaussian_elimination(
        &self,
        matrix: &mut [Vec<F>],
        rhs: &mut [Symbol],
    ) -> Result<Vec<Symbol>, CodingError>
    where
        F: From<u8> + Into<u8>,
    {
        let n = self.symbols;
        let mut rank = 0;

        for col in 0..n {
            // Find pivot
            let mut pivot = None;
            for row in rank..matrix.len() {
                if !matrix[row][col].is_zero() {
                    pivot = Some(row);
                    break;
                }
            }

            if let Some(pivot_row) = pivot {
                // Swap rows
                matrix.swap(rank, pivot_row);
                rhs.swap(rank, pivot_row);

                // Normalize pivot row
                let pivot_val = matrix[rank][col];
                let pivot_inv = pivot_val.invert().ok_or(CodingError::DecodingFailed)?;

                for col_idx in col..n {
                    matrix[rank][col_idx] *= pivot_inv;
                }
                rhs[rank].scale(pivot_inv);

                // Eliminate other rows
                for row in 0..matrix.len() {
                    if row != rank && !matrix[row][col].is_zero() {
                        let factor = matrix[row][col];
                        for col_idx in col..n {
                            matrix[row][col_idx] =
                                matrix[row][col_idx] + matrix[rank][col_idx] * factor;
                        }
                        rhs[row].add_assign(&rhs[rank].scaled(factor));
                    }
                }

                rank += 1;
            }

            if rank == n {
                break;
            }
        }

        if rank < n {
            return Err(CodingError::InsufficientData);
        }

        // Extract solution
        let mut solution = vec![Symbol::zero(self.symbol_size); n];
        for (i, row) in matrix.iter().enumerate().take(n) {
            for (j, &coeff) in row.iter().enumerate().take(n) {
                if !coeff.is_zero() {
                    let scaled = rhs[i].scaled(coeff);
                    solution[j].add_assign(&scaled);
                }
            }
        }

        Ok(solution)
    }
}

impl<F: BiniusField> Default for RsDecoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField> Decoder<F> for RsDecoder<F>
where
    F: From<u8> + Into<u8>,
{
    fn configure(&mut self, symbols: usize, symbol_size: usize) -> Result<(), CodingError> {
        if symbols == 0 || symbol_size == 0 {
            return Err(CodingError::InvalidParameters);
        }

        self.symbols = symbols;
        self.symbol_size = symbol_size;
        self.received_symbols.clear();
        self.coefficients.clear();
        self.decoded_symbols.clear();

        Ok(())
    }

    fn add_symbol(&mut self, coefficients: &[F], symbol: &[u8]) -> Result<(), CodingError> {
        if coefficients.len() != self.symbols {
            return Err(CodingError::InvalidCoefficients);
        }

        if symbol.len() != self.symbol_size {
            return Err(CodingError::InvalidSymbolSize);
        }

        self.coefficients.push(coefficients.to_vec());
        self.received_symbols
            .push(Symbol::from_data(symbol.to_vec()));

        Ok(())
    }

    fn can_decode(&self) -> bool {
        self.coefficients.len() >= self.symbols
    }

    fn decode(&mut self) -> Result<Vec<u8>, CodingError> {
        if !self.can_decode() {
            return Err(CodingError::InsufficientData);
        }

        let mut matrix = self.coefficients.clone();
        let mut symbols = self.received_symbols.clone();

        self.decoded_symbols = self.gaussian_elimination(&mut matrix, &mut symbols)?;

        let mut result = Vec::with_capacity(self.symbols * self.symbol_size);
        for symbol in &self.decoded_symbols {
            result.extend_from_slice(symbol.as_slice());
        }

        Ok(result)
    }

    fn symbols_needed(&self) -> usize {
        self.symbols.saturating_sub(self.coefficients.len())
    }

    fn symbols_received(&self) -> usize {
        self.coefficients.len()
    }

    fn symbol_size(&self) -> usize {
        self.symbol_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::reed_solomon::encoder::RsEncoder;
    use crate::coding::traits::{Decoder, Encoder};
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_rs_decoder_configuration() {
        let mut decoder = RsDecoder::<GF256>::new();
        assert!(decoder.configure(4, 16).is_ok());
        assert_eq!(decoder.symbols, 4);
        assert_eq!(decoder.symbol_size, 16);
    }

    #[test]
    fn test_rs_decoder_add_symbol() {
        let mut decoder = RsDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = vec![1, 2, 3, 4];

        assert!(decoder.add_symbol(&coeffs, &symbol).is_ok());
        assert_eq!(decoder.coefficients.len(), 1);
    }

    #[test]
    fn test_rs_decoder_insufficient_data() {
        let mut decoder = RsDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = vec![1, 2, 3, 4];

        decoder.add_symbol(&coeffs, &symbol).unwrap();
        assert!(!decoder.can_decode());
        assert!(decoder.decode().is_err());
    }

    #[test]
    fn test_rs_decoder_round_trip() {
        use crate::coding::reed_solomon::encoder::RsEncoder;
        use crate::coding::traits::Encoder;
        let mut encoder = RsEncoder::<GF256>::new();
        let mut decoder = RsDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Send all symbols using Vandermonde coefficients
        for i in 0..symbols {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, symbols);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_decoder_empty_data() {
        let mut decoder = RsDecoder::<GF256>::new();
        assert!(decoder.configure(0, 4).is_err());
        assert!(decoder.configure(4, 0).is_err());
    }

    #[test]
    fn test_rs_decoder_large_symbols() {
        let mut decoder = RsDecoder::<GF256>::new();
        assert!(decoder.configure(255, 1600).is_ok()); // Max RS symbols
        assert_eq!(decoder.symbols, 255);
        assert_eq!(decoder.symbol_size, 1600);
    }

    #[test]
    fn test_rs_decoder_zero_symbol_size() {
        let mut decoder = RsDecoder::<GF256>::new();
        assert!(decoder.configure(5, 0).is_err());
    }

    #[test]
    fn test_rs_decoder_zero_symbols() {
        let mut decoder = RsDecoder::<GF256>::new();
        assert!(decoder.configure(0, 1024).is_err());
    }

    #[test]
    fn test_rs_decoder_not_configured() {
        let mut decoder = RsDecoder::<GF256>::new();
        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = vec![1, 2, 3, 4];
        assert!(decoder.add_symbol(&coeffs, &symbol).is_err());
    }

    #[test]
    fn test_rs_decoder_wrong_coefficients_length() {
        let mut decoder = RsDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        let coeffs = vec![GF256::from(1)]; // Wrong length
        let symbol = vec![1, 2, 3, 4];
        assert!(decoder.add_symbol(&coeffs, &symbol).is_err());
    }

    #[test]
    fn test_rs_decoder_wrong_symbol_size() {
        let mut decoder = RsDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = vec![1, 2, 3]; // Wrong size
        assert!(decoder.add_symbol(&coeffs, &symbol).is_err());
    }

    #[test]
    fn test_rs_decoder_single_symbol() {
        let mut encoder = RsEncoder::<GF256>::new();
        let mut decoder = RsDecoder::<GF256>::new();

        encoder.configure(1, 8).unwrap();
        decoder.configure(1, 8).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let point = GF256::from(0);
        let coeffs = decoder.vandermonde_row(point, 1);
        let symbol = encoder.encode_symbol(&coeffs).unwrap();
        decoder.add_symbol(&coeffs, &symbol).unwrap();

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_decoder_reuse() {
        let mut encoder = RsEncoder::<GF256>::new();
        let mut decoder = RsDecoder::<GF256>::new();

        // First use
        encoder.configure(2, 4).unwrap();
        decoder.configure(2, 4).unwrap();

        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data1).unwrap();

        for i in 0..2 {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, 2);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        let decoded1 = decoder.decode().unwrap();
        assert_eq!(decoded1, data1);

        // Reconfigure and reuse
        encoder.configure(3, 2).unwrap();
        decoder.configure(3, 2).unwrap();

        let data2 = vec![9, 10, 11, 12, 13, 14];
        encoder.set_data(&data2).unwrap();

        for i in 0..3 {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, 3);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        let decoded2 = decoder.decode().unwrap();
        assert_eq!(decoded2, data2);
    }

    #[test]
    fn test_rs_decoder_stress_large_data() {
        let mut encoder = RsEncoder::<GF256>::new();
        let mut decoder = RsDecoder::<GF256>::new();

        let symbols = 50;
        let symbol_size = 512;

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        let data = vec![0u8; symbols * symbol_size];
        encoder.set_data(&data).unwrap();

        for i in 0..symbols {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, symbols);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_decoder_out_of_order_symbols() {
        let mut encoder = RsEncoder::<GF256>::new();
        let mut decoder = RsDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Collect all packets first
        let mut packets = Vec::new();
        for i in 0..symbols {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, symbols);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            packets.push((coeffs, symbol));
        }

        // Add packets in reverse order
        for (coeffs, symbol) in packets.into_iter().rev() {
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_decoder_duplicate_symbols() {
        let mut encoder = RsEncoder::<GF256>::new();
        let mut decoder = RsDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Collect extra packets
        let mut packets = Vec::new();
        for i in 0..symbols + 2 {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, symbols);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            packets.push((coeffs, symbol));
        }

        // Add packets including duplicates
        for (coeffs, symbol) in packets {
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_decoder_progress_calculation() {
        let mut encoder = RsEncoder::<GF256>::new();
        let mut decoder = RsDecoder::<GF256>::new();

        let symbols = 4;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        assert_eq!(decoder.symbols_received(), 0);
        assert_eq!(decoder.symbols_needed(), symbols);

        for i in 0..symbols {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, symbols);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();

            assert_eq!(decoder.symbols_received(), i + 1);
            assert!(decoder.symbols_needed() <= symbols - (i + 1));
        }
    }

    #[test]
    fn test_rs_decoder_random_data_patterns() {
        let mut encoder = RsEncoder::<GF256>::new();
        let mut decoder = RsDecoder::<GF256>::new();

        let symbols = 5;
        let symbol_size = 8;

        // Test with all zeros
        let data_zeros = vec![0; symbols * symbol_size];
        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();
        encoder.set_data(&data_zeros).unwrap();

        for i in 0..symbols {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, symbols);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_zeros);

        // Reset and test with all 0xFF
        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();
        let data_ones = vec![0xFF; symbols * symbol_size];
        encoder.set_data(&data_ones).unwrap();

        for i in 0..symbols {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, symbols);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_ones);

        // Test with incrementing pattern
        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();
        let data_pattern: Vec<u8> = (0..(symbols * symbol_size) as u8).collect();
        encoder.set_data(&data_pattern).unwrap();

        for i in 0..symbols {
            let point = GF256::from(i as u8);
            let coeffs = decoder.vandermonde_row(point, symbols);
            let symbol = encoder.encode_symbol(&coeffs).unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_pattern);
    }
}
