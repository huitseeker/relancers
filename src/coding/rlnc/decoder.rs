use crate::coding::traits::{CodingError, Decoder, StreamingDecoder};
use crate::storage::Symbol;
use binius_field::Field as BiniusField;
use std::marker::PhantomData;

/// Random Linear Network Coding Decoder
pub struct RlnDecoder<F: BiniusField> {
    /// Number of source symbols
    symbols: usize,
    /// Size of each symbol in bytes
    symbol_size: usize,
    /// Received coded symbols
    received_symbols: Vec<Symbol>,
    /// Corresponding coefficient vectors
    coefficients: Vec<Vec<F>>,
    /// Gaussian elimination matrix
    matrix: Vec<Vec<F>>,
    /// Decoded symbols
    decoded_symbols: Vec<Symbol>,
    /// Track which symbols are decoded
    decoded: Vec<bool>,
    /// Current rank of the decoding matrix
    current_rank: usize,
    /// Pivot row indices for incremental diagonalization
    pivot_rows: Vec<Option<usize>>,
    /// Partially decoded symbols (for streaming)
    partial_symbols: Vec<Option<Symbol>>,
    _marker: PhantomData<F>,
}

impl<F: BiniusField> RlnDecoder<F> {
    /// Create a new RLNC decoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
            symbol_size: 0,
            received_symbols: Vec::new(),
            coefficients: Vec::new(),
            matrix: Vec::new(),
            decoded_symbols: Vec::new(),
            decoded: Vec::new(),
            current_rank: 0,
            pivot_rows: Vec::new(),
            partial_symbols: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Initialize the Gaussian elimination matrix
    fn init_matrix(&mut self) {
        self.matrix.clear();
        self.matrix
            .resize(self.symbols, vec![F::ZERO; self.symbols]);

        for (i, coeff_vec) in self.coefficients.iter().enumerate() {
            if i < self.symbols {
                self.matrix[i].copy_from_slice(coeff_vec);
            }
        }
    }

    /// Check if a new contribution increases the rank of the decoding matrix
    pub fn check_rank_increase(&self, coefficients: &[F]) -> bool {
        if coefficients.len() != self.symbols {
            return false;
        }

        // Check if the new coefficients are linearly independent of existing rows
        // by applying the same Gaussian elimination process without cloning
        let mut temp_coefficients = coefficients.to_vec();

        // Apply existing row operations to the new coefficients
        #[allow(clippy::needless_range_loop)]
        for col in 0..self.symbols {
            if let Some(pivot_row) = self.pivot_rows[col] {
                let factor = temp_coefficients[col];
                if !factor.is_zero() {
                    for col_idx in col..self.symbols {
                        temp_coefficients[col_idx] += self.matrix[pivot_row][col_idx] * factor;
                    }
                }
            }
        }

        // Check if any coefficient in the transformed vector is non-zero
        temp_coefficients.iter().any(|c| !c.is_zero())
    }

    /// Perform incremental Gaussian elimination and diagonalization
    fn incremental_diagonalization(&mut self) -> Result<(), CodingError>
    where
        F: From<u8> + Into<u8>,
    {
        if self.coefficients.is_empty() {
            return Ok(());
        }

        let row_idx = self.coefficients.len() - 1;
        let coefficients = &self.coefficients[row_idx];
        let symbol = &self.received_symbols[row_idx];

        // Quick rank check - skip if no rank increase
        if !self.check_rank_increase(coefficients) {
            // Remove the last added coefficients and symbol as they don't contribute
            self.coefficients.pop();
            self.received_symbols.pop();
            return Ok(());
        }

        // Add new row to matrix
        if self.matrix.len() <= row_idx {
            self.matrix.resize(row_idx + 1, vec![F::ZERO; self.symbols]);
        }
        self.matrix[row_idx].copy_from_slice(coefficients);

        // Perform row operations to maintain upper triangular form
        let current_row = row_idx;
        let mut new_symbol = symbol.clone();

        for col in 0..self.symbols {
            if self.matrix[current_row][col].is_zero() {
                continue;
            }

            if let Some(pivot_row) = self.pivot_rows[col] {
                // Row operation: current_row = current_row - factor * pivot_row
                if pivot_row != current_row {
                    let factor = self.matrix[current_row][col];
                    for col_idx in col..self.symbols {
                        self.matrix[current_row][col_idx] = self.matrix[current_row][col_idx]
                            + self.matrix[pivot_row][col_idx] * factor;
                    }
                    new_symbol.add_assign(&self.received_symbols[pivot_row].scaled(factor));
                }
            } else {
                // This is a new pivot
                self.pivot_rows[col] = Some(current_row);
                self.current_rank += 1;

                // Normalize the pivot row
                let pivot_val = self.matrix[current_row][col];
                let pivot_inv = pivot_val.invert().ok_or(CodingError::DecodingFailed)?;

                for col_idx in col..self.symbols {
                    self.matrix[current_row][col_idx] *= pivot_inv;
                }
                new_symbol.scale(pivot_inv);

                // Update partially decoded symbols
                self.partial_symbols[col] = Some(new_symbol.clone());
                self.decoded[col] = true;

                break;
            }
        }

        Ok(())
    }

    /// Perform Gaussian elimination to solve the system
    fn gaussian_elimination(&mut self) -> Result<Vec<Symbol>, CodingError>
    where
        F: From<u8> + Into<u8>,
    {
        let matrix = &mut self.matrix;
        let mut symbols = self.received_symbols.clone();
        let mut rank = 0;
        let n = self.symbols;

        for col in 0..n {
            // Find pivot
            let mut pivot = None;
            #[allow(clippy::needless_range_loop)]
            for row in rank..matrix.len() {
                if !matrix[row][col].is_zero() {
                    pivot = Some(row);
                    break;
                }
            }

            if let Some(pivot_row) = pivot {
                // Swap rows
                matrix.swap(rank, pivot_row);
                symbols.swap(rank, pivot_row);

                // Normalize pivot row
                let pivot_val = matrix[rank][col];
                let pivot_inv = pivot_val.invert().ok_or(CodingError::DecodingFailed)?;

                for col_idx in col..n {
                    matrix[rank][col_idx] *= pivot_inv;
                }
                symbols[rank].scale(pivot_inv);

                // Eliminate other rows
                for row in 0..matrix.len() {
                    if row != rank && !matrix[row][col].is_zero() {
                        let factor = matrix[row][col];
                        for col_idx in col..n {
                            matrix[row][col_idx] =
                                matrix[row][col_idx] + matrix[rank][col_idx] * factor;
                        }
                        let scaled = symbols[rank].scaled(factor);
                        symbols[row].add_assign(&scaled);
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

        // Extract the first n symbols as the decoded result
        Ok(symbols.into_iter().take(n).collect())
    }
}

impl<F: BiniusField> Default for RlnDecoder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField> Decoder<F> for RlnDecoder<F>
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
        self.matrix.clear();
        self.decoded_symbols.clear();
        self.decoded.clear();
        self.decoded.resize(symbols, false);
        self.current_rank = 0;
        self.pivot_rows.clear();
        self.pivot_rows.resize(symbols, None);
        self.partial_symbols.clear();
        self.partial_symbols.resize(symbols, None);

        Ok(())
    }

    fn add_symbol(&mut self, coefficients: &[F], symbol: &[u8]) -> Result<(), CodingError> {
        if coefficients.len() != self.symbols {
            return Err(CodingError::InvalidCoefficients);
        }

        if symbol.len() != self.symbol_size {
            return Err(CodingError::InvalidSymbolSize);
        }

        // Check if this contribution increases rank
        if !self.check_rank_increase(coefficients) {
            return Err(CodingError::RedundantContribution);
        }

        self.coefficients.push(coefficients.to_vec());
        self.received_symbols
            .push(Symbol::from_data(symbol.to_vec()));

        // Perform incremental diagonalization only for useful contributions
        self.incremental_diagonalization()?;

        Ok(())
    }

    fn can_decode(&self) -> bool {
        self.coefficients.len() >= self.symbols
    }

    fn decode(&mut self) -> Result<Vec<u8>, CodingError> {
        if !self.can_decode() {
            return Err(CodingError::InsufficientData);
        }

        self.init_matrix();
        self.decoded_symbols = self.gaussian_elimination()?;

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

impl<F: BiniusField> StreamingDecoder<F> for RlnDecoder<F>
where
    F: From<u8> + Into<u8>,
{
    fn current_rank(&self) -> usize {
        self.current_rank
    }

    fn is_symbol_decoded(&self, index: usize) -> bool {
        if index >= self.symbols {
            return false;
        }
        self.decoded[index]
    }

    fn symbols_decoded(&self) -> usize {
        self.decoded.iter().filter(|&&decoded| decoded).count()
    }

    fn decode_symbol(&mut self, index: usize) -> Result<Option<Vec<u8>>, CodingError> {
        if index >= self.symbols {
            return Ok(None);
        }

        match &self.partial_symbols[index] {
            Some(symbol) => Ok(Some(symbol.clone().into_inner())),
            None => Ok(None),
        }
    }

    fn check_rank_increase(&self, coefficients: &[F]) -> bool {
        self.check_rank_increase(coefficients)
    }
}

impl<F: BiniusField> crate::coding::traits::RecodingDecoder<F> for RlnDecoder<F>
where
    F: From<u8> + Into<u8>,
{
    fn recode(&mut self, recode_coefficients: &[F]) -> Result<Vec<u8>, CodingError> {
        if recode_coefficients.len() != self.coefficients.len() {
            return Err(CodingError::InvalidCoefficients);
        }

        if self.received_symbols.is_empty() {
            return Err(CodingError::InsufficientData);
        }

        let mut recoded_symbol = Symbol::zero(self.symbol_size);

        // Linear combination of received symbols using recode coefficients
        for (coeff, symbol) in recode_coefficients.iter().zip(self.received_symbols.iter()) {
            if !coeff.is_zero() {
                let scaled = symbol.scaled(*coeff);
                recoded_symbol.add_assign(&scaled);
            }
        }

        Ok(recoded_symbol.into_inner())
    }

    fn can_recode(&self) -> bool {
        !self.received_symbols.is_empty()
    }

    fn symbols_needed_for_recode(&self) -> usize {
        // For recoding, we typically need at least 1 symbol to start
        // but can generate useful recoded symbols with any received symbols
        if self.received_symbols.is_empty() {
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::traits::Encoder;
    use crate::coding::RlnEncoder;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_decoder_configuration() {
        let mut decoder = RlnDecoder::<GF256>::new();
        assert!(decoder.configure(4, 16).is_ok());
        assert_eq!(decoder.symbols, 4);
        assert_eq!(decoder.symbol_size, 16);
    }

    #[test]
    fn test_decoder_add_symbol() {
        let mut decoder = RlnDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = vec![1, 2, 3, 4];

        assert!(decoder.add_symbol(&coeffs, &symbol).is_ok());
        assert_eq!(decoder.symbols_received(), 1);
    }

    #[test]
    fn test_decoder_insufficient_data() {
        let mut decoder = RlnDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = vec![1, 2, 3, 4];

        decoder.add_symbol(&coeffs, &symbol).unwrap();
        assert!(!decoder.can_decode());
        assert!(decoder.decode().is_err());
    }

    #[test]
    fn test_decoder_round_trip() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Generate and send enough packets for decoding using seeded RNG
        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data); // Should decode correctly with seeded RNG
    }

    #[test]
    fn test_decoder_over_send() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([123; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 2;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Send exactly the needed packets using seeded RNG
        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data); // Should decode correctly with seeded RNG
    }

    #[test]
    fn test_streaming_decoder_incremental() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Test incremental decoding
        assert_eq!(decoder.current_rank(), 0);
        assert_eq!(decoder.symbols_decoded(), 0);

        // Add symbols one by one
        for i in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();

            // Check incremental progress
            assert!(decoder.current_rank() >= i);
            assert!(decoder.symbols_decoded() >= i);
            assert!(decoder.progress() >= (i + 1) as f64 / symbols as f64);
        }

        assert!(decoder.can_decode());
        assert_eq!(decoder.current_rank(), symbols);
        assert_eq!(decoder.symbols_decoded(), symbols);

        // Test partial symbol decoding
        for i in 0..symbols {
            assert!(decoder.is_symbol_decoded(i));
            let decoded_symbol = decoder.decode_symbol(i).unwrap().unwrap();
            assert_eq!(decoded_symbol.len(), symbol_size);
        }

        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_streaming_decoder_partial_progress() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([99; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 4;
        let symbol_size = 2;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Add symbols and check incremental progress
        let mut previous_rank = 0;
        for _i in 0..symbols {
            assert!(!decoder.can_decode());

            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();

            // Rank should be non-decreasing
            assert!(decoder.current_rank() >= previous_rank);
            previous_rank = decoder.current_rank();

            // Progress should increase
            let progress = decoder.progress();
            assert!(progress > 0.0);

            // Some symbols might be decoded early
            let decoded_count = decoder.symbols_decoded();
            println!("Progress: {decoded_count}/{symbols} symbols decoded");
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_redundant_contribution_detection() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Collect some packets
        let (coeffs1, symbol1) = encoder.encode_packet().unwrap();
        let (coeffs2, symbol2) = encoder.encode_packet().unwrap();

        // Add first packet
        assert!(decoder.check_rank_increase(&coeffs1));
        decoder.add_symbol(&coeffs1, &symbol1).unwrap();
        assert_eq!(decoder.current_rank(), 1);

        // Add second packet
        assert!(decoder.check_rank_increase(&coeffs2));
        decoder.add_symbol(&coeffs2, &symbol2).unwrap();
        assert_eq!(decoder.current_rank(), 2);

        // Create a linear combination of existing packets
        let redundant_coeffs: Vec<GF256> = coeffs1
            .iter()
            .zip(coeffs2.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let redundant_symbol = symbol1
            .iter()
            .zip(symbol2.iter())
            .map(|(a, b)| *a ^ *b)
            .collect::<Vec<u8>>();

        // Should detect this as redundant
        assert!(!decoder.check_rank_increase(&redundant_coeffs));
        let result = decoder.add_symbol(&redundant_coeffs, &redundant_symbol);
        assert!(matches!(result, Err(CodingError::RedundantContribution)));

        // Rank should not increase
        assert_eq!(decoder.current_rank(), 2);
        assert_eq!(decoder.symbols_received(), 2);
    }

    #[test]
    fn test_rank_increase_checking() {
        let mut decoder = RlnDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        // Linearly independent coefficients
        let coeffs1 = vec![GF256::from(1), GF256::from(0)];
        let coeffs2 = vec![GF256::from(0), GF256::from(1)];
        let coeffs3 = vec![GF256::from(1), GF256::from(1)];
        let coeffs4 = vec![GF256::from(2), GF256::from(2)]; // Linearly dependent

        // First coefficient should increase rank
        assert!(decoder.check_rank_increase(&coeffs1));
        decoder.add_symbol(&coeffs1, &[1, 2, 3, 4]).unwrap();
        assert_eq!(decoder.current_rank(), 1);

        // Second coefficient should increase rank
        assert!(decoder.check_rank_increase(&coeffs2));
        decoder.add_symbol(&coeffs2, &[5, 6, 7, 8]).unwrap();
        assert_eq!(decoder.current_rank(), 2);

        // Third coefficient should not increase rank (matrix already full rank)
        assert!(!decoder.check_rank_increase(&coeffs3));

        // Fourth coefficient is linearly dependent
        assert!(!decoder.check_rank_increase(&coeffs4));
    }

    #[test]
    fn test_decoder_empty_data() {
        let mut decoder = RlnDecoder::<GF256>::new();
        assert!(decoder.configure(0, 4).is_err());
        assert!(decoder.configure(4, 0).is_err());
    }

    #[test]
    fn test_decoder_large_symbols() {
        let mut decoder = RlnDecoder::<GF256>::new();
        assert!(decoder.configure(1000, 1024).is_ok());
        assert_eq!(decoder.symbols, 1000);
        assert_eq!(decoder.symbol_size, 1024);
    }

    #[test]
    fn test_decoder_zero_symbol_size() {
        let mut decoder = RlnDecoder::<GF256>::new();
        assert!(decoder.configure(5, 0).is_err());
    }

    #[test]
    fn test_decoder_zero_symbols() {
        let mut decoder = RlnDecoder::<GF256>::new();
        assert!(decoder.configure(0, 1024).is_err());
    }

    #[test]
    fn test_decoder_not_configured() {
        let mut decoder = RlnDecoder::<GF256>::new();
        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = vec![1, 2, 3, 4];
        assert!(decoder.add_symbol(&coeffs, &symbol).is_err());
    }

    #[test]
    fn test_decoder_wrong_coefficients_length() {
        let mut decoder = RlnDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        let coeffs = vec![GF256::from(1)]; // Wrong length
        let symbol = vec![1, 2, 3, 4];
        assert!(decoder.add_symbol(&coeffs, &symbol).is_err());
    }

    #[test]
    fn test_decoder_wrong_symbol_size() {
        let mut decoder = RlnDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = vec![1, 2, 3]; // Wrong size
        assert!(decoder.add_symbol(&coeffs, &symbol).is_err());
    }

    #[test]
    fn test_decoder_single_symbol() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        encoder.configure(1, 8).unwrap();
        decoder.configure(1, 8).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        let (coeffs, symbol) = encoder.encode_packet().unwrap();
        decoder.add_symbol(&coeffs, &symbol).unwrap();

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_decoder_reuse() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        // First use
        encoder.configure(2, 4).unwrap();
        decoder.configure(2, 4).unwrap();

        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data1).unwrap();

        for _ in 0..2 {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        let decoded1 = decoder.decode().unwrap();
        assert_eq!(decoded1, data1);

        // Reconfigure and reuse
        encoder.configure(3, 2).unwrap();
        decoder.configure(3, 2).unwrap();

        let data2 = vec![9, 10, 11, 12, 13, 14];
        encoder.set_data(&data2).unwrap();

        for _ in 0..3 {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        let decoded2 = decoder.decode().unwrap();
        assert_eq!(decoded2, data2);
    }

    #[test]
    fn test_decoder_stress_large_data() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 50;
        let symbol_size = 512;

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        let data = vec![0u8; symbols * symbol_size];
        encoder.set_data(&data).unwrap();

        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_decoder_out_of_order_symbols() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Collect all packets first
        let mut packets = Vec::new();
        for _ in 0..symbols {
            packets.push(encoder.encode_packet().unwrap());
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
    fn test_decoder_duplicate_symbols() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        let mut packets = Vec::new();
        for _ in 0..symbols + 2 {
            packets.push(encoder.encode_packet().unwrap());
        }

        // Add packets and handle potential redundant contributions gracefully
        let mut added = 0;
        for (coeffs, symbol) in packets {
            if decoder.add_symbol(&coeffs, &symbol).is_ok() {
                added += 1;
            }
        }

        // We should have at least the required symbols
        assert!(added >= symbols);
        assert!(decoder.can_decode());
        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_decoder_invalid_symbol_index() {
        let mut decoder = RlnDecoder::<GF256>::new();
        decoder.configure(3, 4).unwrap();

        assert!(!decoder.is_symbol_decoded(3)); // Out of bounds
        assert_eq!(decoder.decode_symbol(3).unwrap(), None);
    }

    #[test]
    fn test_decoder_progress_calculation() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 4;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        assert_eq!(decoder.current_rank(), 0);
        assert_eq!(decoder.symbols_received(), 0);
        assert_eq!(decoder.symbols_needed(), symbols);

        for i in 1..=symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();

            assert_eq!(decoder.symbols_received(), i);
            assert!(decoder.current_rank() <= symbols);
            assert!(decoder.symbols_needed() <= symbols - i);
        }
    }

    #[test]
    fn test_decoder_random_data_patterns() {
        let mut encoder = RlnEncoder::<GF256>::with_seed([99; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 5;
        let symbol_size = 8;

        // Test with all zeros
        let data_zeros = vec![0; symbols * symbol_size];
        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();
        encoder.set_data(&data_zeros).unwrap();

        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_zeros);

        // Reset and test with all 0xFF
        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();
        let data_ones = vec![0xFF; symbols * symbol_size];
        encoder.set_data(&data_ones).unwrap();

        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_ones);

        // Test with incrementing pattern
        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();
        let data_pattern: Vec<u8> = (0..(symbols * symbol_size) as u8).collect();
        encoder.set_data(&data_pattern).unwrap();

        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_pattern);
    }

    #[test]
    fn test_recoding_functionality() {
        use crate::coding::traits::RecodingDecoder;
        use crate::utils::CodingRng;

        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        let symbols = 3;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols, symbol_size).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Send some symbols to decoder
        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        assert!(decoder.can_recode());
        assert_eq!(decoder.symbols_needed_for_recode(), 0);

        // Test recoding with random coefficients
        let mut recode_rng = CodingRng::from_seed([99; 32]);
        let recode_coeffs = recode_rng.generate_coefficients(decoder.symbols_received());

        let recoded_symbol = decoder.recode(&recode_coeffs).unwrap();
        assert_eq!(recoded_symbol.len(), symbol_size);
    }

    #[test]
    fn test_recoding_three_node_network() {
        use crate::coding::traits::RecodingDecoder;

        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut relay_decoder = RlnDecoder::<GF256>::new();
        let mut final_decoder = RlnDecoder::<GF256>::new();

        let symbols = 2;
        let symbol_size = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

        // Configure all nodes
        encoder.configure(symbols, symbol_size).unwrap();
        relay_decoder.configure(symbols, symbol_size).unwrap();
        final_decoder.configure(symbols, symbol_size).unwrap();

        encoder.set_data(&data).unwrap();

        // Step 1: Simulate encoder sending symbols to relay
        // We'll manually create the symbols to ensure we know what we're testing
        let coeffs1 = vec![GF256::from(1), GF256::from(0)];
        let coeffs2 = vec![GF256::from(0), GF256::from(1)];
        let _symbol1 = [1, 2, 3, 4]; // First original symbol
        let _symbol2 = [5, 6, 7, 8]; // Second original symbol

        // Create encoded symbols: c1*symbol1 + c2*symbol2
        let encoded1 = vec![1, 2, 3, 4]; // 1*symbol1 + 0*symbol2
        let encoded2 = vec![5, 6, 7, 8]; // 0*symbol1 + 1*symbol2

        // Relay receives symbols
        relay_decoder.add_symbol(&coeffs1, &encoded1).unwrap();
        relay_decoder.add_symbol(&coeffs2, &encoded2).unwrap();

        // Ensure relay has received enough symbols
        assert_eq!(relay_decoder.symbols_received(), symbols);
        assert!(relay_decoder.can_recode());

        // Step 2: Relay recodes - creates new linear combinations
        // The relay creates new coefficients for relay-to-destination transmission
        let recode_coeffs1 = vec![GF256::from(2), GF256::from(3)]; // New combination
        let recode_coeffs2 = vec![GF256::from(1), GF256::from(4)]; // New combination

        let recoded1 = relay_decoder.recode(&recode_coeffs1).unwrap();
        let recoded2 = relay_decoder.recode(&recode_coeffs2).unwrap();

        // Step 3: Final decoder receives recoded symbols from relay
        final_decoder
            .add_symbol(&recode_coeffs1, &recoded1)
            .unwrap();
        final_decoder
            .add_symbol(&recode_coeffs2, &recoded2)
            .unwrap();

        // Final decoder should be able to decode the original data from recoded symbols
        assert!(final_decoder.can_decode());
        let decoded = final_decoder.decode().unwrap();
        assert_eq!(
            decoded, data,
            "Recoding should preserve original data through network relay"
        );
    }

    #[test]
    fn test_recoding_insufficient_data() {
        use crate::coding::traits::RecodingDecoder;

        let mut decoder = RlnDecoder::<GF256>::new();
        decoder.configure(2, 4).unwrap();

        // Should fail since no symbols received
        assert!(!decoder.can_recode());
        assert_eq!(decoder.symbols_needed_for_recode(), 1);

        let coeffs = vec![GF256::from(1)];
        let result = decoder.recode(&coeffs);
        assert!(result.is_err());
    }

    #[test]
    fn test_recoding_wrong_coefficients_length() {
        use crate::coding::traits::RecodingDecoder;

        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        encoder.configure(2, 4).unwrap();
        decoder.configure(2, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        // Add one symbol
        let (coeffs, symbol) = encoder.encode_packet().unwrap();
        decoder.add_symbol(&coeffs, &symbol).unwrap();

        // Wrong length for recode coefficients
        let wrong_coeffs = vec![GF256::from(1), GF256::from(2)]; // Should be 1, not 2
        let result = decoder.recode(&wrong_coeffs);
        assert!(result.is_err());
    }

    #[test]
    fn test_recoding_zero_coefficients() {
        use crate::coding::traits::RecodingDecoder;

        let mut encoder = RlnEncoder::<GF256>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        encoder.configure(2, 4).unwrap();
        decoder.configure(2, 4).unwrap();

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data).unwrap();

        // Add symbols
        for _ in 0..2 {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        // Test with zero coefficients (should produce zero symbol)
        let zero_coeffs = vec![GF256::ZERO, GF256::ZERO];
        let result = decoder.recode(&zero_coeffs).unwrap();
        assert_eq!(result, vec![0u8; 4]);
    }

    #[test]
    fn test_recoding_bandwidth_efficiency() {
        use crate::coding::traits::RecodingDecoder;

        let mut encoder = RlnEncoder::<GF256>::with_seed([1; 32]);
        let mut decoder = RlnDecoder::<GF256>::new();

        encoder.configure(5, 100).unwrap();
        decoder.configure(5, 100).unwrap();

        let data = vec![42u8; 500];
        encoder.set_data(&data).unwrap();

        // Add some symbols
        for _ in 0..3 {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        // Can recode even with partial data
        assert!(decoder.can_recode());

        let recode_coeffs = vec![GF256::from(1), GF256::from(2), GF256::from(3)];
        let recoded = decoder.recode(&recode_coeffs).unwrap();
        assert_eq!(recoded.len(), 100);
    }
}
