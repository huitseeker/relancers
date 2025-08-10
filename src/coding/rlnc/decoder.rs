use crate::coding::rlnc::optimized_matrix::OptimizedMatrix;
use crate::coding::traits::{CodingError, Decoder, StreamingDecoder};
use crate::storage::Symbol;
use binius_field::Field as BiniusField;

/// Random Linear Network Coding Decoder
pub struct RlnDecoder<F: BiniusField, const M: usize> {
    /// Number of source symbols
    symbols: usize,
    /// Received coded symbols
    received_symbols: Vec<Symbol<F, M>>,
    /// Corresponding coefficient vectors
    coefficients: Vec<Vec<F>>,
    /// Optimized Gaussian elimination matrix with RREF maintenance
    matrix: OptimizedMatrix<F>,
    /// Decoded symbols
    decoded_symbols: Vec<Symbol<F, M>>,
    /// Track which symbols are decoded
    decoded: Vec<bool>,
    /// Current rank of the decoding matrix
    current_rank: usize,
    /// Pivot row indices for incremental diagonalization
    pivot_rows: Vec<Option<usize>>,
    /// Partially decoded symbols (for streaming)
    partial_symbols: Vec<Option<Symbol<F, M>>>,
}

impl<F: BiniusField, const M: usize> RlnDecoder<F, M> {
    /// Create a new RLNC decoder
    pub fn new() -> Self {
        Self {
            symbols: 0,
            received_symbols: Vec::new(),
            coefficients: Vec::new(),
            matrix: OptimizedMatrix::new(0),
            decoded_symbols: Vec::new(),
            decoded: Vec::new(),
            current_rank: 0,
            pivot_rows: Vec::new(),
            partial_symbols: Vec::new(),
        }
    }

    /// Initialize the Gaussian elimination matrix
    fn init_matrix(&mut self) {
        self.matrix.clear();
        self.matrix = OptimizedMatrix::new(self.symbols);

        // Add all coefficients to the optimized matrix
        for coeff_vec in &self.coefficients {
            let _ = self.matrix.add_row(coeff_vec);
        }

        // Update current rank to match matrix rank
        self.current_rank = self.matrix.rank();
    }

    /// Check if a new contribution increases the rank of the decoding matrix
    pub fn check_rank_increase(&self, coefficients: &[F]) -> bool {
        if coefficients.len() != self.symbols {
            return false;
        }

        // Use the optimized matrix's rank increase check
        self.matrix.check_rank_increase(coefficients)
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
        let _symbol = &self.received_symbols[row_idx];

        // Use optimized matrix to add row and maintain RREF
        let rank_increase = self.matrix.add_row(coefficients)?;

        if !rank_increase {
            // Remove the last added coefficients and symbol as they don't contribute
            self.coefficients.pop();
            self.received_symbols.pop();
            return Ok(());
        }

        // Update current rank from optimized matrix
        self.current_rank = self.matrix.rank();

        // Update pivot positions and decoded symbols
        for col in 0..self.symbols {
            if let Some(pivot_row) = self.matrix.pivot_row(col) {
                self.pivot_rows[col] = Some(pivot_row);
                self.decoded[col] = true;

                // Update partially decoded symbols based on RREF
                let mut new_symbol = Symbol::<F, M>::zero();

                // Build the solution for this column
                let row_coefficients = self.matrix.get_row(pivot_row);
                for (coeff_idx, coeff) in row_coefficients.iter().enumerate() {
                    if !coeff.is_zero() && coeff_idx < self.received_symbols.len() {
                        let scaled = self.received_symbols[coeff_idx].scaled(*coeff);
                        new_symbol.add_assign(&scaled);
                    }
                }

                self.partial_symbols[col] = Some(new_symbol);
            }
        }

        Ok(())
    }

    /// Perform Gaussian elimination to solve the system using the optimized matrix
    fn gaussian_elimination(&mut self) -> Result<Vec<Symbol<F, M>>, CodingError>
    where
        F: From<u8> + Into<u8>,
    {
        if !self.can_decode() {
            return Err(CodingError::InsufficientData);
        }

        // Build a temporary matrix for solving the system
        let mut temp_matrix = OptimizedMatrix::new(self.symbols);

        // Add all coefficients to the temporary matrix
        for coeff_vec in &self.coefficients {
            let _ = temp_matrix.add_row(coeff_vec);
        }

        if !temp_matrix.is_full_rank() {
            return Err(CodingError::InsufficientData);
        }

        // Now we need to solve the system Ax = b where:
        // A is our coefficient matrix, b is our received symbols

        // Create a mapping from pivot positions to original symbols
        let mut pivot_map = vec![0; self.symbols];
        let mut used_pivots = vec![false; self.coefficients.len()];

        for col in 0..self.symbols {
            if let Some(pivot_row) = temp_matrix.pivot_row(col) {
                pivot_map[col] = pivot_row;
                used_pivots[pivot_row] = true;
            }
        }

        // Ensure we have exactly the required symbols
        let mut selected_symbols = Vec::new();
        let mut selected_coefficients = Vec::new();

        for src_idx in 0..self.symbols {
            let pivot_row = pivot_map[src_idx];
            selected_symbols.push(self.received_symbols[pivot_row].clone());
            selected_coefficients.push(self.coefficients[pivot_row].clone());
        }

        // Now solve the system using the selected symbols
        let mut matrix = vec![vec![F::ZERO; self.symbols]; self.symbols];
        for (i, coeffs) in selected_coefficients.iter().enumerate() {
            matrix[i].copy_from_slice(coeffs);
        }

        // Perform Gaussian elimination on the selected matrix
        let mut symbols = selected_symbols;
        let n = self.symbols;

        for col in 0..n {
            // Find pivot
            let mut pivot = None;
            for row in col..n {
                if !matrix[row][col].is_zero() {
                    pivot = Some(row);
                    break;
                }
            }

            if let Some(pivot_row) = pivot {
                // Swap rows
                matrix.swap(col, pivot_row);
                symbols.swap(col, pivot_row);

                // Normalize pivot row
                let pivot_val = matrix[col][col];
                let pivot_inv = pivot_val.invert().ok_or(CodingError::DecodingFailed)?;

                for col_idx in col..n {
                    matrix[col][col_idx] *= pivot_inv;
                }
                symbols[col].scale(pivot_inv);

                // Eliminate other rows
                for row in 0..n {
                    if row != col && !matrix[row][col].is_zero() {
                        let factor = matrix[row][col];
                        for col_idx in col..n {
                            matrix[row][col_idx] =
                                matrix[row][col_idx] + matrix[col][col_idx] * factor;
                        }
                        let scaled = symbols[col].scaled(factor);
                        symbols[row].add_assign(&scaled);
                    }
                }
            }
        }

        Ok(symbols)
    }
}

impl<F: BiniusField, const M: usize> Default for RlnDecoder<F, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField, const M: usize> Decoder<F, M> for RlnDecoder<F, M>
where
    F: From<u8> + Into<u8>,
{
    fn configure(&mut self, symbols: usize) -> Result<(), CodingError> {
        if symbols == 0 || M == 0 {
            return Err(CodingError::InvalidParameters);
        }

        self.symbols = symbols;
        self.received_symbols.clear();
        self.coefficients.clear();
        self.matrix = OptimizedMatrix::new(symbols);
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

    fn add_symbol(
        &mut self,
        coefficients: &[F],
        symbol: &crate::storage::Symbol<F, M>,
    ) -> Result<(), CodingError> {
        if coefficients.len() != self.symbols {
            return Err(CodingError::InvalidCoefficients);
        }

        // Check if this contribution increases rank
        if !self.check_rank_increase(coefficients) {
            return Err(CodingError::RedundantContribution);
        }

        self.coefficients.push(coefficients.to_vec());
        self.received_symbols.push(symbol.clone());

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

        let mut result = Vec::with_capacity(self.symbols * M);
        for symbol in &self.decoded_symbols {
            for field_element in symbol.as_slice() {
                let byte: u8 = (*field_element).into();
                result.push(byte);
            }
        }

        Ok(result)
    }

    fn symbols_needed(&self) -> usize {
        self.symbols.saturating_sub(self.coefficients.len())
    }

    fn symbols_received(&self) -> usize {
        self.coefficients.len()
    }
}

impl<F: BiniusField, const M: usize> StreamingDecoder<F, M> for RlnDecoder<F, M>
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

    fn decode_symbol(
        &mut self,
        index: usize,
    ) -> Result<Option<crate::storage::Symbol<F, M>>, CodingError> {
        if index >= self.symbols {
            return Ok(None);
        }

        Ok(self.partial_symbols[index].clone())
    }

    fn check_rank_increase(&self, coefficients: &[F]) -> bool {
        self.check_rank_increase(coefficients)
    }
}

impl<F: BiniusField, const M: usize> crate::coding::traits::RecodingDecoder<F, M>
    for RlnDecoder<F, M>
where
    F: From<u8> + Into<u8>,
{
    fn recode(
        &mut self,
        recode_coefficients: &[F],
    ) -> Result<crate::storage::Symbol<F, M>, CodingError> {
        if recode_coefficients.len() != self.coefficients.len() {
            return Err(CodingError::InvalidCoefficients);
        }

        if self.received_symbols.is_empty() {
            return Err(CodingError::InsufficientData);
        }

        let mut recoded_symbol = Symbol::<F, M>::zero();

        // Linear combination of received symbols using recode coefficients
        for (coeff, symbol) in recode_coefficients.iter().zip(self.received_symbols.iter()) {
            if !coeff.is_zero() {
                let scaled = symbol.scaled(*coeff);
                recoded_symbol.add_assign(&scaled);
            }
        }

        Ok(recoded_symbol)
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
        let mut decoder = RlnDecoder::<GF256, 16>::new();
        assert!(decoder.configure(4).is_ok());
        assert_eq!(decoder.symbols, 4);
    }

    #[test]
    fn test_decoder_add_symbol() {
        let mut decoder = RlnDecoder::<GF256, 4>::new();
        decoder.configure(2).unwrap();

        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = Symbol::<GF256, 4>::from_data([GF256::from(1), GF256::from(2), GF256::from(3), GF256::from(4)]);

        assert!(decoder.add_symbol(&coeffs, &symbol).is_ok());
        assert_eq!(decoder.symbols_received(), 1);
    }

    #[test]
    fn test_decoder_insufficient_data() {
        let mut decoder = RlnDecoder::<GF256, 4>::new();
        decoder.configure(2).unwrap();

        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = Symbol::<GF256, 4>::from_data([GF256::from(1), GF256::from(2), GF256::from(3), GF256::from(4)]);

        decoder.add_symbol(&coeffs, &symbol).unwrap();
        assert!(!decoder.can_decode());
        assert!(decoder.decode().is_err());
    }

    #[test]
    fn test_decoder_round_trip() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([123; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 2;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
            let _decoded_symbol = decoder.decode_symbol(i).unwrap().unwrap();
        }

        let decoded = decoder.decode().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_streaming_decoder_partial_progress() {
        let mut encoder = RlnEncoder::<GF256, 2>::with_seed([99; 32]);
        let mut decoder = RlnDecoder::<GF256, 2>::new();

        let symbols = 4;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
        let redundant_symbol: [GF256; 4] = [
            symbol1[0] + symbol2[0],
            symbol1[1] + symbol2[1],
            symbol1[2] + symbol2[2],
            symbol1[3] + symbol2[3],
        ];

        // Should detect this as redundant
        assert!(!decoder.check_rank_increase(&redundant_coeffs));
        let redundant_symbol_obj = Symbol::<GF256, 4>::from_data(redundant_symbol);
        let result = decoder.add_symbol(&redundant_coeffs, &redundant_symbol_obj);
        assert!(matches!(result, Err(CodingError::RedundantContribution)));

        // Rank should not increase
        assert_eq!(decoder.current_rank(), 2);
        assert_eq!(decoder.symbols_received(), 2);
    }

    #[test]
    fn test_rank_increase_checking() {
        let mut decoder = RlnDecoder::<GF256, 4>::new();
        decoder.configure(2).unwrap();

        // Linearly independent coefficients
        let coeffs1 = vec![GF256::from(1), GF256::from(0)];
        let coeffs2 = vec![GF256::from(0), GF256::from(1)];
        let coeffs3 = vec![GF256::from(1), GF256::from(1)];
        let coeffs4 = vec![GF256::from(2), GF256::from(2)]; // Linearly dependent

        // First coefficient should increase rank
        assert!(decoder.check_rank_increase(&coeffs1));
        decoder
            .add_symbol(&coeffs1, &Symbol::<GF256, 4>::from_data([GF256::from(1), GF256::from(2), GF256::from(3), GF256::from(4)]))
            .unwrap();
        assert_eq!(decoder.current_rank(), 1);

        // Second coefficient should increase rank
        assert!(decoder.check_rank_increase(&coeffs2));
        decoder
            .add_symbol(&coeffs2, &Symbol::<GF256, 4>::from_data([GF256::from(5), GF256::from(6), GF256::from(7), GF256::from(8)]))
            .unwrap();
        assert_eq!(decoder.current_rank(), 2);

        // Third coefficient should not increase rank (matrix already full rank)
        assert!(!decoder.check_rank_increase(&coeffs3));

        // Fourth coefficient is linearly dependent
        assert!(!decoder.check_rank_increase(&coeffs4));
    }

    #[test]
    fn test_decoder_empty_data() {
        let mut decoder = RlnDecoder::<GF256, 4>::new();
        assert!(decoder.configure(0).is_err());
    }

    #[test]
    fn test_decoder_large_symbols() {
        let mut decoder = RlnDecoder::<GF256, 1024>::new();
        assert!(decoder.configure(1000).is_ok());
        assert_eq!(decoder.symbols, 1000);
    }

    #[test]
    fn test_decoder_zero_symbol_size() {
        let mut decoder = RlnDecoder::<GF256, 0>::new();
        assert!(decoder.configure(5).is_err());
    }

    #[test]
    fn test_decoder_zero_symbols() {
        let mut decoder = RlnDecoder::<GF256, 1024>::new();
        assert!(decoder.configure(0).is_err());
    }

    #[test]
    fn test_decoder_not_configured() {
        let mut decoder = RlnDecoder::<GF256, 4>::new();
        let coeffs = vec![GF256::from(1), GF256::from(0)];
        let symbol = Symbol::<GF256, 4>::from_data([GF256::from(1), GF256::from(2), GF256::from(3), GF256::from(4)]);
        assert!(decoder.add_symbol(&coeffs, &symbol).is_err());
    }

    #[test]
    fn test_decoder_wrong_coefficients_length() {
        let mut decoder = RlnDecoder::<GF256, 4>::new();
        decoder.configure(2).unwrap();

        let coeffs = vec![GF256::from(1)]; // Wrong length
        let symbol = Symbol::<GF256, 4>::from_data([GF256::from(1), GF256::from(2), GF256::from(3), GF256::from(4)]);
        assert!(decoder.add_symbol(&coeffs, &symbol).is_err());
    }

    #[test]
    fn test_decoder_single_symbol() {
        let mut encoder = RlnEncoder::<GF256, 8>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 8>::new();

        encoder.configure(1).unwrap();
        decoder.configure(1).unwrap();

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
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        // First use
        encoder.configure(2).unwrap();
        decoder.configure(2).unwrap();

        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        encoder.set_data(&data1).unwrap();

        for _ in 0..2 {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        let decoded1 = decoder.decode().unwrap();
        assert_eq!(decoded1, data1);

        // Reconfigure and reuse
        // only multiples of the original size
        encoder.configure(3).unwrap();
        decoder.configure(3).unwrap();

        let data2 = vec![9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
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
        let mut encoder = RlnEncoder::<GF256, 512>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 512>::new();

        let symbols = 50;
        let symbol_size = 512;

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
        let mut decoder = RlnDecoder::<GF256, 4>::new();
        decoder.configure(3).unwrap();

        assert!(!decoder.is_symbol_decoded(3)); // Out of bounds
        assert_eq!(decoder.decode_symbol(3).unwrap(), None);
    }

    #[test]
    fn test_decoder_progress_calculation() {
        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 4;
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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
        let mut encoder = RlnEncoder::<GF256, 8>::with_seed([99; 32]);
        let mut decoder = RlnDecoder::<GF256, 8>::new();

        let symbols = 5;
        let symbol_size = 8;

        // Test with all zeros
        let data_zeros = vec![0; symbols * symbol_size];
        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();
        encoder.set_data(&data_zeros).unwrap();

        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_zeros);

        // Reset and test with all 0xFF
        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();
        let data_ones = vec![0xFF; symbols * symbol_size];
        encoder.set_data(&data_ones).unwrap();

        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_ones);

        // Test with incrementing pattern
        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();
        let data_pattern: Vec<u8> = (0..(symbols * symbol_size) as u8).collect();
        encoder.set_data(&data_pattern).unwrap();

        for _ in 0..symbols {
            let (coeffs, symbol) = encoder.encode_packet().unwrap();
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }
        assert_eq!(decoder.decode().unwrap(), data_pattern);
    }

    #[test]
    fn test_seeded_encoder_decoder_compatibility() {
        // Test that seeded encoders produce outputs that decoders can decode
        let mut encoder = RlnEncoder::<GF256, 16>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 16>::new();

        let symbols = 4;
        let data = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

        encoder.set_data(&data).unwrap();

        // Generate exactly the required symbols using seeded RNG
        let mut packets = Vec::new();
        for _ in 0..symbols {
            packets.push(encoder.encode_packet().unwrap());
        }

        // Verify deterministic behavior - same seed should produce same packets
        let mut encoder2 = RlnEncoder::<GF256, 16>::with_seed([42; 32]);
        encoder2.configure(symbols).unwrap();
        encoder2.set_data(&data).unwrap();

        for (original_coeffs, original_symbol) in &packets {
            let (new_coeffs, new_symbol) = encoder2.encode_packet().unwrap();
            assert_eq!(original_coeffs, &new_coeffs);
            assert_eq!(original_symbol, &new_symbol);
        }

        // Decode using the packets
        for (coeffs, symbol) in packets {
            decoder.add_symbol(&coeffs, &symbol).unwrap();
        }

        let decoded = decoder.decode().unwrap();
        assert_eq!(
            decoded, data,
            "Seeded encoder should produce decodable output"
        );
    }

    #[test]
    fn test_different_seeds_produce_different_outputs() {
        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        let mut encoder1 = RlnEncoder::<GF256, 4>::with_seed([1; 32]);
        let mut encoder2 = RlnEncoder::<GF256, 4>::with_seed([2; 32]);

        encoder1.configure(symbols).unwrap();
        encoder2.configure(symbols).unwrap();

        encoder1.set_data(&data).unwrap();
        encoder2.set_data(&data).unwrap();

        let (coeffs1, symbol1) = encoder1.encode_packet().unwrap();
        let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();

        // Different seeds should produce different outputs
        assert!(coeffs1 != coeffs2 || symbol1 != symbol2);

        // But both should be valid for decoding
        let mut decoder1 = RlnDecoder::<GF256, 4>::new();
        let mut decoder2 = RlnDecoder::<GF256, 4>::new();

        decoder1.configure(symbols).unwrap();
        decoder2.configure(symbols).unwrap();

        // Collect enough packets from each encoder
        let mut packets1 = Vec::new();
        let mut packets2 = Vec::new();

        for _ in 0..symbols {
            packets1.push(encoder1.encode_packet().unwrap());
            packets2.push(encoder2.encode_packet().unwrap());
        }

        // Decode
        for (coeffs, symbol) in packets1 {
            decoder1.add_symbol(&coeffs, &symbol).unwrap();
        }
        for (coeffs, symbol) in packets2 {
            decoder2.add_symbol(&coeffs, &symbol).unwrap();
        }

        let decoded1 = decoder1.decode().unwrap();
        let decoded2 = decoder2.decode().unwrap();

        assert_eq!(decoded1, data);
        assert_eq!(decoded2, data);
    }

    #[test]
    fn test_recoding_functionality() {
        use crate::coding::traits::RecodingDecoder;
        use crate::utils::CodingRng;

        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 3;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        encoder.configure(symbols).unwrap();
        decoder.configure(symbols).unwrap();

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

        let _recoded_symbol = decoder.recode(&recode_coeffs).unwrap();
    }

    #[test]
    fn test_recoding_three_node_network() {
        use crate::coding::traits::RecodingDecoder;

        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut relay_decoder = RlnDecoder::<GF256, 4>::new();
        let mut final_decoder = RlnDecoder::<GF256, 4>::new();

        let symbols = 2;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

        // Configure all nodes
        encoder.configure(symbols).unwrap();
        relay_decoder.configure(symbols).unwrap();
        final_decoder.configure(symbols).unwrap();

        encoder.set_data(&data).unwrap();

        // Step 1: Simulate encoder sending symbols to relay
        // We'll manually create the symbols to ensure we know what we're testing
        let coeffs1 = vec![GF256::from(1), GF256::from(0)];
        let coeffs2 = vec![GF256::from(0), GF256::from(1)];
        let _symbol1 = [1, 2, 3, 4]; // First original symbol
        let _symbol2 = [5, 6, 7, 8]; // Second original symbol

        // Create encoded symbols: c1*symbol1 + c2*symbol2
        let encoded1 = Symbol::<GF256, 4>::from_data([GF256::from(1), GF256::from(2), GF256::from(3), GF256::from(4)]); // 1*symbol1 + 0*symbol2
        let encoded2 = Symbol::<GF256, 4>::from_data([GF256::from(5), GF256::from(6), GF256::from(7), GF256::from(8)]); // 0*symbol1 + 1*symbol2

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

        let mut decoder = RlnDecoder::<GF256, 4>::new();
        decoder.configure(2).unwrap();

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

        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        encoder.configure(2).unwrap();
        decoder.configure(2).unwrap();

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

        let mut encoder = RlnEncoder::<GF256, 4>::with_seed([42; 32]);
        let mut decoder = RlnDecoder::<GF256, 4>::new();

        encoder.configure(2).unwrap();
        decoder.configure(2).unwrap();

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
        let expected: [u8; 4] = [0, 0, 0, 0];
        let actual: [u8; 4] = result.into_inner().map(|f| f.into());
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_recoding_bandwidth_efficiency() {
        use crate::coding::traits::RecodingDecoder;

        let mut encoder = RlnEncoder::<GF256, 100>::with_seed([1; 32]);
        let mut decoder = RlnDecoder::<GF256, 100>::new();

        encoder.configure(5).unwrap();
        decoder.configure(5).unwrap();

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
        let _recoded = decoder.recode(&recode_coeffs).unwrap();
    }
}
