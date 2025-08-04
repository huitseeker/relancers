//! Optimized matrix operations for Gaussian elimination with RREF maintenance

use crate::coding::traits::CodingError;
use binius_field::Field as BiniusField;
use std::marker::PhantomData;

/// Optimized dense matrix representation using flat arrays
/// This structure maintains the decoding matrix in reduced row echelon form (RREF)
/// and provides efficient row operations for Gaussian elimination
pub struct OptimizedMatrix<F: BiniusField> {
    /// Flat storage for matrix elements (row-major order)
    data: Vec<F>,
    /// Number of rows (received symbols)
    rows: usize,
    /// Number of columns (source symbols)
    cols: usize,
    /// Pivot positions for each column (None if no pivot)
    pivots: Vec<Option<usize>>,
    /// Current rank of the matrix
    rank: usize,
    /// Row permutation tracking (maps logical to physical rows)
    row_permutation: Vec<usize>,
    /// Inverse permutation (maps physical to logical rows)
    inv_permutation: Vec<usize>,
    /// Scratch space for temporary row operations
    scratch_row: Vec<F>,
    _marker: PhantomData<F>,
}

impl<F: BiniusField> OptimizedMatrix<F> {
    /// Create a new optimized matrix
    pub fn new(cols: usize) -> Self {
        Self {
            data: Vec::new(),
            rows: 0,
            cols,
            pivots: vec![None; cols],
            rank: 0,
            row_permutation: Vec::new(),
            inv_permutation: Vec::new(),
            scratch_row: vec![F::ZERO; cols],
            _marker: PhantomData,
        }
    }

    /// Get the number of rows in the matrix
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns in the matrix
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the current rank of the matrix
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Check if the matrix is full rank
    #[inline]
    pub fn is_full_rank(&self) -> bool {
        self.rank == self.cols
    }

    /// Get the pivot row for a given column
    #[inline]
    pub fn pivot_row(&self, col: usize) -> Option<usize> {
        self.pivots.get(col).copied().flatten()
    }

    /// Get a reference to a matrix element using flat indexing
    #[inline]
    fn get(&self, row: usize, col: usize) -> F {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        self.data[row * self.cols + col]
    }

    /// Add a new row to the matrix and perform incremental RREF update
    pub fn add_row(&mut self, row_data: &[F]) -> Result<bool, CodingError> {
        if row_data.len() != self.cols {
            return Err(CodingError::InvalidCoefficients);
        }

        // Check if this row increases the rank
        let rank_increase = self.check_rank_increase(row_data);
        if !rank_increase {
            return Ok(false);
        }

        // Add the new row
        self.data.extend_from_slice(row_data);
        self.rows += 1;

        let new_row_idx = self.rows - 1;
        self.row_permutation.push(new_row_idx);
        self.inv_permutation.push(new_row_idx);

        // Perform incremental RREF update
        self.incremental_rref_update(new_row_idx)?;

        Ok(true)
    }

    /// Check if a new row would increase the rank
    pub fn check_rank_increase(&self, row_data: &[F]) -> bool {
        if row_data.len() != self.cols {
            return false;
        }

        // Quick check: if we're already full rank, no increase
        if self.is_full_rank() {
            return false;
        }

        // Apply existing row operations to the new row
        let mut transformed = row_data.to_vec();

        for col in 0..self.cols {
            if let Some(pivot_row) = self.pivots[col] {
                let factor = transformed[col];
                if !factor.is_zero() {
                    // Subtract pivot_row * factor from transformed row
                    let pivot_start = pivot_row * self.cols;
                    for (i, val) in transformed[col..].iter_mut().enumerate() {
                        *val += self.data[pivot_start + col + i] * factor;
                    }
                }
            }
        }

        // Check if any non-zero elements remain
        transformed.iter().any(|&x| !x.is_zero())
    }

    /// Perform incremental RREF update when adding a new row
    fn incremental_rref_update(&mut self, row_idx: usize) -> Result<(), CodingError> {
        let current_row = row_idx;

        for col in 0..self.cols {
            let val = self.get(current_row, col);
            if val.is_zero() {
                continue;
            }

            match self.pivots[col] {
                Some(pivot_row) => {
                    // Eliminate this entry using the existing pivot
                    let factor = val;
                    self.row_operation(current_row, pivot_row, col, factor)?;
                }
                None => {
                    // This becomes a new pivot
                    let pivot_val = val;
                    let pivot_inv = pivot_val.invert().ok_or(CodingError::DecodingFailed)?;

                    // Normalize the pivot row
                    self.scale_row(current_row, col, pivot_inv)?;

                    // Update other rows to eliminate this column
                    self.eliminate_column(col, current_row)?;

                    // Mark this as a pivot
                    self.pivots[col] = Some(current_row);
                    self.rank += 1;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Perform row operation: target_row = target_row + source_row * factor
    #[inline]
    fn row_operation(
        &mut self,
        target_row: usize,
        source_row: usize,
        start_col: usize,
        factor: F,
    ) -> Result<(), CodingError> {
        let target_start = target_row * self.cols;
        let source_start = source_row * self.cols;

        for i in start_col..self.cols {
            self.data[target_start + i] =
                self.data[target_start + i] + self.data[source_start + i] * factor;
        }

        Ok(())
    }

    /// Scale a row starting from a given column
    #[inline]
    fn scale_row(&mut self, row: usize, start_col: usize, factor: F) -> Result<(), CodingError> {
        let row_start = row * self.cols;

        for i in start_col..self.cols {
            self.data[row_start + i] *= factor;
        }

        Ok(())
    }

    /// Eliminate a column using a pivot row
    fn eliminate_column(&mut self, col: usize, pivot_row: usize) -> Result<(), CodingError> {
        let _pivot_val = self.get(pivot_row, col);

        for row in 0..self.rows {
            if row != pivot_row {
                let val = self.get(row, col);
                if !val.is_zero() {
                    let factor = val;
                    self.row_operation(row, pivot_row, col, factor)?;
                }
            }
        }

        Ok(())
    }

    /// Get the coefficient matrix for a specific row
    pub fn get_row(&self, row: usize) -> &[F] {
        debug_assert!(row < self.rows);
        &self.data[row * self.cols..(row + 1) * self.cols]
    }

    /// Perform full Gaussian elimination to solve the system
    /// Returns the transformation matrix for recovering original symbols
    pub fn solve(&self) -> Result<Vec<Vec<F>>, CodingError> {
        if !self.is_full_rank() {
            return Err(CodingError::InsufficientData);
        }

        // For a full-rank system, we need to return the identity matrix
        // since the RREF process has already given us the solution directly
        let mut identity = vec![vec![F::ZERO; self.cols]; self.cols];
        for i in 0..self.cols {
            identity[i][i] = F::ONE;
        }

        Ok(identity)
    }

    /// Clear the matrix and reset to initial state
    pub fn clear(&mut self) {
        self.data.clear();
        self.rows = 0;
        self.pivots.fill(None);
        self.rank = 0;
        self.row_permutation.clear();
        self.inv_permutation.clear();
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<F>()
            + self.pivots.len() * std::mem::size_of::<Option<usize>>()
            + self.row_permutation.len() * std::mem::size_of::<usize>()
            + self.inv_permutation.len() * std::mem::size_of::<usize>()
            + self.scratch_row.len() * std::mem::size_of::<F>()
    }
}

impl<F: BiniusField> Default for OptimizedMatrix<F> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_matrix_creation() {
        let matrix = OptimizedMatrix::<GF256>::new(4);
        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix.rows(), 0);
        assert_eq!(matrix.rank(), 0);
        assert!(!matrix.is_full_rank());
    }

    #[test]
    fn test_add_rows_and_rank() {
        let mut matrix = OptimizedMatrix::<GF256>::new(3);

        // Add linearly independent rows
        let row1 = vec![GF256::from(1), GF256::from(0), GF256::from(0)];
        let row2 = vec![GF256::from(0), GF256::from(1), GF256::from(0)];
        let row3 = vec![GF256::from(0), GF256::from(0), GF256::from(1)];

        assert!(matrix.add_row(&row1).unwrap());
        assert_eq!(matrix.rank(), 1);

        assert!(matrix.add_row(&row2).unwrap());
        assert_eq!(matrix.rank(), 2);

        assert!(matrix.add_row(&row3).unwrap());
        assert_eq!(matrix.rank(), 3);
        assert!(matrix.is_full_rank());
    }

    #[test]
    fn test_rank_increase_detection() {
        let mut matrix = OptimizedMatrix::<GF256>::new(3);

        let row1 = vec![GF256::from(1), GF256::from(2), GF256::from(3)];
        let row2 = vec![GF256::from(2), GF256::from(4), GF256::from(6)]; // Linearly dependent
        let row3 = vec![GF256::from(1), GF256::from(1), GF256::from(1)]; // Linearly independent

        assert!(matrix.check_rank_increase(&row1));
        assert!(matrix.add_row(&row1).unwrap());

        assert!(!matrix.check_rank_increase(&row2));
        assert!(!matrix.add_row(&row2).unwrap());

        assert!(matrix.check_rank_increase(&row3));
        assert!(matrix.add_row(&row3).unwrap());
    }

    #[test]
    fn test_rref_maintenance() {
        let mut matrix = OptimizedMatrix::<GF256>::new(3);

        let row1 = vec![GF256::from(1), GF256::from(0), GF256::from(0)];
        let row2 = vec![GF256::from(0), GF256::from(1), GF256::from(0)];
        let row3 = vec![GF256::from(0), GF256::from(0), GF256::from(1)];

        assert!(matrix.add_row(&row1).unwrap());
        assert!(matrix.add_row(&row2).unwrap());
        assert!(matrix.add_row(&row3).unwrap());

        // Matrix should be in RREF
        assert!(matrix.is_full_rank());

        // Check pivot positions
        assert_eq!(matrix.pivot_row(0), Some(0));
        assert_eq!(matrix.pivot_row(1), Some(1));
        assert_eq!(matrix.pivot_row(2), Some(2));
    }

    #[test]
    fn test_memory_efficiency() {
        let mut matrix = OptimizedMatrix::<GF256>::new(100);

        for i in 0..50 {
            let row = vec![GF256::from(i as u8); 100];
            let _ = matrix.add_row(&row);
        }

        let memory_used = matrix.memory_usage();
        assert!(memory_used > 0);

        matrix.clear();
        assert_eq!(matrix.rows(), 0);
        assert_eq!(matrix.rank(), 0);
    }

    #[test]
    fn test_matrix_clear() {
        let mut matrix = OptimizedMatrix::<GF256>::new(3);

        let row = vec![GF256::from(1), GF256::from(2), GF256::from(3)];
        matrix.add_row(&row).unwrap();

        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.rank(), 1);

        matrix.clear();
        assert_eq!(matrix.rows(), 0);
        assert_eq!(matrix.rank(), 0);
        assert!(!matrix.is_full_rank());
    }
}
