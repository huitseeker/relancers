use binius_field::Field as BiniusField;
use std::ops::{Index, IndexMut};

/// A symbol is a fixed-size chunk of data in a network coding context
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Symbol<F: BiniusField, const M: usize> {
    data: [F; M],
}

impl<F: BiniusField, const M: usize> Symbol<F, M> {
    /// Create a new symbol with all elements set to zero
    pub fn new() -> Self {
        Self { data: [F::ZERO; M] }
    }

    /// Create a new symbol from existing field elements
    pub fn from_data(data: [F; M]) -> Self {
        Self { data }
    }

    /// Create a zero symbol
    pub fn zero() -> Self {
        Self::new()
    }

    /// Get the size of the symbol in bytes (const)
    pub const fn len() -> usize {
        M
    }

    /// Check if the symbol is empty (always false for M > 0)
    pub const fn is_empty() -> bool {
        M == 0
    }

    /// Get the underlying data as a slice
    pub fn as_slice(&self) -> &[F] {
        &self.data
    }

    /// Get the underlying data as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [F] {
        &mut self.data
    }

    /// Get the underlying data
    pub fn into_inner(self) -> [F; M] {
        self.data
    }

    /// Add another symbol to this one (element-wise addition for field elements)
    pub fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    /// Scale this symbol by a field element
    pub fn scale(&mut self, scalar: F) {
        if scalar.is_zero() {
            for elem in &mut self.data {
                *elem = F::ZERO;
            }
        } else if scalar != F::ONE {
            for elem in &mut self.data {
                *elem *= scalar;
            }
        }
    }

    /// Create a copy of this symbol scaled by a field element
    pub fn scaled(&self, scalar: F) -> Self {
        let mut result = self.clone();
        result.scale(scalar);
        result
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [F; M] {
        &mut self.data
    }

    /// Convert from a slice of bytes (panics if slice is wrong size)
    pub fn from_bytes(slice: &[u8]) -> Self
    where
        F: From<u8>,
    {
        assert_eq!(slice.len(), M, "Slice length must match symbol size");
        let mut data = [F::ZERO; M];
        for (i, &byte) in slice.iter().enumerate() {
            data[i] = F::from(byte);
        }
        Self { data }
    }

    /// Create a symbol filled with a specific field element
    pub fn filled(value: F) -> Self {
        Self { data: [value; M] }
    }
}

impl<F: BiniusField, const M: usize> Default for Symbol<F, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BiniusField, const M: usize> Index<usize> for Symbol<F, M> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<F: BiniusField, const M: usize> IndexMut<usize> for Symbol<F, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<F: BiniusField, const M: usize> From<[F; M]> for Symbol<F, M> {
    fn from(data: [F; M]) -> Self {
        Self::from_data(data)
    }
}

impl<F: BiniusField, const M: usize> From<Symbol<F, M>> for [F; M] {
    fn from(symbol: Symbol<F, M>) -> Self {
        symbol.into_inner()
    }
}

impl<F: BiniusField, const M: usize> AsRef<[F]> for Symbol<F, M> {
    fn as_ref(&self) -> &[F] {
        &self.data
    }
}

impl<F: BiniusField, const M: usize> AsMut<[F]> for Symbol<F, M> {
    fn as_mut(&mut self) -> &mut [F] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_symbol_creation() {
        let symbol = Symbol::<GF256, 10>::new();
        assert_eq!(symbol.as_slice(), &[GF256::ZERO; 10]);
    }

    #[test]
    fn test_symbol_from_data() {
        let data = [GF256::from(1), GF256::from(2), GF256::from(3), GF256::from(4), GF256::from(5)];
        let symbol = Symbol::<GF256, 5>::from_data(data);
        assert_eq!(symbol.as_slice(), data.as_slice());
    }

    #[test]
    fn test_symbol_add_assign() {
        let mut a = Symbol::<GF256, 5>::from_data([
            GF256::from(1),
            GF256::from(2),
            GF256::from(3),
            GF256::from(4),
            GF256::from(5),
        ]);
        let b = Symbol::<GF256, 5>::from_data([
            GF256::from(5),
            GF256::from(4),
            GF256::from(3),
            GF256::from(2),
            GF256::from(1),
        ]);

        a.add_assign(&b);
        let expected = [
            GF256::from(4),
            GF256::from(6),
            GF256::from(0),
            GF256::from(6),
            GF256::from(4),
        ];
        assert_eq!(a.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_symbol_scaling() {
        let mut symbol = Symbol::<GF256, 5>::from_data([
            GF256::from(1),
            GF256::from(2),
            GF256::from(3),
            GF256::from(4),
            GF256::from(5),
        ]);
        symbol.scale(GF256::from(2));

        assert_eq!(symbol[0], GF256::from(2));
        assert_eq!(symbol[1], GF256::from(4));
    }

    #[test]
    fn test_symbol_scaling_gf256() {
        let mut symbol = Symbol::<GF256, 3>::from_data([
            GF256::from(0x02),
            GF256::from(0x03),
            GF256::from(0x04),
        ]);
        let scalar = GF256::from(0x03);
        symbol.scale(scalar);

        let expected_0 = GF256::from(0x02) * scalar;
        let expected_1 = GF256::from(0x03) * scalar;
        let expected_2 = GF256::from(0x04) * scalar;

        assert_eq!(symbol[0], expected_0);
        assert_eq!(symbol[1], expected_1);
        assert_eq!(symbol[2], expected_2);
    }

    #[test]
    fn test_symbol_len_const() {
        assert_eq!(Symbol::<GF256, 16>::len(), 16);
        assert_eq!(Symbol::<GF256, 1024>::len(), 1024);
    }

    #[test]
    fn test_symbol_from_bytes() {
        let bytes = [1, 2, 3, 4, 5];
        let symbol = Symbol::<GF256, 5>::from_bytes(&bytes);
        let expected = [
            GF256::from(1),
            GF256::from(2),
            GF256::from(3),
            GF256::from(4),
            GF256::from(5),
        ];
        assert_eq!(symbol.as_slice(), expected.as_slice());
    }
}    
