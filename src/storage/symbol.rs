use binius_field::{underlier::WithUnderlier, Field as BiniusField};
use std::ops::{Index, IndexMut};

/// A symbol is a fixed-size chunk of data in a network coding context
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    data: Vec<u8>,
}

impl Symbol {
    /// Create a new symbol with the given size
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
        }
    }

    /// Create a new symbol from existing data
    pub fn from_data(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Create a zero symbol of given size
    pub fn zero(size: usize) -> Self {
        Self::new(size)
    }

    /// Get the size of the symbol in bytes
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the symbol is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the underlying data as a slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get the underlying data as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get the underlying data
    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }

    /// Add another symbol to this one (element-wise XOR for GF(256))
    pub fn add_assign(&mut self, other: &Symbol) {
        assert_eq!(self.len(), other.len(), "Symbols must have same size");

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a ^= *b;
        }
    }

    /// Scale this symbol by a field element
    pub fn scale<F>(&mut self, scalar: F)
    where
        F: BiniusField + WithUnderlier<Underlier = u8>,
    {
        if scalar.is_zero() {
            for byte in &mut self.data {
                *byte = 0;
            }
        } else if scalar != F::ONE {
            // Use proper field multiplication with Binius
            for byte in &mut self.data {
                let field_byte = F::from_underlier(*byte);
                let scaled = field_byte * scalar;
                *byte = scaled.to_underlier();
            }
        }
    }

    /// Create a copy of this symbol scaled by a field element
    pub fn scaled<F>(&self, scalar: F) -> Self
    where
        F: BiniusField + WithUnderlier<Underlier = u8>,
    {
        let mut result = self.clone();
        result.scale(scalar);
        result
    }
}

impl Index<usize> for Symbol {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Symbol {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl From<Vec<u8>> for Symbol {
    fn from(data: Vec<u8>) -> Self {
        Self::from_data(data)
    }
}

impl From<Symbol> for Vec<u8> {
    fn from(symbol: Symbol) -> Self {
        symbol.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_symbol_creation() {
        let symbol = Symbol::new(10);
        assert_eq!(symbol.len(), 10);
        assert_eq!(symbol.as_slice(), &[0u8; 10]);
    }

    #[test]
    fn test_symbol_from_data() {
        let data = vec![1, 2, 3, 4, 5];
        let symbol = Symbol::from_data(data.clone());
        assert_eq!(symbol.as_slice(), data.as_slice());
    }

    #[test]
    fn test_symbol_add_assign() {
        let mut a = Symbol::from_data(vec![1, 2, 3, 4, 5]);
        let b = Symbol::from_data(vec![5, 4, 3, 2, 1]);

        a.add_assign(&b);
        assert_eq!(a.as_slice(), &[4, 6, 0, 6, 4]);
    }

    #[test]
    fn test_symbol_scaling() {
        let mut symbol = Symbol::from_data(vec![1, 2, 3, 4, 5]);
        symbol.scale(GF256::from(2));

        // Note: This uses actual Binius GF(256) multiplication with AESTowerField8b
        // Results depend on actual field arithmetic
        let result = symbol.as_slice();
        assert_eq!(result[0], 2); // 1 * 2 = 2
        assert_eq!(result[1], 4); // 2 * 2 = 4 (actual AESTowerField8b result)
        assert_eq!(result[2], 6); // 3 * 2 = 6 (actual AESTowerField8b result)
        assert_eq!(result[3], 8); // 4 * 2 = 8
        assert_eq!(result[4], 10); // 5 * 2 = 10
    }

    #[test]
    fn test_symbol_scaling_gf256() {
        let mut symbol = Symbol::from_data(vec![0x02, 0x03, 0x04]);
        let scalar = GF256::from(0x03);
        symbol.scale(scalar);

        // Test actual Binius GF(256) multiplication
        let result = symbol.as_slice();
        let expected_0: u8 = (GF256::from(0x02) * scalar).into();
        let expected_1: u8 = (GF256::from(0x03) * scalar).into();
        let expected_2: u8 = (GF256::from(0x04) * scalar).into();

        assert_eq!(result[0], expected_0);
        assert_eq!(result[1], expected_1);
        assert_eq!(result[2], expected_2);
    }
}
