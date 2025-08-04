use binius_field::{underlier::WithUnderlier, Field as BiniusField};
use std::ops::{Index, IndexMut};

/// A symbol is a fixed-size chunk of data in a network coding context
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Symbol<const N: usize> {
    data: [u8; N],
}

impl<const N: usize> Symbol<N> {
    /// Create a new symbol with all bytes set to 0
    pub fn new() -> Self {
        Self { data: [0u8; N] }
    }

    /// Create a new symbol from existing data
    pub fn from_data(data: [u8; N]) -> Self {
        Self { data }
    }

    /// Create a zero symbol
    pub fn zero() -> Self {
        Self::new()
    }

    /// Get the size of the symbol in bytes (const)
    pub const fn len() -> usize {
        N
    }

    /// Check if the symbol is empty (always false for N > 0)
    pub const fn is_empty() -> bool {
        N == 0
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
    pub fn into_inner(self) -> [u8; N] {
        self.data
    }

    /// Add another symbol to this one (element-wise XOR for GF(256))
    pub fn add_assign(&mut self, other: &Self) {
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
            self.data = [0u8; N];
        } else if scalar != F::ONE {
            #[inline(always)]
            fn scale_byte<F>(byte: &mut u8, scalar: F)
            where
                F: BiniusField + WithUnderlier<Underlier = u8>,
            {
                if *byte == 0 {
                    return;
                }
                let field_byte = F::from_underlier(*byte);
                let scaled = field_byte * scalar;
                *byte = scaled.to_underlier();
            }

            for byte in &mut self.data {
                scale_byte(byte, scalar);
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

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [u8; N] {
        &mut self.data
    }

    /// Convert from a slice of bytes (panics if slice is wrong size)
    pub fn from_slice(slice: &[u8]) -> Self {
        assert_eq!(slice.len(), N, "Slice length must match symbol size");
        let mut data = [0u8; N];
        data.copy_from_slice(slice);
        Self { data }
    }

    /// Create a symbol filled with a specific value
    pub fn filled(value: u8) -> Self {
        Self { data: [value; N] }
    }
}

impl<const N: usize> Default for Symbol<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Index<usize> for Symbol<N> {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> IndexMut<usize> for Symbol<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const N: usize> From<[u8; N]> for Symbol<N> {
    fn from(data: [u8; N]) -> Self {
        Self::from_data(data)
    }
}

impl<const N: usize> From<Symbol<N>> for [u8; N] {
    fn from(symbol: Symbol<N>) -> Self {
        symbol.into_inner()
    }
}

impl<const N: usize> AsRef<[u8]> for Symbol<N> {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl<const N: usize> AsMut<[u8]> for Symbol<N> {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::AESTowerField8b as GF256;

    #[test]
    fn test_symbol_creation() {
        let symbol = Symbol::<10>::new();
        assert_eq!(symbol.as_slice(), &[0u8; 10]);
    }

    #[test]
    fn test_symbol_from_data() {
        let data = [1, 2, 3, 4, 5];
        let symbol = Symbol::<5>::from_data(data);
        assert_eq!(symbol.as_slice(), data.as_slice());
    }

    #[test]
    fn test_symbol_add_assign() {
        let mut a = Symbol::<5>::from_data([1, 2, 3, 4, 5]);
        let b = Symbol::<5>::from_data([5, 4, 3, 2, 1]);

        a.add_assign(&b);
        assert_eq!(a.as_slice(), &[4, 6, 0, 6, 4]);
    }

    #[test]
    fn test_symbol_scaling() {
        let mut symbol = Symbol::<5>::from_data([1, 2, 3, 4, 5]);
        symbol.scale(GF256::from(2));

        assert_eq!(symbol.as_slice()[0], 2);
        assert_eq!(symbol.as_slice()[1], 4);
    }

    #[test]
    fn test_symbol_scaling_gf256() {
        let mut symbol = Symbol::<3>::from_data([0x02, 0x03, 0x04]);
        let scalar = GF256::from(0x03);
        symbol.scale(scalar);

        let expected_0: u8 = (GF256::from(0x02) * scalar).into();
        let expected_1: u8 = (GF256::from(0x03) * scalar).into();
        let expected_2: u8 = (GF256::from(0x04) * scalar).into();

        assert_eq!(symbol.as_slice()[0], expected_0);
        assert_eq!(symbol.as_slice()[1], expected_1);
        assert_eq!(symbol.as_slice()[2], expected_2);
    }

    #[test]
    fn test_symbol_len_const() {
        assert_eq!(Symbol::<16>::len(), 16);
        assert_eq!(Symbol::<1024>::len(), 1024);
    }
}
