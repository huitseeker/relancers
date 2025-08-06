use binius_field::{underlier::WithUnderlier, Field as BiniusField};
use std::ops::{Index, IndexMut};

/// A symbol is a fixed-size chunk of data in a network coding context
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Symbol<F, const M: usize>
where
    F: BiniusField,
{
    data: [F; M],
}

impl<F, const M: usize> Symbol<F, M>
where
    F: BiniusField + WithUnderlier<Underlier = u8>,
{
    /// Create a new symbol with all elements set to zero
    pub fn new() -> Self {
        Self { data: [F::ZERO; M] }
    }

    /// Create a zero symbol
    pub fn zero() -> Self {
        Self::new()
    }

    /// Get the size of the symbol in elements (const)
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

    /// Add another symbol to this one (element-wise addition)
    pub fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    /// Scale this symbol by a field element
    pub fn scale(&mut self, scalar: F) {
        if scalar.is_zero() {
            self.data = [F::ZERO; M];
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
    pub fn from_bytes(slice: &[u8]) -> Self {
        assert_eq!(slice.len(), M, "Slice length must match symbol size");
        let mut data = [F::ZERO; M];
        for (i, &byte) in slice.iter().enumerate() {
            data[i] = F::from_underlier(byte);
        }
        Self { data }
    }

    /// Convert from a slice of field elements (panics if slice is wrong size)
    pub fn from_field_elements(elements: &[F]) -> Self {
        assert_eq!(elements.len(), M, "Slice length must match symbol size");
        let mut data = [F::ZERO; M];
        data.copy_from_slice(elements);
        Self { data }
    }

    /// Create a symbol filled with a specific byte value
    pub fn filled_from_byte(value: u8) -> Self {
        let field_value = F::from_underlier(value);
        Self {
            data: [field_value; M],
        }
    }
}

impl<F, const M: usize> Default for Symbol<F, M>
where
    F: BiniusField + WithUnderlier<Underlier = u8>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F, const M: usize> Index<usize> for Symbol<F, M>
where
    F: BiniusField + WithUnderlier<Underlier = u8>,
{
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<F, const M: usize> IndexMut<usize> for Symbol<F, M>
where
    F: BiniusField + WithUnderlier<Underlier = u8>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<F, const M: usize> From<[u8; M]> for Symbol<F, M>
where
    F: BiniusField + WithUnderlier<Underlier = u8>,
{
    fn from(data: [u8; M]) -> Self {
        let mut field_data = [F::ZERO; M];
        for (i, &byte) in data.iter().enumerate() {
            field_data[i] = F::from_underlier(byte);
        }
        Symbol { data: field_data }
    }
}

impl<F, const M: usize> From<Symbol<F, M>> for [F; M]
where
    F: BiniusField + WithUnderlier<Underlier = u8>,
{
    fn from(symbol: Symbol<F, M>) -> Self {
        symbol.into_inner()
    }
}

impl<F, const M: usize> AsRef<[F]> for Symbol<F, M>
where
    F: BiniusField + WithUnderlier<Underlier = u8>,
{
    fn as_ref(&self) -> &[F] {
        &self.data
    }
}

impl<F, const M: usize> AsMut<[F]> for Symbol<F, M>
where
    F: BiniusField + WithUnderlier<Underlier = u8>,
{
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
        assert_eq!(symbol.into_inner().map(|f| f.to_underlier()), [0u8; 10]);
    }

    #[test]
    fn test_symbol_from_data() {
        let data = [1, 2, 3, 4, 5];
        let symbol: Symbol<GF256, 5> = Symbol::<GF256, 5>::from(data);
        assert_eq!(
            symbol.into_inner().map(|f| f.to_underlier()),
            data.as_slice()
        );
    }

    #[test]
    fn test_symbol_add_assign() {
        let mut a = Symbol::<GF256, 5>::from([1, 2, 3, 4, 5]);
        let b = Symbol::<GF256, 5>::from([5, 4, 3, 2, 1]);

        a.add_assign(&b);
        assert_eq!(a.into_inner().map(|f| f.to_underlier()), [4, 6, 0, 6, 4]);
    }

    #[test]
    fn test_symbol_scaling() {
        let mut symbol = Symbol::<GF256, 5>::from([1, 2, 3, 4, 5]);
        symbol.scale(GF256::from(2));

        assert_eq!(symbol.as_slice()[0].to_underlier(), 2);
        assert_eq!(symbol.as_slice()[1].to_underlier(), 4);
    }

    #[test]
    fn test_symbol_scaling_gf256() {
        let mut symbol = Symbol::<GF256, 3>::from([0x02, 0x03, 0x04]);
        let scalar = GF256::from(0x03);
        symbol.scale(scalar);

        let expected_0: u8 = (GF256::from(0x02) * scalar).into();
        let expected_1: u8 = (GF256::from(0x03) * scalar).into();
        let expected_2: u8 = (GF256::from(0x04) * scalar).into();

        assert_eq!(symbol.as_slice()[0].to_underlier(), expected_0);
        assert_eq!(symbol.as_slice()[1].to_underlier(), expected_1);
        assert_eq!(symbol.as_slice()[2].to_underlier(), expected_2);
    }

    #[test]
    fn test_symbol_len_const() {
        assert_eq!(Symbol::<GF256, 16>::len(), 16);
        assert_eq!(Symbol::<GF256, 1024>::len(), 1024);
    }
}
