//! Utility functions for network coding

/// Random number generation utilities
pub mod rand;

/// Precomputed AES GF(2^8) multiplication table
pub mod mul_table;

pub use rand::CodingRng;
