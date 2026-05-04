//! Utility functions for network coding

/// Random number generation utilities
pub mod rand;

/// Precomputed AES GF(2^8) multiplication table
pub mod mul_table;

/// SIMD-accelerated GF(2^8) operations
pub mod simd;

pub use rand::CodingRng;
