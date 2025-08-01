//! Network coding implementations

/// Reed-Solomon error correcting codes
pub mod reed_solomon;
/// Random Linear Network Coding implementation
pub mod rlnc;
/// Sparse coefficient generation for RLNC
pub mod sparse;
/// Core coding traits and error types
pub mod traits;

pub use reed_solomon::{RsDecoder, RsEncoder};
pub use rlnc::{RlnDecoder, RlnEncoder};
pub use sparse::{SparseCoeffGenerator, SparseConfig};
pub use traits::{Decoder, Encoder};
