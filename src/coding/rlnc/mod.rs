//! Random Linear Network Coding (RLNC) implementation

mod decoder;
mod encoder;
mod optimized_matrix;

pub use decoder::RlnDecoder;
pub use encoder::RlnEncoder;
pub use optimized_matrix::OptimizedMatrix;
