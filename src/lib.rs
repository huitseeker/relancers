//! A Rust implementation of network coding algorithms
//!
//! This crate provides efficient implementations of network coding schemes including:
//! - Random Linear Network Coding (RLNC)
//! - Reed-Solomon codes

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]

pub use binius_field::{AESTowerField8b, BinaryField, Field as BiniusField};

pub mod coding;
pub mod storage;
pub mod utils;
