# Relancers

⚠️ **Alpha Software** - This crate is in early development and has not been security audited. Use in production at your own risk.

High-performance Random Linear Network Coding (RLNC) for Rust, powered by [Binius](https://github.com/IrreducibleOSS/binius) with AVX-512 acceleration.

[![Crates.io](https://img.shields.io/crates/v/relancers.svg)](https://crates.io/crates/relancers)
[![Docs.rs](https://img.shields.io/docsrs/relancers)](https://docs.rs/relancers)
[![License](https://img.shields.io/crates/l/relancers.svg)](#license)

## Performance

**Relancers** leverages [Binius](https://github.com/IrreducibleOSS/binius) for cutting-edge finite field arithmetic with **AVX-512 acceleration**. For optimal performance:

```bash
cargo +nightly bench --features "nightly_features"
```

Requires nightly Rust for SIMD optimizations.

## Quick Start

```rust
use relancers::coding::{RlnEncoder, RlnDecoder, Encoder, Decoder};
use binius_field::AESTowerField8b as GF256;

// 3× faster than dense RLNC with sparse coefficients
let mut encoder = RlnEncoder::<GF256, 1024>::new();
encoder.configure(16).unwrap();
encoder.set_data(&data).unwrap();
let (coeffs, symbol) = encoder.encode_packet().unwrap();
```

## Features

- **RLNC** (Random Linear Network Coding) - dense and sparse coefficients
- **Sparse RLNC** - configurable sparsity levels (0.1-1.0)
- **Reed-Solomon** systematic codes *(testing only, not optimized)*
- **Streaming decoder** with partial recovery
- **Seeded encoders** for deterministic testing (requires non-zero seed)
- **AVX-512 acceleration** via Binius
- **GF(256)** finite field arithmetic

## Usage

### Basic RLNC
```rust
use relancers::coding::{RlnEncoder, Encoder};
use binius_field::AESTowerField8b as GF256;

let mut encoder = RlnEncoder::<GF256, 1024>::new();
encoder.configure(16).unwrap();
encoder.set_data(&data).unwrap();
let (coefficients, symbol) = encoder.encode_packet().unwrap();
```

### Sparse RLNC
```rust
use relancers::coding::{RlnEncoder, SparseConfig};
use binius_field::AESTowerField8b as GF256;

let mut encoder = RlnEncoder::<GF256, 1024>::new();
encoder.configure_with_sparsity(16, Some(0.3)).unwrap(); // 30% sparsity
// or
encoder.configure(16).unwrap();
encoder.set_sparsity(0.3); // Configure sparsity after configuration
```

### Seeded Encoders for Deterministic Testing

For reproducible coefficient generation, use seeded encoders with a **non-zero seed**. Zero seeds (`[0u8; 32]`) are treated as "no seed specified" and will use system entropy.

```rust
use relancers::coding::{RlnEncoder, Encoder};
use binius_field::AESTowerField8b as GF256;

// Deterministic encoding with seed
let seed = [42u8; 32]; // Non-zero seed required
let mut encoder = RlnEncoder::<GF256, 1024>::with_seed(seed);
encoder.configure(4).unwrap();
encoder.set_data(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();

// Same seed produces identical results
let mut encoder2 = RlnEncoder::<GF256, 1024>::with_seed(seed);
encoder2.configure(4).unwrap();
encoder2.set_data(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();

let (coeffs1, symbol1) = encoder.encode_packet().unwrap();
let (coeffs2, symbol2) = encoder2.encode_packet().unwrap();
assert_eq!(coeffs1, coeffs2); // Identical coefficients
assert_eq!(symbol1, symbol2); // Identical symbols
```

### Reed-Solomon (Testing Only)
⚠️ **Note**: The Reed-Solomon implementation is for testing purposes only and has not been optimized for performance.

```rust
use relancers::coding::{RsEncoder, RsDecoder, Encoder, Decoder};
use binius_field::AESTowerField8b as GF256;

let mut encoder = RsEncoder::<GF256, 1024>::new();
encoder.configure(16).unwrap();
encoder.set_data(&data).unwrap();
let (coeffs, symbol) = encoder.encode_packet().unwrap();
```

## Performance

Requires nightly Rust for SIMD optimizations:

```bash
cargo +nightly bench --features "nightly_features"
```

## License

Apache-2.0
