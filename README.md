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
use binius_field::BinaryField8b as GF256;

// 3× faster than dense RLNC with sparse coefficients
let mut encoder = RlnEncoder::<GF256>::new();
encoder.configure(16, 1024).unwrap();
encoder.set_data(&data).unwrap();
let (coeffs, symbol) = encoder.encode_packet().unwrap();
```

## Features

- **RLNC** (Random Linear Network Coding) - dense and sparse coefficients
- **Reed-Solomon** systematic codes *(testing only, not optimized)*
- **Streaming decoder** with partial recovery
- **AVX-512 acceleration** via Binius
- **GF(256)** finite field arithmetic

## Usage

### Basic RLNC
```rust
let mut encoder = RlnEncoder::<GF256>::new();
encoder.configure(16, 1024)?;
encoder.set_data(&data)?;
let packet = encoder.encode_packet()?;
```

### Sparse RLNC
```rust
use relancers::coding::{SparseRlnEncoder, SparseConfig};
let encoder = SparseRlnEncoder::with_sparse_config(SparseConfig::new(0.3));
```

### Reed-Solomon (Testing Only)
⚠️ **Note**: The Reed-Solomon implementation is for testing purposes only and has not been optimized for performance.

```rust
use relancers::coding::{RsEncoder, RsDecoder};
let mut encoder = RsEncoder::<GF256>::new();
encoder.configure(16, 1024)?;
encoder.set_data(&data)?;
```

## Performance

Requires nightly Rust for SIMD optimizations:

```bash
cargo +nightly bench --features "nightly_features"
```

## License

Apache-2.0
