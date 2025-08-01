# Relancers

Random Linear Network Coding for Rust. Powered by Binius for AVX-512 acceleration.

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

- **RLNC** with dense/sparse coefficients
- **Reed-Solomon** systematic codes  
- **Seed-based encoding** for deterministic generation
- **Streaming decoder** with partial recovery
- **GF(256)** via Binius field arithmetic

## Binius Integration

Powered by Binius's high-performance finite field library:
- AVX-512 vectorization
- Optimized polynomial arithmetic
- Memory-efficient symbol storage
- Nightly features required for SIMD

## Examples

### Basic RLNC
```rust
let mut encoder = RlnEncoder::<GF256>::new();
encoder.configure(16, 1024)?;
encoder.set_data(&data)?;
let packet = encoder.encode_packet()?;
```

### Sparse RLNC (3× faster)
```rust
use relancers::coding::{SparseRlnEncoder, SparseConfig};

let encoder = SparseRlnEncoder::with_sparse_config(SparseConfig::new(0.3));
```

### Reed-Solomon
```rust
use relancers::coding::{RsEncoder, RsDecoder};

let mut encoder = RsEncoder::<GF256>::new();
encoder.configure(16, 1024)?;
encoder.set_data(&data)?;
```

## License

MIT OR Apache-2.0

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).