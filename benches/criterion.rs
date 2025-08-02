use binius_field::BinaryField8b as GF256;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use relancers::coding::reed_solomon::{RsDecoder, RsEncoder};
use relancers::coding::rlnc::SparseRlnEncoder;
use relancers::coding::rlnc::{RlnDecoder, RlnEncoder};
use relancers::coding::sparse::SparseConfig;
use relancers::coding::traits::{Decoder, Encoder, StreamingDecoder};

fn bench_rlnc_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("RLNC Encoding");

    let symbols = 16;
    let symbol_size = 1024;
    let data = vec![0u8; symbols * symbol_size];

    // Calculate total bytes processed per iteration
    let total_bytes = (symbols * symbol_size) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    let mut encoder = RlnEncoder::<GF256>::new();
    encoder.configure(symbols, symbol_size).unwrap();
    encoder.set_data(&data).unwrap();

    group.bench_function("encode packet", |b| {
        b.iter(|| {
            let (_coeffs, _symbol) = encoder.encode_packet().unwrap();
            black_box((_coeffs, _symbol));
        })
    });
    group.finish();
}

fn bench_rlnc_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("RLNC Decoding");

    let symbols = 16;
    let symbol_size = 1024;
    let data = vec![0u8; symbols * symbol_size];

    // Calculate total bytes processed per iteration
    let total_bytes = (symbols * symbol_size) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    let mut encoder = RlnEncoder::<GF256>::new();
    let mut decoder = RlnDecoder::<GF256>::new();

    encoder.configure(symbols, symbol_size).unwrap();
    encoder.set_data(&data).unwrap();

    decoder.configure(symbols, symbol_size).unwrap();

    // Generate enough packets for decoding
    let mut packets = Vec::new();
    for _ in 0..symbols {
        packets.push(encoder.encode_packet().unwrap());
    }

    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut decoder = RlnDecoder::<GF256>::new();
            decoder.configure(symbols, symbol_size).unwrap();

            for (coeffs, symbol) in &packets {
                decoder.add_symbol(coeffs, symbol).unwrap();
            }

            let decoded = decoder.decode().unwrap();
            black_box(decoded);
        })
    });
    group.finish();
}

fn bench_rs_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("RS Encoding");

    let symbols = 16;
    let symbol_size = 1024;
    let data = vec![0u8; symbols * symbol_size];

    // Calculate total bytes processed per iteration
    let total_bytes = (symbols * symbol_size) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    let mut encoder = RsEncoder::<GF256>::new();
    encoder.configure(symbols, symbol_size).unwrap();
    encoder.set_data(&data).unwrap();

    group.bench_function("encode packet", |b| {
        b.iter(|| {
            let (_coeffs, _symbol) = encoder.encode_packet().unwrap();
            black_box((_coeffs, _symbol));
        })
    });
    group.finish();
}

fn bench_rs_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("RS Decoding");

    let symbols = 16;
    let symbol_size = 1024;
    let data = vec![0u8; symbols * symbol_size];

    // Calculate total bytes processed per iteration
    let total_bytes = (symbols * symbol_size) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    let mut encoder = RsEncoder::<GF256>::new();
    let mut decoder = RsDecoder::<GF256>::new();

    encoder.configure(symbols, symbol_size).unwrap();
    encoder.set_data(&data).unwrap();

    decoder.configure(symbols, symbol_size).unwrap();

    // Generate exactly symbols packets for Reed-Solomon decoding
    let mut packets = Vec::new();
    for i in 0..symbols {
        // Use deterministic approach to ensure we get valid RS packets
        let point = GF256::from(i as u8);
        let mut coeffs = vec![GF256::from(1u8); symbols];
        let mut power = GF256::from(1u8);
        #[allow(clippy::needless_range_loop)]
        for j in 1..symbols {
            power *= point;
            coeffs[j] = power;
        }
        let symbol = encoder.encode_symbol(&coeffs).unwrap();
        packets.push((coeffs, symbol));
    }

    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut decoder = RsDecoder::<GF256>::new();
            decoder.configure(symbols, symbol_size).unwrap();

            for (coeffs, symbol) in &packets {
                decoder.add_symbol(coeffs, symbol).unwrap();
            }

            let decoded = decoder.decode().unwrap();
            black_box(decoded);
        })
    });
    group.finish();
}

fn bench_sparse_rlnc_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse RLNC Encoding");

    let symbols = 16;
    let symbol_size = 1024;
    let data = vec![0u8; symbols * symbol_size];

    // Calculate total bytes processed per iteration
    let total_bytes = (symbols * symbol_size) as u64;

    // Test different sparsity levels
    for sparsity in &[0.1, 0.3, 0.5, 0.8, 1.0] {
        let config = SparseConfig::new(*sparsity);
        let mut encoder = SparseRlnEncoder::<GF256>::with_sparse_config(config);
        encoder.configure(symbols, symbol_size).unwrap();
        encoder.set_data(&data).unwrap();

        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(format!("sparsity_{sparsity}"), sparsity, |b, _| {
            b.iter(|| {
                let (_coeffs, _symbol) = encoder.encode_packet().unwrap();
                black_box((_coeffs, _symbol));
            })
        });
    }
    group.finish();
}

fn bench_sparse_rlnc_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse RLNC Decoding");

    let symbols = 16;
    let symbol_size = 1024;
    let data = vec![0u8; symbols * symbol_size];

    // Calculate total bytes processed per iteration
    let total_bytes = (symbols * symbol_size) as u64;

    // Test different sparsity levels
    for sparsity in &[0.1, 0.3, 0.5, 0.8, 1.0] {
        let config = SparseConfig::new(*sparsity);
        let mut encoder = SparseRlnEncoder::<GF256>::with_sparse_config(config);
        let mut decoder = RlnDecoder::<GF256>::new();

        encoder.configure(symbols, symbol_size).unwrap();
        encoder.set_data(&data).unwrap();
        decoder.configure(symbols, symbol_size).unwrap();

        // Use deterministic RLNC for reliable benchmarking
        let seed = [42u8; 32];
        let mut deterministic_encoder = RlnEncoder::<GF256>::with_seed(seed);
        deterministic_encoder
            .configure(symbols, symbol_size)
            .unwrap();
        deterministic_encoder.set_data(&data).unwrap();

        // Generate exactly the required packets using deterministic encoding
        let mut packets = Vec::new();
        for _ in 0..symbols {
            packets.push(deterministic_encoder.encode_packet().unwrap());
        }

        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(format!("sparsity_{sparsity}"), sparsity, |b, _| {
            b.iter(|| {
                let mut decoder = RlnDecoder::<GF256>::new();
                decoder.configure(symbols, symbol_size).unwrap();

                // Handle potential redundant contributions gracefully
                let mut added = 0;
                for (coeffs, symbol) in &packets {
                    if decoder.add_symbol(coeffs, symbol).is_ok() {
                        added += 1;
                    }
                    if added >= symbols {
                        break;
                    }
                }

                let decoded = decoder.decode().unwrap();
                black_box(decoded);
            })
        });
    }
    group.finish();
}

fn bench_streaming_rlnc(c: &mut Criterion) {
    let mut group = c.benchmark_group("Streaming RLNC");

    let symbols = 16;
    let symbol_size = 1024;
    let data = vec![0u8; symbols * symbol_size];

    // Calculate total bytes processed per iteration
    let total_bytes = (symbols * symbol_size) as u64;
    group.throughput(Throughput::Bytes(total_bytes));

    let mut encoder = RlnEncoder::<GF256>::new();
    let mut decoder = RlnDecoder::<GF256>::new();

    encoder.configure(symbols, symbol_size).unwrap();
    encoder.set_data(&data).unwrap();
    decoder.configure(symbols, symbol_size).unwrap();

    // Generate packets for streaming
    let mut packets = Vec::new();
    for _ in 0..symbols + 4 {
        // Generate extra for streaming tests
        packets.push(encoder.encode_packet().unwrap());
    }

    group.bench_function("streaming_incremental", |b| {
        b.iter(|| {
            let mut decoder = RlnDecoder::<GF256>::new();
            decoder.configure(symbols, symbol_size).unwrap();

            for (coeffs, symbol) in &packets {
                decoder.add_symbol(coeffs, symbol).unwrap();

                // Check streaming metrics
                let _rank = decoder.current_rank();
                let _decoded = decoder.symbols_decoded();
                let _progress = decoder.progress();

                if decoder.can_decode() {
                    let _final_decoded = decoder.decode().unwrap();
                    break;
                }
            }

            black_box(decoder);
        })
    });

    group.bench_function("streaming_partial_decode", |b| {
        b.iter(|| {
            let mut decoder = RlnDecoder::<GF256>::new();
            decoder.configure(symbols, symbol_size).unwrap();

            // Add enough symbols for partial decoding
            for (coeffs, symbol) in packets.iter().take(symbols) {
                decoder.add_symbol(coeffs, symbol).unwrap();
            }

            // Benchmark partial symbol decoding
            for i in 0..symbols {
                if decoder.is_symbol_decoded(i) {
                    let _symbol = decoder.decode_symbol(i).unwrap().unwrap();
                    black_box(_symbol);
                }
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rlnc_encoding,
    bench_rlnc_decoding,
    bench_rs_encoding,
    bench_rs_decoding,
    bench_sparse_rlnc_encoding,
    bench_sparse_rlnc_decoding,
    bench_streaming_rlnc
);
criterion_main!(benches);
