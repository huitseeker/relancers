use binius_field::{underlier::WithUnderlier, AESTowerField8b as GF256};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Benchmark to compare conversion performance between From/Into vs WithUnderlier

fn bench_from_into_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Conversion Comparison");

    // Test data
    let test_bytes: Vec<u8> = (0..255).cycle().take(1024).collect();
    let test_fields: Vec<GF256> = test_bytes.iter().map(|b| GF256::from(*b)).collect();

    // Benchmark From/Into style (old approach)
    group.bench_function("from_u8", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(test_bytes.len());
            for byte in test_bytes.iter() {
                result.push(GF256::from(*byte));
            }
            black_box(result)
        })
    });

    // Benchmark WithUnderlier style (new approach)
    group.bench_function("from_underlier", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(test_bytes.len());
            for byte in test_bytes.iter() {
                result.push(GF256::from_underlier(*byte));
            }
            black_box(result)
        })
    });

    // Benchmark Into<u8> style (old approach)
    group.bench_function("into_u8", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(test_fields.len());
            for field in test_fields.iter() {
                result.push(<GF256 as Into<u8>>::into(*field));
            }
            black_box(result)
        })
    });

    // Benchmark to_underlier style (new approach)
    group.bench_function("to_underlier", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(test_fields.len());
            for field in test_fields.iter() {
                result.push(field.to_underlier());
            }
            black_box(result)
        })
    });

    // Benchmark combined conversion (typical usage pattern)
    group.bench_function("combined_from_into", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(test_bytes.len());
            for byte in test_bytes.iter() {
                let field = GF256::from(*byte);
                let back: u8 = field.into();
                result.push(back);
            }
            black_box(result)
        })
    });

    group.bench_function("combined_underlier", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(test_bytes.len());
            for byte in test_bytes.iter() {
                let field = GF256::from_underlier(*byte);
                let back = field.to_underlier();
                result.push(back);
            }
            black_box(result)
        })
    });

    group.finish();
}

// Test actual field operations that use these conversions
fn bench_field_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Field Operations");

    let test_bytes: Vec<u8> = (0..=255).collect();
    let test_coeffs: Vec<GF256> = (0..32).map(|i| GF256::from(i as u8)).collect();

    // Simulate symbol scaling (like in Symbol::scale)
    group.bench_function("symbol_scaling_old", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(test_bytes.len());
            for byte in test_bytes.iter() {
                let field_byte = GF256::from(*byte);
                let scaled = field_byte * test_coeffs[0];
                result.push(<GF256 as Into<u8>>::into(scaled));
            }
            black_box(result)
        })
    });

    group.bench_function("symbol_scaling_new", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(test_bytes.len());
            for byte in test_bytes.iter() {
                let field_byte = GF256::from_underlier(*byte);
                let scaled = field_byte * test_coeffs[0];
                result.push(scaled.to_underlier());
            }
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_from_into_conversion, bench_field_operations);
criterion_main!(benches);
