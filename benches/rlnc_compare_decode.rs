use binius_field::AESTowerField8b as GF256;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rand::Rng;
use relancers::coding::rlnc::{RlnDecoder, RlnEncoder};
use relancers::coding::traits::{Decoder, Encoder};
use std::{fmt::Debug, time::Duration};

#[derive(Clone, Copy)]
struct RLNCConfig {
    data_byte_len: usize,
    piece_count: usize,
}

fn bytes_to_human_readable(bytes: usize) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut bytes = bytes as f64;
    let mut unit_index = 0;
    while bytes >= 1024.0 && unit_index < units.len() - 1 {
        bytes /= 1024.0;
        unit_index += 1;
    }
    format!("{:.1}{}", bytes, units[unit_index])
}

impl Debug for RLNCConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!(
            "{}/{}-pieces",
            &bytes_to_human_readable(self.data_byte_len),
            self.piece_count
        ))
    }
}

macro_rules! bench_decode_one {
    ($group:ident, $data_len:literal, $pieces:literal, $ssize:literal) => {{
        let config = RLNCConfig { data_byte_len: $data_len, piece_count: $pieces };
        let mut rng = rand::rng();
        let data: Vec<u8> = (0..$data_len).map(|_| rng.random()).collect();
        let mut encoder = RlnEncoder::<GF256, $ssize>::new();
        encoder.configure($pieces).unwrap();
        encoder.set_data(&data).unwrap();
        let num_pieces_to_produce = $pieces * 2;
        let packets: Vec<(Vec<GF256>, _)> = (0..num_pieces_to_produce)
            .map(|_| encoder.encode_packet().unwrap())
            .collect();
        $group.measurement_time(Duration::from_secs(20));
        $group.sample_size(100);
        $group.throughput(Throughput::Bytes(
            (($pieces as u64) + ($ssize as u64)) * ($pieces as u64),
        ));
        $group.bench_function(format!("{:?}", config), |b| {
            b.iter_batched(
                || RlnDecoder::<GF256, $ssize>::new(),
                |mut decoder| {
                    decoder.configure($pieces).unwrap();
                    let mut packet_iter = packets.iter();
                    while !black_box(&decoder).can_decode() {
                        let (coeffs, symbol) = packet_iter.next().unwrap();
                        let _ = black_box(&mut decoder).add_symbol(coeffs, symbol);
                    }
                    let decoded = decoder.decode().unwrap();
                    black_box(decoded);
                },
                BatchSize::LargeInput,
            );
        });
    }};
}

fn decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    bench_decode_one!(group, 524_288, 16, 32_768);
    bench_decode_one!(group, 1_048_576, 32, 32_768);
    bench_decode_one!(group, 2_097_152, 64, 32_768);
    bench_decode_one!(group, 4_194_304, 128, 32_768);
    bench_decode_one!(group, 8_388_608, 256, 32_768);
    bench_decode_one!(group, 16_777_216, 512, 32_768);
    bench_decode_one!(group, 33_554_432, 1024, 32_768);
    group.finish();
}

criterion_group!(rlnc_compare_decoder, decode);
criterion_main!(rlnc_compare_decoder);
