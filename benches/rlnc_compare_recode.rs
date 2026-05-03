use binius_field::AESTowerField8b as GF256;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::Rng;
use relancers::coding::rlnc::{RlnDecoder, RlnEncoder};
use relancers::coding::traits::{Decoder, Encoder, RecodingDecoder};
use std::{fmt::Debug, time::Duration};

#[derive(Clone, Copy)]
struct RecodeConfig {
    data_byte_len: usize,
    piece_count: usize,
    recoding_with_piece_count: usize,
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

impl Debug for RecodeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!(
            "{}/{}-pieces/{}-pieces",
            &bytes_to_human_readable(self.data_byte_len),
            self.piece_count,
            self.recoding_with_piece_count
        ))
    }
}

macro_rules! bench_recode_one {
    ($group:ident, $data_len:literal, $pieces:literal, $recoding:literal, $ssize:literal) => {{
        let config = RecodeConfig {
            data_byte_len: $data_len,
            piece_count: $pieces,
            recoding_with_piece_count: $recoding,
        };
        let mut rng = rand::rng();
        let data: Vec<u8> = (0..$data_len).map(|_| rng.random()).collect();
        let mut encoder = RlnEncoder::<GF256, $ssize>::new();
        encoder.configure($pieces).unwrap();
        encoder.set_data(&data).unwrap();
        let packets: Vec<(Vec<GF256>, _)> = (0..$recoding)
            .map(|_| encoder.encode_packet().unwrap())
            .collect();
        let mut decoder = RlnDecoder::<GF256, $ssize>::new();
        decoder.configure($pieces).unwrap();
        for (coeffs, symbol) in &packets {
            decoder.add_symbol(coeffs, symbol).unwrap();
        }
        $group.measurement_time(Duration::from_secs(20));
        $group.sample_size(100);
        $group.throughput(Throughput::Bytes(
            (($pieces as u64) + ($ssize as u64)) * (($recoding as u64) + 1),
        ));
        $group.bench_function(format!("{:?}", config), |b| {
            b.iter(|| {
                let recode_coeffs: Vec<GF256> = (0..decoder.symbols_received())
                    .map(|_| GF256::from(rng.random::<u8>()))
                    .collect();
                let recoded = black_box(&mut decoder).recode(&recode_coeffs).unwrap();
                black_box(recoded);
            });
        });
    }};
}

fn recode(c: &mut Criterion) {
    let mut group = c.benchmark_group("recode");
    bench_recode_one!(group, 524_288, 16, 8, 32_768);
    bench_recode_one!(group, 1_048_576, 32, 16, 32_768);
    bench_recode_one!(group, 2_097_152, 64, 32, 32_768);
    bench_recode_one!(group, 4_194_304, 128, 64, 32_768);
    bench_recode_one!(group, 8_388_608, 256, 128, 32_768);
    bench_recode_one!(group, 16_777_216, 512, 256, 32_768);
    bench_recode_one!(group, 33_554_432, 1024, 512, 32_768);
    group.finish();
}

criterion_group!(rlnc_compare_recoder, recode);
criterion_main!(rlnc_compare_recoder);
