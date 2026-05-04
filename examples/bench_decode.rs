use binius_field::AESTowerField8b as GF256;
use relancers::coding::rlnc::{RlnDecoder, RlnEncoder};
use relancers::coding::traits::{Decoder, Encoder};

fn main() {
    let mut encoder = RlnEncoder::<GF256, 32768>::with_seed([42; 32]);
    encoder.configure(16).unwrap();
    let data = vec![0u8; 16 * 32768];
    encoder.set_data(&data).unwrap();

    let packets: Vec<_> = (0..16).map(|_| encoder.encode_packet().unwrap()).collect();

    let mut decoder = RlnDecoder::<GF256, 32768>::new();
    decoder.configure(16).unwrap();
    for (coeffs, symbol) in &packets {
        decoder.add_symbol(coeffs, symbol).unwrap();
    }

    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let mut d = RlnDecoder::<GF256, 32768>::new();
        d.configure(16).unwrap();
        for (coeffs, symbol) in &packets {
            d.add_symbol(coeffs, symbol).unwrap();
        }
        let _ = d.decode().unwrap();
    }
    let elapsed = start.elapsed();
    println!("full pipeline avg: {:?}", elapsed / 10000);

    // Now measure with bigger symbol size
    let mut encoder2 = RlnEncoder::<GF256, 65536>::with_seed([42; 32]);
    encoder2.configure(16).unwrap();
    let data2 = vec![0u8; 16 * 65536];
    encoder2.set_data(&data2).unwrap();
    let packets2: Vec<_> = (0..16).map(|_| encoder2.encode_packet().unwrap()).collect();

    let mut decoder2 = RlnDecoder::<GF256, 65536>::new();
    decoder2.configure(16).unwrap();
    for (coeffs, symbol) in &packets2 {
        decoder2.add_symbol(coeffs, symbol).unwrap();
    }

    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let mut d = RlnDecoder::<GF256, 65536>::new();
        d.configure(16).unwrap();
        for (coeffs, symbol) in &packets2 {
            d.add_symbol(coeffs, symbol).unwrap();
        }
        let _ = d.decode().unwrap();
    }
    let elapsed = start.elapsed();
    println!("full pipeline 64KB avg: {:?}", elapsed / 10000);
}
