use relancers::storage::Symbol;

fn main() {
    let a = Symbol::<32768>::from_data([1u8; 32768]);
    let b = Symbol::<32768>::from_data([2u8; 32768]);
    let mut c = Symbol::<32768>::zero();

    // Warmup
    for _ in 0..1000 {
        c = a;
        c.scale_add_assign_aes(&b, binius_field::AESTowerField8b::from(0x55));
    }

    let start = std::time::Instant::now();
    for _ in 0..100000 {
        c = a;
        c.scale_add_assign_aes(&b, binius_field::AESTowerField8b::from(0x55));
    }
    let elapsed = start.elapsed();
    println!("scale_add_assign avg: {:?}", elapsed / 100000);

    // Test with different scalars
    for scalar in [0x01u8, 0x02, 0x55, 0xFF] {
        let start = std::time::Instant::now();
        for _ in 0..100000 {
            c = a;
            c.scale_add_assign_aes(&b, binius_field::AESTowerField8b::from(scalar));
        }
        let elapsed = start.elapsed();
        println!(
            "scale_add_assign scalar={:02x} avg: {:?}",
            scalar,
            elapsed / 100000
        );
    }

    // Prevent `c` from being optimized away
    std::hint::black_box(c);
}
