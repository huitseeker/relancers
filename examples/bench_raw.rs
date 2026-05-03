fn main() {
    let start = std::time::Instant::now();
    let mut sum = 0u64;
    for _ in 0..1_000_000 {
        if std::is_x86_feature_detected!("avx512vbmi") {
            sum += 1;
        } else if std::is_x86_feature_detected!("avx2") {
            sum += 2;
        } else {
            sum += 3;
        }
    }
    let elapsed = start.elapsed();
    println!("feature detect avg: {:?}", elapsed / 1_000_000);
    println!("sum = {}", sum);
}
