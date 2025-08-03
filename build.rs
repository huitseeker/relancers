use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Detect target architecture
    let target = env::var("TARGET").unwrap();
    println!("cargo:warning=Building for target: {}", target);

    // Check if we're on x86_64
    if target.contains("x86_64") {
        println!(
            "cargo:warning=x86_64 detected - AVX-512 optimizations available with nightly features"
        );

        // Check for nightly toolchain
        let version = env::var("CARGO_PKG_VERSION").unwrap_or_default();
        println!("cargo:warning=Building with Rust version info: {}", version);

        // Set feature flags for optimal performance
        if let Ok(rustflags) = env::var("RUSTFLAGS") {
            println!("cargo:warning=Current RUSTFLAGS: {}", rustflags);
        } else {
            println!("cargo:warning=Consider setting RUSTFLAGS='-C target-cpu=native -C target-feature=+avx512f,+gfni,+pclmulqdq'");
        }
    } else {
        println!("cargo:warning=Non-x86_64 architecture detected - using portable implementations");
    }

    // Check for nightly features
    let features = env::var("CARGO_FEATURE_NIGHTLY_FEATURES").unwrap_or_default();
    if !features.is_empty() {
        println!("cargo:warning=Nightly features enabled - optimal performance paths available");
    } else {
        println!("cargo:warning=Nightly features disabled - using portable implementations");
    }
}
