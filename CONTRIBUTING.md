# Contributing to Relancers

Thank you for your interest in contributing to Relancers\! This document provides guidelines for contributors.

## Getting Started

### Prerequisites

- **Rust nightly**: Required for Binius AVX-512 acceleration
- **Git**: For version control
- **Cargo**: Rust's package manager

### Setup

```bash
git clone https://github.com/relancers/relancers.git
cd relancers
rustup override set nightly
cargo test --features "nightly_features"
```

## Development Workflow

### Testing

```bash
# Run all tests
cargo test --features "nightly_features"

# Run specific test
cargo test test_name --features "nightly_features"

# Run benchmarks (requires nightly)
cargo +nightly bench --features "nightly_features"
```

### Code Quality

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --all-targets --all-features

# Build documentation
cargo doc --all-features
```

### Making Changes

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Make** your changes
4. **Test** your changes: `cargo test --features "nightly_features"`
5. **Commit** your changes: `git commit -m "feat: add your feature"`
6. **Push** to your branch: `git push origin feature/your-feature`
7. **Create** a Pull Request

## Pull Request Guidelines

### Before Submitting

- [ ] Code compiles with warnings
- [ ] All tests pass: `cargo test --features "nightly_features"`
- [ ] Code is formatted: `cargo fmt`
- [ ] Clippy passes: `cargo clippy --all-targets --all-features`
- [ ] Documentation is updated if needed
- [ ] Examples work correctly

### PR Description
Include:
- **What** the change does
- **Why** it's needed
- **How** it was tested
- **Performance** impact (if applicable)

### Commit Messages
Follow conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` adding or modifying tests
- `refactor:` code refactoring
- `perf:` performance improvements

## Code Guidelines

### Style
- Follow Rust standard formatting (rustfmt)
- Use meaningful variable names
- Add documentation for public APIs
- Include examples in doc comments

### Performance
- **Benchmark** significant changes: `cargo +nightly bench`
- **Profile** before optimizing
- **Document** performance characteristics
- **Test** under realistic workloads

### Safety
- Prefer safe code when possible
- Document any unsafe blocks
- Ensure memory safety in all cases
- Validate inputs thoroughly

## Reporting Issues

### Bug Reports
Include:
- **Minimal reproduction** example
- **Expected** vs **actual** behavior
- **Environment** details (OS, Rust version)
- **Error messages** or stack traces

### Feature Requests
Include:
- **Use case** description
- **Proposed** API
- **Performance** considerations
- **Backward compatibility** implications

## Questions?
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and design discussions
- **Documentation**: Check docs.rs for API reference

## License
By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
