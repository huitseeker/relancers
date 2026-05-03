#!/bin/bash
set -euo pipefail
cargo test --lib 2>&1 | tail -20
