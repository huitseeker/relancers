#!/bin/bash
set -euo pipefail

# Build release benchmarks
cargo build --bench rlnc_compare_encode --bench rlnc_compare_decode --bench rlnc_compare_recode --release >/dev/null 2>&1

ENCODE_BIN=$(ls target/release/deps/rlnc_compare_encode-* | grep -v '\.d$' | head -1)
DECODE_BIN=$(ls target/release/deps/rlnc_compare_decode-* | grep -v '\.d$' | head -1)
RECODE_BIN=$(ls target/release/deps/rlnc_compare_recode-* | grep -v '\.d$' | head -1)

# Run encode benchmark — 1.0MB/32-pieces
ENCODE_OUT=$("$ENCODE_BIN" --bench "1.0MB/32-pieces" 2>&1 | grep -E "time:\s+\[" | head -1 || true)
ENCODE_MS=$(echo "$ENCODE_OUT" | sed -n 's/.*time:.*\[\([0-9.]*\)\s*ms.*/\1/p')
if [ -z "$ENCODE_MS" ]; then
    ENCODE_MS=$(echo "$ENCODE_OUT" | sed -n 's/.*time:.*\[\([0-9.]*\)\s*µs.*/\1/p')
    if [ -n "$ENCODE_MS" ]; then
        ENCODE_MS=$(awk "BEGIN {printf \"%.4f\", $ENCODE_MS/1000}")
    fi
fi

# Run decode benchmark — 512.0KB/16-pieces
DECODE_OUT=$("$DECODE_BIN" --bench "512.0KB/16-pieces" 2>&1 | grep -E "time:\s+\[" | head -1 || true)
DECODE_MS=$(echo "$DECODE_OUT" | sed -n 's/.*time:.*\[\([0-9.]*\)\s*ms.*/\1/p')
if [ -z "$DECODE_MS" ]; then
    DECODE_MS=$(echo "$DECODE_OUT" | sed -n 's/.*time:.*\[\([0-9.]*\)\s*µs.*/\1/p')
    if [ -n "$DECODE_MS" ]; then
        DECODE_MS=$(awk "BEGIN {printf \"%.4f\", $DECODE_MS/1000}")
    fi
fi

# Run recode benchmark — 1.0MB/32-pieces/16-pieces
RECODE_OUT=$("$RECODE_BIN" --bench "1.0MB/32-pieces/16-pieces" 2>&1 | grep -E "time:\s+\[" | head -1 || true)
RECODE_MS=$(echo "$RECODE_OUT" | sed -n 's/.*time:.*\[\([0-9.]*\)\s*ms.*/\1/p')
if [ -z "$RECODE_MS" ]; then
    RECODE_MS=$(echo "$RECODE_OUT" | sed -n 's/.*time:.*\[\([0-9.]*\)\s*µs.*/\1/p')
    if [ -n "$RECODE_MS" ]; then
        RECODE_MS=$(awk "BEGIN {printf \"%.4f\", $RECODE_MS/1000}")
    fi
fi

ENCODE_MS=${ENCODE_MS:-0}
DECODE_MS=${DECODE_MS:-0}
RECODE_MS=${RECODE_MS:-0}

echo "METRIC encode_time_ms=$ENCODE_MS"
echo "METRIC decode_time_ms=$DECODE_MS"
echo "METRIC recode_time_ms=$RECODE_MS"
