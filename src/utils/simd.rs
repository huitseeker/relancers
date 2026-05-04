//! SIMD-accelerated GF(2^8) operations for AESTowerField8b.
//! Uses the split-table technique (low/high nibble) with SSSE3 `_mm_shuffle_epi8`.
//! Uses GFNI (`_mm_gf2p8mul_epi8`) when available for 1-instruction multiplication.
//! Adapted from the gf-complete / rlnc approach.

use std::sync::OnceLock;

/// Precomputed split multiplication tables for AESTowerField8b.
/// `LOW[s][i]  = s * i`       (i in 0..16, treated as low-nibble element)
/// `HIGH[s][i] = s * (i << 4)` (i in 0..16, treated as high-nibble element)
struct SimdMulTables {
    low: [[u8; 16]; 256],
    high: [[u8; 16]; 256],
}

fn get_tables() -> &'static SimdMulTables {
    static TABLES: OnceLock<SimdMulTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut low = [[0u8; 16]; 256];
        let mut high = [[0u8; 16]; 256];
        let full = &super::mul_table::MUL_TABLE;
        for s in 0..256 {
            for n in 0..16 {
                low[s][n] = full[s][n];
                high[s][n] = full[s][n << 4];
            }
        }
        SimdMulTables { low, high }
    })
}

/// `dst ^= src` using 128-bit XOR (SSSE3).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn add_assign_ssse3(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let mut i = 0;
    while i + 64 <= n {
        let d0 = _mm_loadu_si128(dst.as_ptr().add(i).cast());
        let d1 = _mm_loadu_si128(dst.as_ptr().add(i + 16).cast());
        let d2 = _mm_loadu_si128(dst.as_ptr().add(i + 32).cast());
        let d3 = _mm_loadu_si128(dst.as_ptr().add(i + 48).cast());

        let s0 = _mm_loadu_si128(src.as_ptr().add(i).cast());
        let s1 = _mm_loadu_si128(src.as_ptr().add(i + 16).cast());
        let s2 = _mm_loadu_si128(src.as_ptr().add(i + 32).cast());
        let s3 = _mm_loadu_si128(src.as_ptr().add(i + 48).cast());

        _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), _mm_xor_si128(d0, s0));
        _mm_storeu_si128(dst.as_mut_ptr().add(i + 16).cast(), _mm_xor_si128(d1, s1));
        _mm_storeu_si128(dst.as_mut_ptr().add(i + 32).cast(), _mm_xor_si128(d2, s2));
        _mm_storeu_si128(dst.as_mut_ptr().add(i + 48).cast(), _mm_xor_si128(d3, s3));

        i += 64;
    }
    while i + 16 <= n {
        let d = _mm_loadu_si128(dst.as_ptr().add(i).cast());
        let s = _mm_loadu_si128(src.as_ptr().add(i).cast());
        _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), _mm_xor_si128(d, s));
        i += 16;
    }
    for j in i..n {
        dst[j] ^= src[j];
    }
}

/// `dst += src * scalar` using split-table SSSE3.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn scale_add_assign_ssse3(dst: &mut [u8], src: &[u8], scalar: u8, tables: &SimdMulTables) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let l_tbl = _mm_loadu_si128(tables.low[scalar as usize].as_ptr().cast());
    let h_tbl = _mm_loadu_si128(tables.high[scalar as usize].as_ptr().cast());
    let nibble_mask = _mm_set1_epi8(0x0f);

    let n = dst.len();
    let mut i = 0;
    while i + 64 <= n {
        // Process 4 x 16-byte chunks per iteration
        for k in 0..4 {
            let src_vec = _mm_loadu_si128(src.as_ptr().add(i + k * 16).cast());
            let src_lo = _mm_and_si128(src_vec, nibble_mask);
            let src_hi = _mm_and_si128(_mm_srli_epi64(src_vec, 4), nibble_mask);
            let prod_lo = _mm_shuffle_epi8(l_tbl, src_lo);
            let prod_hi = _mm_shuffle_epi8(h_tbl, src_hi);
            let prod = _mm_xor_si128(prod_lo, prod_hi);
            let dst_vec = _mm_loadu_si128(dst.as_ptr().add(i + k * 16).cast());
            _mm_storeu_si128(dst.as_mut_ptr().add(i + k * 16).cast(), _mm_xor_si128(dst_vec, prod));
        }
        i += 64;
    }
    while i + 16 <= n {
        let src_vec = _mm_loadu_si128(src.as_ptr().add(i).cast());
        let src_lo = _mm_and_si128(src_vec, nibble_mask);
        let src_hi = _mm_and_si128(_mm_srli_epi64(src_vec, 4), nibble_mask);
        let prod_lo = _mm_shuffle_epi8(l_tbl, src_lo);
        let prod_hi = _mm_shuffle_epi8(h_tbl, src_hi);
        let prod = _mm_xor_si128(prod_lo, prod_hi);
        let dst_vec = _mm_loadu_si128(dst.as_ptr().add(i).cast());
        _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), _mm_xor_si128(dst_vec, prod));
        i += 16;
    }
    // Scalar tail
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] ^= table[src[j] as usize];
    }
}

/// `dst *= scalar` using split-table SSSE3.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn scale_ssse3(dst: &mut [u8], scalar: u8, tables: &SimdMulTables) {
    use std::arch::x86_64::*;
    let l_tbl = _mm_loadu_si128(tables.low[scalar as usize].as_ptr().cast());
    let h_tbl = _mm_loadu_si128(tables.high[scalar as usize].as_ptr().cast());
    let nibble_mask = _mm_set1_epi8(0x0f);

    let n = dst.len();
    let mut i = 0;
    while i + 64 <= n {
        for k in 0..4 {
            let vec = _mm_loadu_si128(dst.as_ptr().add(i + k * 16).cast());
            let vec_lo = _mm_and_si128(vec, nibble_mask);
            let vec_hi = _mm_and_si128(_mm_srli_epi64(vec, 4), nibble_mask);
            let prod_lo = _mm_shuffle_epi8(l_tbl, vec_lo);
            let prod_hi = _mm_shuffle_epi8(h_tbl, vec_hi);
            _mm_storeu_si128(dst.as_mut_ptr().add(i + k * 16).cast(), _mm_xor_si128(prod_lo, prod_hi));
        }
        i += 64;
    }
    while i + 16 <= n {
        let vec = _mm_loadu_si128(dst.as_ptr().add(i).cast());
        let vec_lo = _mm_and_si128(vec, nibble_mask);
        let vec_hi = _mm_and_si128(_mm_srli_epi64(vec, 4), nibble_mask);
        let prod_lo = _mm_shuffle_epi8(l_tbl, vec_lo);
        let prod_hi = _mm_shuffle_epi8(h_tbl, vec_hi);
        _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), _mm_xor_si128(prod_lo, prod_hi));
        i += 16;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] = table[dst[j] as usize];
    }
}

// ------------------------------------------------------------------
// AVX2 (256-bit) variants
// ------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_assign_avx2(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let mut i = 0;
    while i + 128 <= n {
        for k in 0..4 {
            let d = _mm256_loadu_si256(dst.as_ptr().add(i + k * 32).cast());
            let s = _mm256_loadu_si256(src.as_ptr().add(i + k * 32).cast());
            _mm256_storeu_si256(dst.as_mut_ptr().add(i + k * 32).cast(), _mm256_xor_si256(d, s));
        }
        i += 128;
    }
    while i + 32 <= n {
        let d = _mm256_loadu_si256(dst.as_ptr().add(i).cast());
        let s = _mm256_loadu_si256(src.as_ptr().add(i).cast());
        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_xor_si256(d, s));
        i += 32;
    }
    for j in i..n {
        dst[j] ^= src[j];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn scale_add_assign_avx2(dst: &mut [u8], src: &[u8], scalar: u8, tables: &SimdMulTables) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let l_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.low[scalar as usize].as_ptr().cast()));
    let h_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.high[scalar as usize].as_ptr().cast()));
    let nibble_mask = _mm256_set1_epi8(0x0f);

    let n = dst.len();
    let mut i = 0;
    while i + 128 <= n {
        for k in 0..4 {
            let src_vec = _mm256_loadu_si256(src.as_ptr().add(i + k * 32).cast());
            let src_lo = _mm256_and_si256(src_vec, nibble_mask);
            let src_hi = _mm256_and_si256(_mm256_srli_epi64(src_vec, 4), nibble_mask);
            let prod_lo = _mm256_shuffle_epi8(l_tbl, src_lo);
            let prod_hi = _mm256_shuffle_epi8(h_tbl, src_hi);
            let prod = _mm256_xor_si256(prod_lo, prod_hi);
            let dst_vec = _mm256_loadu_si256(dst.as_ptr().add(i + k * 32).cast());
            _mm256_storeu_si256(dst.as_mut_ptr().add(i + k * 32).cast(), _mm256_xor_si256(dst_vec, prod));
        }
        i += 128;
    }
    while i + 32 <= n {
        let src_vec = _mm256_loadu_si256(src.as_ptr().add(i).cast());
        let src_lo = _mm256_and_si256(src_vec, nibble_mask);
        let src_hi = _mm256_and_si256(_mm256_srli_epi64(src_vec, 4), nibble_mask);
        let prod_lo = _mm256_shuffle_epi8(l_tbl, src_lo);
        let prod_hi = _mm256_shuffle_epi8(h_tbl, src_hi);
        let prod = _mm256_xor_si256(prod_lo, prod_hi);
        let dst_vec = _mm256_loadu_si256(dst.as_ptr().add(i).cast());
        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_xor_si256(dst_vec, prod));
        i += 32;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] ^= table[src[j] as usize];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(dst: &mut [u8], scalar: u8, tables: &SimdMulTables) {
    use std::arch::x86_64::*;
    let l_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.low[scalar as usize].as_ptr().cast()));
    let h_tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.high[scalar as usize].as_ptr().cast()));
    let nibble_mask = _mm256_set1_epi8(0x0f);

    let n = dst.len();
    let mut i = 0;
    while i + 128 <= n {
        for k in 0..4 {
            let vec = _mm256_loadu_si256(dst.as_ptr().add(i + k * 32).cast());
            let vec_lo = _mm256_and_si256(vec, nibble_mask);
            let vec_hi = _mm256_and_si256(_mm256_srli_epi64(vec, 4), nibble_mask);
            let prod_lo = _mm256_shuffle_epi8(l_tbl, vec_lo);
            let prod_hi = _mm256_shuffle_epi8(h_tbl, vec_hi);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i + k * 32).cast(), _mm256_xor_si256(prod_lo, prod_hi));
        }
        i += 128;
    }
    while i + 32 <= n {
        let vec = _mm256_loadu_si256(dst.as_ptr().add(i).cast());
        let vec_lo = _mm256_and_si256(vec, nibble_mask);
        let vec_hi = _mm256_and_si256(_mm256_srli_epi64(vec, 4), nibble_mask);
        let prod_lo = _mm256_shuffle_epi8(l_tbl, vec_lo);
        let prod_hi = _mm256_shuffle_epi8(h_tbl, vec_hi);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_xor_si256(prod_lo, prod_hi));
        i += 32;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] = table[dst[j] as usize];
    }
}

// ------------------------------------------------------------------
// AVX-512 (512-bit) variants — requires AVX512VBMI for _mm512_shuffle_epi8
// ------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn add_assign_avx512(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let mut i = 0;
    while i + 256 <= n {
        for k in 0..4 {
            let d = _mm512_loadu_si512(dst.as_ptr().add(i + k * 64).cast());
            let s = _mm512_loadu_si512(src.as_ptr().add(i + k * 64).cast());
            _mm512_storeu_si512(dst.as_mut_ptr().add(i + k * 64).cast(), _mm512_xor_si512(d, s));
        }
        i += 256;
    }
    while i + 64 <= n {
        let d = _mm512_loadu_si512(dst.as_ptr().add(i).cast());
        let s = _mm512_loadu_si512(src.as_ptr().add(i).cast());
        _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), _mm512_xor_si512(d, s));
        i += 64;
    }
    for j in i..n {
        dst[j] ^= src[j];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn scale_add_assign_avx512(dst: &mut [u8], src: &[u8], scalar: u8, tables: &SimdMulTables) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let l_tbl = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.low[scalar as usize].as_ptr().cast()));
    let h_tbl = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.high[scalar as usize].as_ptr().cast()));
    let nibble_mask = _mm512_set1_epi8(0x0f);

    let n = dst.len();
    let mut i = 0;
    while i + 256 <= n {
        for k in 0..4 {
            let src_vec = _mm512_loadu_si512(src.as_ptr().add(i + k * 64).cast());
            let src_lo = _mm512_and_si512(src_vec, nibble_mask);
            let src_hi = _mm512_and_si512(_mm512_srli_epi64(src_vec, 4), nibble_mask);
            let prod_lo = _mm512_shuffle_epi8(l_tbl, src_lo);
            let prod_hi = _mm512_shuffle_epi8(h_tbl, src_hi);
            let prod = _mm512_xor_si512(prod_lo, prod_hi);
            let dst_vec = _mm512_loadu_si512(dst.as_ptr().add(i + k * 64).cast());
            _mm512_storeu_si512(dst.as_mut_ptr().add(i + k * 64).cast(), _mm512_xor_si512(dst_vec, prod));
        }
        i += 256;
    }
    while i + 64 <= n {
        let src_vec = _mm512_loadu_si512(src.as_ptr().add(i).cast());
        let src_lo = _mm512_and_si512(src_vec, nibble_mask);
        let src_hi = _mm512_and_si512(_mm512_srli_epi64(src_vec, 4), nibble_mask);
        let prod_lo = _mm512_shuffle_epi8(l_tbl, src_lo);
        let prod_hi = _mm512_shuffle_epi8(h_tbl, src_hi);
        let prod = _mm512_xor_si512(prod_lo, prod_hi);
        let dst_vec = _mm512_loadu_si512(dst.as_ptr().add(i).cast());
        _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), _mm512_xor_si512(dst_vec, prod));
        i += 64;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] ^= table[src[j] as usize];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn scale_avx512(dst: &mut [u8], scalar: u8, tables: &SimdMulTables) {
    use std::arch::x86_64::*;
    let l_tbl = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.low[scalar as usize].as_ptr().cast()));
    let h_tbl = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.high[scalar as usize].as_ptr().cast()));
    let nibble_mask = _mm512_set1_epi8(0x0f);

    let n = dst.len();
    let mut i = 0;
    while i + 256 <= n {
        for k in 0..4 {
            let vec = _mm512_loadu_si512(dst.as_ptr().add(i + k * 64).cast());
            let vec_lo = _mm512_and_si512(vec, nibble_mask);
            let vec_hi = _mm512_and_si512(_mm512_srli_epi64(vec, 4), nibble_mask);
            let prod_lo = _mm512_shuffle_epi8(l_tbl, vec_lo);
            let prod_hi = _mm512_shuffle_epi8(h_tbl, vec_hi);
            _mm512_storeu_si512(dst.as_mut_ptr().add(i + k * 64).cast(), _mm512_xor_si512(prod_lo, prod_hi));
        }
        i += 256;
    }
    while i + 64 <= n {
        let vec = _mm512_loadu_si512(dst.as_ptr().add(i).cast());
        let vec_lo = _mm512_and_si512(vec, nibble_mask);
        let vec_hi = _mm512_and_si512(_mm512_srli_epi64(vec, 4), nibble_mask);
        let prod_lo = _mm512_shuffle_epi8(l_tbl, vec_lo);
        let prod_hi = _mm512_shuffle_epi8(h_tbl, vec_hi);
        _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), _mm512_xor_si512(prod_lo, prod_hi));
        i += 64;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] = table[dst[j] as usize];
    }
}

// ------------------------------------------------------------------
// GFNI (1-instruction GF(2^8) multiply) variants
// AESTowerField8b is natively compatible with GFNI's AES polynomial.
// ------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "gfni,avx512f,avx512bw,avx512vl")]
unsafe fn scale_add_assign_gfni512(dst: &mut [u8], src: &[u8], scalar: u8) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let scalar_vec = _mm512_set1_epi8(scalar as i8);
    const CHUNK: usize = 256;

    let mut dst_iter = dst.chunks_exact_mut(CHUNK);
    let mut src_iter = src.chunks_exact(CHUNK);
    for (dst_chunk, src_chunk) in dst_iter.by_ref().zip(src_iter.by_ref()) {
        let (s0, s1, s2, s3) = {
            let (c0, rest) = src_chunk.split_at_unchecked(64);
            let (c1, rest) = rest.split_at_unchecked(64);
            let (c2, c3) = rest.split_at_unchecked(64);
            (c0, c1, c2, c3)
        };
        let s0v = _mm512_loadu_si512(s0.as_ptr().cast());
        let s1v = _mm512_loadu_si512(s1.as_ptr().cast());
        let s2v = _mm512_loadu_si512(s2.as_ptr().cast());
        let s3v = _mm512_loadu_si512(s3.as_ptr().cast());

        let p0 = _mm512_gf2p8mul_epi8(s0v, scalar_vec);
        let p1 = _mm512_gf2p8mul_epi8(s1v, scalar_vec);
        let p2 = _mm512_gf2p8mul_epi8(s2v, scalar_vec);
        let p3 = _mm512_gf2p8mul_epi8(s3v, scalar_vec);

        let (d0, d1, d2, d3) = {
            let (c0, rest) = dst_chunk.split_at_mut_unchecked(64);
            let (c1, rest) = rest.split_at_mut_unchecked(64);
            let (c2, c3) = rest.split_at_mut_unchecked(64);
            (c0, c1, c2, c3)
        };
        let d0v = _mm512_loadu_si512(d0.as_ptr().cast());
        let d1v = _mm512_loadu_si512(d1.as_ptr().cast());
        let d2v = _mm512_loadu_si512(d2.as_ptr().cast());
        let d3v = _mm512_loadu_si512(d3.as_ptr().cast());

        _mm512_storeu_si512(d0.as_mut_ptr().cast(), _mm512_xor_si512(d0v, p0));
        _mm512_storeu_si512(d1.as_mut_ptr().cast(), _mm512_xor_si512(d1v, p1));
        _mm512_storeu_si512(d2.as_mut_ptr().cast(), _mm512_xor_si512(d2v, p2));
        _mm512_storeu_si512(d3.as_mut_ptr().cast(), _mm512_xor_si512(d3v, p3));
    }
    for (d, s) in dst_iter.into_remainder().iter_mut().zip(src_iter.remainder().iter()) {
        let table = &super::mul_table::MUL_TABLE[scalar as usize];
        *d ^= table[*s as usize];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "gfni,avx512f,avx512bw,avx512vl")]
unsafe fn scale_gfni512(dst: &mut [u8], scalar: u8) {
    use std::arch::x86_64::*;
    let scalar_vec = _mm512_set1_epi8(scalar as i8);
    const CHUNK: usize = 256;

    let mut iter = dst.chunks_exact_mut(CHUNK);
    for chunk in iter.by_ref() {
        let (c0, c1, c2, c3) = {
            let (c0, rest) = chunk.split_at_mut_unchecked(64);
            let (c1, rest) = rest.split_at_mut_unchecked(64);
            let (c2, c3) = rest.split_at_mut_unchecked(64);
            (c0, c1, c2, c3)
        };
        let v0 = _mm512_loadu_si512(c0.as_ptr().cast());
        let v1 = _mm512_loadu_si512(c1.as_ptr().cast());
        let v2 = _mm512_loadu_si512(c2.as_ptr().cast());
        let v3 = _mm512_loadu_si512(c3.as_ptr().cast());

        _mm512_storeu_si512(c0.as_mut_ptr().cast(), _mm512_gf2p8mul_epi8(v0, scalar_vec));
        _mm512_storeu_si512(c1.as_mut_ptr().cast(), _mm512_gf2p8mul_epi8(v1, scalar_vec));
        _mm512_storeu_si512(c2.as_mut_ptr().cast(), _mm512_gf2p8mul_epi8(v2, scalar_vec));
        _mm512_storeu_si512(c3.as_mut_ptr().cast(), _mm512_gf2p8mul_epi8(v3, scalar_vec));
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for d in iter.into_remainder() {
        *d = table[*d as usize];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "gfni,avx2")]
unsafe fn scale_add_assign_gfni256(dst: &mut [u8], src: &[u8], scalar: u8) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let scalar_vec = _mm256_set1_epi8(scalar as i8);
    let n = dst.len();
    let mut i = 0;
    while i + 128 <= n {
        for k in 0..4 {
            let src_vec = _mm256_loadu_si256(src.as_ptr().add(i + k * 32).cast());
            let prod = _mm256_gf2p8mul_epi8(src_vec, scalar_vec);
            let dst_vec = _mm256_loadu_si256(dst.as_ptr().add(i + k * 32).cast());
            _mm256_storeu_si256(dst.as_mut_ptr().add(i + k * 32).cast(), _mm256_xor_si256(dst_vec, prod));
        }
        i += 128;
    }
    while i + 32 <= n {
        let src_vec = _mm256_loadu_si256(src.as_ptr().add(i).cast());
        let prod = _mm256_gf2p8mul_epi8(src_vec, scalar_vec);
        let dst_vec = _mm256_loadu_si256(dst.as_ptr().add(i).cast());
        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_xor_si256(dst_vec, prod));
        i += 32;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] ^= table[src[j] as usize];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "gfni,avx2")]
unsafe fn scale_gfni256(dst: &mut [u8], scalar: u8) {
    use std::arch::x86_64::*;
    let scalar_vec = _mm256_set1_epi8(scalar as i8);
    let n = dst.len();
    let mut i = 0;
    while i + 128 <= n {
        for k in 0..4 {
            let vec = _mm256_loadu_si256(dst.as_ptr().add(i + k * 32).cast());
            let prod = _mm256_gf2p8mul_epi8(vec, scalar_vec);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i + k * 32).cast(), prod);
        }
        i += 128;
    }
    while i + 32 <= n {
        let vec = _mm256_loadu_si256(dst.as_ptr().add(i).cast());
        let prod = _mm256_gf2p8mul_epi8(vec, scalar_vec);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), prod);
        i += 32;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] = table[dst[j] as usize];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "gfni")]
unsafe fn scale_add_assign_gfni128(dst: &mut [u8], src: &[u8], scalar: u8) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let scalar_vec = _mm_set1_epi8(scalar as i8);
    let n = dst.len();
    let mut i = 0;
    while i + 64 <= n {
        for k in 0..4 {
            let src_vec = _mm_loadu_si128(src.as_ptr().add(i + k * 16).cast());
            let prod = _mm_gf2p8mul_epi8(src_vec, scalar_vec);
            let dst_vec = _mm_loadu_si128(dst.as_ptr().add(i + k * 16).cast());
            _mm_storeu_si128(dst.as_mut_ptr().add(i + k * 16).cast(), _mm_xor_si128(dst_vec, prod));
        }
        i += 64;
    }
    while i + 16 <= n {
        let src_vec = _mm_loadu_si128(src.as_ptr().add(i).cast());
        let prod = _mm_gf2p8mul_epi8(src_vec, scalar_vec);
        let dst_vec = _mm_loadu_si128(dst.as_ptr().add(i).cast());
        _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), _mm_xor_si128(dst_vec, prod));
        i += 16;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] ^= table[src[j] as usize];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "gfni")]
unsafe fn scale_gfni128(dst: &mut [u8], scalar: u8) {
    use std::arch::x86_64::*;
    let scalar_vec = _mm_set1_epi8(scalar as i8);
    let n = dst.len();
    let mut i = 0;
    while i + 64 <= n {
        for k in 0..4 {
            let vec = _mm_loadu_si128(dst.as_ptr().add(i + k * 16).cast());
            let prod = _mm_gf2p8mul_epi8(vec, scalar_vec);
            _mm_storeu_si128(dst.as_mut_ptr().add(i + k * 16).cast(), prod);
        }
        i += 64;
    }
    while i + 16 <= n {
        let vec = _mm_loadu_si128(dst.as_ptr().add(i).cast());
        let prod = _mm_gf2p8mul_epi8(vec, scalar_vec);
        _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), prod);
        i += 16;
    }
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    for j in i..n {
        dst[j] = table[dst[j] as usize];
    }
}

// ------------------------------------------------------------------
// Public wrappers with runtime feature detection
// ------------------------------------------------------------------

/// Unchecked `scale_simd`. Caller must ensure `scalar != 0` and `scalar != 1`.
#[inline]
pub fn scale_simd_unchecked(dst: &mut [u8], scalar: u8) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("gfni") {
        unsafe { scale_gfni512(dst, scalar) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("gfni") {
        unsafe { scale_gfni256(dst, scalar) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("gfni") {
        unsafe { scale_gfni128(dst, scalar) };
        return;
    }
    let tables = get_tables();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512vbmi") {
        unsafe { scale_avx512(dst, scalar, tables) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        unsafe { scale_avx2(dst, scalar, tables) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("ssse3") {
        unsafe { scale_ssse3(dst, scalar, tables) };
        return;
    }
    // Fallback
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    let mut i = 0;
    let n = dst.len();
    while i + 8 <= n {
        dst[i] = table[dst[i] as usize];
        dst[i + 1] = table[dst[i + 1] as usize];
        dst[i + 2] = table[dst[i + 2] as usize];
        dst[i + 3] = table[dst[i + 3] as usize];
        dst[i + 4] = table[dst[i + 4] as usize];
        dst[i + 5] = table[dst[i + 5] as usize];
        dst[i + 6] = table[dst[i + 6] as usize];
        dst[i + 7] = table[dst[i + 7] as usize];
        i += 8;
    }
    for j in i..n {
        dst[j] = table[dst[j] as usize];
    }
}

/// Unchecked `scale_add_assign_simd`. Caller must ensure `scalar != 0` and `scalar != 1`.
#[inline]
pub fn scale_add_assign_simd_unchecked(dst: &mut [u8], src: &[u8], scalar: u8) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("gfni") {
        unsafe { scale_add_assign_gfni512(dst, src, scalar) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("gfni") {
        unsafe { scale_add_assign_gfni256(dst, src, scalar) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("gfni") {
        unsafe { scale_add_assign_gfni128(dst, src, scalar) };
        return;
    }
    let tables = get_tables();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512vbmi") {
        unsafe { scale_add_assign_avx512(dst, src, scalar, tables) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        unsafe { scale_add_assign_avx2(dst, src, scalar, tables) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("ssse3") {
        unsafe { scale_add_assign_ssse3(dst, src, scalar, tables) };
        return;
    }
    // Fallback
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    let mut i = 0;
    let n = dst.len();
    while i + 8 <= n {
        dst[i] ^= table[src[i] as usize];
        dst[i + 1] ^= table[src[i + 1] as usize];
        dst[i + 2] ^= table[src[i + 2] as usize];
        dst[i + 3] ^= table[src[i + 3] as usize];
        dst[i + 4] ^= table[src[i + 4] as usize];
        dst[i + 5] ^= table[src[i + 5] as usize];
        dst[i + 6] ^= table[src[i + 6] as usize];
        dst[i + 7] ^= table[src[i + 7] as usize];
        i += 8;
    }
    for j in i..n {
        dst[j] ^= table[src[j] as usize];
    }
}

#[inline]
pub fn add_assign_simd(dst: &mut [u8], src: &[u8]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512vbmi") {
        unsafe { add_assign_avx512(dst, src) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        unsafe { add_assign_avx2(dst, src) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("ssse3") {
        unsafe { add_assign_ssse3(dst, src) };
        return;
    }
    // Fallback
    let mut i = 0;
    let n = dst.len();
    while i + 8 <= n {
        dst[i] ^= src[i];
        dst[i + 1] ^= src[i + 1];
        dst[i + 2] ^= src[i + 2];
        dst[i + 3] ^= src[i + 3];
        dst[i + 4] ^= src[i + 4];
        dst[i + 5] ^= src[i + 5];
        dst[i + 6] ^= src[i + 6];
        dst[i + 7] ^= src[i + 7];
        i += 8;
    }
    for j in i..n {
        dst[j] ^= src[j];
    }
}

#[inline]
pub fn scale_add_assign_simd(dst: &mut [u8], src: &[u8], scalar: u8) {
    if scalar == 0 {
        return;
    }
    if scalar == 1 {
        add_assign_simd(dst, src);
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("gfni") {
        unsafe { scale_add_assign_gfni512(dst, src, scalar) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("gfni") {
        unsafe { scale_add_assign_gfni256(dst, src, scalar) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("gfni") {
        unsafe { scale_add_assign_gfni128(dst, src, scalar) };
        return;
    }
    let tables = get_tables();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512vbmi") {
        unsafe { scale_add_assign_avx512(dst, src, scalar, tables) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        unsafe { scale_add_assign_avx2(dst, src, scalar, tables) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("ssse3") {
        unsafe { scale_add_assign_ssse3(dst, src, scalar, tables) };
        return;
    }
    // Fallback
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    let mut i = 0;
    let n = dst.len();
    while i + 8 <= n {
        dst[i] ^= table[src[i] as usize];
        dst[i + 1] ^= table[src[i + 1] as usize];
        dst[i + 2] ^= table[src[i + 2] as usize];
        dst[i + 3] ^= table[src[i + 3] as usize];
        dst[i + 4] ^= table[src[i + 4] as usize];
        dst[i + 5] ^= table[src[i + 5] as usize];
        dst[i + 6] ^= table[src[i + 6] as usize];
        dst[i + 7] ^= table[src[i + 7] as usize];
        i += 8;
    }
    for j in i..n {
        dst[j] ^= table[src[j] as usize];
    }
}

#[inline]
pub fn scale_simd(dst: &mut [u8], scalar: u8) {
    if scalar == 0 {
        dst.fill(0);
        return;
    }
    if scalar == 1 {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("gfni") {
        unsafe { scale_gfni512(dst, scalar) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("gfni") {
        unsafe { scale_gfni256(dst, scalar) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("gfni") {
        unsafe { scale_gfni128(dst, scalar) };
        return;
    }
    let tables = get_tables();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512vbmi") {
        unsafe { scale_avx512(dst, scalar, tables) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        unsafe { scale_avx2(dst, scalar, tables) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("ssse3") {
        unsafe { scale_ssse3(dst, scalar, tables) };
        return;
    }
    // Fallback
    let table = &super::mul_table::MUL_TABLE[scalar as usize];
    let mut i = 0;
    let n = dst.len();
    while i + 8 <= n {
        dst[i] = table[dst[i] as usize];
        dst[i + 1] = table[dst[i + 1] as usize];
        dst[i + 2] = table[dst[i + 2] as usize];
        dst[i + 3] = table[dst[i + 3] as usize];
        dst[i + 4] = table[dst[i + 4] as usize];
        dst[i + 5] = table[dst[i + 5] as usize];
        dst[i + 6] = table[dst[i + 6] as usize];
        dst[i + 7] = table[dst[i + 7] as usize];
        i += 8;
    }
    for j in i..n {
        dst[j] = table[dst[j] as usize];
    }
}
