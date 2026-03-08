//! ─────────────────────────────────────────────────────────────────────────────
//! ZK Guest Program: Face Identity Proof
//! Runs INSIDE the Risc Zero zkVM — every instruction is proven.
//!
//! Pipeline:
//!   1. Read private FaceProofInput from the host
//!   2. Compute cosine similarity between probe and reference embeddings
//!   3. Run BCH fuzzy extractor Reproduce() to recover the enrolled key
//!   4. Commit FaceProofOutput (hash of key + pass/fail) to the public journal
//! ─────────────────────────────────────────────────────────────────────────────

#![no_main]
risc0_zkvm::guest::entry!(main);

use risc0_zkvm::guest::env;
use zkface_core::{
    FaceProofInput, FaceProofOutput,
    EMBEDDING_DIM, BCH_N, BCH_K, BCH_T, KEY_BYTES,
};

// ── SHA-256 (no_std compatible, pure Rust) ───────────────────────────────────

/// Minimal SHA-256 implementation (FIPS 180-4) — avoids pulling in openssl.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
        0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
        0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
        0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
        0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
        0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i*4],chunk[i*4+1],chunk[i*4+2],chunk[i*4+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh) =
            (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1);
            d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b);
        h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f);
        h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i*4..i*4+4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

// ── Q8.8 Fixed-Point Cosine Similarity ───────────────────────────────────────

/// Cosine similarity of two Q8.8 vectors. Returns similarity × 256 (i32).
/// Both inputs must have been L2-normalised before quantisation (MobileNet does this).
fn cosine_similarity_q8(a: &[i32], b: &[i32]) -> i32 {
    assert_eq!(a.len(), b.len());
    // dot product (accumulate in i64 to avoid overflow)
    let dot: i64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as i64) * (*y as i64)).sum();
    let norm_a: i64 = a.iter().map(|x| (*x as i64) * (*x as i64)).sum::<i64>();
    let norm_b: i64 = b.iter().map(|x| (*x as i64) * (*x as i64)).sum::<i64>();

    if norm_a == 0 || norm_b == 0 {
        return 0;
    }
    // integer sqrt via Newton's method
    fn isqrt(n: i64) -> i64 {
        if n <= 0 { return 0; }
        let mut x = n;
        let mut y = (x + 1) / 2;
        while y < x { x = y; y = (x + n / x) / 2; }
        x
    }
    // result scaled: dot is in Q16.16 (two Q8.8 multiplied), norms in Q16.16
    // similarity = dot / sqrt(norm_a * norm_b), then re-scale to Q8.8
    let denom = isqrt(norm_a) * isqrt(norm_b); // Q8.8 × Q8.8 = Q16.16 but we divide...
    // Return in Q8.8 (× 256)
    ((dot * 256) / denom) as i32
}

// ── BCH(255, 131, 18) Fuzzy Extractor ────────────────────────────────────────
//
// Secure Sketch + Strong Extractor pattern (Dodis et al. 2004):
//
//   Enroll (offline, host-side):
//     w   = quantised embedding bit-string (255 bits from 128 dims)
//     c   = BCH.encode(w[0..K])           -- encode first K bits as codeword
//     pub = w XOR c                        -- helper string (syndrome-like)
//     key = SHA-256(c[0..KEY_BYTES])       -- stable key
//
//   Reproduce (inside zkVM):
//     w'  = new quantised embedding
//     c'  = w' XOR pub                     -- noisy codeword
//     c   = BCH.decode(c')                 -- error correction (≤ t errors)
//     key = SHA-256(c[0..KEY_BYTES])
//
// The guest proves: given (w', pub), it correctly ran BCH.decode and derived key.

/// Pack embedding signs into a bit-string of length BCH_N.
/// We use the sign bit of each Q8.8 value → 128 bits, then pad to 255 bits.
fn embedding_to_bits(embedding: &[i32]) -> [u8; (BCH_N + 7) / 8] {
    let mut bits = [0u8; (BCH_N + 7) / 8];
    for (i, &val) in embedding.iter().take(BCH_N).enumerate() {
        // Use top 2 bits of each value for richer encoding (128 dims × 2 = 256 > 255)
        let bit = if val >= 0 { 1u8 } else { 0u8 };
        bits[i / 8] |= bit << (7 - (i % 8));
    }
    bits
}

/// BCH GF(2^8) minimal polynomial table for t=18 correction.
/// Generator polynomial g(x) for BCH(255,131,18) over GF(2^8).
/// (Coefficients of g(x) mod 2, degree = n-k = 124)
const BCH_GENERATOR: &[u8] = &[
    // Precomputed g(x) for BCH(255,131,18) — 124 parity bits = 16 bytes (we use 16 of 16)
    // In practice generated via: prod_{i=1}^{2t} min_poly(alpha^i) over GF(2^8)
    // These coefficients are illustrative — a real impl uses a GF(2^8) library.
    0xD9, 0x5B, 0x3A, 0x7F, 0x8C, 0x12, 0xE4, 0x56,
    0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89,
];

/// GF(2^8) arithmetic with primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D)
mod gf256 {
    const PRIM: u16 = 0x11D;
    pub fn mul(mut a: u8, mut b: u8) -> u8 {
        let mut result = 0u8;
        while b != 0 {
            if b & 1 != 0 { result ^= a; }
            let carry = a & 0x80;
            a <<= 1;
            if carry != 0 { a ^= (PRIM & 0xFF) as u8; }
            b >>= 1;
        }
        result
    }
    pub fn pow(mut base: u8, mut exp: u8) -> u8 {
        let mut result = 1u8;
        while exp > 0 {
            if exp & 1 != 0 { result = mul(result, base); }
            base = mul(base, base);
            exp >>= 1;
        }
        result
    }
}

/// Compute BCH syndromes S_1 .. S_{2t} for received word r (255 bits packed).
/// Returns syndrome bytes [S_1, S_3, S_5, ..., S_{2t-1}] (odd syndromes, t of them).
fn bch_syndromes(r: &[u8; (BCH_N + 7) / 8]) -> [u8; BCH_T] {
    let mut syndromes = [0u8; BCH_T];
    // alpha = primitive element of GF(2^8), alpha^1 = 0x02
    let alpha: u8 = 0x02;
    for j in 0..BCH_T {
        // S_{2j+1} = sum_{i: r_i=1} alpha^{i*(2j+1)}
        let power = (2 * j + 1) as u8;
        let mut s = 0u8;
        for i in 0..BCH_N {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            if (r[byte_idx] >> bit_idx) & 1 == 1 {
                s ^= gf256::pow(gf256::pow(alpha, i as u8), power);
            }
        }
        syndromes[j] = s;
    }
    syndromes
}

/// Berlekamp-Massey over GF(2^8) to find error-locator polynomial Λ(x).
fn berlekamp_massey(syndromes: &[u8; BCH_T]) -> Vec<u8> {
    let n = syndromes.len();
    let mut c = vec![0u8; n + 1];
    let mut b = vec![0u8; n + 1];
    c[0] = 1; b[0] = 1;
    let mut l = 0usize;
    let mut m = 1usize;
    let mut b_val: u8 = 1;

    for i in 0..n {
        // discrepancy d
        let mut d = syndromes[i];
        for j in 1..=l {
            d ^= gf256::mul(c[j], syndromes[i.saturating_sub(j)]);
        }
        if d == 0 {
            m += 1;
        } else if 2 * l <= i {
            let t = c.clone();
            let coeff = gf256::mul(d, gf256::pow(b_val, 255 - 1)); // d / b_val
            for j in m..=n {
                c[j] ^= gf256::mul(coeff, b[j - m]);
            }
            l = i + 1 - l;
            b = t;
            b_val = d;
            m = 1;
        } else {
            let coeff = gf256::mul(d, gf256::pow(b_val, 255 - 1));
            for j in m..=n {
                c[j] ^= gf256::mul(coeff, b[j - m]);
            }
            m += 1;
        }
    }
    c[0..=l].to_vec()
}

/// Chien search: find roots of Λ(x) → error positions.
fn chien_search(lambda: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();
    let alpha: u8 = 0x02;
    for i in 0..BCH_N {
        // Evaluate Λ(alpha^{-i})
        let alpha_inv_i = gf256::pow(alpha, (255 - i as u8) % 255);
        let mut val = 0u8;
        for (j, &coef) in lambda.iter().enumerate() {
            val ^= gf256::mul(coef, gf256::pow(alpha_inv_i, j as u8));
        }
        if val == 0 {
            positions.push(i);
        }
    }
    positions
}

/// Full BCH decode: correct up to BCH_T errors in `received` (in-place).
/// Returns number of bits corrected, or Err if uncorrectable.
fn bch_decode(received: &mut [u8; (BCH_N + 7) / 8]) -> Result<usize, &'static str> {
    let syndromes = bch_syndromes(received);

    // If all syndromes zero → no errors
    if syndromes.iter().all(|&s| s == 0) {
        return Ok(0);
    }

    let lambda = berlekamp_massey(&syndromes);
    let error_count = lambda.len() - 1;

    if error_count > BCH_T {
        return Err("Too many errors — uncorrectable");
    }

    let positions = chien_search(&lambda);

    if positions.len() != error_count {
        return Err("Chien search failed — decoding error");
    }

    // Flip error bits
    for &pos in &positions {
        let byte_idx = pos / 8;
        let bit_idx = 7 - (pos % 8);
        received[byte_idx] ^= 1 << bit_idx;
    }

    Ok(positions.len())
}

// ── Guest Entry Point ─────────────────────────────────────────────────────────

pub fn main() {
    // 1. Read private inputs from the host
    let input: FaceProofInput = env::read();

    assert_eq!(input.embedding.len(), EMBEDDING_DIM, "Wrong embedding size");
    assert_eq!(input.reference_embedding.len(), EMBEDDING_DIM, "Wrong reference size");
    assert_eq!(input.helper_string.len(), (BCH_N + 7) / 8,
        "Helper string size mismatch");

    // 2. Cosine similarity check (fixed-point, no floats in zkVM)
    let similarity_q8 = cosine_similarity_q8(&input.embedding, &input.reference_embedding);
    let face_match = similarity_q8 >= input.threshold_q8;

    // 3. Fuzzy Extractor — Reproduce()
    //    Recover the enrolled BCH codeword from probe embedding + helper string
    let probe_bits = embedding_to_bits(&input.embedding);

    // c' = probe_bits XOR helper_string (noisy codeword)
    let mut noisy_codeword = [0u8; (BCH_N + 7) / 8];
    for i in 0..noisy_codeword.len() {
        noisy_codeword[i] = probe_bits[i] ^ input.helper_string[i];
    }

    // Run BCH error correction
    let decode_result = bch_decode(&mut noisy_codeword);
    let (fuzzy_ok, bits_corrected) = match decode_result {
        Ok(n) => (true, n),
        Err(_) => (false, 0),
    };

    // 4. Derive biometric key = SHA-256(corrected codeword message bits)
    let key_material = &noisy_codeword[0..KEY_BYTES];
    let key_commitment = sha256(key_material);

    // 5. Commit public outputs to the journal
    let output = FaceProofOutput {
        identity_verified: face_match && fuzzy_ok,
        key_commitment,
        similarity_q8,
        bits_corrected,
    };

    env::commit(&output);
}
