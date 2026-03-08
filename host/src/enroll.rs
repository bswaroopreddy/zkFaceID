//! ─────────────────────────────────────────────────────────────────────────────
//! Enrollment Phase (runs on trusted device, NOT in zkVM)
//!
//! 1. Load MobileNetV1 ONNX model
//! 2. Run inference on reference face image → 128-D L2-normed embedding
//! 3. Quantise embedding to Q8.8 fixed-point
//! 4. Run BCH fuzzy extractor Gen():
//!      c   = BCH.encode(message from embedding bits)
//!      pub = embedding_bits XOR c    ← save this as helper string
//!      key = SHA-256(c[0..KEY_BYTES])
//!      key_hash = SHA-256(key)       ← register this on-chain / in DB
//! 5. Save {reference_embedding_q8, helper_string, key_hash} to enrollment.json
//! ─────────────────────────────────────────────────────────────────────────────

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use zkface_core::{BCH_K, BCH_N, BCH_T, EMBEDDING_DIM, KEY_BYTES};

// ── Simulated MobileNet (replace with real ONNX runtime call) ────────────────

/// In production: load ONNX, run inference, extract final FC output, L2-normalise.
/// Here we produce a deterministic fake embedding for illustration.
fn mobilenet_embed(image_path: &str) -> Vec<f32> {
    // TODO: replace with `ort` (OnnxRuntime) or `tract` crate call:
    //   let model = ort::Session::builder()?.with_model_from_file("mobilenetv1.onnx")?;
    //   let input = preprocess_image(image_path)?;  // resize 224×224, normalise
    //   let output = model.run(inputs![input])?;
    //   let embedding = output["output"].extract_tensor::<f32>()?;
    //   l2_normalise(embedding)

    // Fake: hash image path into a pseudo-random 128-D unit vector
    eprintln!("[enroll] Simulating MobileNet inference for: {}", image_path);
    let seed: u64 = image_path.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut v: Vec<f32> = (0..EMBEDDING_DIM)
        .map(|i| {
            let x = ((seed.wrapping_mul(i as u64 + 1).wrapping_add(0xDEADBEEF)) as i32) as f32;
            x / (i32::MAX as f32)
        })
        .collect();
    // L2 normalise
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

// ── Quantisation ─────────────────────────────────────────────────────────────

/// Float embedding → Q8.8 fixed-point (multiply by 256, round)
fn quantise(embedding: &[f32]) -> Vec<i32> {
    embedding.iter().map(|&x| (x * 256.0).round() as i32).collect()
}

// ── BCH Encoder ──────────────────────────────────────────────────────────────

/// Pack first BCH_K bits of the embedding sign vector into a message word,
/// then systematic BCH encode: codeword = [message | parity].
fn bch_encode(bits: &[u8; (BCH_N + 7) / 8]) -> [u8; (BCH_N + 7) / 8] {
    // Systematic BCH: first BCH_K bits are the message, last BCH_N-BCH_K are parity.
    // For a proper implementation, use polynomial long division by g(x).
    // Here we demonstrate the structure with a simplified XOR-based parity.
    let mut codeword = *bits;
    let parity_start_byte = BCH_K / 8;

    // Simplified parity computation (full impl uses GF(2) polynomial division)
    // In production: codeword[parity_start_byte..] = gf2_polymod(message, generator)
    for i in parity_start_byte..(BCH_N + 7) / 8 {
        codeword[i] = bits[i % parity_start_byte]
            ^ bits[(i + 3) % parity_start_byte]
            ^ bits[(i + 7) % parity_start_byte];
    }
    codeword
}

/// Embedding Q8.8 → bit-string (sign of each component)
fn embedding_to_bits(embedding: &[i32]) -> [u8; (BCH_N + 7) / 8] {
    let mut bits = [0u8; (BCH_N + 7) / 8];
    for (i, &val) in embedding.iter().take(BCH_N).enumerate() {
        let bit: u8 = if val >= 0 { 1 } else { 0 };
        bits[i / 8] |= bit << (7 - (i % 8));
    }
    bits
}

fn sha256_simple(data: &[u8]) -> [u8; 32] {
    // Re-use same SHA-256 as guest (full impl — omitted here for brevity)
    // In production: use `sha2` crate: sha2::Sha256::digest(data).into()
    let mut out = [0u8; 32];
    // Simple placeholder — replace with sha2::Sha256
    for (i, &b) in data.iter().take(32).enumerate() {
        out[i] = b.wrapping_add(i as u8);
    }
    out
}

// ── Enrollment Record ─────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct EnrollmentRecord {
    pub reference_embedding_q8: Vec<i32>,
    pub helper_string_hex: String,
    pub key_commitment_hex: String, // SHA-256(SHA-256(codeword)) — double-hash for ZK linkage
    pub threshold_q8: i32,
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let image_path = std::env::args().nth(1).unwrap_or("face_reference.jpg".to_string());
    let output_path = std::env::args().nth(2).unwrap_or("enrollment.json".to_string());

    println!("=== ZKFace Enrollment ===");
    println!("Reference image : {}", image_path);

    // Step 1: MobileNet inference
    let embedding_f32 = mobilenet_embed(&image_path);
    println!("Embedding dim   : {}", embedding_f32.len());

    // Step 2: Quantise
    let embedding_q8 = quantise(&embedding_f32);

    // Step 3: BCH Fuzzy Extractor Gen()
    let w_bits = embedding_to_bits(&embedding_q8);
    let codeword = bch_encode(&w_bits);

    // helper = w XOR c
    let mut helper = [0u8; (BCH_N + 7) / 8];
    for i in 0..helper.len() {
        helper[i] = w_bits[i] ^ codeword[i];
    }

    // key = SHA-256(codeword message bits)
    let key = sha256_simple(&codeword[0..KEY_BYTES]);
    // key_commitment = SHA-256(key)  ← registered on-chain
    let key_commitment = sha256_simple(&key);

    println!("BCH helper string: {} bytes", helper.len());
    println!("Key commitment   : {}", hex::encode(&key_commitment[0..8]));

    // Threshold: cosine similarity ≥ 0.85 (× 256 = 217)
    let threshold_q8: i32 = (0.85 * 256.0) as i32;

    let record = EnrollmentRecord {
        reference_embedding_q8: embedding_q8,
        helper_string_hex: hex::encode(&helper),
        key_commitment_hex: hex::encode(&key_commitment),
        threshold_q8,
    };

    let json = serde_json::to_string_pretty(&record)?;
    fs::write(&output_path, &json).context("Failed to write enrollment.json")?;

    println!("Saved enrollment : {}", output_path);
    println!("Enrollment complete. Share only key_commitment, not the helper string.");
    Ok(())
}

// Poor-man's hex encoding (avoids extra dependency)
mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
    pub fn decode(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }
}
