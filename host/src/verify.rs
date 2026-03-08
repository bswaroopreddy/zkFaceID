//! ─────────────────────────────────────────────────────────────────────────────
//! Verification Phase — generates and verifies the Risc Zero ZK proof.
//!
//! 1. Load enrollment.json (reference embedding + helper string + key commitment)
//! 2. Run MobileNet on probe face image → probe embedding
//! 3. Build FaceProofInput (private witness)
//! 4. Execute guest program inside Risc Zero zkVM → receipt
//! 5. Verify receipt cryptographically
//! 6. Extract FaceProofOutput from journal
//! 7. Check key_commitment matches enrollment record
//! ─────────────────────────────────────────────────────────────────────────────

use anyhow::{bail, Context, Result};
use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};
use serde::{Deserialize, Serialize};
use std::fs;
use zkface_core::{BCH_N, EMBEDDING_DIM, FaceProofInput, FaceProofOutput, KEY_BYTES};
use methods::{FACE_VERIFY_ELF, FACE_VERIFY_ID};

// Re-use enrollment record type from enroll binary
#[derive(Deserialize)]
struct EnrollmentRecord {
    pub reference_embedding_q8: Vec<i32>,
    pub helper_string_hex: String,
    pub key_commitment_hex: String,
    pub threshold_q8: i32,
}

// ── Simulated MobileNet (same mock as enroll.rs — replace with real ONNX) ────

fn mobilenet_embed(image_path: &str) -> Vec<f32> {
    eprintln!("[verify] Simulating MobileNet inference for: {}", image_path);
    // Slightly perturbed version of enrollment embedding to simulate real biometric noise
    let seed: u64 = image_path.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    // Use a different seed offset to simulate probe vs reference
    let probe_seed = seed.wrapping_add(42);
    let mut v: Vec<f32> = (0..EMBEDDING_DIM)
        .map(|i| {
            let x = ((probe_seed.wrapping_mul(i as u64 + 1).wrapping_add(0xDEADBEEF)) as i32) as f32;
            x / (i32::MAX as f32)
        })
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn quantise(embedding: &[f32]) -> Vec<i32> {
    embedding.iter().map(|&x| (x * 256.0).round() as i32).collect()
}

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

// ── Proof Generation ─────────────────────────────────────────────────────────

fn generate_proof(input: &FaceProofInput) -> Result<Receipt> {
    println!("[prove] Building executor environment...");

    let env = ExecutorEnv::builder()
        .write(input)
        .context("Failed to serialise FaceProofInput")?
        .build()
        .context("Failed to build ExecutorEnv")?;

    println!("[prove] Running guest program inside zkVM...");
    println!("[prove] (This may take 30-120 seconds on CPU — use GPU prover for production)");

    let prover = default_prover();
    let prove_start = std::time::Instant::now();
    let receipt = prover
        .prove(env, FACE_VERIFY_ELF)
        .context("zkVM execution / proving failed")?
        .receipt;
    let prove_time = prove_start.elapsed();
    println!("[prove] Proof generated successfully!");
    println!("[prove] ⏱  Proving time : {:.2?}", prove_time);
    Ok(receipt)
}

// ── Proof Verification ────────────────────────────────────────────────────────

fn verify_proof(receipt: &Receipt) -> Result<FaceProofOutput> {
    println!("[verify] Verifying receipt cryptographically...");

    let verify_start = std::time::Instant::now();
    receipt
        .verify(FACE_VERIFY_ID)
        .context("Receipt verification failed — proof is INVALID")?;
    let verify_time = verify_start.elapsed();
    println!("[verify] Receipt is cryptographically valid.");
    println!("[verify] ⏱  Verifier time: {:.2?}", verify_time);

    let output: FaceProofOutput = receipt.journal.decode()
        .context("Failed to decode journal")?;

    Ok(output)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let enrollment_path = std::env::args().nth(1).unwrap_or("enrollment.json".to_string());
    let probe_image = std::env::args().nth(2).unwrap_or("face_probe.jpg".to_string());
    let receipt_path = std::env::args().nth(3).unwrap_or("proof.bin".to_string());

    println!("=== ZKFace Verification ===");
    println!("Enrollment file : {}", enrollment_path);
    println!("Probe image     : {}", probe_image);

    // 1. Load enrollment record
    let enrollment_json = fs::read_to_string(&enrollment_path)
        .context("Cannot read enrollment.json — run enroll first")?;
    let enrollment: EnrollmentRecord = serde_json::from_str(&enrollment_json)?;

    let helper_bytes = hex_decode(&enrollment.helper_string_hex);
    assert_eq!(helper_bytes.len(), (BCH_N + 7) / 8);

    // 2. Probe embedding
    let probe_f32 = mobilenet_embed(&probe_image);
    let probe_q8 = quantise(&probe_f32);

    // 3. Assemble private witness
    let input = FaceProofInput {
        embedding: probe_q8,
        reference_embedding: enrollment.reference_embedding_q8.clone(),
        helper_string: helper_bytes,
        threshold_q8: enrollment.threshold_q8,
    };

    // 4. Generate ZK proof
    let receipt = generate_proof(&input)?;

    // 5. Save receipt to disk (can be sent to any verifier)
    let receipt_bytes = bincode::serialize(&receipt)
        .unwrap_or_else(|_| vec![]);
    if !receipt_bytes.is_empty() {
        fs::write(&receipt_path, &receipt_bytes).ok();
        println!("[prove] Receipt saved to: {}", receipt_path);
    }

    // 6. Verify proof and read public outputs
    let output = verify_proof(&receipt)?;

    // 7. Check key commitment matches enrollment
    let enrolled_commitment = hex_decode(&enrollment.key_commitment_hex);
    let commitments_match = output.key_commitment == enrolled_commitment.as_slice();

    // ── Results ───────────────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════╗");
    println!("║        ZK IDENTITY PROOF RESULT      ║");
    println!("╠══════════════════════════════════════╣");
    println!("║ Identity verified : {:<17} ║", output.identity_verified);
    println!("║ Key commitment OK : {:<17} ║", commitments_match);
    println!("║ Cosine similarity : {:<17.4} ║",
        output.similarity_q8 as f32 / 256.0);
    println!("║ Threshold         : {:<17.4} ║",
        enrollment.threshold_q8 as f32 / 256.0);
    println!("║ BCH bits corrected: {:<17} ║", output.bits_corrected);
    println!("║ Key commitment    : {}... ║",
        &enrollment.key_commitment_hex[..16]);
    println!("╚══════════════════════════════════════╝");

    if output.identity_verified && commitments_match {
        println!("\n✅ PROOF VALID — Identity confirmed without revealing face data or key.");
    } else {
        println!("\n❌ PROOF FAILED — Identity not confirmed.");
        if !output.identity_verified {
            println!("   Reason: Face similarity below threshold or BCH decoding failed.");
        }
        if !commitments_match {
            println!("   Reason: Recovered key does not match enrollment commitment.");
        }
    }

    Ok(())
}
