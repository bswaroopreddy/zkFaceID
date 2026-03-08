/// Shared types between Host and Guest (zkVM)
/// These are committed to the public journal of the proof.

use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────────

/// MobileNetV1 output embedding dimension (final FC layer → 128-D L2-normed)
pub const EMBEDDING_DIM: usize = 128;

/// BCH(255,131,18) over GF(2^8):
///   - n=255 codeword bits, k=131 message bits, t=18 correctable bit-errors
pub const BCH_N: usize = 255;
pub const BCH_K: usize = 131;
pub const BCH_T: usize = 18;

/// Number of bytes in the helper string (syndrome / parity bits stored at enroll)
pub const HELPER_BYTES: usize = (BCH_N - BCH_K + 7) / 8; // 16 bytes

/// Biometric key length in bytes (derived from BCH codeword message bits)
pub const KEY_BYTES: usize = (BCH_K + 7) / 8; // 17 bytes → we use 16 for clean AES key

// ── Guest Inputs (private witness) ──────────────────────────────────────────

/// Everything the guest needs — kept private, never revealed outside the zkVM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceProofInput {
    /// Raw face embedding from MobileNet (fixed-point Q8.8, i.e. *256)
    /// Shape: [EMBEDDING_DIM]
    pub embedding: Vec<i32>,

    /// Enrolled reference embedding (also Q8.8)
    pub reference_embedding: Vec<i32>,

    /// BCH helper string produced during enrollment (public at enroll time)
    pub helper_string: Vec<u8>,

    /// Cosine similarity threshold × 256 (e.g. 0.85 → 217)
    pub threshold_q8: i32,
}

// ── Guest Outputs (public journal) ───────────────────────────────────────────

/// What gets committed to the public receipt journal.
/// The verifier sees ONLY this — no face data, no keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceProofOutput {
    /// True iff probe embedding is within threshold of reference AND
    /// the fuzzy extractor successfully reproduced the enrolled key.
    pub identity_verified: bool,

    /// Commitment (SHA-256) of the reproduced biometric key.
    /// Lets the verifier check a pre-registered key hash without learning the key.
    pub key_commitment: [u8; 32],

    /// Cosine similarity achieved (Q8.8 fixed-point), for auditability.
    pub similarity_q8: i32,

    /// Number of bit-errors corrected by BCH during this session.
    pub bits_corrected: usize,
}
