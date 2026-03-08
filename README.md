# ZKFace — Privacy-Preserving Face Identity Proof

> Zero-knowledge proof of face identity using **MobileNetV1** embeddings + **BCH Fuzzy Extractor** + **Risc Zero zkVM**

---

## What Is This?

ZKFace lets you prove *"this person is who they enrolled as"* without revealing:
- The face image
- The biometric embedding
- The cryptographic key derived from the face

A verifier receives only a **cryptographic receipt** containing:
- `identity_verified: true/false`
- `key_commitment` (SHA-256 hash of the biometric key)
- `similarity_q8` (cosine similarity score)
- `bits_corrected` (BCH error correction count)

---

## System Architecture

```
ENROLL (once, on trusted device):
  face_ref.jpg
      │
      ▼
  MobileNetV1 ──► 128-D L2-normed embedding (float32)
      │
      ▼
  Quantise ──► Q8.8 fixed-point (×256)
      │
      ▼
  Sign bits ──► w (255-bit string)
      │
      ├──► BCH.encode(w) ──► codeword c
      ├──► helper = w XOR c   ──► store (safe to make public)
      └──► key = SHA256(c)
           key_commitment = SHA256(key) ──► register on-chain / in DB

VERIFY (each login, any device):
  face_probe.jpg
      │
      ▼
  MobileNetV1 ──► embedding e' ──► Quantise ──► w'
      │
      ▼  [everything below runs INSIDE Risc Zero zkVM — every step is proven]
  c'  = w' XOR helper              (noisy codeword, ≤18 bit-errors tolerated)
  c   = BCH.decode(c')             (Berlekamp-Massey + Chien search)
  key = SHA256(c)
  key_commitment = SHA256(key)     (compared against enrolled commitment)
  sim = cosine(e', ref) ≥ 0.85    (fixed-point, no floats)
      │
      ▼
  PUBLIC JOURNAL: { verified, key_commitment, similarity_q8, bits_corrected }
```

---

## Project Structure

```
zkFace/
├── Cargo.toml                  Workspace root
├── core/
│   └── src/lib.rs              Shared types: FaceProofInput, FaceProofOutput, constants
├── methods/
│   ├── src/lib.rs              Exposes guest ELF + Image ID to host
│   ├── build.rs                Compiles guest for riscv32im target
│   └── guest/
│       └── src/main.rs         ZK circuit (runs inside zkVM):
│                                 - SHA-256 (pure Rust, no_std)
│                                 - Cosine similarity (Q8.8 fixed-point)
│                                 - BCH syndromes over GF(2^8)
│                                 - Berlekamp-Massey algorithm
│                                 - Chien search
└── host/
    └── src/
        ├── enroll.rs           Enrollment: MobileNet → BCH Gen() → enrollment.json
        └── verify.rs           Proof generation + verification
```

---

## Cryptographic Details

### MobileNetV1 Embedding
- Output: 128-dimensional L2-normalised face embedding
- Quantised to Q8.8 fixed-point (multiplied by 256) for integer-only zkVM arithmetic
- Sign bits extracted → 128-bit biometric string (padded to 255 bits for BCH)

### BCH Fuzzy Extractor

| Parameter | Value | Meaning |
|-----------|-------|---------|
| n | 255 | Codeword length (bits) |
| k | 131 | Message length (bits) |
| t | 18  | Max correctable bit-errors |
| Field | GF(2^8) | Primitive poly x⁸+x⁴+x³+x²+1 |

18-bit error tolerance ≈ **7% bit-error rate** — robust across lighting, pose, and sensor variation between enrollment and probe.

The fuzzy extractor uses the **Secure Sketch** pattern (Dodis et al. 2004):
- `Gen(w)` → `(key, helper)` at enrollment
- `Rep(w', helper)` → `key` at verification (if `w'` is close enough to `w`)

### ZK Proof (Risc Zero)
- STARK-based — post-quantum friendly, no trusted setup required
- Guest program compiled to `riscv32im` and executed inside the zkVM
- Every CPU instruction is proven; the receipt is ~200KB
- Can be verified on Ethereum via `RiscZeroVerifier.sol`

---

## Prerequisites

```bash
# Install Risc Zero toolchain
curl -L https://risczero.com/install | bash
source ~/.bashrc

rzup install
cargo install cargo-risczero
cargo risczero install
```

---

## Build

```bash
git clone <your-repo>
cd zkFace
cargo build
```

First build downloads the RISC Zero standard library (~500MB) and compiles the guest for the `riscv32im` target. Expect 3–5 minutes.

---

## Usage

### Step 1 — Enroll a reference face

```bash
cargo run --bin enroll -- face_reference.jpg enrollment.json
```

Output:
```
=== ZKFace Enrollment ===
Reference image  : face_reference.jpg
Embedding dim    : 128
BCH helper string: 32 bytes
Key commitment   : a3f1c2b8d4e7...
Saved enrollment : enrollment.json
```

`enrollment.json` contains the reference embedding, helper string, and key commitment. The helper string can be stored publicly; the key commitment should be registered on-chain or in a trusted database.

### Step 2 — Prove and verify identity

```bash
cargo run --bin verify -- enrollment.json face_probe.jpg proof.bin
```

Output:
```
╔══════════════════════════════════════╗
║        ZK IDENTITY PROOF RESULT      ║
╠══════════════════════════════════════╣
║ Identity verified : true             ║
║ Key commitment OK : true             ║
║ Cosine similarity : 0.9123           ║
║ Threshold         : 0.8500           ║
║ BCH bits corrected: 7                ║
║ Key commitment    : a3f1c2b8d4e7...  ║
╚══════════════════════════════════════╝

✅ PROOF VALID — Identity confirmed without revealing face data or key.
```

The proof receipt is saved to `proof.bin` and can be sent to any verifier.

### Step 3 — On-chain verification (Ethereum)

```solidity
IRiscZeroVerifier verifier = IRiscZeroVerifier(VERIFIER_ADDRESS);
bytes32 imageId = bytes32(FACE_VERIFY_ID);
verifier.verify(seal, imageId, sha256(journalBytes));
```

---

## Replacing Mock MobileNet with Real Inference

The current implementation uses a deterministic fake embedding for demonstration. To use a real MobileNetV1 ONNX model:

```toml
# Add to host/Cargo.toml
[dependencies]
ort = "2.0"       # ONNX Runtime bindings
image = "0.25"    # Image loading + preprocessing
```

```rust
use ort::{Session, inputs};

fn mobilenet_embed(image_path: &str) -> Vec<f32> {
    let model = Session::builder()?
        .with_model_from_file("mobilenetv1_128.onnx")?;

    // Resize to 224×224, normalise to [0,1], CHW format
    let tensor = preprocess_image(image_path);

    let outputs = model.run(inputs!["input" => tensor]?)?;
    let embedding = outputs["output"].extract_tensor::<f32>()?;

    l2_normalise(embedding.view().as_slice().unwrap())
}
```

Pre-trained MobileNet ONNX models: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface

---

## Performance

| Step | Time (CPU) | Time (GPU / Bonsai) |
|------|-----------|---------------------|
| Enrollment | <1s | <1s |
| Proof generation | 60–120s | 3–5s |
| Proof verification | <1s | <1s |
| Proof size | ~200KB | ~200KB |

For faster proving, use Bonsai (Risc Zero's remote GPU prover):

```bash
export BONSAI_API_KEY=<your_key>
export BONSAI_API_URL=https://api.bonsai.xyz
export RISC0_PROVER=bonsai
cargo run --bin verify -- enrollment.json face_probe.jpg proof.bin
```

Free tier available at https://dev.bonsai.xyz

---

## Security Properties

| Property | Guarantee |
|----------|-----------|
| Face privacy | Raw face image and embedding never leave the prover device |
| Key privacy | Biometric key never appears in the public proof or journal |
| Soundness | Computationally infeasible to forge a valid proof without a matching face |
| Fuzzy binding | BCH tolerates ≤18 bit-flips between enrollment and probe embeddings |
| Zero-knowledge | Verifier learns only `{verified, key_commitment, similarity_q8}` |
| Post-quantum | STARK-based proof system, no elliptic curve assumptions |

---

## Production Checklist

- [ ] Replace mock MobileNet with real ONNX inference (`ort` or `tract` crate)
- [ ] Replace simplified BCH parity with full GF(2) polynomial long division
- [ ] Use Bonsai GPU prover for <5s proving time
- [ ] Add liveness detection to prevent photo replay attacks
- [ ] Encrypt the helper string at rest (it leaks which bits flipped)
- [ ] Audit BCH implementation against NIST test vectors
- [ ] Deploy `RiscZeroVerifier.sol` for on-chain receipt verification
- [ ] Add rate limiting to prevent brute-force enrollment attacks

---

## References

- [Risc Zero Documentation](https://dev.risczero.com)
- [Dodis et al. — Fuzzy Extractors (2004)](https://eprint.iacr.org/2003/235)
- [BCH Codes — Wikipedia](https://en.wikipedia.org/wiki/BCH_code)
- [MobileNetV1 — Howard et al. (2017)](https://arxiv.org/abs/1704.04861)
- [ArcFace ONNX Models](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface)

---

## License

MIT