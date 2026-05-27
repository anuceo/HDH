use blake3;
use crate::round;
use crate::state::State;

/// Rate in bytes: 5888 bits (25+25+25+17 lanes × 64 bits).
pub const RATE_BYTES: usize = 736;
/// Capacity in bytes: 512 bits (s4[17..25]).
pub const CAPACITY_BYTES: usize = 64;
/// Default digest length: 512 bits.
pub const OUTPUT_BYTES: usize = 64;

// ── Permutation ────────────────────────────────────────────────────────────────

/// Apply 6 rounds of the HDH permutation (χ → θ → Φ with round constant).
pub fn permute(s: State) -> State {
    (0..6u64).fold(s, |acc, idx| round(acc, idx))
}

// ── IV ─────────────────────────────────────────────────────────────────────────

fn iv_lane(share: usize, lane: usize) -> u64 {
    // Domain string unique per (share, lane): breaks all starting symmetry.
    let tag = format!("HDH-IV-s{share}-l{lane}");
    let h = blake3::hash(tag.as_bytes());
    u64::from_le_bytes(h.as_bytes()[0..8].try_into().unwrap())
}

/// Initialize the state with BLAKE3-derived per-share/per-lane IV constants.
/// Every cell gets a unique 64-bit seed, playing the role SHA-256's sqrt-of-prime IVs.
pub fn iv_state() -> State {
    let s1: [u64; 25] = core::array::from_fn(|i| iv_lane(0, i));
    let s2: [u64; 25] = core::array::from_fn(|i| iv_lane(1, i));
    let s3: [u64; 25] = core::array::from_fn(|i| iv_lane(2, i));
    let s4: [u64; 25] = core::array::from_fn(|i| iv_lane(3, i));
    let parity: [u64; 25] = core::array::from_fn(|i| s1[i] ^ s2[i] ^ s3[i] ^ s4[i]);
    State { s1, s2, s3, s4, parity }
}

// ── Absorption ─────────────────────────────────────────────────────────────────

/// Rate layout (92 rate lanes × 64 bits = 5888 bits):
///   s1[0..25]  (1600 b) · s2[0..25]  (1600 b) · s3[0..25]  (1600 b) · s4[0..17]  (1088 b)
/// Capacity (never touched during absorption):
///   s4[17..25] (512 b)
fn absorb_block(mut state: State, block: &[u8; RATE_BYTES]) -> State {
    let mut pos = 0;
    macro_rules! xor_share {
        ($share:ident, $range:expr) => {
            for i in $range {
                state.$share[i] ^= u64::from_le_bytes(
                    block[pos..pos + 8].try_into().unwrap(),
                );
                pos += 8;
            }
        };
    }
    xor_share!(s1, 0..25);
    xor_share!(s2, 0..25);
    xor_share!(s3, 0..25);
    xor_share!(s4, 0..17);
    // Recompute parity for all lanes after rate XOR (multiple shares may have changed per lane).
    for i in 0..25 {
        state.parity[i] = state.s1[i] ^ state.s2[i] ^ state.s3[i] ^ state.s4[i];
    }
    permute(state)
}

// ── Padding ────────────────────────────────────────────────────────────────────

/// Apply pad10*1: append 0x01, pad with 0x00 to block boundary, XOR 0x80 into final byte.
///
/// Every input produces at least one block. A message that fills a complete block
/// receives an additional all-pad block to preserve prefix-freeness.
fn pad_blocks(data: &[u8]) -> Vec<[u8; RATE_BYTES]> {
    let mut buf: Vec<u8> = data.to_vec();
    buf.push(0x01);
    let rem = buf.len() % RATE_BYTES;
    let extra = if rem == 0 { RATE_BYTES } else { RATE_BYTES - rem };
    buf.resize(buf.len() + extra, 0x00);
    *buf.last_mut().unwrap() ^= 0x80;
    buf.chunks_exact(RATE_BYTES)
        .map(|c| c.try_into().unwrap())
        .collect()
}

// ── Squeeze ────────────────────────────────────────────────────────────────────

/// Extract OUTPUT_BYTES from the first lanes of the rate portion (s1[0..8]).
fn squeeze_output(state: &State) -> [u8; OUTPUT_BYTES] {
    let mut out = [0u8; OUTPUT_BYTES];
    for (i, &lane) in state.s1[..8].iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
    }
    out
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Hash `data` with HDH and return a 64-byte (512-bit) digest.
pub fn hash(data: &[u8]) -> [u8; OUTPUT_BYTES] {
    let mut state = iv_state();
    for block in pad_blocks(data) {
        state = absorb_block(state, &block);
    }
    squeeze_output(&state)
}

/// Keyed PRF mode.
///
/// The 64-byte key is XOR'd into the capacity lanes (s4[17..25]) — the portion
/// never exposed during squeezing — before absorption begins. The key material
/// stays hidden throughout the sponge lifetime.
pub fn prf(key: &[u8; CAPACITY_BYTES], data: &[u8]) -> [u8; OUTPUT_BYTES] {
    let mut state = iv_state();
    for (i, chunk) in key.chunks_exact(8).enumerate() {
        state.s4[17 + i] ^= u64::from_le_bytes(chunk.try_into().unwrap());
    }
    // Keep parity consistent for the capacity lanes after key injection.
    for i in 17..25 {
        state.parity[i] = state.s1[i] ^ state.s2[i] ^ state.s3[i] ^ state.s4[i];
    }
    for block in pad_blocks(data) {
        state = absorb_block(state, &block);
    }
    squeeze_output(&state)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        let data = b"HDH test vector";
        assert_eq!(hash(data), hash(data));
    }

    #[test]
    fn hash_empty_input_does_not_panic() {
        let out = hash(b"");
        assert_ne!(out, [0u8; OUTPUT_BYTES], "hash of empty input must not be all-zero");
    }

    #[test]
    fn hash_single_bit_flip_changes_output() {
        let mut data = [0u8; 64];
        let a = hash(&data);
        data[0] ^= 1;
        let b = hash(&data);
        assert_ne!(a, b, "single-bit flip in input must change the digest");
        // Expect near-50% bit change (avalanche)
        let flipped: u32 = a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum();
        assert!(
            flipped > 100,
            "avalanche: only {flipped}/512 bits changed — output barely differs after input flip"
        );
    }

    #[test]
    fn hash_known_answer_empty() {
        let got = hash(b"");
        assert_eq!(got, HDH_KAT_EMPTY, "hash(empty) KAT mismatch");
    }

    #[test]
    fn hash_known_answer_abc() {
        let got = hash(b"abc");
        assert_eq!(got, HDH_KAT_ABC, "hash(\"abc\") KAT mismatch");
    }

    #[test]
    fn prf_known_answer() {
        // Non-zero key: single 0x01 byte in first position.
        let mut key = [0u8; CAPACITY_BYTES];
        key[0] = 0x01;
        let got = prf(&key, b"abc");
        assert_eq!(got, HDH_KAT_PRF_ABC, "prf KAT mismatch");
    }

    #[test]
    fn prf_differs_from_hash_with_nonzero_key() {
        // A non-zero key XORs into the BLAKE3-seeded capacity lanes, producing
        // a different starting point than the unkeyed hash.
        let mut key = [0u8; CAPACITY_BYTES];
        key[0] = 0x01;
        assert_ne!(
            hash(b"abc"),
            prf(&key, b"abc"),
            "PRF with non-zero key must differ from unkeyed hash"
        );
    }

    #[test]
    fn prf_different_keys_produce_different_output() {
        let data = b"message";
        let mut key_a = [0u8; CAPACITY_BYTES];
        let mut key_b = [0u8; CAPACITY_BYTES];
        key_a[0] = 0x01;
        key_b[0] = 0x02;
        assert_ne!(prf(&key_a, data), prf(&key_b, data));
    }

    #[test]
    fn prf_is_deterministic() {
        let data = b"message";
        let key = [42u8; CAPACITY_BYTES];
        assert_eq!(prf(&key, data), prf(&key, data));
    }

    // KAT values — generated by `cargo run --bin gen_kat`.
    static HDH_KAT_EMPTY: [u8; OUTPUT_BYTES] = [
        0x08, 0xcc, 0xd4, 0x2a, 0x5c, 0xfd, 0x07, 0xf9,
        0xe2, 0x8a, 0xb8, 0xa0, 0x57, 0x38, 0xfd, 0x6f,
        0x48, 0xf0, 0xf0, 0x87, 0x38, 0x61, 0x8b, 0x77,
        0x91, 0x31, 0x4d, 0x29, 0x38, 0x6e, 0xec, 0xfe,
        0xa9, 0xa0, 0xe0, 0xb1, 0x03, 0xd2, 0x7b, 0x28,
        0x38, 0x51, 0x8c, 0xe9, 0x4f, 0x86, 0xa6, 0x54,
        0x5a, 0x23, 0x8d, 0x61, 0xc4, 0x3d, 0x80, 0xbe,
        0x81, 0x8c, 0xd1, 0x16, 0x0e, 0x6f, 0x14, 0xc3,
    ];

    static HDH_KAT_ABC: [u8; OUTPUT_BYTES] = [
        0xb3, 0x26, 0x21, 0xec, 0x01, 0x20, 0x84, 0x81,
        0x9e, 0xb5, 0xc2, 0xd7, 0xc2, 0x67, 0x21, 0xf0,
        0x00, 0x51, 0x81, 0x3c, 0xda, 0xb2, 0xd3, 0x3f,
        0x8e, 0x1d, 0x07, 0x70, 0xc7, 0x13, 0xa8, 0xd1,
        0x60, 0x6a, 0x19, 0x43, 0x7a, 0x64, 0x1b, 0xed,
        0x28, 0x85, 0xae, 0x86, 0x18, 0x3e, 0x61, 0xa1,
        0x78, 0x2a, 0xb5, 0x06, 0x10, 0x03, 0x27, 0xfa,
        0x35, 0xe7, 0xe9, 0x3f, 0x03, 0xc0, 0x65, 0x76,
    ];

    // PRF with key = [0x01, 0x00, 0x00, ..., 0x00].
    static HDH_KAT_PRF_ABC: [u8; OUTPUT_BYTES] = [
        0xcd, 0x9c, 0x95, 0x14, 0xbf, 0x29, 0xcf, 0x2e,
        0xa0, 0xf3, 0xa3, 0x5f, 0xfb, 0xae, 0xe7, 0xf7,
        0xe3, 0x9a, 0x0c, 0xb5, 0xbb, 0xe5, 0x84, 0x4b,
        0x70, 0x1d, 0x5e, 0xb9, 0xf3, 0x50, 0x9f, 0x3a,
        0x83, 0x93, 0x86, 0x37, 0xbf, 0xe5, 0x89, 0x4f,
        0x33, 0x31, 0x25, 0x62, 0xd2, 0xaf, 0x20, 0x1f,
        0xc3, 0x14, 0xde, 0xff, 0xd2, 0x80, 0xf9, 0x9b,
        0xd7, 0x42, 0x53, 0x27, 0x5c, 0xb1, 0xb7, 0x48,
    ];
}
