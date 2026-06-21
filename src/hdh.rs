/// HDH (Hypo-Diffused Hash) — full algorithm implementation.
///
/// State : 25 lanes × 256 bits = 6 400 bits  (Lane = [u64; 4])
/// Rate  : 23 lanes             = 5 888 bits  (736 bytes absorbed per block)
/// Cap   : 2  lanes             =   512 bits  (never touched during absorb)
/// Output: first 2 rate lanes   =   512 bits  (64 bytes)
/// Rounds: 4 per permutation call
///
/// Per-round layer order: θ → χ → φ → entropy-fold → feedforward → RC

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::sync::OnceLock;

// ── Dimensions ────────────────────────────────────────────────────────────────

pub const LANES: usize = 25;
pub const RATE_LANES: usize = 23;
pub const RATE_BYTES: usize = RATE_LANES * 32; // 736 bytes = 5888 bits
pub const OUTPUT_BYTES: usize = 64;            // 512 bits = first 2 lanes
pub const ROUNDS: usize = 4;

// ── χ mixing constants (Fibonacci / golden-ratio hashing) ────────────────────

pub const C1: u64 = 0x9E3779B97F4A7C15;
pub const C2: u64 = 0xBF58476D1CE4E5B9;

// ── Types ─────────────────────────────────────────────────────────────────────

/// 256-bit lane: four 64-bit words stored little-endian.
pub type Lane = [u64; 4];

/// Full 6 400-bit permutation state: 25 lanes.
pub type State = [Lane; LANES];

// ── Lane helpers ──────────────────────────────────────────────────────────────

#[inline(always)]
pub fn xor_lane(a: Lane, b: Lane) -> Lane {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}

/// Word-level rotation: rotate each 64-bit word in the lane left by r bits.
#[inline(always)]
pub fn rotl_lane(lane: Lane, r: u32) -> Lane {
    [
        lane[0].rotate_left(r),
        lane[1].rotate_left(r),
        lane[2].rotate_left(r),
        lane[3].rotate_left(r),
    ]
}

// ── Round constants ───────────────────────────────────────────────────────────

static RC_CACHE: OnceLock<[Lane; ROUNDS]> = OnceLock::new();

fn round_constants() -> &'static [Lane; ROUNDS] {
    RC_CACHE.get_or_init(|| {
        let mut rcs = [[0u64; 4]; ROUNDS];
        for (r, rc) in rcs.iter_mut().enumerate() {
            let mut h = Shake256::default();
            h.update(b"HDH-RC");
            h.update(&[r as u8]);
            let mut rdr = h.finalize_xof();
            let mut buf = [0u8; 32];
            rdr.read(&mut buf);
            for (w, chunk) in buf.chunks_exact(8).enumerate() {
                rc[w] = u64::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        rcs
    })
}

/// Return the round constant for round r (exposed for structural tests).
pub fn rc(r: usize) -> Lane {
    round_constants()[r]
}

// ── Layer θ — linear ring diffusion ──────────────────────────────────────────
//
//   T[i] = S[i] ⊕ S[i−7] ⊕ S[i−1] ⊕ S[i+1] ⊕ S[i+7]   (indices mod 25)
//   Branch number B(θ) = 6.

pub fn layer_theta(state: &mut State) {
    let prev = *state;
    for i in 0..LANES {
        let n = xor_lane(
            xor_lane(prev[(i + LANES - 7) % LANES], prev[(i + LANES - 1) % LANES]),
            xor_lane(prev[(i + 1) % LANES], prev[(i + 7) % LANES]),
        );
        state[i] = xor_lane(prev[i], n);
    }
}

/// Activity propagation of θ over a 25-bit lane-activity bitmask.
/// Used by structural test S1 for exhaustive branch-number verification.
pub fn theta_activity(a: u32) -> u32 {
    // Cyclic 25-bit rotation (left by n positions).
    fn rot25(x: u32, n: u32) -> u32 {
        ((x << n) | (x >> (25 - n))) & 0x1FF_FFFF
    }
    // A_out[i] = A_in[i] ^ A_in[i-7] ^ A_in[i-1] ^ A_in[i+1] ^ A_in[i+7]
    // In bitmask terms: shift right by k  ≡  the bit at position i comes from i+k.
    a ^ rot25(a, 7) ^ rot25(a, 1) ^ rot25(a, 24) ^ rot25(a, 18)
}

// ── Layer χ — multiplication-based nonlinear mixing ───────────────────────────

#[inline(always)]
pub fn chi_word(mut x: u64) -> u64 {
    x ^= x.rotate_left(13);
    x ^= x.rotate_left(29);
    x = x.wrapping_mul(C1);
    x ^= x >> 17;
    x = x.wrapping_mul(C2);
    x
}

pub fn layer_chi(state: &mut State) {
    for lane in state.iter_mut() {
        for w in lane.iter_mut() {
            *w = chi_word(*w);
        }
    }
}

// ── Layer φ — state-dependent cross-lane routing ─────────────────────────────
//
//   selector = lane[i][0] ⊕ lane[i][3]   (word 0 XOR word 3)
//   j        = selector % 25
//   rot      = (selector >> 5) % 64
//   S[i]    ^= rotl_lane(S[j], rot)

pub fn layer_phi(state: &mut State) {
    let prev = *state;
    for i in 0..LANES {
        let selector = prev[i][0] ^ prev[i][3];
        let j = (selector as usize) % LANES;
        let rot = ((selector >> 5) % 64) as u32;
        state[i] = xor_lane(state[i], rotl_lane(prev[j], rot));
    }
}

// ── Layer entropy-fold — long-range cross-lane XOR ───────────────────────────

pub fn layer_entropy_fold(state: &mut State) {
    let prev = *state;
    for i in 0..LANES {
        let f1 = rotl_lane(prev[(i + 11) % LANES], 17);
        let f2 = rotl_lane(prev[(i + 17) % LANES], 41);
        state[i] = xor_lane(state[i], xor_lane(f1, f2));
    }
}

// ── Permutation (4 rounds) ────────────────────────────────────────────────────

pub fn permute(state: &mut State) {
    let rcs = round_constants();
    for r in 0..ROUNDS {
        let prev = *state;              // snapshot before this round
        layer_theta(state);
        layer_chi(state);
        layer_phi(state);
        layer_entropy_fold(state);
        // Feedforward: XOR round-input back into round-output
        for i in 0..LANES {
            state[i] = xor_lane(state[i], prev[i]);
        }
        // Round constant injected into lane 0
        state[0] = xor_lane(state[0], rcs[r]);
    }
}

/// Reduced-round permutation — runs only `rounds` of the 4 available.
/// Used by security tests (SC2, SC3, SC4).
pub fn permute_r(state: &mut State, rounds: usize) {
    let rcs = round_constants();
    for r in 0..rounds {
        let prev = *state;
        layer_theta(state);
        layer_chi(state);
        layer_phi(state);
        layer_entropy_fold(state);
        for i in 0..LANES {
            state[i] = xor_lane(state[i], prev[i]);
        }
        state[0] = xor_lane(state[0], rcs[r % ROUNDS]);
    }
}

// ── Sponge: padding ───────────────────────────────────────────────────────────
//
// pad10*1: append 0x01, zero-fill to RATE_BYTES boundary, XOR 0x80 into last byte.

pub fn pad_blocks(data: &[u8]) -> Vec<[u8; RATE_BYTES]> {
    let mut buf = data.to_vec();
    buf.push(0x01);
    let rem = buf.len() % RATE_BYTES;
    let extra = if rem == 0 { RATE_BYTES } else { RATE_BYTES - rem };
    buf.resize(buf.len() + extra, 0x00);
    *buf.last_mut().unwrap() ^= 0x80;
    buf.chunks_exact(RATE_BYTES)
        .map(|c| c.try_into().unwrap())
        .collect()
}

// ── Sponge: absorb / squeeze ──────────────────────────────────────────────────

fn absorb(state: &mut State, block: &[u8; RATE_BYTES]) {
    for lane in 0..RATE_LANES {
        let base = lane * 32;
        for w in 0..4 {
            let off = base + w * 8;
            let word = u64::from_le_bytes(block[off..off + 8].try_into().unwrap());
            state[lane][w] ^= word;
        }
    }
    permute(state);
}

fn absorb_r(state: &mut State, block: &[u8; RATE_BYTES], rounds: usize) {
    for lane in 0..RATE_LANES {
        let base = lane * 32;
        for w in 0..4 {
            let off = base + w * 8;
            let word = u64::from_le_bytes(block[off..off + 8].try_into().unwrap());
            state[lane][w] ^= word;
        }
    }
    permute_r(state, rounds);
}

fn squeeze(state: &State) -> [u8; OUTPUT_BYTES] {
    let mut out = [0u8; OUTPUT_BYTES];
    for lane in 0..2 {
        for w in 0..4 {
            let off = lane * 32 + w * 8;
            out[off..off + 8].copy_from_slice(&state[lane][w].to_le_bytes());
        }
    }
    out
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Full HDH-512 hash: produces a 64-byte (512-bit) digest.
pub fn hash(data: &[u8]) -> [u8; OUTPUT_BYTES] {
    let mut state: State = [[0u64; 4]; LANES];
    for block in pad_blocks(data) {
        absorb(&mut state, &block);
    }
    squeeze(&state)
}

/// Reduced-round hash — same sponge structure but permutation runs only `rounds`.
pub fn hash_r(data: &[u8], rounds: usize) -> [u8; OUTPUT_BYTES] {
    let mut state: State = [[0u64; 4]; LANES];
    for block in pad_blocks(data) {
        absorb_r(&mut state, &block, rounds);
    }
    squeeze(&state)
}

/// Extendable-output function (XOF) mode: squeeze `len` bytes.
pub fn xof(data: &[u8], len: usize) -> Vec<u8> {
    let mut state: State = [[0u64; 4]; LANES];
    for block in pad_blocks(data) {
        absorb(&mut state, &block);
    }
    let mut out = Vec::with_capacity(len);
    let mut remaining = len;
    while remaining > 0 {
        let chunk = squeeze(&state);
        let take = remaining.min(OUTPUT_BYTES);
        out.extend_from_slice(&chunk[..take]);
        remaining -= take;
        if remaining > 0 {
            permute(&mut state);
        }
    }
    out
}

// ── Toy 8-bit χ — for algebraic degree analysis (SC1) ────────────────────────

pub fn chi_word_toy(mut x: u8) -> u8 {
    x ^= x.rotate_left(1);
    x ^= x.rotate_left(3);
    x = x.wrapping_mul(0xD5);
    x ^= x >> 2;
    x = x.wrapping_mul(0x4F);
    x
}

/// Apply `rounds` iterations of chi_word_toy to x.
pub fn chi_toy_r(mut x: u8, rounds: usize) -> u8 {
    for _ in 0..rounds {
        x = chi_word_toy(x);
    }
    x
}
