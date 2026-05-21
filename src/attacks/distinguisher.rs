/// Reduced-round distinguisher search for the HDH permutation.
///
/// For each round count r ∈ {1, 2, 3} the module tests whether the r-round
/// function is statistically distinguishable from a uniformly random permutation
/// on 6400-bit (4 × 25 × 64) inputs.
///
/// Four independent distinguishing strategies:
///
/// 1. **Avalanche completeness**: flip each of `n` randomly selected input bits
///    and measure the average fraction of the 6400 output bits that change.
///    For a random permutation every input bit independently flips ~50% of
///    output bits.  "Completeness" = fraction of input bits whose average change
///    fraction falls in [0.4, 0.6].  Grows from ~0 at round 1 (poor diffusion)
///    to ~1 at round 2–3 (complete mixing).
///
/// 2. **Output-bit balance**: over `n` random input states, measure how
///    balanced each of a sampled set of output bit positions is.
///    |Pr[bit = 1] − ½| should be ≤ noise floor (≈ 2/√samples) for an
///    indistinguishable function.
///
/// 3. **Full-state linear bias**: for random full-state linear mask pairs
///    (λ_in, λ_out), measure |Pr[parity(λ_in · in) = parity(λ_out · out)] − ½|.
///    Any nonzero bias is a linear distinguisher.
///
/// 4. **Chi4 zero-sum property** (exact algebraic, 4-bit model): for a Boolean
///    function of algebraic degree ≤ d, the XOR sum of outputs over any affine
///    subspace of dimension > d is zero.  chi4 has max degree 5 (from ANF
///    analysis), so dim-6 cosets must sum to 0.  Failures indicate a degree
///    miscalculation or structural error.

use crate::algorithm::state::State;
use rand::Rng;

// ── Internal helpers ────────────────────────────────────────────────────────

pub fn random_state(rng: &mut impl Rng) -> State {
    let mut s = State::zero();
    for i in 0..25 {
        s.s1[i] = rng.gen();
        s.s2[i] = rng.gen();
        s.s3[i] = rng.gen();
        s.s4[i] = rng.gen();
        s.parity[i] = s.s1[i] ^ s.s2[i] ^ s.s3[i] ^ s.s4[i];
    }
    s
}

/// Apply `n` consecutive rounds indexed 0, 1, …, n−1.
pub fn apply_rounds(state: State, n: usize) -> State {
    (0..n as u64).fold(state, |s, idx| crate::algorithm::round(s, idx))
}

/// Total bits across 4 shares × 25 lanes × 64 bits = 6400.
pub const STATE_BITS: usize = 4 * 25 * 64;

/// Extract bit `idx` (0 ≤ idx < STATE_BITS) from a state.
/// Layout: share 0 = s1 (bits 0..1599), share 1 = s2, 2 = s3, 3 = s4.
fn get_bit(s: &State, idx: usize) -> u64 {
    let (share, rem) = (idx / 1600, idx % 1600);
    let (lane, bp) = (rem / 64, rem % 64);
    let w = match share { 0 => s.s1[lane], 1 => s.s2[lane], 2 => s.s3[lane], _ => s.s4[lane] };
    (w >> bp) & 1
}

/// Flip bit `idx` (0 ≤ idx < STATE_BITS) and update the affected parity lane.
fn flip_bit(s: &mut State, idx: usize) {
    let (share, rem) = (idx / 1600, idx % 1600);
    let (lane, bp) = (rem / 64, rem % 64);
    let mask = 1u64 << bp;
    match share { 0 => s.s1[lane] ^= mask, 1 => s.s2[lane] ^= mask,
                  2 => s.s3[lane] ^= mask, _ => s.s4[lane] ^= mask }
    s.parity[lane] = s.s1[lane] ^ s.s2[lane] ^ s.s3[lane] ^ s.s4[lane];
}

/// Bit-level Hamming distance between two states (4 × 25 × 64 = 6400 bits).
fn hamming_dist(a: &State, b: &State) -> usize {
    (0..25)
        .map(|i| {
            (a.s1[i] ^ b.s1[i]).count_ones()
                + (a.s2[i] ^ b.s2[i]).count_ones()
                + (a.s3[i] ^ b.s3[i]).count_ones()
                + (a.s4[i] ^ b.s4[i]).count_ones()
        })
        .sum::<u32>() as usize
}

/// Full-state linear parity under a 4×25-word mask.
fn state_parity(s: &State, m: &[[u64; 25]; 4]) -> u64 {
    let mut p = 0u64;
    for i in 0..25 {
        p ^= (s.s1[i] & m[0][i]).count_ones() as u64
            ^ (s.s2[i] & m[1][i]).count_ones() as u64
            ^ (s.s3[i] & m[2][i]).count_ones() as u64
            ^ (s.s4[i] & m[3][i]).count_ones() as u64;
    }
    p & 1
}

// ── 1. Avalanche completeness ───────────────────────────────────────────────

pub struct AvalancheStats {
    pub round_count: usize,
    pub input_bits_tested: usize,
    pub samples_per_bit: usize,
    /// Per-bit average output-change fraction (length = input_bits_tested).
    pub per_bit_frac: Vec<f64>,
    /// Min / mean / max of per_bit_frac.
    pub min_frac: f64,
    pub mean_frac: f64,
    pub max_frac: f64,
    /// Fraction of input bits whose avg change fraction lies in [0.4, 0.6].
    /// 0.0 = no bit triggers ~50% change; 1.0 = all bits do (full avalanche).
    pub completeness: f64,
}

/// Measure avalanche completeness: for each of `n_bits` randomly selected
/// input bit positions, compute the mean Hamming-distance fraction over
/// `samples` random input states.
pub fn measure_avalanche(
    rounds: usize,
    n_bits: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> AvalancheStats {
    let bit_positions: Vec<usize> =
        (0..n_bits).map(|_| rng.gen_range(0..STATE_BITS)).collect();

    let mut per_bit_frac = Vec::with_capacity(n_bits);

    for &pos in &bit_positions {
        let total: usize = (0..samples)
            .map(|_| {
                let base = random_state(rng);
                let mut flip = base.clone();
                flip_bit(&mut flip, pos);
                let o_base = apply_rounds(base, rounds);
                let o_flip = apply_rounds(flip, rounds);
                hamming_dist(&o_base, &o_flip)
            })
            .sum();
        per_bit_frac.push(total as f64 / (samples * STATE_BITS) as f64);
    }

    let min_frac = per_bit_frac.iter().cloned().fold(1.0f64, f64::min);
    let max_frac = per_bit_frac.iter().cloned().fold(0.0f64, f64::max);
    let mean_frac = per_bit_frac.iter().sum::<f64>() / n_bits as f64;
    let completeness =
        per_bit_frac.iter().filter(|&&f| f >= 0.4 && f <= 0.6).count() as f64 / n_bits as f64;

    AvalancheStats {
        round_count: rounds,
        input_bits_tested: n_bits,
        samples_per_bit: samples,
        per_bit_frac,
        min_frac,
        mean_frac,
        max_frac,
        completeness,
    }
}

// ── 2. Output-bit balance ───────────────────────────────────────────────────

pub struct BalanceStats {
    pub round_count: usize,
    pub output_bits_sampled: usize,
    pub samples: usize,
    /// Maximum |Pr[bit = 1] − ½| across all sampled bits.
    pub max_abs_bias: f64,
    /// Mean |Pr[bit = 1] − ½| across all sampled bits.
    pub mean_abs_bias: f64,
    /// Statistical noise floor 2/√samples (2-sigma single-bit threshold).
    pub noise_floor: f64,
    /// Number of sampled bits whose |bias| exceeds the noise floor.
    pub bits_above_floor: usize,
}

/// Measure output-bit balance: over `samples` random inputs, compute
/// |Pr[bit = 1] − ½| for each of `n_bits` randomly selected output positions.
pub fn measure_output_balance(
    rounds: usize,
    n_bits: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> BalanceStats {
    let bit_positions: Vec<usize> =
        (0..n_bits).map(|_| rng.gen_range(0..STATE_BITS)).collect();

    let outputs: Vec<State> = (0..samples)
        .map(|_| apply_rounds(random_state(rng), rounds))
        .collect();

    let noise_floor = 2.0 / (samples as f64).sqrt();
    let mut biases: Vec<f64> = Vec::with_capacity(n_bits);

    for &pos in &bit_positions {
        let ones: usize = outputs.iter().map(|s| get_bit(s, pos) as usize).sum();
        biases.push((ones as f64 / samples as f64 - 0.5).abs());
    }

    let max_abs_bias = biases.iter().cloned().fold(0.0f64, f64::max);
    let mean_abs_bias = biases.iter().sum::<f64>() / n_bits as f64;
    let bits_above_floor = biases.iter().filter(|&&b| b > noise_floor).count();

    BalanceStats {
        round_count: rounds,
        output_bits_sampled: n_bits,
        samples,
        max_abs_bias,
        mean_abs_bias,
        noise_floor,
        bits_above_floor,
    }
}

// ── 3. Full-state linear bias ───────────────────────────────────────────────

pub struct LinearDistStats {
    pub round_count: usize,
    pub masks_tested: usize,
    pub samples: usize,
    pub max_bias: f64,
    pub avg_bias: f64,
    /// Statistical noise floor 1/√samples (1-sigma threshold).
    pub noise_floor: f64,
}

/// Measure full-state linear bias: for `n_masks` random (λ_in, λ_out) pairs,
/// compute |Pr[parity(λ_in·input) = parity(λ_out·output)] − ½|.
pub fn measure_linear_bias(
    rounds: usize,
    n_masks: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> LinearDistStats {
    let mut max_bias = 0.0f64;
    let mut total_bias = 0.0f64;

    for _ in 0..n_masks {
        let m_in: [[u64; 25]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.gen::<u64>()));
        let m_out: [[u64; 25]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.gen::<u64>()));

        let agree: usize = (0..samples)
            .filter(|_| {
                let s = random_state(rng);
                let out = apply_rounds(s.clone(), rounds);
                state_parity(&s, &m_in) == state_parity(&out, &m_out)
            })
            .count();

        let bias = (agree as f64 / samples as f64 - 0.5).abs();
        if bias > max_bias { max_bias = bias; }
        total_bias += bias;
    }

    LinearDistStats {
        round_count: rounds,
        masks_tested: n_masks,
        samples,
        max_bias,
        avg_bias: total_bias / n_masks as f64,
        noise_floor: 1.0 / (samples as f64).sqrt(),
    }
}

// ── 4. Chi4 zero-sum property (exact algebraic) ────────────────────────────

fn chi4(packed: u16) -> u16 {
    let x1 = (packed & 0xF) as u8;
    let x2 = ((packed >> 4) & 0xF) as u8;
    let x3 = ((packed >> 8) & 0xF) as u8;
    let x4 = ((packed >> 12) & 0xF) as u8;
    let g = (x1.wrapping_mul(x2) ^ x3.wrapping_mul(x4)) & 0xF;
    let o1 = (x1 ^ g) & 0xF;
    let o2 = (x2 ^ ((g << 1) | (g >> 3)) & 0xF) & 0xF;
    let o3 = (x3 ^ ((g << 2) | (g >> 2)) & 0xF) & 0xF;
    let o4 = (x4 ^ ((g << 3) | (g >> 1)) & 0xF) & 0xF;
    (o1 as u16) | ((o2 as u16) << 4) | ((o3 as u16) << 8) | ((o4 as u16) << 12)
}

pub struct ZeroSumStats {
    /// Dimension of the affine subspaces used (coset size = 2^dim).
    pub dim: usize,
    pub cosets_tested: usize,
    /// Number of cosets where XOR-sum of chi4 outputs ≠ 0.
    /// Must be 0 for dim > max_algebraic_degree.
    pub nonzero_sum_count: usize,
    /// XOR-sum of the zero-valued output (all zeros means all cosets passed).
    pub accumulated_xor: u16,
}

/// Test the zero-sum property of chi4 over `n_cosets` random `dim`-dimensional
/// affine cosets.  For chi4 of max algebraic degree 5, every coset with
/// dim ≥ 6 must yield XOR-sum = 0 over all 2^dim outputs.
///
/// Also serves as a negative control when called with dim = 5 (should have
/// some nonzero sums, confirming the function has degree > 4).
pub fn check_zero_sum_chi4(dim: usize, n_cosets: usize, rng: &mut impl Rng) -> ZeroSumStats {
    assert!(dim <= 15, "dim must be < 16 (coset must be a proper subset of GF(2)^16)");
    let coset_size = 1usize << dim;
    let mut nonzero = 0usize;
    let mut accumulated = 0u16;

    for _ in 0..n_cosets {
        // Random basis vectors in GF(2)^16; allow collisions — they reduce
        // effective dimension but cannot produce false nonzero sums.
        let basis: Vec<u16> = (0..dim).map(|_| rng.gen::<u16>()).collect();
        let offset: u16 = rng.gen();

        let xor_sum: u16 = (0..coset_size as u16)
            .map(|mask| {
                let x = basis
                    .iter()
                    .enumerate()
                    .fold(offset, |acc, (k, &bv)| if (mask >> k) & 1 == 1 { acc ^ bv } else { acc });
                chi4(x)
            })
            .fold(0u16, |acc, v| acc ^ v);

        if xor_sum != 0 { nonzero += 1; }
        accumulated ^= xor_sum;
    }

    ZeroSumStats { dim, cosets_tested: n_cosets, nonzero_sum_count: nonzero, accumulated_xor: accumulated }
}
