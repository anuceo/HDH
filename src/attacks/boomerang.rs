/// Boomerang-style second-order differential analysis for reduced-round HDH.
///
/// Probes the second-order differential (boomerang sum)
///   D²F(x; α, β) = F(x) ⊕ F(x⊕α) ⊕ F(x⊕β) ⊕ F(x⊕α⊕β)
///
/// For a degree-d function D²F has degree ≤ d−2.
/// - 1-round HDH (deg ≤ 3): D²F has degree ≤ 1 → constant in x for fixed (α,β);
///   Hamming weight shows structured, sub-random values.
/// - 2-round HDH (deg > 4): D²F has degree ≥ 2 → pseudorandom HW near 3200.
///
/// Four categories:
/// 1. Boomerang sum HW distribution (avg, min, max, stddev, fraction low).
/// 2. Projected boomerang excess (k-bit projection zero-count vs random).
/// 3. Structured-difference boomerang (single-bit α vs random α HW comparison).
/// 4. Boomerang-rectangle probability (quartet matching count vs random).

use crate::attacks::distinguisher::{apply_rounds, random_state, STATE_BITS};
use crate::state::State;
use rand::Rng;

// ── Local helpers ────────────────────────────────────────────────────────────

fn xor_states(a: &State, b: &State) -> State {
    let mut out = State::zero();
    for i in 0..25 {
        out.s1[i] = a.s1[i] ^ b.s1[i];
        out.s2[i] = a.s2[i] ^ b.s2[i];
        out.s3[i] = a.s3[i] ^ b.s3[i];
        out.s4[i] = a.s4[i] ^ b.s4[i];
        out.parity[i] = out.s1[i] ^ out.s2[i] ^ out.s3[i] ^ out.s4[i];
    }
    out
}

fn hamming_weight_state(s: &State) -> usize {
    (0..25)
        .map(|i| {
            s.s1[i].count_ones()
                + s.s2[i].count_ones()
                + s.s3[i].count_ones()
                + s.s4[i].count_ones()
        })
        .sum::<u32>() as usize
}

fn flip_bit(s: &mut State, idx: usize) {
    let (share, rem) = (idx / 1600, idx % 1600);
    let (lane, bp) = (rem / 64, rem % 64);
    let mask = 1u64 << bp;
    match share {
        0 => s.s1[lane] ^= mask,
        1 => s.s2[lane] ^= mask,
        2 => s.s3[lane] ^= mask,
        _ => s.s4[lane] ^= mask,
    }
    s.parity[lane] = s.s1[lane] ^ s.s2[lane] ^ s.s3[lane] ^ s.s4[lane];
}

fn get_bit(s: &State, idx: usize) -> u8 {
    let (share, rem) = (idx / 1600, idx % 1600);
    let (lane, bp) = (rem / 64, rem % 64);
    let w = match share {
        0 => s.s1[lane],
        1 => s.s2[lane],
        2 => s.s3[lane],
        _ => s.s4[lane],
    };
    ((w >> bp) & 1) as u8
}

fn project(s: &State, positions: &[usize]) -> u64 {
    debug_assert!(positions.len() <= 64);
    positions.iter().enumerate().fold(0u64, |acc, (bit, &pos)| {
        acc | ((get_bit(s, pos) as u64) << bit)
    })
}

/// Compute the boomerang sum D²F(x; α, β) over `total_rounds` applied rounds.
fn boomerang_sum(x: &State, alpha: &State, beta: &State, total_rounds: usize) -> State {
    let x_a = xor_states(x, alpha);
    let x_b = xor_states(x, beta);
    let x_ab = xor_states(&x_a, beta);

    let fx = apply_rounds(x.clone(), total_rounds);
    let fa = apply_rounds(x_a, total_rounds);
    let fb = apply_rounds(x_b, total_rounds);
    let fab = apply_rounds(x_ab, total_rounds);

    // D²F = F(x) ⊕ F(x⊕α) ⊕ F(x⊕β) ⊕ F(x⊕α⊕β)
    xor_states(&xor_states(&fx, &fa), &xor_states(&fb, &fab))
}

// ── Category 1: Boomerang sum HW distribution ────────────────────────────────

pub struct BoomerangSumStats {
    pub rounds_forward: usize,
    pub rounds_backward: usize,
    pub samples: usize,
    pub avg_hw: f64,
    pub min_hw: usize,
    pub max_hw: usize,
    pub hw_std_dev: f64,
    /// Fraction of samples with HW < STATE_BITS/4 (below 1600).
    pub frac_low_hw: f64,
    /// Random-permutation expected HW (STATE_BITS/2).
    pub expected_hw: f64,
}

/// Measure HW distribution of the boomerang sum over `samples` random (x, α, β).
pub fn test_boomerang_sum(
    rounds_forward: usize,
    rounds_backward: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> BoomerangSumStats {
    let total = rounds_forward + rounds_backward;
    let low_threshold = STATE_BITS / 4;
    let expected = STATE_BITS as f64 / 2.0;

    let mut hws: Vec<usize> = Vec::with_capacity(samples);

    for _ in 0..samples {
        let x = random_state(rng);
        let alpha = random_state(rng);
        let beta = random_state(rng);
        let s = boomerang_sum(&x, &alpha, &beta, total);
        hws.push(hamming_weight_state(&s));
    }

    let avg_hw = hws.iter().sum::<usize>() as f64 / samples as f64;
    let min_hw = *hws.iter().min().unwrap_or(&0);
    let max_hw = *hws.iter().max().unwrap_or(&0);
    let variance = hws.iter().map(|&h| {
        let d = h as f64 - avg_hw;
        d * d
    }).sum::<f64>() / samples as f64;
    let hw_std_dev = variance.sqrt();
    let frac_low_hw = hws.iter().filter(|&&h| h < low_threshold).count() as f64 / samples as f64;

    BoomerangSumStats {
        rounds_forward,
        rounds_backward,
        samples,
        avg_hw,
        min_hw,
        max_hw,
        hw_std_dev,
        frac_low_hw,
        expected_hw: expected,
    }
}

// ── Category 2: Projected boomerang excess ───────────────────────────────────

pub struct ProjectedBoomerangStats {
    pub total_rounds: usize,
    pub projection_bits: usize,
    pub samples: usize,
    /// Observed fraction of samples where k-bit projection = 0.
    pub zero_frac: f64,
    /// Expected fraction for a random function: 1/2^k.
    pub expected_zero_frac: f64,
    /// log2(observed / expected); positive = more zeros than random (structure).
    pub log2_excess: f64,
}

/// Project the boomerang sum to `k` random bit positions and count zero projections.
pub fn test_projected_boomerang(
    total_rounds: usize,
    projection_bits: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> ProjectedBoomerangStats {
    assert!(projection_bits <= 64);

    let positions: Vec<usize> = (0..projection_bits)
        .map(|_| rng.gen_range(0..STATE_BITS))
        .collect();

    let expected_zero_frac = 1.0 / (1u64 << projection_bits) as f64;

    let mut zero_count = 0usize;
    for _ in 0..samples {
        let x = random_state(rng);
        let alpha = random_state(rng);
        let beta = random_state(rng);
        let s = boomerang_sum(&x, &alpha, &beta, total_rounds);
        if project(&s, &positions) == 0 {
            zero_count += 1;
        }
    }

    let zero_frac = zero_count as f64 / samples as f64;
    let log2_excess = if zero_frac > 0.0 && expected_zero_frac > 0.0 {
        (zero_frac / expected_zero_frac).log2()
    } else if zero_count == 0 {
        f64::NEG_INFINITY
    } else {
        0.0
    };

    ProjectedBoomerangStats {
        total_rounds,
        projection_bits,
        samples,
        zero_frac,
        expected_zero_frac,
        log2_excess,
    }
}

// ── Category 3: Structured-difference boomerang ──────────────────────────────

pub struct StructuredBoomerangStats {
    pub total_rounds: usize,
    pub samples: usize,
    /// Average HW of D²F when α is a single-bit flip.
    pub avg_hw_single_bit_alpha: f64,
    /// Average HW of D²F when α is a fully random state.
    pub avg_hw_random_alpha: f64,
    /// Reduction in avg HW: random_alpha − single_bit_alpha.
    /// Positive → single-bit input differences produce smaller boomerang sums.
    pub hw_reduction: f64,
}

/// Compare boomerang sum HW for single-bit α vs fully random α.
///
/// Lane-local χ structure means single-bit differences propagate with fewer
/// cross-lane interactions, potentially reducing the HW of D²F at low rounds.
pub fn test_structured_boomerang(
    total_rounds: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> StructuredBoomerangStats {
    let mut hw_single = 0usize;
    let mut hw_random = 0usize;

    for _ in 0..samples {
        let x = random_state(rng);
        let beta = random_state(rng);

        // Single-bit α: flip one random bit in an otherwise zero state.
        let mut alpha_single = State::zero();
        let bit_pos = rng.gen_range(0..STATE_BITS);
        flip_bit(&mut alpha_single, bit_pos);

        // Random α.
        let alpha_random = random_state(rng);

        let s_single = boomerang_sum(&x, &alpha_single, &beta, total_rounds);
        let s_random = boomerang_sum(&x, &alpha_random, &beta, total_rounds);

        hw_single += hamming_weight_state(&s_single);
        hw_random += hamming_weight_state(&s_random);
    }

    let avg_hw_single_bit_alpha = hw_single as f64 / samples as f64;
    let avg_hw_random_alpha = hw_random as f64 / samples as f64;

    StructuredBoomerangStats {
        total_rounds,
        samples,
        avg_hw_single_bit_alpha,
        avg_hw_random_alpha,
        hw_reduction: avg_hw_random_alpha - avg_hw_single_bit_alpha,
    }
}

// ── Category 4: Boomerang-rectangle probability ──────────────────────────────

pub struct BoomerangRectStats {
    pub rounds_forward: usize,
    pub n_left: usize,
    pub n_right: usize,
    /// Observed matching quartet count.
    pub matching_quartets: usize,
    /// Expected under a random permutation: n_left * n_right / 2^STATE_BITS.
    pub expected_quartets: f64,
    /// log2(observed+1 / expected+1); positive → more matches than random.
    pub log2_excess: f64,
}

/// Count boomerang-rectangle quartet matches.
///
/// Fix a random α.  For `n_left` "left" inputs x_i and `n_right` "right"
/// inputs x'_j, compute the forward-half images under `rounds_forward` rounds:
///   y_i = F0(x_i),  y_iα = F0(x_i ⊕ α)
///   y'_j = F0(x'_j), y'_jα = F0(x'_j ⊕ α)
///
/// A matching quartet (i, j) satisfies y_i ⊕ y'_j = y_iα ⊕ y'_jα, which
/// simplifies to δ_i = δ_j where δ_i = y_i ⊕ y_iα.
///
/// Count pairs (i, j) with equal intermediate differences δ_i = δ_j projected
/// to `proj_bits` bit positions (to keep comparison tractable).
pub fn test_boomerang_rect(
    rounds_forward: usize,
    n_left: usize,
    n_right: usize,
    proj_bits: usize,
    rng: &mut impl Rng,
) -> BoomerangRectStats {
    assert!(proj_bits <= 64);

    let alpha = random_state(rng);
    let positions: Vec<usize> = (0..proj_bits)
        .map(|_| rng.gen_range(0..STATE_BITS))
        .collect();

    // Compute projected intermediate differences for the left set.
    let left_deltas: Vec<u64> = (0..n_left)
        .map(|_| {
            let x = random_state(rng);
            let x_a = xor_states(&x, &alpha);
            let y = apply_rounds(x, rounds_forward);
            let y_a = apply_rounds(x_a, rounds_forward);
            project(&xor_states(&y, &y_a), &positions)
        })
        .collect();

    // Compute projected intermediate differences for the right set.
    let right_deltas: Vec<u64> = (0..n_right)
        .map(|_| {
            let x = random_state(rng);
            let x_a = xor_states(&x, &alpha);
            let y = apply_rounds(x, rounds_forward);
            let y_a = apply_rounds(x_a, rounds_forward);
            project(&xor_states(&y, &y_a), &positions)
        })
        .collect();

    // Count matching pairs: δ_i == δ_j.
    // Use a frequency map for efficiency.
    let mut freq: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for &d in &left_deltas {
        *freq.entry(d).or_insert(0) += 1;
    }
    let matching_quartets: usize = right_deltas
        .iter()
        .map(|d| freq.get(d).copied().unwrap_or(0))
        .sum();

    // Expected matches for a random permutation: each of n_left * n_right pairs
    // has probability 1/2^proj_bits of matching (random deltas agree on k bits).
    let expected_quartets = (n_left * n_right) as f64 / (1u64 << proj_bits) as f64;

    let log2_excess = ((matching_quartets + 1) as f64 / (expected_quartets + 1.0)).log2();

    BoomerangRectStats {
        rounds_forward,
        n_left,
        n_right,
        matching_quartets,
        expected_quartets,
        log2_excess,
    }
}
