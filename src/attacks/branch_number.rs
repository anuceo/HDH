/// Branch number analysis for the θ (theta) linear diffusion layer of HDH.
///
/// The theta function acts as a GF(2) linear map on 25-bit activity vectors:
///   out[i] = in[i] ^ in[(i+24)%25] ^ in[(i+1)%25] ^ in[(i+18)%25] ^ in[(i+7)%25]
///
/// The branch number B(θ) = min over all nonzero x of (wt(x) + wt(θ(x))).
/// For a single active input lane (wt=1), exactly 5 output lanes are activated (wt=5),
/// giving B(θ) = 6.

use crate::attacks::distinguisher::{apply_rounds, random_state};
use crate::state::State;
use rand::Rng;

// ── Theta activity map ───────────────────────────────────────────────────────

/// Apply theta as a GF(2) linear activity map on a 25-bit vector.
/// Bit i of the output = bit i ^ bit (i+24)%25 ^ bit (i+1)%25 ^ bit (i+18)%25 ^ bit (i+7)%25
pub fn theta_activity(x: u32) -> u32 {
    let mut out = 0u32;
    for i in 0..25 {
        let xi = (x >> i) & 1;
        let x_prev = (x >> ((i + 24) % 25)) & 1;
        let x_next = (x >> ((i + 1) % 25)) & 1;
        let x_p7 = (x >> ((i + 7) % 25)) & 1;
        let x_p18 = (x >> ((i + 18) % 25)) & 1;
        let bit = xi ^ x_prev ^ x_next ^ x_p7 ^ x_p18;
        out |= bit << i;
    }
    out
}

// ── Exact branch number computation ─────────────────────────────────────────

pub struct ThetaBranchNumber {
    /// The branch number B(θ) = min_{x≠0}(wt(x) + wt(θ(x))).
    pub branch_number: usize,
    /// The input weight at which the minimum is achieved.
    pub achieved_at_input_weight: usize,
    /// The output weight at which the minimum is achieved.
    pub achieved_at_output_weight: usize,
    /// Total number of nonzero patterns checked (= 2^25 - 1).
    pub patterns_checked: u64,
}

/// Compute the exact branch number of θ by exhaustively checking all 2^25 − 1
/// nonzero 25-bit activity patterns.
pub fn theta_branch_number_exact() -> ThetaBranchNumber {
    let mut min_sum = usize::MAX;
    let mut best_in_wt = 0;
    let mut best_out_wt = 0;
    let patterns_checked: u64 = (1u64 << 25) - 1;

    for x in 1u32..(1u32 << 25) {
        let y = theta_activity(x);
        let wt_in = x.count_ones() as usize;
        let wt_out = y.count_ones() as usize;
        let sum = wt_in + wt_out;
        if sum < min_sum {
            min_sum = sum;
            best_in_wt = wt_in;
            best_out_wt = wt_out;
        }
    }

    ThetaBranchNumber {
        branch_number: min_sum,
        achieved_at_input_weight: best_in_wt,
        achieved_at_output_weight: best_out_wt,
        patterns_checked,
    }
}

// ── Branch number by input weight ────────────────────────────────────────────

/// For each input weight w from 1 to max_weight, compute the minimum output
/// weight and minimum branch sum.
/// Returns Vec of (input_wt, min_output_wt, min_sum).
pub fn theta_branch_by_weight() -> Vec<(usize, usize, usize)> {
    // Track per-weight minimums
    let mut min_out: [usize; 26] = [usize::MAX; 26];
    let mut min_sum: [usize; 26] = [usize::MAX; 26];

    for x in 1u32..(1u32 << 25) {
        let y = theta_activity(x);
        let wt_in = x.count_ones() as usize;
        let wt_out = y.count_ones() as usize;
        let sum = wt_in + wt_out;
        if wt_out < min_out[wt_in] {
            min_out[wt_in] = wt_out;
        }
        if sum < min_sum[wt_in] {
            min_sum[wt_in] = sum;
        }
    }

    (1..=25)
        .filter(|&w| min_out[w] != usize::MAX)
        .map(|w| (w, min_out[w], min_sum[w]))
        .collect()
}

// ── Sampled round branch number ───────────────────────────────────────────────

pub struct RoundBranchStats {
    pub rounds: usize,
    pub samples: usize,
    /// Minimum branch sum (wt_in + wt_out) observed.
    pub min_sum: usize,
    /// Average branch sum.
    pub avg_sum: f64,
    /// Minimum active input lane count observed.
    pub min_in: usize,
    /// Minimum active output lane count observed.
    pub min_out: usize,
}

/// Count the number of active lanes (lanes where any share XOR != 0).
pub fn active_lanes(a: &State, b: &State) -> usize {
    (0..25)
        .filter(|&i| {
            (a.s1[i] ^ b.s1[i]) != 0
                || (a.s2[i] ^ b.s2[i]) != 0
                || (a.s3[i] ^ b.s3[i]) != 0
                || (a.s4[i] ^ b.s4[i]) != 0
        })
        .count()
}

/// Sample the branch number of the r-round HDH permutation by generating
/// single-bit-flip differences and measuring active lanes before and after.
pub fn round_branch_sampled(rounds: usize, samples: usize, rng: &mut impl Rng) -> RoundBranchStats {
    let mut min_sum = usize::MAX;
    let mut min_in = usize::MAX;
    let mut min_out = usize::MAX;
    let mut total_sum = 0usize;

    for _ in 0..samples {
        // Generate a single-bit-flip difference: flip one bit in one lane of s1.
        let base = random_state(rng);
        let mut flipped = base.clone();
        // Pick a random lane and flip one bit within that lane.
        let lane: usize = rng.gen_range(0..25);
        let bit: u32 = rng.gen_range(0..64);
        flipped.s1[lane] ^= 1u64 << bit;
        flipped.parity[lane] = flipped.s1[lane] ^ flipped.s2[lane] ^ flipped.s3[lane] ^ flipped.s4[lane];

        let wt_in = active_lanes(&base, &flipped);

        let out_base = apply_rounds(base, rounds);
        let out_flipped = apply_rounds(flipped, rounds);
        let wt_out = active_lanes(&out_base, &out_flipped);

        let sum = wt_in + wt_out;
        if sum < min_sum { min_sum = sum; }
        if wt_in < min_in { min_in = wt_in; }
        if wt_out < min_out { min_out = wt_out; }
        total_sum += sum;
    }

    RoundBranchStats {
        rounds,
        samples,
        min_sum,
        avg_sum: total_sum as f64 / samples as f64,
        min_in,
        min_out,
    }
}
