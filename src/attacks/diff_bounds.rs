/// Formal upper bounds on differential probability for r-round HDH.
///
/// Uses empirical sampling with a fixed input difference (s1[0] = 1, single
/// bit flip in the first lane) to construct 95% confidence upper bounds via
/// the one-sided Wald interval.
///
/// For k == 0 successes: bound = 1 - 0.05^(1/n)  (exact inversion of
///   the one-sided binomial test at significance 0.05).
/// For k > 0: bound = p_hat + 1.645 * sqrt(p_hat*(1-p_hat)/n).

use crate::attacks::distinguisher::{apply_rounds, random_state};
use crate::state::State;
use rand::Rng;
use std::collections::HashMap;

// ── XOR helper ───────────────────────────────────────────────────────────────

/// XOR two states together, updating parity.
pub fn xor_states(a: &State, b: &State) -> State {
    let mut out = a.clone();
    for i in 0..25 {
        out.s1[i] ^= b.s1[i];
        out.s2[i] ^= b.s2[i];
        out.s3[i] ^= b.s3[i];
        out.s4[i] ^= b.s4[i];
    }
    for i in 0..25 {
        out.parity[i] = out.s1[i] ^ out.s2[i] ^ out.s3[i] ^ out.s4[i];
    }
    out
}

// ── Fingerprint ──────────────────────────────────────────────────────────────

/// Compute a fingerprint of a differential: XOR of s1[0..8] with rotated accumulation.
fn diff_fingerprint(a: &State, b: &State) -> u64 {
    let diff = xor_states(a, b);
    let mut fp = 0u64;
    for i in 0..8 {
        fp ^= diff.s1[i].rotate_left((i as u32) * 7);
    }
    fp
}

// ── Confidence upper bound ───────────────────────────────────────────────────

/// One-sided 95% confidence upper bound on a proportion.
/// k = observed successes, n = total samples.
fn confidence_upper_bound_95(k: usize, n: usize) -> f64 {
    if k == 0 {
        // Exact zero-count bound: 1 - alpha^(1/n) with alpha = 0.05
        1.0 - 0.05_f64.powf(1.0 / n as f64)
    } else {
        let p_hat = k as f64 / n as f64;
        (p_hat + 1.645 * (p_hat * (1.0 - p_hat) / n as f64).sqrt()).min(1.0)
    }
}

// ── Public structs ───────────────────────────────────────────────────────────

pub struct DiffProbBound {
    pub rounds: usize,
    pub samples: usize,
    /// Maximum hit count across all seen output differentials.
    pub max_count: usize,
    /// Maximum empirical probability: max_count / samples.
    pub empirical_max_prob: f64,
    /// 95% one-sided Wald upper bound on the max differential probability.
    pub upper_bound_95: f64,
    /// Log2 of the upper bound.
    pub upper_bound_log2: f64,
    /// Number of unique output differentials observed.
    pub unique_diffs: usize,
}

// ── Core computation ─────────────────────────────────────────────────────────

/// Estimate a 95% upper bound on the differential probability for `rounds`-round HDH
/// with a fixed input difference (s1[0] ^= 1).
pub fn compute_diff_bound(rounds: usize, samples: usize, rng: &mut impl Rng) -> DiffProbBound {
    let mut counts: HashMap<u64, usize> = HashMap::new();

    for _ in 0..samples {
        let base = random_state(rng);
        // Fixed input difference: flip bit 0 of lane 0 of share s1.
        let mut flipped = base.clone();
        flipped.s1[0] ^= 1u64;
        flipped.parity[0] = flipped.s1[0] ^ flipped.s2[0] ^ flipped.s3[0] ^ flipped.s4[0];

        let out_base = apply_rounds(base, rounds);
        let out_flipped = apply_rounds(flipped, rounds);

        let fp = diff_fingerprint(&out_base, &out_flipped);
        *counts.entry(fp).or_insert(0) += 1;
    }

    let max_count = counts.values().copied().max().unwrap_or(0);
    let unique_diffs = counts.len();
    let empirical_max_prob = max_count as f64 / samples as f64;
    let upper_bound_95 = confidence_upper_bound_95(max_count, samples);
    let upper_bound_log2 = upper_bound_95.log2();

    DiffProbBound {
        rounds,
        samples,
        max_count,
        empirical_max_prob,
        upper_bound_95,
        upper_bound_log2,
        unique_diffs,
    }
}

/// Sweep over round counts 1..=max_rounds, computing differential bounds.
pub fn diff_bound_sweep(max_rounds: usize, samples: usize, rng: &mut impl Rng) -> Vec<DiffProbBound> {
    (1..=max_rounds)
        .map(|r| compute_diff_bound(r, samples, rng))
        .collect()
}
