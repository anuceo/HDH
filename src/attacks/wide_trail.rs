/// Wide-trail strategy: track active S-box (lane) counts through r rounds
/// for minimal input differences.
///
/// The wide-trail argument gives a lower bound on the number of active S-boxes
/// over r rounds as (B-1)^r where B is the branch number of the linear layer.
/// For θ with B=6, the analytical bound is min(25, 5^r).

use crate::attacks::distinguisher::{apply_rounds, random_state};
use crate::state::State;
use rand::Rng;

/// Maximum log2 differential probability for a single active chi lane.
/// Empirically measured from HDH's chi differential uniformity.
pub const LOG2_P_MAX_CHI: f64 = -15.6;

// ── Active lane counting ─────────────────────────────────────────────────────

/// Count the number of lanes where any share XOR != 0.
pub fn active_lane_count(a: &State, b: &State) -> usize {
    (0..25)
        .filter(|&i| {
            (a.s1[i] ^ b.s1[i]) != 0
                || (a.s2[i] ^ b.s2[i]) != 0
                || (a.s3[i] ^ b.s3[i]) != 0
                || (a.s4[i] ^ b.s4[i]) != 0
        })
        .count()
}

// ── Analytical minimum ───────────────────────────────────────────────────────

/// Analytical minimum number of active lanes over `round` rounds,
/// using the wide-trail bound: min(25, (branch_number - 1)^round).
pub fn analytical_min_active(round: usize, branch_number: usize) -> usize {
    let base = (branch_number - 1) as u64;
    let bound = base.saturating_pow(round as u32) as usize;
    bound.min(25)
}

// ── Trail entry ──────────────────────────────────────────────────────────────

pub struct TrailEntry {
    pub rounds: usize,
    pub min_active: usize,
    pub avg_active: f64,
    /// Fraction of samples where all 25 lanes are active.
    pub full_activation_frac: f64,
    /// Implied log2 differential probability: min_active * LOG2_P_MAX_CHI.
    pub implied_log2_prob: f64,
}

// ── Wide-trail sweep ─────────────────────────────────────────────────────────

/// For each round count from 1 to max_rounds, measure the distribution of
/// active lane counts using single-bit-flip differences.
pub fn wide_trail_sweep(max_rounds: usize, samples: usize, rng: &mut impl Rng) -> Vec<TrailEntry> {
    let mut entries = Vec::with_capacity(max_rounds);

    for rounds in 1..=max_rounds {
        let mut min_active = 26usize;
        let mut total_active = 0usize;
        let mut full_count = 0usize;

        for _ in 0..samples {
            // Single-bit-flip difference in a random lane
            let base = random_state(rng);
            let mut flipped = base.clone();
            let lane: usize = rng.gen_range(0..25);
            let bit: u32 = rng.gen_range(0..64);
            flipped.s1[lane] ^= 1u64 << bit;
            flipped.parity[lane] = flipped.s1[lane]
                ^ flipped.s2[lane]
                ^ flipped.s3[lane]
                ^ flipped.s4[lane];

            let out_base = apply_rounds(base, rounds);
            let out_flipped = apply_rounds(flipped, rounds);
            let active = active_lane_count(&out_base, &out_flipped);

            if active < min_active { min_active = active; }
            total_active += active;
            if active == 25 { full_count += 1; }
        }

        if min_active == 26 { min_active = 0; }

        let avg_active = total_active as f64 / samples as f64;
        let full_activation_frac = full_count as f64 / samples as f64;
        let implied_log2_prob = min_active as f64 * LOG2_P_MAX_CHI;

        entries.push(TrailEntry {
            rounds,
            min_active,
            avg_active,
            full_activation_frac,
            implied_log2_prob,
        });
    }

    entries
}
