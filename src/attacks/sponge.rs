/// Sponge construction security analysis for the HDH permutation.
///
/// Models the 6400-bit HDH state as the underlying permutation of a sponge
/// construction, computing standard security bounds (Bertoni et al. 2011) and
/// mapping empirical attack results to round-count security claims.

use crate::attacks::distinguisher::{apply_rounds, random_state};
use rand::Rng;
use std::collections::HashMap;

// ── 1. SpongeParams / SpongeSecurityClaims ──────────────────────────────────

#[derive(Clone, Debug)]
pub struct SpongeParams {
    pub state_bits: usize,    // b = 6400
    pub rate_bits: usize,     // r
    pub capacity_bits: usize, // c = b - r
}

#[derive(Clone, Debug)]
pub struct SpongeSecurityClaims {
    pub params: SpongeParams,
    /// Collision resistance: 2^(c/2) → security = c/2 bits.
    pub collision_security_bits: f64,
    /// Preimage resistance: min(c/2, output_bits).
    pub preimage_security_bits: f64,
    /// Second preimage: same formula as preimage.
    pub second_preimage_security_bits: f64,
    /// Indifferentiability from ROM: 2^(c/2) → c/2 bits.
    pub indifferentiability_bits: f64,
    /// Throughput: r / b (fraction of state absorbed per call).
    pub throughput_fraction: f64,
    /// Security efficiency: collision_security_bits / state_bits.
    pub security_efficiency: f64,
}

pub fn compute_security(params: SpongeParams, output_bits: usize) -> SpongeSecurityClaims {
    let c = params.capacity_bits as f64;
    let b = params.state_bits as f64;
    let r = params.rate_bits as f64;
    let n = output_bits as f64;

    let collision_bits = c / 2.0;
    let preimage_bits = collision_bits.min(n);
    let throughput = r / b;
    let efficiency = collision_bits / b;

    SpongeSecurityClaims {
        collision_security_bits: collision_bits,
        preimage_security_bits: preimage_bits,
        second_preimage_security_bits: preimage_bits,
        indifferentiability_bits: collision_bits,
        throughput_fraction: throughput,
        security_efficiency: efficiency,
        params,
    }
}

// ── 2. SecuritySweep ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SecuritySweepEntry {
    pub rate_bits: usize,
    pub capacity_bits: usize,
    pub collision_bits: f64,
    pub throughput_fraction: f64,
    pub meets_128bit: bool, // collision_bits >= 128
    pub meets_256bit: bool, // collision_bits >= 256
}

#[derive(Debug)]
pub struct SecuritySweep {
    pub entries: Vec<SecuritySweepEntry>,
    /// Maximum rate r such that c/2 >= 128 (i.e., c >= 256 → r <= b - 256).
    pub min_rate_for_128bit: usize,
    /// Maximum rate r such that c/2 >= 256 (i.e., c >= 512 → r <= b - 512).
    pub min_rate_for_256bit: usize,
    pub state_bits: usize,
}

/// Candidate rate values to sweep.
const SWEEP_RATES: &[usize] = &[64, 128, 256, 512, 1024, 2048, 3200, 4096, 5120, 6144];

pub fn sweep_security_tradeoffs(state_bits: usize) -> SecuritySweep {
    let mut entries = Vec::with_capacity(SWEEP_RATES.len());
    let mut max_rate_128 = 0usize;
    let mut max_rate_256 = 0usize;

    for &r in SWEEP_RATES {
        if r >= state_bits {
            // Zero or negative capacity — skip.
            continue;
        }
        let c = state_bits - r;
        let col = c as f64 / 2.0;
        let meets_128 = col >= 128.0;
        let meets_256 = col >= 256.0;

        if meets_128 && r > max_rate_128 {
            max_rate_128 = r;
        }
        if meets_256 && r > max_rate_256 {
            max_rate_256 = r;
        }

        entries.push(SecuritySweepEntry {
            rate_bits: r,
            capacity_bits: c,
            collision_bits: col,
            throughput_fraction: r as f64 / state_bits as f64,
            meets_128bit: meets_128,
            meets_256bit: meets_256,
        });
    }

    SecuritySweep {
        entries,
        min_rate_for_128bit: max_rate_128,
        min_rate_for_256bit: max_rate_256,
        state_bits,
    }
}

// ── 3. RoundSecurityMap ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RoundAttackStatus {
    pub rounds: usize,
    pub degree_bound: &'static str,
    /// true = a distinguisher exists (structural weakness).
    pub integral_distinguisher: bool,
    /// true = forward/backward separable (MITM applicable).
    pub mitm_separable: bool,
    /// Measured log2 excess in biclique matching; 0 = random behaviour.
    pub biclique_excess_bits: f64,
    /// false = at least one attack succeeds; true = all known attacks fail.
    pub passes_security_bar: bool,
}

#[derive(Debug)]
pub struct RoundSecurityMap {
    pub rounds: Vec<RoundAttackStatus>,
    /// Smallest r where passes_security_bar is true for all r' >= r.
    pub min_secure_rounds: usize,
    /// Recommended deployment round count (min_secure × safety_margin).
    pub recommended_rounds: usize,
    pub safety_margin_factor: f64,
}

/// Hard-coded from the empirical attack suite results.
pub fn build_round_security_map() -> RoundSecurityMap {
    let rounds = vec![
        RoundAttackStatus {
            rounds: 1,
            degree_bound: "≤ 3",
            integral_distinguisher: true,
            mitm_separable: true,
            biclique_excess_bits: 8.9,
            passes_security_bar: false,
        },
        RoundAttackStatus {
            rounds: 2,
            degree_bound: "> 4",
            integral_distinguisher: false,
            mitm_separable: false,
            biclique_excess_bits: 0.5,
            passes_security_bar: true,
        },
        RoundAttackStatus {
            rounds: 3,
            degree_bound: "> 4 (estimated stronger)",
            integral_distinguisher: false,
            mitm_separable: false,
            biclique_excess_bits: 0.0,
            passes_security_bar: true,
        },
    ];

    let min_secure = rounds
        .iter()
        .find(|r| r.passes_security_bar)
        .map(|r| r.rounds)
        .unwrap_or(usize::MAX);

    let safety_margin = 2.0f64;
    let recommended = ((min_secure as f64 * safety_margin).ceil() as usize).max(min_secure + 1);

    RoundSecurityMap {
        rounds,
        min_secure_rounds: min_secure,
        recommended_rounds: recommended,
        safety_margin_factor: safety_margin,
    }
}

// ── 4. StatePartitionAnalysis ───────────────────────────────────────────────

#[derive(Debug)]
pub struct StatePartitionAnalysis {
    pub state_bits: usize,
    /// (security_bits, min_capacity_bits): capacity required for each target.
    pub security_capacity_pairs: Vec<(usize, usize)>,
    /// (security_bits, max_rate_bits, throughput_fraction).
    pub max_rate_per_security: Vec<(usize, usize, f64)>,
    /// Recommended capacity for 256-bit collision security.
    pub recommended_capacity: usize, // 512
    /// Recommended rate = state_bits - recommended_capacity.
    pub recommended_rate: usize,     // 5888
    /// Throughput = recommended_rate / state_bits.
    pub recommended_throughput: f64, // 5888/6400
}

pub fn analyze_state_partition(state_bits: usize) -> StatePartitionAnalysis {
    // min capacity = 2 × security_bits
    let security_levels: &[(usize, usize)] = &[(128, 256), (192, 384), (256, 512), (512, 1024)];

    let security_capacity_pairs: Vec<(usize, usize)> = security_levels.to_vec();

    let max_rate_per_security: Vec<(usize, usize, f64)> = security_levels
        .iter()
        .map(|&(sec, min_cap)| {
            let max_rate = if state_bits > min_cap { state_bits - min_cap } else { 0 };
            let throughput = max_rate as f64 / state_bits as f64;
            (sec, max_rate, throughput)
        })
        .collect();

    let recommended_capacity = 512usize;
    let recommended_rate = state_bits.saturating_sub(recommended_capacity);
    let recommended_throughput = recommended_rate as f64 / state_bits as f64;

    StatePartitionAnalysis {
        state_bits,
        security_capacity_pairs,
        max_rate_per_security,
        recommended_capacity,
        recommended_rate,
        recommended_throughput,
    }
}

// ── 5. BirthdayBoundCheck ───────────────────────────────────────────────────

#[derive(Debug)]
pub struct BirthdayBoundCheck {
    /// Number of output bits projected to (hash window).
    pub projection_bits: usize,
    pub samples: usize,
    pub actual_collisions: usize,
    /// Classic birthday formula: samples*(samples-1) / 2^(projection_bits+1).
    pub expected_collisions: f64,
    /// actual / expected.
    pub ratio: f64,
    /// Holds when actual <= 3 × expected (birthday bound is not exceeded).
    pub within_3x: bool,
}

/// Extract a `projection_bits`-wide window from the first lane of share 1 of
/// the output state.  Keeps the implementation simple and deterministic.
fn project_state(s: &crate::state::State, projection_bits: usize) -> u64 {
    let mask = if projection_bits >= 64 { u64::MAX } else { (1u64 << projection_bits) - 1 };
    s.s1[0] & mask
}

/// Run `samples` permutation evaluations (2 rounds — empirically secure),
/// count collisions in the `projection_bits`-bit output window, and compare
/// against the birthday-bound expectation.
pub fn check_birthday_bound(
    projection_bits: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> BirthdayBoundCheck {
    assert!(projection_bits <= 64, "projection_bits must be <= 64");

    let mut counts: HashMap<u64, usize> = HashMap::new();

    for _ in 0..samples {
        let s = random_state(rng);
        let out = apply_rounds(s, 2);
        let proj = project_state(&out, projection_bits);
        *counts.entry(proj).or_insert(0) += 1;
    }

    // Count pairs sharing the same projection value.
    let actual_collisions: usize = counts.values().map(|&c| c * c.saturating_sub(1) / 2).sum();

    // Expected collisions = C(N, 2) / 2^projection_bits.
    let expected = (samples as f64 * (samples as f64 - 1.0)) / (2.0 * (1u64 << projection_bits) as f64);

    let ratio = if expected > 0.0 { actual_collisions as f64 / expected } else { 0.0 };

    BirthdayBoundCheck {
        projection_bits,
        samples,
        actual_collisions,
        expected_collisions: expected,
        ratio,
        within_3x: actual_collisions as f64 <= 3.0 * expected.max(1.0),
    }
}
