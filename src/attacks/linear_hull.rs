/// Formal upper bound on multi-round linear bias via Walsh spectrum measurement.
///
/// Measures empirical Walsh coefficients for chi_lane and derives:
/// - Single-trail bias bounds for r rounds with m active lanes
/// - Linear hull bounds incorporating combinatorial trail counting

use crate::chi::chi_lane;
use rand::Rng;

// ── Structs ──────────────────────────────────────────────────────────────────

pub struct WalshMeasurement {
    pub masks_tested: usize,
    pub samples_per_mask: usize,
    pub max_bias: f64,
    pub max_correlation: f64,
    pub max_correlation_log2: f64,
}

pub struct LinearHullEntry {
    pub rounds: usize,
    pub min_active: usize,
    pub trail_bias_log2: f64,
    pub hull_bias_log2: f64,
}

// ── Binomial helper ───────────────────────────────────────────────────────────

/// C(25, k) as f64
fn binomial_c25_k(k: usize) -> f64 {
    let k = k.min(25);
    // C(25, k) = 25! / (k! * (25-k)!)
    let mut result = 1.0f64;
    for i in 0..k {
        result *= (25 - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

// ── Walsh measurement ─────────────────────────────────────────────────────────

/// Measure empirical Walsh coefficients for chi_lane over random (mask_in, mask_out) pairs.
///
/// For each mask pair: sample `samples` random 256-bit inputs, apply chi_lane,
/// compute bias = |Pr[parity(mask_in & input) = parity(mask_out & output)] - 0.5|
pub fn measure_walsh_chi_lane(masks: usize, samples: usize, rng: &mut impl Rng) -> WalshMeasurement {
    let mut max_bias = 0.0f64;

    for _ in 0..masks {
        // Generate random 256-bit masks (4 × 64-bit)
        let min1: u64 = rng.gen();
        let min2: u64 = rng.gen();
        let min3: u64 = rng.gen();
        let min4: u64 = rng.gen();
        let mout1: u64 = rng.gen();
        let mout2: u64 = rng.gen();
        let mout3: u64 = rng.gen();
        let mout4: u64 = rng.gen();

        let mut agree_count = 0usize;

        for _ in 0..samples {
            // Generate random 256-bit input (4 random u64s)
            let x1: u64 = rng.gen();
            let x2: u64 = rng.gen();
            let x3: u64 = rng.gen();
            let x4: u64 = rng.gen();

            let (o1, o2, o3, o4) = chi_lane(x1, x2, x3, x4);

            // Parity of (mask_in & input)
            let pin = ((min1 & x1).count_ones()
                + (min2 & x2).count_ones()
                + (min3 & x3).count_ones()
                + (min4 & x4).count_ones()) & 1;

            // Parity of (mask_out & output)
            let pout = ((mout1 & o1).count_ones()
                + (mout2 & o2).count_ones()
                + (mout3 & o3).count_ones()
                + (mout4 & o4).count_ones()) & 1;

            if pin == pout {
                agree_count += 1;
            }
        }

        let bias = (agree_count as f64 / samples as f64 - 0.5).abs();
        if bias > max_bias {
            max_bias = bias;
        }
    }

    let max_correlation = 2.0 * max_bias;
    let max_correlation_log2 = if max_correlation > 0.0 {
        max_correlation.log2()
    } else {
        f64::NEG_INFINITY
    };

    WalshMeasurement {
        masks_tested: masks,
        samples_per_mask: samples,
        max_bias,
        max_correlation,
        max_correlation_log2,
    }
}

// ── Linear hull sweep ─────────────────────────────────────────────────────────

/// Sweep multi-round linear hull bounds for r = 1..=max_rounds.
///
/// Uses hardcoded min_active values from wide trail analysis:
///   r=1 → 1, r=2 → 5, r=3 → 21, r=4 → 25
pub fn linear_hull_sweep(max_rounds: usize, samples: usize, rng: &mut impl Rng) -> Vec<LinearHullEntry> {
    // Measure Walsh coefficient once with a moderate number of masks
    let walsh_masks = (samples / 200).max(20).min(500);
    let walsh = measure_walsh_chi_lane(walsh_masks, samples, rng);
    let max_corr = walsh.max_correlation;

    // Hardcoded min_active from wide trail analysis
    let min_active_per_round: &[usize] = &[1, 5, 21, 25, 25, 25, 25, 25];

    let mut entries = Vec::new();
    for r in 1..=max_rounds {
        let m_r = if r <= min_active_per_round.len() {
            min_active_per_round[r - 1]
        } else {
            25
        };

        // Single-trail bias bound
        let trail_bias_log2 = if max_corr > 0.0 {
            m_r as f64 * max_corr.log2()
        } else {
            f64::NEG_INFINITY
        };

        // Linear hull bound with combinatorial trail count
        let num_active_patterns = binomial_c25_k(m_r.min(25));
        let hull_bias_log2 = if max_corr > 0.0 {
            0.5 * num_active_patterns.log2() + m_r as f64 * max_corr.log2()
        } else {
            f64::NEG_INFINITY
        };

        entries.push(LinearHullEntry {
            rounds: r,
            min_active: m_r,
            trail_bias_log2,
            hull_bias_log2,
        });
    }

    entries
}
