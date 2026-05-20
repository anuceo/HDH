/// Empirical differential uniformity analysis of the χ nonlinear core.
///
/// For a fixed input difference Δ = (Δx1,Δx2,Δx3,Δx4) we sample many random
/// inputs and record the distribution of output differences ΔG = quad(x)⊕quad(x⊕Δ).
/// A low maximum collision fraction means no differential shortcut exists.
use crate::chi::quad;
use rand::Rng;
use std::collections::HashMap;

pub struct DiffStats {
    pub input_diff: (u64, u64, u64, u64),
    pub samples: usize,
    pub unique_output_diffs: usize,
    pub max_collision_count: usize,
    /// max_collision_count / samples; ideal approaches 1/2^64 ≈ 0
    pub max_bias: f64,
}

pub fn analyze_differential(
    delta: (u64, u64, u64, u64),
    samples: usize,
    rng: &mut impl Rng,
) -> DiffStats {
    let mut counts: HashMap<u64, usize> = HashMap::new();
    let (da, db, dc, dd) = delta;

    for _ in 0..samples {
        let a: u64 = rng.gen();
        let b: u64 = rng.gen();
        let c: u64 = rng.gen();
        let d: u64 = rng.gen();
        let delta_g = quad(a, b, c, d) ^ quad(a ^ da, b ^ db, c ^ dc, d ^ dd);
        *counts.entry(delta_g).or_insert(0) += 1;
    }

    let max_count = counts.values().copied().max().unwrap_or(0);
    DiffStats {
        input_diff: delta,
        samples,
        unique_output_diffs: counts.len(),
        max_collision_count: max_count,
        max_bias: max_count as f64 / samples as f64,
    }
}

/// Returns the worst-case bias across a batch of random non-zero input differences.
pub fn max_differential_bias(
    diff_count: usize,
    samples_each: usize,
    rng: &mut impl Rng,
) -> f64 {
    let mut worst: f64 = 0.0;
    for _ in 0..diff_count {
        // Ensure at least one Δx component is nonzero.
        let delta = loop {
            let d = (rng.gen::<u64>(), rng.gen::<u64>(), rng.gen::<u64>(), rng.gen::<u64>());
            if d != (0, 0, 0, 0) {
                break d;
            }
        };
        let stats = analyze_differential(delta, samples_each, rng);
        if stats.max_bias > worst {
            worst = stats.max_bias;
        }
    }
    worst
}
