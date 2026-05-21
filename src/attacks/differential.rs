/// Empirical differential uniformity analysis of the χ nonlinear core.
///
/// For a fixed input difference Δ = (Δx1,Δx2,Δx3,Δx4) we sample many random
/// inputs and record the distribution of output differences ΔG = quad(x)⊕quad(x⊕Δ).
/// A low maximum collision fraction means no differential shortcut exists.
use crate::algorithm::chi::{quad, chi_lane};
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

// ── Linear bias (from linear.rs) ──────────────────────────────────────────────

/// Empirical linear-bias (Walsh coefficient) analysis of the full χ lane.
///
/// For linear masks λ = (λ1,λ2,λ3,λ4) on input and μ = (μ1,μ2,μ3,μ4) on
/// output, we measure |Pr[parity(λ·in) = parity(μ·out)] − ½|.  A perfectly
/// nonlinear function has zero bias for all non-trivial masks.

pub struct LinearBiasStats {
    pub masks_tested: usize,
    pub samples_per_mask: usize,
    pub max_bias: f64,
    pub avg_bias: f64,
}

fn bit_parity(x: u64) -> u64 {
    x.count_ones() as u64 & 1
}

fn lane_parity(l1: u64, l2: u64, l3: u64, l4: u64, x1: u64, x2: u64, x3: u64, x4: u64) -> u64 {
    bit_parity(l1 & x1) ^ bit_parity(l2 & x2) ^ bit_parity(l3 & x3) ^ bit_parity(l4 & x4)
}

pub fn analyze_linear_bias(
    masks_in: &[(u64, u64, u64, u64)],
    masks_out: &[(u64, u64, u64, u64)],
    samples: usize,
    rng: &mut impl Rng,
) -> LinearBiasStats {
    assert_eq!(masks_in.len(), masks_out.len());
    let mut biases = Vec::with_capacity(masks_in.len());

    for (&(l1, l2, l3, l4), &(m1, m2, m3, m4)) in masks_in.iter().zip(masks_out.iter()) {
        let mut agree = 0usize;
        for _ in 0..samples {
            let (x1, x2, x3, x4): (u64, u64, u64, u64) =
                (rng.gen(), rng.gen(), rng.gen(), rng.gen());
            let (o1, o2, o3, o4) = chi_lane(x1, x2, x3, x4);
            let p_in = lane_parity(l1, l2, l3, l4, x1, x2, x3, x4);
            let p_out = lane_parity(m1, m2, m3, m4, o1, o2, o3, o4);
            if p_in == p_out {
                agree += 1;
            }
        }
        biases.push((agree as f64 / samples as f64 - 0.5).abs());
    }

    let max_bias = biases.iter().cloned().fold(0.0f64, f64::max);
    let avg_bias = biases.iter().sum::<f64>() / biases.len() as f64;
    LinearBiasStats {
        masks_tested: masks_in.len(),
        samples_per_mask: samples,
        max_bias,
        avg_bias,
    }
}

/// Generates `count` random non-trivial mask pairs and runs the bias test.
pub fn sample_linear_bias(count: usize, samples: usize, rng: &mut impl Rng) -> LinearBiasStats {
    let mut mi: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(count);
    let mut mo: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(count);
    for _ in 0..count {
        mi.push((rng.gen(), rng.gen(), rng.gen(), rng.gen()));
        mo.push((rng.gen(), rng.gen(), rng.gen(), rng.gen()));
    }
    analyze_linear_bias(&mi, &mo, samples, rng)
}

// ── Jacobian rank (from jacobian.rs) ──────────────────────────────────────────

/// GF(2) Jacobian rank of the full 64-bit χ lane.
///
/// The Jacobian J of a vector Boolean function f: GF(2)^n → GF(2)^m at a
/// point x is the m×n matrix where J[i,j] = f_i(x⊕eⱼ) ⊕ f_i(x).  For
/// chi_lane this is a 256×256 matrix over GF(2).
///
/// rank(J) ≈ n means the local linearization covers all output dimensions —
/// no solvable linear shortcut exists at that point.
/// rank(J) << n means an attacker could exploit the rank deficiency to recover
/// inputs from outputs with linear-algebraic work.

/// 256-bit row (4 × u64 words).
type Row256 = [u64; 4];

fn bit_is_set(row: &Row256, col: usize) -> bool {
    (row[col / 64] >> (col % 64)) & 1 == 1
}

fn gf2_rank_256(mut rows: Vec<Row256>) -> usize {
    let n = rows.len();
    let mut pivot = 0usize;
    for col in 0..256 {
        let found = (pivot..n).find(|&r| bit_is_set(&rows[r], col));
        if let Some(r) = found {
            rows.swap(pivot, r);
            let prow: Row256 = rows[pivot]; // Copy (Row256 is Copy)
            for r2 in 0..n {
                if r2 != pivot && bit_is_set(&rows[r2], col) {
                    rows[r2][0] ^= prow[0];
                    rows[r2][1] ^= prow[1];
                    rows[r2][2] ^= prow[2];
                    rows[r2][3] ^= prow[3];
                }
            }
            pivot += 1;
        }
    }
    pivot
}

/// Compute rank(J_chi(x)) for a specific input point.
/// Requires 257 evaluations of chi_lane.
pub fn jacobian_rank_at_point(x1: u64, x2: u64, x3: u64, x4: u64) -> usize {
    let (b0, b1, b2, b3) = chi_lane(x1, x2, x3, x4);

    let rows: Vec<Row256> = (0..256)
        .map(|bit| {
            let mask = 1u64 << (bit % 64);
            let (nx1, nx2, nx3, nx4) = match bit / 64 {
                0 => (x1 ^ mask, x2, x3, x4),
                1 => (x1, x2 ^ mask, x3, x4),
                2 => (x1, x2, x3 ^ mask, x4),
                _ => (x1, x2, x3, x4 ^ mask),
            };
            let (f0, f1, f2, f3) = chi_lane(nx1, nx2, nx3, nx4);
            [b0 ^ f0, b1 ^ f1, b2 ^ f2, b3 ^ f3]
        })
        .collect();

    gf2_rank_256(rows)
}

pub struct JacobianStats {
    pub points_tested: usize,
    pub min_rank: usize,
    pub avg_rank: f64,
    pub max_theoretical_rank: usize,
    /// max_theoretical_rank − min_rank.  Zero = full rank at every tested point.
    pub worst_rank_deficit: usize,
}

pub fn analyze_jacobian_rank(points: usize, rng: &mut impl Rng) -> JacobianStats {
    let ranks: Vec<usize> = (0..points)
        .map(|_| jacobian_rank_at_point(rng.gen(), rng.gen(), rng.gen(), rng.gen()))
        .collect();

    let min_rank = *ranks.iter().min().unwrap();
    let avg_rank = ranks.iter().sum::<usize>() as f64 / ranks.len() as f64;

    JacobianStats {
        points_tested: points,
        min_rank,
        avg_rank,
        max_theoretical_rank: 256,
        worst_rank_deficit: 256 - min_rank,
    }
}
