/// Empirical linear-bias (Walsh coefficient) analysis of the full χ lane.
///
/// For linear masks λ = (λ1,λ2,λ3,λ4) on input and μ = (μ1,μ2,μ3,μ4) on
/// output, we measure |Pr[parity(λ·in) = parity(μ·out)] − ½|.  A perfectly
/// nonlinear function has zero bias for all non-trivial masks.
use crate::chi::chi_lane;
use rand::Rng;

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
