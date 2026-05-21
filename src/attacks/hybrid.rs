/// Differential-linear hybrid and second-order (boomerang-style) analysis.
///
/// **Differential-linear correlation**: For a fixed input difference Δ and
/// output linear mask α, measure |Pr_{x}[parity(α·(χ(x)⊕χ(x⊕Δ))) = 0] − ½|.
/// If the attacker can split χ into a "differential" half and a "linear" half
/// with correlated intermediate values, this bias would be nonzero.  For a
/// high-degree nonlinear function it should fall below the statistical floor.
///
/// **Second-order (boomerang-style) differential bias**: measures the output
/// parity bias of the second-order derivative ∂²_{Δ₀,Δ₁}χ(x) =
/// χ(x)⊕χ(x⊕Δ₀)⊕χ(x⊕Δ₁)⊕χ(x⊕Δ₀⊕Δ₁).  For a degree-d function this
/// has degree d−2; at d≥32 it remains high-degree and the parity should be
/// unbiased.  A large bias would indicate a degree-2 "rectangular" structure
/// exploitable by boomerang-rectangle distinguishers.
use crate::algorithm::chi::chi_lane;
use rand::Rng;

fn bit_parity(x: u64) -> u64 {
    x.count_ones() as u64 & 1
}

fn lane_parity_4(
    m1: u64, m2: u64, m3: u64, m4: u64,
    v1: u64, v2: u64, v3: u64, v4: u64,
) -> u64 {
    bit_parity(m1 & v1) ^ bit_parity(m2 & v2) ^ bit_parity(m3 & v3) ^ bit_parity(m4 & v4)
}

// ── Differential-linear bias ───────────────────────────────────────────────

pub struct DiffLinStats {
    pub masks_tested: usize,
    pub samples_per_mask: usize,
    pub max_bias: f64,
    pub avg_bias: f64,
}

/// Sample differential-linear bias over `n_masks` random (Δ, α) pairs.
///
/// For each pair, compute |Pr[parity(α·(χ(x)⊕χ(x⊕Δ))) = 0] − ½| over
/// `samples` random inputs x.
pub fn sample_difflin_bias(n_masks: usize, samples: usize, rng: &mut impl Rng) -> DiffLinStats {
    let mut total_bias = 0.0f64;
    let mut max_bias = 0.0f64;

    for _ in 0..n_masks {
        let (da, db, dc, dd): (u64, u64, u64, u64) =
            (rng.gen(), rng.gen(), rng.gen(), rng.gen());
        let (m1, m2, m3, m4): (u64, u64, u64, u64) =
            (rng.gen(), rng.gen(), rng.gen(), rng.gen());

        let mut agree = 0usize;
        for _ in 0..samples {
            let (x1, x2, x3, x4): (u64, u64, u64, u64) =
                (rng.gen(), rng.gen(), rng.gen(), rng.gen());
            let (o1, o2, o3, o4) = chi_lane(x1, x2, x3, x4);
            let (p1, p2, p3, p4) = chi_lane(x1 ^ da, x2 ^ db, x3 ^ dc, x4 ^ dd);
            let diff_parity =
                lane_parity_4(m1, m2, m3, m4, o1 ^ p1, o2 ^ p2, o3 ^ p3, o4 ^ p4);
            if diff_parity == 0 {
                agree += 1;
            }
        }
        let bias = (agree as f64 / samples as f64 - 0.5).abs();
        if bias > max_bias {
            max_bias = bias;
        }
        total_bias += bias;
    }

    DiffLinStats {
        masks_tested: n_masks,
        samples_per_mask: samples,
        max_bias,
        avg_bias: total_bias / n_masks as f64,
    }
}

// ── Second-order (boomerang-style) differential ────────────────────────────

pub struct SecondOrderStats {
    /// Number of (Δ₀, Δ₁, α) triples tested.
    pub triples_tested: usize,
    pub samples_per_triple: usize,
    /// Max |bias| of parity(α · ∂²_{Δ₀,Δ₁}χ(x)) over all tested triples.
    pub max_bias: f64,
    pub avg_bias: f64,
}

/// Test for boomerang-rectangle exploitable structure via second-order
/// differential parity bias.
///
/// For each random (Δ₀, Δ₁, α) triple, compute
/// |Pr[parity(α·(χ(x)⊕χ(x⊕Δ₀)⊕χ(x⊕Δ₁)⊕χ(x⊕Δ₀⊕Δ₁))) = 0] − ½|.
/// For a function of degree ≥ 3 the second-order derivative is degree ≥ 1
/// and the parity should be unbiased.
pub fn test_second_order_differential(
    n_triples: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> SecondOrderStats {
    let mut total_bias = 0.0f64;
    let mut max_bias = 0.0f64;

    for _ in 0..n_triples {
        let (d0a, d0b, d0c, d0d): (u64, u64, u64, u64) =
            (rng.gen(), rng.gen(), rng.gen(), rng.gen());
        let (d1a, d1b, d1c, d1d): (u64, u64, u64, u64) =
            (rng.gen(), rng.gen(), rng.gen(), rng.gen());
        let (m1, m2, m3, m4): (u64, u64, u64, u64) =
            (rng.gen(), rng.gen(), rng.gen(), rng.gen());

        let mut agree = 0usize;
        for _ in 0..samples {
            let (x1, x2, x3, x4): (u64, u64, u64, u64) =
                (rng.gen(), rng.gen(), rng.gen(), rng.gen());
            let (a1, a2, a3, a4) = chi_lane(x1, x2, x3, x4);
            let (b1, b2, b3, b4) = chi_lane(x1 ^ d0a, x2 ^ d0b, x3 ^ d0c, x4 ^ d0d);
            let (c1, c2, c3, c4) = chi_lane(x1 ^ d1a, x2 ^ d1b, x3 ^ d1c, x4 ^ d1d);
            let (e1, e2, e3, e4) =
                chi_lane(x1 ^ d0a ^ d1a, x2 ^ d0b ^ d1b, x3 ^ d0c ^ d1c, x4 ^ d0d ^ d1d);
            // Second-order derivative: χ(x)⊕χ(x⊕Δ₀)⊕χ(x⊕Δ₁)⊕χ(x⊕Δ₀⊕Δ₁)
            let (s1, s2, s3, s4) = (
                a1 ^ b1 ^ c1 ^ e1,
                a2 ^ b2 ^ c2 ^ e2,
                a3 ^ b3 ^ c3 ^ e3,
                a4 ^ b4 ^ c4 ^ e4,
            );
            if lane_parity_4(m1, m2, m3, m4, s1, s2, s3, s4) == 0 {
                agree += 1;
            }
        }
        let bias = (agree as f64 / samples as f64 - 0.5).abs();
        if bias > max_bias {
            max_bias = bias;
        }
        total_bias += bias;
    }

    SecondOrderStats {
        triples_tested: n_triples,
        samples_per_triple: samples,
        max_bias,
        avg_bias: total_bias / n_triples as f64,
    }
}
