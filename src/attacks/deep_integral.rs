/// Phase 1A — Automatic Integral Distinguishers for HDH.
///
/// Searches for integral structure (balanced sums, zero-sum cubes, partial
/// integrals, high-order derivatives) across 1–5 rounds and cube dimensions
/// 1–7.  Confirms the "2-round closure" hypothesis from integral.rs:
///
///   Round 1: significant integral structure — dim-4 cubes XOR to zero (≥93%)
///             because the algebraic degree is ≤ 3.
///   Round 2: sharp collapse — dim-4 cubes no longer zero-sum (degree > 4).
///   Round 3+: indistinguishable from uniform random permutation.
///
/// Key invariant: for a Boolean function of degree d, D^k F = 0 identically
/// when k > d.  "Integral structure at dimension k" means zero_sum_fraction
/// is HIGH (close to 1.0), which requires k > degree(F).
///
/// Four components:
/// 1. test_affine_subspace    – randomised affine subspace XOR-sum survey.
/// 2. find_integral_dimension – finds the minimum cube dimension where
///                              integral structure first appears.
/// 3. measure_derivative_collapse – tracks zero_frac vs. order; reports the
///                              annihilation order (first order with high zf).
/// 4. sweep_integral_persistence – joint sweep over rounds × dimensions.

use crate::attacks::distinguisher::{apply_rounds, random_state, STATE_BITS};
use crate::state::State;
use rand::Rng;

// ── internal helpers ─────────────────────────────────────────────────────────

/// XOR two states together (pointwise).
fn xor_states(a: &State, b: &State) -> State {
    State {
        s1: std::array::from_fn(|i| a.s1[i] ^ b.s1[i]),
        s2: std::array::from_fn(|i| a.s2[i] ^ b.s2[i]),
        s3: std::array::from_fn(|i| a.s3[i] ^ b.s3[i]),
        s4: std::array::from_fn(|i| a.s4[i] ^ b.s4[i]),
        parity: std::array::from_fn(|i| a.parity[i] ^ b.parity[i]),
    }
}

/// Hamming weight of a state (total set bits across all 4 shares + parity).
fn state_hw(s: &State) -> u32 {
    s.s1.iter().chain(s.s2.iter()).chain(s.s3.iter()).chain(s.s4.iter())
        .map(|w| w.count_ones()).sum()
}

/// Check whether a state is the all-zero state.
fn state_is_zero(s: &State) -> bool { state_hw(s) == 0 }

/// Compute the XOR sum of F over the k-dimensional affine subspace
///   { base ⊕ (e_{i1} ⊕ … ⊕ e_{ij}) : J ⊆ {0,…,k-1} }
/// i.e. the k-th order derivative D^k_basis F(base).
///
/// `basis` must have length k ≤ 7 (2^7 = 128 evaluations per call).
fn xor_sum_over_subspace(base: &State, basis: &[State], rounds: usize) -> State {
    assert!(basis.len() <= 7, "subspace dimension ≤ 7 for tractability");
    let k = basis.len();
    let n = 1usize << k;
    let mut acc = State::zero();
    for mask in 0..n {
        let mut pt = base.clone();
        for (bit, dir) in basis.iter().enumerate() {
            if (mask >> bit) & 1 == 1 {
                pt = xor_states(&pt, dir);
            }
        }
        let out = apply_rounds(pt, rounds);
        acc = xor_states(&acc, &out);
    }
    acc
}

/// Generate a random non-zero basis vector (single random bit set).
fn random_basis_vector(rng: &mut impl Rng) -> State {
    let mut v = State::zero();
    let bit_idx = rng.gen_range(0..STATE_BITS);
    let share = bit_idx / 1600;
    let rem   = bit_idx % 1600;
    let lane  = rem / 64;
    let bp    = rem % 64;
    let mask  = 1u64 << bp;
    match share {
        0 => v.s1[lane] ^= mask,
        1 => v.s2[lane] ^= mask,
        2 => v.s3[lane] ^= mask,
        _ => v.s4[lane] ^= mask,
    }
    v
}

/// Fraction of STATE_BITS output bits that are zero in the XOR sum.
/// 1.0 = all-zero (perfect integral balance); 0.5 = random-like.
fn balanced_bit_fraction(xor_sum: &State) -> f64 {
    let hw = state_hw(xor_sum) as f64;
    1.0 - hw / STATE_BITS as f64
}

// ── 1. AffineSubspaceStats ───────────────────────────────────────────────────

/// Statistics from randomised affine-subspace XOR-sum survey.
#[derive(Debug)]
pub struct AffineSubspaceStats {
    pub rounds: usize,
    pub dim: usize,
    /// Number of independently chosen (base, basis) pairs tested.
    pub n_subspaces: usize,
    /// Fraction of subspaces where XOR sum is the all-zero state.
    /// HIGH (≥ 0.5) indicates integral structure (degree < dim).
    pub zero_sum_fraction: f64,
    /// Average fraction of output bits equal to zero across all XOR sums.
    pub balanced_bit_fraction: f64,
    /// Mean Hamming weight of the XOR sums.
    pub hw_mean: f64,
    /// Standard deviation of the Hamming weights.
    pub hw_std: f64,
    /// Expected hw_mean under uniformly random output (STATE_BITS / 2).
    pub expected_hw_random: f64,
    /// Normalised deficit: (expected_hw_random − hw_mean) / expected_hw_random.
    /// Positive and large → XOR sums have far lower HW than random → structured.
    pub normalised_hw_deficit: f64,
}

/// Survey `n_subspaces` randomly chosen k-dimensional affine subspaces of the
/// HDH input space, evaluate the XOR sum (k-th order derivative) after
/// `rounds` rounds, and collect statistics.
pub fn test_affine_subspace(
    rounds: usize,
    dim: usize,
    n_subspaces: usize,
    rng: &mut impl Rng,
) -> AffineSubspaceStats {
    assert!(dim >= 1 && dim <= 7, "dim must be 1–7");
    let mut zero_count = 0usize;
    let mut hw_sum = 0.0f64;
    let mut hw_sq = 0.0f64;
    let mut bal_sum = 0.0f64;

    for _ in 0..n_subspaces {
        let base = random_state(rng);
        let basis: Vec<State> = (0..dim).map(|_| random_basis_vector(rng)).collect();
        let xs = xor_sum_over_subspace(&base, &basis, rounds);
        if state_is_zero(&xs) { zero_count += 1; }
        let hw = state_hw(&xs) as f64;
        hw_sum += hw;
        hw_sq  += hw * hw;
        bal_sum += balanced_bit_fraction(&xs);
    }

    let n = n_subspaces as f64;
    let hw_mean = hw_sum / n;
    let hw_var  = (hw_sq / n) - hw_mean * hw_mean;
    let hw_std  = hw_var.max(0.0).sqrt();
    let expected = STATE_BITS as f64 / 2.0;

    AffineSubspaceStats {
        rounds,
        dim,
        n_subspaces,
        zero_sum_fraction: zero_count as f64 / n,
        balanced_bit_fraction: bal_sum / n,
        hw_mean,
        hw_std,
        expected_hw_random: expected,
        normalised_hw_deficit: (expected - hw_mean) / expected,
    }
}

// ── 2. IntegralDimResult (greedy cube finder) ────────────────────────────────

/// Result of the greedy integral-dimension search.
#[derive(Debug)]
pub struct IntegralDimResult {
    pub rounds: usize,
    /// Minimum dimension where zero_sum_fraction ≥ 0.5 (integral structure).
    /// Equals degree(F) + 1 for a degree-d function.
    /// None if no structured dimension is found within max_dim.
    pub integral_start_dim: Option<usize>,
    /// Upper bound on algebraic degree: integral_start_dim − 1, or max_dim if None.
    pub degree_upper_bound: usize,
    /// Same as degree_upper_bound (alias used in reporting).
    pub cube_annihilation_depth: usize,
    /// zero_sum_fraction at each tested dimension 1..=max_dim.
    pub dim_fractions: Vec<f64>,
}

/// Greedy search: for each dimension d from 1 to `max_dim`, measure the
/// fraction of randomly chosen d-dimensional affine subspaces whose XOR sum
/// is zero.
///
/// Returns the MINIMUM dimension where this fraction first reaches ≥ 0.5
/// (the algebraic "annihilation point" = degree + 1).  For round 1 HDH
/// (degree ≤ 3) this should be dim = 4.  For round 2+ it is None.
pub fn find_integral_dimension(
    rounds: usize,
    max_dim: usize,
    n_trials: usize,
    rng: &mut impl Rng,
) -> IntegralDimResult {
    let max_dim = max_dim.min(7);
    let mut dim_fractions = Vec::with_capacity(max_dim);

    for d in 1..=max_dim {
        let stats = test_affine_subspace(rounds, d, n_trials, rng);
        dim_fractions.push(stats.zero_sum_fraction);
    }

    // First dim where fraction ≥ 0.5 = algebraic annihilation point.
    let integral_start_dim = dim_fractions.iter()
        .position(|&f| f >= 0.5)
        .map(|i| i + 1);  // 0-indexed → 1-indexed

    let degree_upper_bound = integral_start_dim
        .map(|d| d - 1)
        .unwrap_or(max_dim);

    IntegralDimResult {
        rounds,
        integral_start_dim,
        degree_upper_bound,
        cube_annihilation_depth: degree_upper_bound,
        dim_fractions,
    }
}

// ── 3. DerivativeCollapseStats ───────────────────────────────────────────────

/// Per-order statistics for the derivative collapse measurement.
#[derive(Debug)]
pub struct DerivOrderStats {
    pub order: usize,
    /// Fraction of samples where D^order F = 0 exactly.
    /// LOW (≈ 0) = derivative non-zero; HIGH (≈ 1) = annihilated.
    pub zero_frac: f64,
    pub hw_mean: f64,
    pub hw_std: f64,
}

/// Statistics from adaptive high-order derivative collapse measurement.
#[derive(Debug)]
pub struct DerivativeCollapseStats {
    pub rounds: usize,
    pub per_order: Vec<DerivOrderStats>,
    /// First order where zero_frac ≥ 0.9 (derivative annihilated).
    /// Equals degree(F) + 1 for functions of degree ≤ max_order.
    pub annihilation_order: Option<usize>,
    /// Binary entropy H(zero_frac) at each order (bits, 0–1).
    /// Peaks at the transition from non-zero to zero.
    pub derivative_entropy: Vec<f64>,
}

/// Measure how XOR sums evolve as derivative order increases from 1 to
/// `max_order`.  For each order, independently sample `n_samples` random
/// (base, basis) pairs and record zero_frac / hw statistics.
pub fn measure_derivative_collapse(
    rounds: usize,
    max_order: usize,
    n_samples: usize,
    rng: &mut impl Rng,
) -> DerivativeCollapseStats {
    let max_order = max_order.min(7);
    let mut per_order = Vec::with_capacity(max_order);
    let mut derivative_entropy = Vec::with_capacity(max_order);

    for order in 1..=max_order {
        let stats = test_affine_subspace(rounds, order, n_samples, rng);
        let zf = stats.zero_sum_fraction;
        // Binary entropy H(p) = -p log2 p - (1-p) log2(1-p).
        let ent = if zf <= 0.0 || zf >= 1.0 {
            0.0
        } else {
            -zf * zf.log2() - (1.0 - zf) * (1.0 - zf).log2()
        };
        per_order.push(DerivOrderStats {
            order,
            zero_frac: zf,
            hw_mean: stats.hw_mean,
            hw_std: stats.hw_std,
        });
        derivative_entropy.push(ent);
    }

    // Annihilation order: first where zero_frac ≥ 0.9 (= D^k F ≈ 0 always).
    let annihilation_order = per_order.iter()
        .find(|s| s.zero_frac >= 0.9)
        .map(|s| s.order);

    DerivativeCollapseStats { rounds, per_order, annihilation_order, derivative_entropy }
}

// ── 4. IntegralPersistenceSweep ──────────────────────────────────────────────

/// Single (rounds, dim) data point in the persistence sweep.
#[derive(Debug)]
pub struct PersistenceEntry {
    pub rounds: usize,
    pub dim: usize,
    pub zero_sum_fraction: f64,
    pub balanced_bit_fraction: f64,
    /// True when zero_sum_fraction ≥ 0.5 (= structural integral behaviour).
    pub integral_persists: bool,
}

/// Joint sweep of integral structure over rounds × dimensions.
#[derive(Debug)]
pub struct IntegralPersistenceSweep {
    pub entries: Vec<PersistenceEntry>,
    /// First round where EVERY tested dimension shows random-like behaviour
    /// (zero_sum_fraction < 0.5 for all dims).
    pub closure_round: Option<usize>,
}

/// Sweep `max_rounds` × `dims`, measuring integral structure at each point.
/// `n_subspaces` subspaces are sampled per (rounds, dim) pair.
pub fn sweep_integral_persistence(
    max_rounds: usize,
    dims: &[usize],
    n_subspaces: usize,
    rng: &mut impl Rng,
) -> IntegralPersistenceSweep {
    let mut entries = Vec::new();

    for r in 1..=max_rounds {
        for &d in dims {
            if d > 7 { continue; }
            let stats = test_affine_subspace(r, d, n_subspaces, rng);
            entries.push(PersistenceEntry {
                rounds: r,
                dim: d,
                zero_sum_fraction: stats.zero_sum_fraction,
                balanced_bit_fraction: stats.balanced_bit_fraction,
                integral_persists: stats.zero_sum_fraction >= 0.5,
            });
        }
    }

    // Closure round: first r where all tested dims show random-like behaviour.
    let closure_round = (1..=max_rounds).find(|&r| {
        entries.iter()
            .filter(|e| e.rounds == r)
            .all(|e| !e.integral_persists)
    });

    IntegralPersistenceSweep { entries, closure_round }
}
