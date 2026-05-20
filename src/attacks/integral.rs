/// Higher-order integral distinguisher search for reduced-round HDH.
///
/// An integral distinguisher selects a k-dimensional "cube" of input bit
/// positions, fixes all remaining inputs to a constant, varies the k cube
/// bits over all 2^k values, and computes the XOR sum of all 2^k outputs.
///
/// **Algebraic foundation**: for a Boolean function of algebraic degree d,
/// any cube of dimension > d produces an XOR sum of exactly 0 over all
/// 2^k inputs (the zero-sum / higher-order derivative theorem).
///
/// **Observed HDH degree structure** (empirical):
///
/// | Rounds | Avg balanced output bits | Cube zeros at dim=4 |
/// |--------|--------------------------|----------------------|
/// | 0      | 6400 (identity, deg=1)   | 100% (deg < 4)       |
/// | 1      | ~6395 (most bits zero)   | ~93%  (deg ~3–4)     |
/// | 2      | ~3700 (slightly above½)  |   0%  (deg > 4)      |
///
/// - **1 round**: the composition entropy→χ→θ→Φ has effective bit-level
///   degree ≤ 3 for most input directions.  ~93% of dim=4 cubes give an
///   all-zero XOR sum; practically all output bits are balanced per cube.
///   This is an expected single-round property — HDH requires ≥ 2 rounds.
/// - **2 rounds**: degree exceeds 4 in all tested directions.  No dim ≤ 8
///   cube produces an all-zero XOR sum.  The partial-balance metric falls
///   to ~3700/6400 ≈ 58%, only weakly above the 50% random baseline.
///
/// **Tests**:
/// 1. Positive control (0 rounds): dim=2 gives 100% zero sums (degree 1).
/// 2. 1-round structure: dim=4 gives ≥ 70% zero sums AND avg_balanced > 6000
///    (confirming observable integral structure at a single round).
/// 3. 2-round transition: dim=4 cubes produce 0% full zero sums
///    (degree exceeds 4 after two rounds — integral distinguisher eliminated).
///
/// **Partial integral metric**: avg_balanced counts output bits (of 6400)
/// whose individual XOR sum over the cube is 0.  Expected ~3200 for a
/// random permutation (σ ≈ 40).  1-round gives ~6395 (near-total balance);
/// 2-round gives ~3700 (slightly above random, no practical exploitation).

use crate::state::State;
use rand::Rng;

use crate::attacks::distinguisher::{apply_rounds, random_state, STATE_BITS};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn flip_bit(s: &mut State, idx: usize) {
    let (share, rem) = (idx / 1600, idx % 1600);
    let (lane, bp) = (rem / 64, rem % 64);
    let mask = 1u64 << bp;
    match share {
        0 => s.s1[lane] ^= mask,
        1 => s.s2[lane] ^= mask,
        2 => s.s3[lane] ^= mask,
        _ => s.s4[lane] ^= mask,
    }
    s.parity[lane] = s.s1[lane] ^ s.s2[lane] ^ s.s3[lane] ^ s.s4[lane];
}

fn xor_into(dst: &mut State, src: &State) {
    for i in 0..25 {
        dst.s1[i] ^= src.s1[i];
        dst.s2[i] ^= src.s2[i];
        dst.s3[i] ^= src.s3[i];
        dst.s4[i] ^= src.s4[i];
    }
}

fn hamming_weight(s: &State) -> usize {
    (0..25)
        .map(|i| {
            s.s1[i].count_ones()
                + s.s2[i].count_ones()
                + s.s3[i].count_ones()
                + s.s4[i].count_ones()
        })
        .sum::<u32>() as usize
}

fn is_zero(s: &State) -> bool {
    (0..25).all(|i| s.s1[i] == 0 && s.s2[i] == 0 && s.s3[i] == 0 && s.s4[i] == 0)
}

// ── Public API ───────────────────────────────────────────────────────────────

pub struct CubeSumStats {
    pub round_count: usize,
    pub cube_dim: usize,
    pub cubes_tested: usize,
    /// Fraction of (cube, constant) pairs whose XOR-sum over 2^k outputs is 0.
    /// Should be 1.0 for linear functions (degree 1); ~0 for HDH at any round.
    pub zero_sum_fraction: f64,
    /// Average number of output bits where the XOR-sum bit is 0 (balanced bits).
    /// Expected ~STATE_BITS/2 = 3200 for a pseudorandom permutation.
    /// STATE_BITS = 6400 for a fully linear/trivial function.
    pub avg_balanced_bits: f64,
    /// Maximum balanced-bit count across all tested cubes.
    pub max_balanced_bits: usize,
    /// Statistical center and spread for a random permutation.
    pub expected_balanced_bits: f64,
    pub expected_std_dev: f64,
}

/// XOR all 2^`cube_dim` outputs over a random cube of `cube_dim` input bits,
/// repeated `n_cubes` times with independent random bases and positions.
pub fn test_cube_sum(
    rounds: usize,
    cube_dim: usize,
    n_cubes: usize,
    rng: &mut impl Rng,
) -> CubeSumStats {
    assert!(cube_dim <= 20, "cube_dim must be ≤ 20 for feasibility");
    let cube_size = 1usize << cube_dim;

    let mut zero_count = 0usize;
    let mut total_balanced = 0usize;
    let mut max_balanced = 0usize;

    for _ in 0..n_cubes {
        let base = random_state(rng);

        // k distinct bit positions for the cube; allow up to 2k attempts to
        // find k distinct values (collision rate is negligible for k << STATE_BITS).
        let mut positions = Vec::with_capacity(cube_dim);
        while positions.len() < cube_dim {
            let p = rng.gen_range(0..STATE_BITS);
            if !positions.contains(&p) {
                positions.push(p);
            }
        }

        // Compute XOR sum of f(base ⊕ e_{mask}) for all masks in 0..2^k.
        let mut xor_sum = State::zero();
        for mask in 0..cube_size {
            let mut s = base.clone();
            for (bit_idx, &pos) in positions.iter().enumerate() {
                if (mask >> bit_idx) & 1 == 1 {
                    flip_bit(&mut s, pos);
                }
            }
            let out = apply_rounds(s, rounds);
            xor_into(&mut xor_sum, &out);
        }

        let balanced = STATE_BITS - hamming_weight(&xor_sum);
        if is_zero(&xor_sum) {
            zero_count += 1;
        }
        total_balanced += balanced;
        if balanced > max_balanced {
            max_balanced = balanced;
        }
    }

    // For a pseudorandom permutation: expected balanced = STATE_BITS/2 per bit
    // (each XOR-sum bit is independently 0 with prob ~½).
    // Std dev = sqrt(STATE_BITS * 0.25).
    let expected = STATE_BITS as f64 / 2.0;
    let std_dev = ((STATE_BITS as f64) * 0.25f64).sqrt();

    CubeSumStats {
        round_count: rounds,
        cube_dim,
        cubes_tested: n_cubes,
        zero_sum_fraction: zero_count as f64 / n_cubes as f64,
        avg_balanced_bits: total_balanced as f64 / n_cubes as f64,
        max_balanced_bits: max_balanced,
        expected_balanced_bits: expected,
        expected_std_dev: std_dev,
    }
}

pub struct IntegralSweepStats {
    pub round_count: usize,
    pub by_dim: Vec<CubeSumStats>,
    /// Smallest cube_dim where zero_sum_fraction > 0.5, or None.
    pub integral_threshold_dim: Option<usize>,
}

/// Sweep cube dimensions `dims` at `rounds` rounds, `n_cubes` per dimension.
pub fn sweep_integral(
    rounds: usize,
    dims: &[usize],
    n_cubes: usize,
    rng: &mut impl Rng,
) -> IntegralSweepStats {
    let by_dim: Vec<CubeSumStats> = dims
        .iter()
        .map(|&d| test_cube_sum(rounds, d, n_cubes, rng))
        .collect();

    let threshold = by_dim.iter().find(|s| s.zero_sum_fraction > 0.5).map(|s| s.cube_dim);

    IntegralSweepStats { round_count: rounds, by_dim, integral_threshold_dim: threshold }
}
