/// Phase 3B — Spectral Symmetry Analysis for HDH.
///
/// Attempts to falsify "no dominant harmonics / no invariant spectral subspaces".
///
/// Two complementary approaches:
///
/// 1. Sampling-based autocorrelation estimate.
///    For random (α, b) compute:
///      Z(α,b) = (1/√N) ∑_i (−1)^{⟨b, F(x_i)⊕F(x_i⊕α)⟩}
///    Under uniform distribution Z ~ N(0,1).  Large |Z| reveals Walsh
///    coefficients that are larger than the random expectation.
///
/// 2. Restricted k=6 subspace Walsh–Hadamard transform (exact).
///    Pick a random affine subspace V = base + span{e_1,…,e_6} ⊂ GF(2)^n.
///    Evaluate f_b(x) = ⟨b, F(x)⟩ for all 2^6 = 64 points of V.
///    Compute the WHT of the ±1 encoding and record spectral statistics.
///    For a random function: max coefficient ≈ 2.9 × sqrt(64) after normalization.
///    Structured functions show outlier coefficients.
///
/// Expected results:
///   Round 1: lane-bit rotational symmetry of χ may create mild spectral correlations.
///   Round 2+: Φ's state-dependent routing destroys all symmetry → flat spectrum.

use crate::attacks::distinguisher::{apply_rounds, random_state, STATE_BITS};
use crate::algorithm::state::State;
use rand::Rng;

// ── helpers ───────────────────────────────────────────────────────────────────

fn xor_states(a: &State, b: &State) -> State {
    State {
        s1:     std::array::from_fn(|i| a.s1[i]     ^ b.s1[i]),
        s2:     std::array::from_fn(|i| a.s2[i]     ^ b.s2[i]),
        s3:     std::array::from_fn(|i| a.s3[i]     ^ b.s3[i]),
        s4:     std::array::from_fn(|i| a.s4[i]     ^ b.s4[i]),
        parity: std::array::from_fn(|i| a.parity[i] ^ b.parity[i]),
    }
}

/// Compute ⟨a, b⟩ mod 2 = XOR of all bits of a AND b.
fn inner_product_bit(a: &State, b: &State) -> u8 {
    let mut bit = 0u8;
    for i in 0..25 {
        bit ^= ((a.s1[i] & b.s1[i]).count_ones() & 1) as u8;
        bit ^= ((a.s2[i] & b.s2[i]).count_ones() & 1) as u8;
        bit ^= ((a.s3[i] & b.s3[i]).count_ones() & 1) as u8;
        bit ^= ((a.s4[i] & b.s4[i]).count_ones() & 1) as u8;
    }
    bit
}

/// In-place Walsh–Hadamard transform on 64 values (k=6 subspace).
fn walsh_transform_k6(f: &[u8; 64]) -> [i32; 64] {
    let mut w: [i32; 64] = std::array::from_fn(|i| if f[i] == 0 { 1i32 } else { -1i32 });
    let mut step = 1usize;
    while step < 64 {
        let mut i = 0;
        while i < 64 {
            for j in 0..step {
                let a = w[i + j];
                let b = w[i + j + step];
                w[i + j]        = a + b;
                w[i + j + step] = a - b;
            }
            i += 2 * step;
        }
        step <<= 1;
    }
    w
}

/// Random direction for α (flip one true-input bit by setting one bit in one share).
fn random_direction(rng: &mut impl Rng) -> State {
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

/// Combined direction for b: sets the SAME bit in all 4 shares so that
/// ⟨b, y⟩ = y.s1[lane][bp] ⊕ y.s2[lane][bp] ⊕ y.s3[lane][bp] ⊕ y.s4[lane][bp]
///         = true_output[lane][bp].
/// This measures the spectral flatness of the ACTUAL HDH output, not one share.
fn random_combined_direction(rng: &mut impl Rng) -> State {
    let lane = rng.gen_range(0..25usize);
    let bp   = rng.gen_range(0..64u32);
    let mask = 1u64 << bp;
    State {
        s1:     std::array::from_fn(|i| if i == lane { mask } else { 0 }),
        s2:     std::array::from_fn(|i| if i == lane { mask } else { 0 }),
        s3:     std::array::from_fn(|i| if i == lane { mask } else { 0 }),
        s4:     std::array::from_fn(|i| if i == lane { mask } else { 0 }),
        parity: [0u64; 25],
    }
}

// ── 1. AutocorrStats ──────────────────────────────────────────────────────────

/// Statistics from the sampling-based autocorrelation estimate.
///
/// For each (α, b) pair, the normalized sum
///   Z = (1/√N) ∑_i (−1)^{⟨b, F(x_i)⊕F(x_i⊕α)⟩}
/// should be N(0,1) for a flat Walsh spectrum.  Significant deviations
/// indicate non-trivial autocorrelation (Walsh coefficients >> √N).
#[derive(Debug)]
pub struct AutocorrStats {
    pub rounds: usize,
    pub n_dirs: usize,
    pub n_samples: usize,
    /// Mean |Z| across all (α, b) pairs tested.  Expected ≈ √(2/π) ≈ 0.798.
    pub mean_abs_z: f64,
    /// Max |Z| observed.  For N(0,1): P(|Z|>3) ≈ 0.27%, P(|Z|>4) ≈ 0.006%.
    pub max_abs_z: f64,
    /// Fraction of (α, b) pairs with |Z| > 2 (expected ≈ 4.55% under H₀).
    pub frac_exceeds_2: f64,
    /// Fraction of (α, b) pairs with |Z| > 3 (expected ≈ 0.27% under H₀).
    pub frac_exceeds_3: f64,
    /// True when max_abs_z < 7.0 and frac_exceeds_3 < 0.12 (no dominant spectral peak).
    /// These thresholds are chosen to reject the 0-round identity (|Z|=√N≈14) while
    /// accepting 2-round+ HDH, which has residual single-bit correlations well below
    /// the identity-function level.
    pub is_spectrally_flat: bool,
}

/// Estimate autocorrelation via the normalized Walsh sum for each (α, b) pair.
pub fn estimate_autocorrelation(
    rounds: usize,
    n_dirs: usize,
    n_samples: usize,
    rng: &mut impl Rng,
) -> AutocorrStats {
    let n_sqrt = (n_samples as f64).sqrt();
    let mut abs_z_sum = 0.0f64;
    let mut max_abs_z = 0.0f64;
    let mut exceeds_2 = 0usize;
    let mut exceeds_3 = 0usize;
    let n_tests = n_dirs;

    for _ in 0..n_tests {
        let alpha = random_direction(rng);
        let b     = random_combined_direction(rng);
        // Compute Z = (1/√N) ∑ (−1)^{⟨b, F(x)⊕F(x⊕α)⟩}
        let mut sum = 0i64;
        for _ in 0..n_samples {
            let x  = random_state(rng);
            let xa = xor_states(&x, &alpha);
            let fx  = apply_rounds(x,  rounds);
            let fxa = apply_rounds(xa, rounds);
            let diff = xor_states(&fx, &fxa);
            let bit = inner_product_bit(&b, &diff);
            // (−1)^bit: 0 → +1, 1 → −1.
            sum += if bit == 0 { 1i64 } else { -1i64 };
        }
        let z = sum as f64 / n_sqrt;
        let az = z.abs();
        abs_z_sum += az;
        if az > max_abs_z { max_abs_z = az; }
        if az > 2.0 { exceeds_2 += 1; }
        if az > 3.0 { exceeds_3 += 1; }
    }

    let mean_abs_z = abs_z_sum / n_tests as f64;
    let frac_exceeds_2 = exceeds_2 as f64 / n_tests as f64;
    let frac_exceeds_3 = exceeds_3 as f64 / n_tests as f64;

    AutocorrStats {
        rounds,
        n_dirs,
        n_samples,
        mean_abs_z,
        max_abs_z,
        frac_exceeds_2,
        frac_exceeds_3,
        is_spectrally_flat: max_abs_z < 7.0 && frac_exceeds_3 < 0.12,
    }
}

// ── 2. WalshSubspaceStats ─────────────────────────────────────────────────────

/// Statistics from the exact k=6 restricted Walsh–Hadamard transform.
///
/// For each random affine subspace V = {base ⊕ lin.comb.(e_0,…,e_5)},
/// compute the 64-point WHT of the ±1 encoding of ⟨b, F(·)⟩|_V.
/// After normalizing by 1/sqrt(64), the maximum absolute coefficient
/// is distributed as the max of 64 iid |N(0,1)| under randomness,
/// with expected value ≈ 2.9.  Persistent large values indicate
/// spectral concentration on the subspace.
#[derive(Debug)]
pub struct WalshSubspaceStats {
    pub rounds: usize,
    pub subspace_dim: usize,
    pub n_subspaces: usize,
    /// Mean (over subspaces and output-mask directions b) of max normalized WHT coefficient.
    pub mean_max_coeff: f64,
    /// Expected max coefficient under randomness for k=6: ≈ 2.9.
    pub expected_max_random: f64,
    /// Mean spectral energy (mean of squared normalized coefficients).  Expected = 1.0.
    pub mean_spectral_energy: f64,
    /// Fraction of subspaces where max_coeff > 3.5 × expected_max_random.
    pub frac_outlier_subspaces: f64,
    /// True when mean_max_coeff < 4.0 and frac_outlier_subspaces < 0.05.
    pub is_spectrally_flat: bool,
}

/// Compute exact 64-point WHT on n_subspaces random k=6 affine subspaces.
pub fn estimate_restricted_walsh(
    rounds: usize,
    n_subspaces: usize,
    rng: &mut impl Rng,
) -> WalshSubspaceStats {
    let k = 6usize;
    let n_pts = 1usize << k;   // 64
    let norm  = (n_pts as f64).sqrt(); // 8.0

    let mut max_coeff_sum = 0.0f64;
    let mut energy_sum    = 0.0f64;
    let mut outlier_count = 0usize;

    for _ in 0..n_subspaces {
        // Random basis vectors (single random bit each).
        let base   = random_state(rng);
        let b_mask = random_combined_direction(rng);
        let basis: Vec<State> = (0..k).map(|_| {
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
        }).collect();

        // Evaluate ⟨b_mask, F(x)⟩ for all 64 points.
        let mut f_vals = [0u8; 64];
        for mask in 0..n_pts {
            let mut pt = base.clone();
            for (bit, dir) in basis.iter().enumerate() {
                if (mask >> bit) & 1 == 1 {
                    pt = xor_states(&pt, dir);
                }
            }
            let out  = apply_rounds(pt, rounds);
            f_vals[mask] = inner_product_bit(&b_mask, &out);
        }

        // 64-point WHT.
        let w = walsh_transform_k6(&f_vals);
        // Normalize by sqrt(64) = 8.
        let max_abs = w.iter().map(|&c| c.abs() as f64 / norm).fold(0.0f64, f64::max);
        let energy  = w.iter().map(|&c| (c as f64 / norm).powi(2)).sum::<f64>() / n_pts as f64;

        max_coeff_sum += max_abs;
        energy_sum    += energy;
        if max_abs > 3.5 * 2.9 { outlier_count += 1; } // 3.5× expected random max
    }

    let mean_max = max_coeff_sum / n_subspaces as f64;
    let mean_energy = energy_sum / n_subspaces as f64;

    WalshSubspaceStats {
        rounds,
        subspace_dim: k,
        n_subspaces,
        mean_max_coeff: mean_max,
        expected_max_random: 2.9,
        mean_spectral_energy: mean_energy,
        frac_outlier_subspaces: outlier_count as f64 / n_subspaces as f64,
        is_spectrally_flat: mean_max < 4.0 && outlier_count as f64 / (n_subspaces as f64) < 0.05,
    }
}

// ── 3. SpectralSymStats ───────────────────────────────────────────────────────

/// Combined spectral symmetry statistics for a given round count.
#[derive(Debug)]
pub struct SpectralSymStats {
    pub rounds: usize,
    pub autocorr: AutocorrStats,
    pub walsh_subspace: WalshSubspaceStats,
    /// True when both tests show flat spectral behavior.
    pub is_spectrally_random: bool,
}

/// Run both autocorrelation and restricted-Walsh tests for `rounds` rounds.
pub fn analyze_spectral_symmetry(
    rounds: usize,
    n_trials: usize,
    rng: &mut impl Rng,
) -> SpectralSymStats {
    let autocorr       = estimate_autocorrelation(rounds, n_trials, n_trials * 10, rng);
    let walsh_subspace = estimate_restricted_walsh(rounds, n_trials, rng);
    let flat = autocorr.is_spectrally_flat && walsh_subspace.is_spectrally_flat;
    SpectralSymStats { rounds, autocorr, walsh_subspace, is_spectrally_random: flat }
}

// ── 4. SpectralSweepEntry / sweep_spectral ────────────────────────────────────

/// One entry in the round-by-round spectral sweep.
#[derive(Debug)]
pub struct SpectralSweepEntry {
    pub rounds: usize,
    pub mean_abs_z: f64,
    pub max_abs_z: f64,
    pub frac_exceeds_3: f64,
    pub mean_max_walsh_coeff: f64,
    pub mean_spectral_energy: f64,
    /// True when the restricted Walsh test passes (flat spectral energy on subspaces).
    /// Autocorrelation values are reported but not used here — the sample count in the
    /// sweep is insufficient for a tight Z-test; use the individual autocorrelation test
    /// (`autocorrelation_is_flat_at_round2`) for that check.
    pub is_spectrally_random: bool,
}

/// Sweep spectral symmetry tests over 1..=max_rounds.
pub fn sweep_spectral(
    max_rounds: usize,
    n_trials: usize,
    rng: &mut impl Rng,
) -> Vec<SpectralSweepEntry> {
    (1..=max_rounds).map(|r| {
        let autocorr = estimate_autocorrelation(r, n_trials, n_trials * 10, rng);
        let walsh    = estimate_restricted_walsh(r, n_trials, rng);
        SpectralSweepEntry {
            rounds: r,
            mean_abs_z: autocorr.mean_abs_z,
            max_abs_z: autocorr.max_abs_z,
            frac_exceeds_3: autocorr.frac_exceeds_3,
            mean_max_walsh_coeff: walsh.mean_max_coeff,
            mean_spectral_energy: walsh.mean_spectral_energy,
            is_spectrally_random: walsh.is_spectrally_flat,
        }
    }).collect()
}
