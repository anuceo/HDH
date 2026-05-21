/// Phase 3A — Rotational-XOR Search for HDH.
///
/// Attempts to falsify "Φ destroys all hidden rotational symmetry".
///
/// Three tests:
/// 1. test_rotational_diff    – measures HW(F(x) ⊕ F(Rot_r(x))) for random x.
///    A cipher commuting with rotation gives all-zero XOR; random gives HW ≈ n/2.
/// 2. test_xor_preservation   – for random direction α, measures how consistently
///    D_α F(x) behaves under rotation (correlation across inputs).
/// 3. test_affine_equivalence – variance of D_α F(x) HW across x; low variance =
///    affine-structured output.
///
/// Theoretical analysis:
///   χ commutes with lane-bit rotation (intra-lane bijection).
///   θ does NOT commute with lane-bit rotation (changes parity coupling).
///   θ DOES commute with lane-index rotation (full cyclic lane shift).
///   Φ (state-dependent routing) breaks ALL rotational symmetries.
///
/// Expected result: after 1 round (χ+θ+Φ) the rotational differential output
/// is already near-random for lane-bit rotation; by round 2 fully random for
/// every rotation type.

use crate::attacks::distinguisher::{apply_rounds, random_state, STATE_BITS};
use crate::algorithm::state::State;
use crate::algorithm::chi::phi;
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

fn state_hw(s: &State) -> u32 {
    s.s1.iter().chain(s.s2.iter()).chain(s.s3.iter()).chain(s.s4.iter())
        .map(|w| w.count_ones()).sum()
}

/// Rotate every lane's bits left by `rot` positions (intra-lane rotation).
fn rotate_lane_bits(s: &State, rot: u32) -> State {
    let r = rot % 64;
    if r == 0 { return s.clone(); }
    State {
        s1:     std::array::from_fn(|i| s.s1[i].rotate_left(r)),
        s2:     std::array::from_fn(|i| s.s2[i].rotate_left(r)),
        s3:     std::array::from_fn(|i| s.s3[i].rotate_left(r)),
        s4:     std::array::from_fn(|i| s.s4[i].rotate_left(r)),
        parity: std::array::from_fn(|i| s.parity[i].rotate_left(r)),
    }
}

/// Rotate lane indices by `rot` positions (inter-lane rotation).
fn rotate_lane_index(s: &State, rot: usize) -> State {
    let r = rot % 25;
    if r == 0 { return s.clone(); }
    State {
        s1:     std::array::from_fn(|i| s.s1[(i + 25 - r) % 25]),
        s2:     std::array::from_fn(|i| s.s2[(i + 25 - r) % 25]),
        s3:     std::array::from_fn(|i| s.s3[(i + 25 - r) % 25]),
        s4:     std::array::from_fn(|i| s.s4[(i + 25 - r) % 25]),
        parity: std::array::from_fn(|i| s.parity[(i + 25 - r) % 25]),
    }
}

/// Apply a named rotation with a given amount to a state.
/// rotation_label: "lane_bits" or "lane_index".
fn apply_rotation(s: &State, rotation_label: &str, amount: usize) -> State {
    match rotation_label {
        "lane_bits"  => rotate_lane_bits(s, amount as u32),
        "lane_index" => rotate_lane_index(s, amount),
        _            => s.clone(),
    }
}

/// Random non-zero single-bit state as a derivative direction.
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

// ── 1. RotationalDiffStats ────────────────────────────────────────────────────

/// Statistics from a rotational-differential test.
///
/// For a random input x, measure HW(F(Rot(x)) ⊕ F(x)).
/// If F commutes with Rot: F(Rot(x)) = Rot(F(x)), so the XOR is
/// Rot(F(x)) ⊕ F(x) = HW of a non-trivial state ≠ 0 in general.
/// For a *perfect* equivariant cipher this could still be near n/2;
/// the key discriminator is whether HW is concentrated far from n/2.
#[derive(Debug)]
pub struct RotationalDiffStats {
    pub rounds: usize,
    pub rotation_label: &'static str,
    pub rotation_amount: usize,
    pub n_samples: usize,
    /// Mean HW of F(Rot(x)) ⊕ F(x) across random x.
    /// Random: ≈ STATE_BITS/2.  Structured: << STATE_BITS/2.
    pub hw_mean: f64,
    pub hw_std: f64,
    pub expected_hw_random: f64,
    /// (hw_mean − expected) / expected.  Near 0 = random-like.
    pub normalized_deviation: f64,
    /// Fraction of samples where HW < 0.1 × STATE_BITS (very low weight).
    pub frac_very_low_hw: f64,
    /// |normalized_deviation| < 0.05.
    pub is_random_like: bool,
}

/// Measure the rotational differential for `rounds` rounds of HDH.
/// rotation_label: "lane_bits" or "lane_index".
pub fn test_rotational_diff(
    rounds: usize,
    rotation_label: &'static str,
    rotation_amount: usize,
    n_samples: usize,
    rng: &mut impl Rng,
) -> RotationalDiffStats {
    let mut hw_sum = 0.0f64;
    let mut hw_sq  = 0.0f64;
    let mut very_low_count = 0usize;
    let threshold = (STATE_BITS / 10) as u32;

    for _ in 0..n_samples {
        let x = random_state(rng);
        let rx = apply_rotation(&x, rotation_label, rotation_amount);
        let fx  = apply_rounds(x,  rounds);
        let frx = apply_rounds(rx, rounds);
        let diff_hw = state_hw(&xor_states(&fx, &frx)) as f64;
        hw_sum += diff_hw;
        hw_sq  += diff_hw * diff_hw;
        if diff_hw < threshold as f64 { very_low_count += 1; }
    }

    let n = n_samples as f64;
    let hw_mean = hw_sum / n;
    let hw_var  = (hw_sq / n) - hw_mean * hw_mean;
    let hw_std  = hw_var.max(0.0).sqrt();
    let expected = STATE_BITS as f64 / 2.0;
    let normalized_deviation = (hw_mean - expected) / expected;

    RotationalDiffStats {
        rounds,
        rotation_label,
        rotation_amount,
        n_samples,
        hw_mean,
        hw_std,
        expected_hw_random: expected,
        normalized_deviation,
        frac_very_low_hw: very_low_count as f64 / n,
        is_random_like: normalized_deviation.abs() < 0.05,
    }
}

// ── 2. XorPreservationStats ───────────────────────────────────────────────────

/// Statistics from the XOR-preservation consistency test.
///
/// For a fixed random direction α, evaluate D_α F(x) = F(x) ⊕ F(x⊕α) for
/// many random x.  Under affine (linear) structure D_α F(x) is constant in x.
/// Under random-like structure it varies uniformly.
///
/// consistency_score = 1 − mean_hw_of_pairwise_xors / (STATE_BITS/2).
/// Score ≈ 0: random-like (D_α F varies maximally).
/// Score ≈ 1: constant (affine structure).
#[derive(Debug)]
pub struct XorPreservationStats {
    pub rounds: usize,
    pub n_directions: usize,
    pub n_samples: usize,
    /// Mean HW of D_α F(x_i) ⊕ D_α F(x_j) for random pairs (i≠j) and random α.
    /// Random: ≈ STATE_BITS/2.  Constant-derivative: 0.
    pub mean_pairwise_hw: f64,
    /// 1 − mean_pairwise_hw / (STATE_BITS/2).  0 = random; 1 = fully preserved.
    pub consistency_score: f64,
    /// Fraction of (α, sample-pair) combinations where pairwise HW < threshold.
    pub frac_near_constant: f64,
    /// |consistency_score| < 0.1 → random-like behavior.
    pub is_random_like: bool,
}

/// Test XOR-preservation: measure how much D_α F(x) varies across x.
pub fn test_xor_preservation(
    rounds: usize,
    n_directions: usize,
    n_samples: usize,
    rng: &mut impl Rng,
) -> XorPreservationStats {
    let threshold = (STATE_BITS / 20) as f64; // 5% of STATE_BITS
    let expected_half = STATE_BITS as f64 / 2.0;
    let mut total_pairwise_hw = 0.0f64;
    let mut near_constant_count = 0usize;
    let mut pair_count = 0usize;

    for _ in 0..n_directions {
        let alpha = random_direction(rng);
        // Evaluate D_α F(x_i) for n_samples random x_i.
        let derivs: Vec<State> = (0..n_samples).map(|_| {
            let x = random_state(rng);
            let xa = xor_states(&x, &alpha);
            xor_states(&apply_rounds(x, rounds), &apply_rounds(xa, rounds))
        }).collect();

        // Sample a few pairs to measure pairwise XOR HW.
        let pairs = n_samples.min(20);
        for i in 0..pairs {
            for j in (i + 1)..pairs.min(i + 5) {
                let hw = state_hw(&xor_states(&derivs[i], &derivs[j])) as f64;
                total_pairwise_hw += hw;
                pair_count += 1;
                if hw < threshold { near_constant_count += 1; }
            }
        }
    }

    let mean_pairwise_hw = if pair_count > 0 {
        total_pairwise_hw / pair_count as f64
    } else {
        expected_half
    };
    let consistency_score = 1.0 - mean_pairwise_hw / expected_half;

    XorPreservationStats {
        rounds,
        n_directions,
        n_samples,
        mean_pairwise_hw,
        consistency_score,
        frac_near_constant: near_constant_count as f64 / pair_count.max(1) as f64,
        is_random_like: consistency_score.abs() < 0.1,
    }
}

// ── 3. AffineEquivalenceStats ─────────────────────────────────────────────────

/// Statistics from the affine-equivalence score test.
///
/// For each random direction α, compute the standard deviation of
/// HW(D_α F(x)) across n_samples random x.
/// Low std_hw: output is nearly constant-weight → affine structure.
/// High std_hw ≈ sqrt(STATE_BITS × 0.25): fully random-like output.
///
/// affine_score = 1 − mean(std_hw) / sqrt(STATE_BITS × 0.25).
/// Score near 0 → random-like; score near 1 → strong affine structure.
#[derive(Debug)]
pub struct AffineEquivalenceStats {
    pub rounds: usize,
    pub n_directions: usize,
    pub n_samples: usize,
    /// Mean (over directions) of the standard deviation of HW(D_α F(x)) across x.
    pub mean_std_hw: f64,
    /// Expected std_hw for a uniformly random function: sqrt(STATE_BITS × 0.25).
    pub expected_std_random: f64,
    /// 1 − mean_std_hw / expected_std_random.  0 = random; 1 = affine.
    pub affine_score: f64,
    /// True when affine_score < 0.9, i.e. std_hw > 0.1 × expected (derivative is NOT constant).
    pub no_affine_structure: bool,
}

/// Test affine equivalence: measure how much the HW of D_α F(x) varies across x.
pub fn test_affine_equivalence(
    rounds: usize,
    n_directions: usize,
    n_samples: usize,
    rng: &mut impl Rng,
) -> AffineEquivalenceStats {
    let expected_std = ((STATE_BITS as f64) * 0.25_f64).sqrt();
    let mut std_sum = 0.0f64;

    for _ in 0..n_directions {
        let alpha = random_direction(rng);
        let mut hw_s = 0.0f64;
        let mut hw_sq = 0.0f64;
        for _ in 0..n_samples {
            let x = random_state(rng);
            let xa = xor_states(&x, &alpha);
            let d = xor_states(&apply_rounds(x, rounds), &apply_rounds(xa, rounds));
            let hw = state_hw(&d) as f64;
            hw_s  += hw;
            hw_sq += hw * hw;
        }
        let mean = hw_s / n_samples as f64;
        let var  = (hw_sq / n_samples as f64) - mean * mean;
        std_sum += var.max(0.0).sqrt();
    }

    let mean_std_hw = std_sum / n_directions as f64;
    let affine_score = 1.0 - mean_std_hw / expected_std;

    AffineEquivalenceStats {
        rounds,
        n_directions,
        n_samples,
        mean_std_hw,
        expected_std_random: expected_std,
        affine_score,
        no_affine_structure: affine_score < 0.9,
    }
}

// ── 4. RotationalSweepEntry / sweep_rotational ────────────────────────────────

/// One entry in the joint (rounds × rotation-type) sweep.
#[derive(Debug)]
pub struct RotationalSweepEntry {
    pub rounds: usize,
    pub rotation_label: &'static str,
    pub rotation_amount: usize,
    pub normalized_deviation: f64,
    pub consistency_score: f64,
    pub affine_score: f64,
    /// True when all three tests show random-like behavior.
    pub is_fully_random: bool,
}

/// Sweep rotational symmetry tests over 1..=max_rounds for both rotation types.
pub fn sweep_rotational(
    max_rounds: usize,
    n_samples: usize,
    rng: &mut impl Rng,
) -> Vec<RotationalSweepEntry> {
    let rotation_configs: &[(&'static str, usize)] = &[
        ("lane_bits",  1),
        ("lane_bits",  7),
        ("lane_bits",  13),
        ("lane_index", 1),
        ("lane_index", 5),
        ("lane_index", 12),
    ];

    let mut entries = Vec::new();
    for r in 1..=max_rounds {
        for &(label, amount) in rotation_configs {
            let diff  = test_rotational_diff(r, label, amount, n_samples.min(200), rng);
            let xpres = test_xor_preservation(r, 10, n_samples.min(30), rng);
            let aeq   = test_affine_equivalence(r, 10, n_samples.min(30), rng);
            entries.push(RotationalSweepEntry {
                rounds: r,
                rotation_label: label,
                rotation_amount: amount,
                normalized_deviation: diff.normalized_deviation,
                consistency_score: xpres.consistency_score,
                affine_score: aeq.affine_score,
                is_fully_random: diff.is_random_like
                    && xpres.is_random_like
                    && aeq.no_affine_structure,
            });
        }
    }
    entries
}

fn rotate_state(s: &State, r: usize) -> State {
    let mut out = State::zero();
    for i in 0..25 {
        let src = (i + r) % 25;
        out.s1[i] = s.s1[src];
        out.s2[i] = s.s2[src];
        out.s3[i] = s.s3[src];
        out.s4[i] = s.s4[src];
        out.parity[i] = s.parity[src];
    }
    out
}

fn states_equal_shares(a: &State, b: &State) -> bool {
    a.s1 == b.s1 && a.s2 == b.s2 && a.s3 == b.s3 && a.s4 == b.s4
}

/// Average fraction of output u64 words that match between two states,
/// comparing all 4 × 25 = 100 share words (not parity).
fn share_match_fraction(a: &State, b: &State) -> f64 {
    let matches: usize = a.s1.iter().zip(b.s1.iter()).filter(|(&x, &y)| x == y).count()
        + a.s2.iter().zip(b.s2.iter()).filter(|(&x, &y)| x == y).count()
        + a.s3.iter().zip(b.s3.iter()).filter(|(&x, &y)| x == y).count()
        + a.s4.iter().zip(b.s4.iter()).filter(|(&x, &y)| x == y).count();
    matches as f64 / 100.0
}

// ── Public API ─────────────────────────────────────────────────────────────

pub struct RotationSymmetryStats {
    pub rotations_tested: usize,
    pub samples_per_rotation: usize,
    /// For each r ∈ 1..=24: fraction of samples where φ(rot(S,r)) == rot(φ(S),r) exactly.
    pub exact_equivariance_fractions: Vec<f64>,
    /// For each r: average fraction of matching output u64 words.  ≈ 0 means
    /// independent (no symmetry); > 0 would indicate partial equivariance.
    pub avg_word_match_fractions: Vec<f64>,
    /// Maximum exact equivariance fraction across all rotations.
    pub max_exact_equivariance: f64,
    /// Maximum average word-match fraction across all rotations.
    pub max_avg_word_match: f64,
    /// Expected word-match fraction for two independent 64-bit outputs: ≈ 2^{-64} ≈ 0.
    pub expected_word_match: f64,
}

/// Test whether Φ commutes with lane rotations for all r ∈ {1, …, 24}.
pub fn test_rotational_symmetry(samples: usize, rng: &mut impl Rng) -> RotationSymmetryStats {
    let mut exact_fracs: Vec<f64> = Vec::with_capacity(24);
    let mut word_match_fracs: Vec<f64> = Vec::with_capacity(24);
    let mut max_exact = 0.0f64;
    let mut max_word = 0.0f64;

    // Pre-generate random states.
    let states: Vec<State> = (0..samples)
        .map(|_| {
            let mut s = State::zero();
            for i in 0..25 {
                s.s1[i] = rng.gen();
                s.s2[i] = rng.gen();
                s.s3[i] = rng.gen();
                s.s4[i] = rng.gen();
                s.parity[i] = s.s1[i] ^ s.s2[i] ^ s.s3[i] ^ s.s4[i];
            }
            s
        })
        .collect();

    for r in 1..=24 {
        let mut exact_count = 0usize;
        let mut total_word_match = 0.0f64;

        for s in &states {
            let phi_rot_s = phi(&rotate_state(s, r));
            let rot_phi_s = rotate_state(&phi(s), r);

            if states_equal_shares(&phi_rot_s, &rot_phi_s) {
                exact_count += 1;
            }
            total_word_match += share_match_fraction(&phi_rot_s, &rot_phi_s);
        }

        let ef = exact_count as f64 / samples as f64;
        let wf = total_word_match / samples as f64;
        if ef > max_exact { max_exact = ef; }
        if wf > max_word { max_word = wf; }
        exact_fracs.push(ef);
        word_match_fracs.push(wf);
    }

    RotationSymmetryStats {
        rotations_tested: 24,
        samples_per_rotation: samples,
        exact_equivariance_fractions: exact_fracs,
        avg_word_match_fractions: word_match_fracs,
        max_exact_equivariance: max_exact,
        max_avg_word_match: max_word,
        expected_word_match: 2.0f64.powi(-64),
    }
}

// ── Affine shift test ──────────────────────────────────────────────────────

pub struct AffineShiftStats {
    pub shifts_tested: usize,
    pub samples_per_shift: usize,
    /// For each constant-shift test: fraction of samples where the output
    /// XOR φ(S⊕C)⊕φ(S) is the same as for the first sample (i.e., constant).
    pub max_constant_output_frac: f64,
}

/// Test whether φ(S ⊕ C) ⊕ φ(S) is constant over random S for various C.
///
/// For a linear function, this fraction would be 1.  For a state-dependent
/// routing, it should be ≈ 0 (the output XOR depends on S, not just C).
pub fn test_affine_shift(n_constants: usize, samples: usize, rng: &mut impl Rng) -> AffineShiftStats {
    let mut max_constant_frac = 0.0f64;

    for _ in 0..n_constants {
        // Constant shift applied to s1 only (a representative single-share test).
        let mut c = State::zero();
        for i in 0..25 {
            c.s1[i] = rng.gen();
        }

        // Compute the reference output XOR using the first random state.
        let s0: State = {
            let mut s = State::zero();
            for i in 0..25 {
                s.s1[i] = rng.gen();
                s.s2[i] = rng.gen();
                s.s3[i] = rng.gen();
                s.s4[i] = rng.gen();
                s.parity[i] = s.s1[i] ^ s.s2[i] ^ s.s3[i] ^ s.s4[i];
            }
            s
        };
        let sc0 = {
            let mut s = s0.clone();
            for i in 0..25 { s.s1[i] ^= c.s1[i]; }
            s
        };
        let ref_xor_s1: [u64; 25] = std::array::from_fn(|i| phi(&sc0).s1[i] ^ phi(&s0).s1[i]);

        // Check how many subsequent samples give the same output XOR.
        let mut matches = 0usize;
        for _ in 0..samples {
            let mut s = State::zero();
            for i in 0..25 {
                s.s1[i] = rng.gen();
                s.s2[i] = rng.gen();
                s.s3[i] = rng.gen();
                s.s4[i] = rng.gen();
                s.parity[i] = s.s1[i] ^ s.s2[i] ^ s.s3[i] ^ s.s4[i];
            }
            let mut sc = s.clone();
            for i in 0..25 { sc.s1[i] ^= c.s1[i]; }
            let xor_s1: [u64; 25] = std::array::from_fn(|i| phi(&sc).s1[i] ^ phi(&s).s1[i]);
            if xor_s1 == ref_xor_s1 {
                matches += 1;
            }
        }
        let frac = matches as f64 / samples as f64;
        if frac > max_constant_frac { max_constant_frac = frac; }
    }

    AffineShiftStats {
        shifts_tested: n_constants,
        samples_per_shift: samples,
        max_constant_output_frac: max_constant_frac,
    }
}
