/// Rotational symmetry testing for the Φ state-dependent lane routing.
///
/// Φ routes lane i using two state-derived indices:
///   j = (s1[i] ⊕ s2[i]) mod 25
///   k = (s3[i] ⊕ s4[i]) mod 25
/// and then XORs values from lanes j and k into the share arrays.
///
/// A "rotation-by-r" of the state cyclically permutes its 25 lanes:
///   rotate(S, r).s1[i] = S.s1[(i + r) mod 25]
///
/// **Equivariance test**: does φ(rotate(S, r)) = rotate(φ(S), r)?
/// For a truly equivariant φ this would hold for ALL S; for a non-equivariant
/// φ it should hold with probability ≈ 2^(-4·25·64) ≈ 0 for random S.
///
/// Analysis shows: the routing index j computed from rotate(S, r) at position i
/// equals (s1[(i+r) mod 25] ⊕ s2[(i+r) mod 25]) mod 25 — the same index j
/// as in the original state at position i+r.  But the VALUE fetched from lane j
/// in the rotated state is S.s2[(j+r) mod 25], whereas rotate(φ(S), r) fetches
/// S.s2[j].  These differ unless S.s2[j] = S.s2[(j+r) mod 25], which for
/// 64-bit words has probability 2^{-64} per sample.
///
/// **Affine shift test**: does φ(S ⊕ C) ≈ φ(S) ⊕ C' for some fixed constants
/// C, C'?  For a state-DEPENDENT routing φ, XOR-ing a constant into the state
/// changes all routing indices and is therefore non-affine.  We measure whether
/// any single-share constant shift induces a fixed output offset.
use crate::phi::phi;
use crate::state::State;
use rand::Rng;

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
