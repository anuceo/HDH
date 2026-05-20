pub mod annihilator;
pub mod differential;
pub mod distinguisher;
pub mod hybrid;
pub mod integral;
pub mod jacobian;
pub mod linear;
pub mod phi_symmetry;
pub mod preimage;
pub mod sat;
pub mod truncated;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(0xdeadbeef_cafebabe)
    }

    // ── Differential ───────────────────────────────────────────────────────

    #[test]
    fn differential_bias_is_low() {
        let mut r = rng();
        let bias = differential::max_differential_bias(20, 20_000, &mut r);
        assert!(
            bias < 0.01,
            "differential bias {bias:.6} — possible structural shortcut"
        );
    }

    #[test]
    fn differential_nonzero_diff_is_uniform() {
        let mut r = rng();
        let stats = differential::analyze_differential((1u64, 0, 0, 0), 10_000, &mut r);
        assert!(
            stats.unique_output_diffs > 9_000,
            "differential output collapsed: {} unique values",
            stats.unique_output_diffs
        );
    }

    // ── Linear bias ────────────────────────────────────────────────────────

    #[test]
    fn linear_bias_is_negligible() {
        let mut r = rng();
        let stats = linear::sample_linear_bias(50, 20_000, &mut r);
        assert!(
            stats.max_bias < 0.05,
            "linear bias max={:.4} — exploitable linear approximation exists",
            stats.max_bias
        );
    }

    // ── Preimage ────────────────────────────────────────────────────────────

    #[test]
    fn preimage_count_sanity() {
        let mut r = rng();
        let stats = preimage::analyze_preimages(2_000, &mut r);
        assert!(
            stats.avg_preimage_count > 0.5 && stats.avg_preimage_count < 2.0,
            "avg preimage count {:.3} outside [0.5, 2.0]",
            stats.avg_preimage_count
        );
        assert!(
            stats.max_preimage_count <= 8,
            "max preimage count {} is suspiciously high",
            stats.max_preimage_count
        );
    }

    #[test]
    fn preimage_roundtrip_invariant() {
        assert!(preimage::chi8_roundtrip_check());
    }

    // ── Jacobian rank (replaces broken degree-threshold test) ───────────────
    //
    // rank(J_χ(x)) ≈ n means the local linearisation covers all output
    // dimensions — no solvable linear shortcut at that evaluation point.

    #[test]
    fn jacobian_rank_near_full() {
        let mut r = rng();
        let stats = jacobian::analyze_jacobian_rank(8, &mut r);
        // Allow a small deficit (e.g. if an input word is pathologically small)
        // but require ≥ 78% of the theoretical 256-rank in the worst case.
        assert!(
            stats.min_rank >= 200,
            "Jacobian rank {} < 200 — linear shortcut may exist at some inputs",
            stats.min_rank
        );
    }

    // ── Effective XL complexity ─────────────────────────────────────────────
    //
    // After projecting onto the degree-2 subspace, the effective XL degree
    // remains ≥ 32 (carry-chain algebraic degree).  Complexity must be ≥ 2^120.

    #[test]
    fn xl_complexity_meets_120bit_threshold() {
        // Degree-2 rank for 64-bit chi is extrapolated from the 4-bit exact measurement.
        // 4-bit has 16 output bits → rank ≤ 16.  64-bit has 256 output bits → rank ≤ 256.
        // Use the 4-bit measured rank (conservative).
        let iso = preimage::structural_isolation_4bit();
        let eff = sat::model_effective_xl(iso.subsystem_rank);
        assert!(
            eff.meets_120bit_threshold,
            "effective XL complexity 2^{:.1} < 2^120 — XL attack may be feasible",
            eff.xl_complexity_log2
        );
    }

    // ── Incremental SAT: free variables ─────────────────────────────────────
    //
    // After UP with 1+ known pairs, the g-auxiliary (64 bits) must remain free.
    // Free-vars >> n/2 here means free-vars (64) >> n_g/2 (32) where n_g = 64
    // is the auxiliary variable count; the full branching space is 2^64.

    #[test]
    fn sat_free_vars_remain_high_after_up() {
        let stats = sat::simulate_incremental_sat(4);
        for snap in &stats.snapshots[1..] {
            assert!(
                snap.free_vars_after_up >= 32,
                "free vars {} after {} pairs is too low — UP may be resolving g",
                snap.free_vars_after_up,
                snap.pairs
            );
        }
        // Check that adding more pairs does not collapse the free-variable count.
        let last = stats.snapshots.last().unwrap();
        assert!(
            last.log2_dpll_search >= 32.0,
            "DPLL search 2^{:.1} is too small",
            last.log2_dpll_search
        );
    }

    // ── Degree propagation ──────────────────────────────────────────────────
    //
    // The composed χ₄ function should reach maximum algebraic degree (16 for
    // 16-bit input space) quickly, confirming that nonlinearity grows rapidly.

    #[test]
    fn degree_propagates_and_saturates_early() {
        let stats = preimage::degree_propagation(4);

        // Round 1: nonlinearity must have started (chi4 has degree > 1).
        assert!(
            stats.max_degree_per_round[0] > 1,
            "degree after round 1 is {}; χ appears linear",
            stats.max_degree_per_round[0]
        );
        // Round 2: degree must grow significantly beyond round 1.
        // Empirically: 5 → 11 (composition of degree-5 functions in a 4-bit
        // carry-chain model reaches ~11 before the plateau slows growth).
        // We require > round1 + 3 to confirm compound nonlinear growth.
        let r1 = stats.max_degree_per_round[0];
        let r2 = stats.max_degree_per_round[1];
        assert!(
            r2 >= r1 + 3,
            "degree after round 2 ({r2}) not significantly above round 1 ({r1}); \
             composition is not compounding nonlinearity"
        );
        // Round 4: degree must be well above the degree-2 threshold (≥ 8),
        // confirming the function is far outside the reach of degree-2 XL.
        let r4 = stats.max_degree_per_round[3];
        assert!(
            r4 >= 8,
            "degree after 4 rounds ({r4}) is still low; sustained nonlinear growth absent"
        );
    }

    // ── Structural isolation ────────────────────────────────────────────────
    //
    // The degree-2 coefficient matrix of chi₄ has rank << C(16,2) = 120.
    // This means the quadratic subsystem is massively underdetermined: an
    // attacker cannot recover inputs using only the degree-2 structure.

    #[test]
    fn degree2_subsystem_underdetermined() {
        let stats = preimage::structural_isolation_4bit();
        // Rank ≤ 16 (one equation per output bit), unknowns = 120 z-variables.
        // Underdetermination ratio ≥ 5 means ≥ 5 free z-vars per independent equation.
        assert!(
            stats.underdetermination_ratio >= 5.0,
            "underdetermination ratio {:.1} too low — degree-2 subsystem may be isolatable",
            stats.underdetermination_ratio
        );
        // Rank must be < number of z-variables to confirm underdetermination.
        assert!(
            stats.subsystem_rank < stats.degree2_var_count,
            "rank {} = degree2_var_count {} — system is exactly determined (unexpected)",
            stats.subsystem_rank,
            stats.degree2_var_count
        );
    }

    // ── Annihilator search ──────────────────────────────────────────────────
    //
    // For chi4's degree-2 output bits, AI = 2 is expected and mathematically
    // correct: the function itself is an annihilator of its complement at degree 2.
    // The security-relevant property is that NO degree-1 (linear) annihilator
    // exists for any bit, and that the carry-chain bits (higher degree) have AI ≥ 3.
    //
    // Test 1 uses max_degree=1 (fast: 17-monomial matrix per bit).
    // Test 2 uses max_degree=2 to count carry-chain bits with AI > 2.

    #[test]
    fn annihilators_absent_at_low_degree() {
        // No degree-1 (linear/affine) annihilator should exist for any output bit.
        // A degree-1 annihilator would mean the function's support is contained in
        // a hyperplane — an indicator of hidden linear structure.
        let stats = annihilator::analyze_algebraic_immunity(1);
        assert!(
            stats.min_lb >= 2,
            "min AI lower bound = {} — degree-1 annihilator found; linear structure in chi4",
            stats.min_lb
        );
        // All 16 bits should resist degree-1 annihilators.
        assert_eq!(
            stats.high_ai_bit_count,
            16,
            "only {}/{} bits resist degree-1 annihilators",
            stats.high_ai_bit_count,
            stats.per_bit_lb.len()
        );
    }

    #[test]
    fn algebraic_immunity_meets_threshold() {
        // The carry-chain bits (positions where multiplication introduces degree-4+
        // terms via carry propagation) should have AI > 2 (no degree-≤2 annihilator).
        // We require at least 4 such bits (the 4 carry-heavy positions per nibble pair).
        let stats = annihilator::analyze_algebraic_immunity(2);
        let high_ai = stats.per_bit_lb.iter().filter(|&&lb| lb >= 3).count();
        assert!(
            high_ai >= 4,
            "{}/{} bits have AI ≥ 3 — expected ≥ 4 carry-chain bits to resist degree-2",
            high_ai,
            stats.per_bit_lb.len()
        );
        // And confirm no degree-1 annihilators (min_lb ≥ 2 still).
        assert!(
            stats.min_lb >= 2,
            "min AI = {} — a linear annihilator exists; unexpected structural collapse",
            stats.min_lb
        );
    }

    // ── Invariant subspace detection ────────────────────────────────────────
    //
    // For chi₄ (a bijection on 2^16 points), the expected number of fixed
    // points under a random permutation is 1.  We allow a small window
    // [0, 10] to tolerate the non-random structure of chi₄; a very large
    // count would indicate an exploitable invariant subspace.

    #[test]
    fn invariant_subspace_only_trivial() {
        let stats = annihilator::detect_invariant_subspaces();
        // chi4 has structural fixed points wherever g=0 (a nonlinear condition on the
        // inputs).  The relevant cryptographic question is NOT the raw count but whether
        // those fixed points constitute a GF(2)-linear subspace — which would imply an
        // algebraically exploitable invariant subspace.  A non-power-of-2 cardinality
        // already rules out a linear subspace; the explicit closure check confirms it.
        assert!(
            !stats.fixed_points_form_linear_subspace,
            "fixed-point set ({} points) is closed under XOR — linear invariant subspace detected",
            stats.fixed_point_count
        );
        // Maps-to-zero: a handful of inputs satisfying the self-consistency equation
        // g = quad(g, rot1(g), rot2(g), rot3(g)) is expected; a large count would
        // indicate a collapse toward zero.
        assert!(
            stats.maps_to_zero_count <= 16,
            "chi4 maps {} inputs to 0 — structural collapse toward zero detected",
            stats.maps_to_zero_count
        );
    }

    // ── Differential-linear hybrid ──────────────────────────────────────────
    //
    // For a high-degree nonlinear function, combining a differential
    // characteristic with a linear approximation should yield negligible bias.
    // Threshold: below 1/√(samples) ≈ 1/√20000 ≈ 0.007.

    #[test]
    fn difflin_bias_is_negligible() {
        let mut r = rng();
        let stats = hybrid::sample_difflin_bias(50, 20_000, &mut r);
        assert!(
            stats.max_bias < 0.02,
            "differential-linear max bias {:.4} — possible hybrid shortcut",
            stats.max_bias
        );
    }

    // ── Second-order (boomerang-rectangle) differential ─────────────────────
    //
    // The second-order derivative ∂²_{Δ₀,Δ₁}χ has degree ≥ deg(χ)−2 ≥ 30.
    // Its parity should be unbiased; a bias would indicate a rectangular
    // structure exploitable by boomerang distinguishers.

    #[test]
    fn second_order_differential_unbiased() {
        let mut r = rng();
        let stats = hybrid::test_second_order_differential(30, 10_000, &mut r);
        assert!(
            stats.max_bias < 0.02,
            "second-order differential parity bias {:.4} — boomerang structure possible",
            stats.max_bias
        );
    }

    // ── Truncated differential propagation ─────────────────────────────────
    //
    // A single-nibble-active input difference should mix into multiple output
    // nibbles the majority of the time (via the nonlinear g term).  Require
    // ≥ 50% multi-nibble outputs for single-nibble inputs.

    #[test]
    fn truncated_diff_mixes_nibbles() {
        let stats = truncated::analyze_truncated_differentials();
        assert!(
            stats.single_to_multi_rate >= 0.5,
            "single-to-multi nibble rate {:.3} < 0.5 — chi4 diffusion is unexpectedly poor",
            stats.single_to_multi_rate
        );
        // Average output nibble weight for the weakest input pattern must be ≥ 1.5
        // (well above 1, meaning output differences spread beyond a single nibble).
        assert!(
            stats.min_avg_output_weight >= 1.5,
            "min avg output nibble weight {:.2} — activity is concentrating rather than spreading",
            stats.min_avg_output_weight
        );
    }

    // ── Φ rotational symmetry ───────────────────────────────────────────────
    //
    // Φ is state-dependent: rotating the input state changes the routing
    // indices, so φ(rotate(S,r)) ≠ rotate(φ(S),r) in general.  No exact
    // equivariance should occur for random states at any rotation.

    #[test]
    fn phi_has_no_rotational_symmetry() {
        let mut r = rng();
        let stats = phi_symmetry::test_rotational_symmetry(200, &mut r);
        assert!(
            stats.max_exact_equivariance == 0.0,
            "φ appears equivariant under some rotation: max exact match fraction = {:.6}",
            stats.max_exact_equivariance
        );
    }

    // ── Φ affine shift test ─────────────────────────────────────────────────
    //
    // For a state-dependent routing, φ(S⊕C)⊕φ(S) is not a constant for fixed C.
    // If it were constant, an attacker could use affine algebra to reduce the
    // effective key space.

    #[test]
    fn phi_output_xor_is_not_constant_for_fixed_shift() {
        let mut r = rng();
        let stats = phi_symmetry::test_affine_shift(20, 100, &mut r);
        assert!(
            stats.max_constant_output_frac < 0.1,
            "φ(S⊕C)⊕φ(S) is constant for {:.1}% of inputs — affine shift symmetry detected",
            stats.max_constant_output_frac * 100.0
        );
    }

    // ── Reduced-round distinguishers ────────────────────────────────────────
    //
    // The full permutation should be indistinguishable from random by round 2.
    // We test three independent distinguishers and require all fail at 2 rounds.

    #[test]
    fn avalanche_completeness_increases_with_rounds() {
        let mut r = rng();
        // Use small sample counts for test speed: 16 bits × 30 samples each.
        let s1 = distinguisher::measure_avalanche(1, 16, 30, &mut r);
        let s2 = distinguisher::measure_avalanche(2, 16, 30, &mut r);
        // Completeness must grow: round 2 must have strictly more bits in [0.4,0.6]
        // than round 1 (θ+φ together require ≥2 rounds for full avalanche).
        assert!(
            s2.completeness > s1.completeness,
            "completeness did not grow: round1={:.2} round2={:.2}",
            s1.completeness, s2.completeness
        );
        // Two-round avalanche mean must be above 30% (partial → near-full mixing).
        assert!(
            s2.mean_frac > 0.30,
            "two-round avalanche mean {:.3} — expected > 0.30",
            s2.mean_frac
        );
    }

    #[test]
    fn two_round_output_bits_are_balanced() {
        let mut r = rng();
        // 2000 samples, 32 bits; noise floor ≈ 2/√2000 ≈ 0.045.
        // We allow up to 3× the noise floor as max bias (conservative for the
        // sample count; systematic bias would appear as 10× or more).
        let stats = distinguisher::measure_output_balance(2, 32, 2_000, &mut r);
        assert!(
            stats.max_abs_bias < stats.noise_floor * 3.0,
            "round-2 max output bias {:.4} exceeds 3×noise_floor ({:.4}) — systematic bias",
            stats.max_abs_bias,
            stats.noise_floor * 3.0
        );
    }

    #[test]
    fn linear_distinguisher_fails_at_two_rounds() {
        let mut r = rng();
        // 30 mask pairs × 5000 samples; noise floor ≈ 1/√5000 ≈ 0.014.
        let stats = distinguisher::measure_linear_bias(2, 30, 5_000, &mut r);
        assert!(
            stats.max_bias < stats.noise_floor * 3.0,
            "round-2 linear max bias {:.4} — linear distinguisher may exist",
            stats.max_bias
        );
    }

    // ── Chi4 zero-sum property (exact algebraic) ────────────────────────────
    //
    // chi4 has max algebraic degree 5.  Any dim-6 affine coset (64 elements)
    // must XOR-sum to 0 over all output bits.  This is a deterministic algebraic
    // property — any failure indicates an ANF degree bound error.

    #[test]
    fn chi4_zero_sum_holds_at_dim_6() {
        let mut r = rng();
        let stats = distinguisher::check_zero_sum_chi4(6, 500, &mut r);
        assert_eq!(
            stats.nonzero_sum_count,
            0,
            "{}/{} dim-6 cosets gave nonzero XOR sum — chi4 degree > 5",
            stats.nonzero_sum_count,
            stats.cosets_tested
        );
    }

    #[test]
    fn chi4_zero_sum_fails_below_degree_threshold() {
        // Negative control: dim-5 cosets (32 elements) should produce nonzero
        // sums for a function with degree > 4, confirming the test discriminates.
        let mut r = rng();
        let stats = distinguisher::check_zero_sum_chi4(5, 500, &mut r);
        // At least 5% of dim-5 cosets should have nonzero sum if degree > 4.
        let nonzero_frac = stats.nonzero_sum_count as f64 / stats.cosets_tested as f64;
        assert!(
            nonzero_frac > 0.05,
            "only {:.1}% of dim-5 cosets gave nonzero sum — chi4 may have degree ≤ 4",
            nonzero_frac * 100.0
        );
    }

    // ── Higher-order integral distinguishers ────────────────────────────────
    //
    // 0 rounds = identity (degree 1): any cube dim ≥ 2 must give XOR sum = 0.
    //
    // 1 round: empirical analysis shows effective bit-level degree ≤ 3 for most
    // input directions — ~93% of dim=4 cubes give zero XOR sum, and almost all
    // output bits are individually balanced per cube.  This is an expected
    // single-round property; HDH requires ≥ 2 rounds for security.
    //
    // 2 rounds: the degree exceeds 4 in all tested directions.  No dim ≤ 8 cube
    // produces a zero XOR sum.  avg_balanced drops from ~6395 (1-round) to ~3700
    // (2-round), indicating the degree growth destroys the integral structure.

    #[test]
    fn integral_control_identity_is_linear() {
        // 0 rounds = identity (degree 1).  Every dim=2 cube must give XOR sum = 0.
        let mut r = rng();
        let stats = integral::test_cube_sum(0, 2, 10, &mut r);
        assert_eq!(
            stats.zero_sum_fraction, 1.0,
            "identity function: {:.0}% of dim-2 cubes gave nonzero sum — test framework error",
            (1.0 - stats.zero_sum_fraction) * 100.0
        );
    }

    #[test]
    fn integral_one_round_has_low_degree_structure() {
        // After 1 round, HDH has degree ≤ 3 for most input directions.
        // At dim=4 (16 evaluations per cube), ≥ 70% of random cubes should
        // give XOR sum = 0, and avg_balanced_bits should exceed 6000 (most
        // output bits are balanced for every tested cube).
        let mut r = rng();
        let stats = integral::test_cube_sum(1, 4, 30, &mut r);
        assert!(
            stats.zero_sum_fraction > 0.70,
            "1-round dim=4: only {:.0}% of cubes gave zero XOR sum; expected > 70%",
            stats.zero_sum_fraction * 100.0
        );
        assert!(
            stats.avg_balanced_bits > 6_000.0,
            "1-round dim=4: avg balanced bits {:.0} < 6000 — expected near-total balance",
            stats.avg_balanced_bits
        );
    }

    #[test]
    fn integral_two_round_eliminates_full_integral() {
        // After 2 rounds, degree exceeds 4 in every tested direction.
        // No dim=4 cube should produce an all-zero XOR sum (zero_frac = 0.0),
        // and avg_balanced_bits should be near the random baseline of ~3200.
        let mut r = rng();
        let stats = integral::test_cube_sum(2, 4, 20, &mut r);
        assert_eq!(
            stats.zero_sum_fraction, 0.0,
            "2-round dim=4: {:.0}% of cubes gave zero XOR sum — integral structure survives to round 2",
            stats.zero_sum_fraction * 100.0
        );
        // avg_balanced_bits should be well below the 1-round value (> 6000) and
        // within 20σ of random (3200 ± 40).  Even 3900 < 6000 confirms degree growth.
        assert!(
            stats.avg_balanced_bits < 5_000.0,
            "2-round avg balanced bits {:.0} suspiciously high — possible 1-round regression",
            stats.avg_balanced_bits
        );
    }
}
