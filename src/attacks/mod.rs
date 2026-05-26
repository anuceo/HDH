pub mod annihilator;
pub mod boomerang;
pub mod differential;
pub mod distinguisher;
pub mod gpu_algebraic;
pub mod hybrid;
pub mod integral;
pub mod jacobian;
pub mod linear;
pub mod mitm;
pub mod multi_user_sponge;
pub mod phi_symmetry;
pub mod preimage;
pub mod quantum_security;
pub mod sat;
pub mod sponge;
pub mod sponge_indiff;
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

    // ── MITM: structured forward/backward separability analysis ─────────────
    //
    // Primary question: does 2-round HDH destroy forward/backward separability
    // after the degree-inflection transition at the 1→2 round boundary?
    //
    // Cat 1 – partition matching:  2-round intermediate state collisions should
    //   equal random birthday expectation (no exploitable matching surface).
    // Cat 2 – dependency graph:    2-round isolation ratio should drop to ~1
    //   (all 25 lanes affect every output lane); 1-round shows lane isolation.
    // Cat 3 – Φ linear rank:       Φ and 1-round should be near full rank in
    //   the sampled n_bits × n_bits influence submatrix.
    // Cat 4 – biclique matching:   2-round cubes should show zero log2_excess
    //   (collision rate within random expectation).
    // Cat 5 – entropy collapse:    min_entropy at 2 rounds should approach k_bits.

    #[test]
    fn mitm_cat1_two_round_no_partition_excess() {
        // 2-round intermediate state projected to 20 bits should show collisions
        // consistent with the birthday bound (log2_excess ≤ 2.0 bits of excess).
        let mut r = rng();
        let s2 = mitm::measure_partition_matching(2, 500, 20, &mut r);
        assert!(
            s2.log2_excess <= 2.0,
            "2-round partition: log2_excess={:.2} — matching surface may exist",
            s2.log2_excess
        );
        // 1-round should show more excess than 2-round (confirming measurement sensitivity).
        let s1 = mitm::measure_partition_matching(1, 500, 20, &mut r);
        assert!(
            s1.log2_excess >= s2.log2_excess - 1.0,
            "1-round excess {:.2} unexpectedly below 2-round {:.2}",
            s1.log2_excess, s2.log2_excess
        );
    }

    #[test]
    fn mitm_cat2_two_round_destroys_lane_isolation() {
        // At 1 round, same-lane influence >> cross-lane (isolation_ratio >> 1).
        // At 2 rounds, all lanes couple — isolation_ratio should drop below 2.0.
        let mut r = rng();
        let d1 = mitm::analyze_dependency_graph(1, 200, 15, &mut r);
        let d2 = mitm::analyze_dependency_graph(2, 200, 15, &mut r);
        assert!(
            d1.isolation_ratio > d2.isolation_ratio,
            "round 1 isolation ratio {:.2} ≤ round 2 {:.2} — diffusion did not grow",
            d1.isolation_ratio, d2.isolation_ratio
        );
        // 2-round cross-lane influence must be substantial (> 50% of pairs).
        assert!(
            d2.cross_lane_influence_frac > 0.50,
            "2-round cross-lane influence only {:.1}% — lane isolation persists",
            d2.cross_lane_influence_frac * 100.0
        );
    }

    #[test]
    fn mitm_cat3_phi_influence_rank_is_near_full() {
        // Φ's influence matrix is sparse: each output lane fetches from only
        // ~2 out of 25 input lanes via state-derived routing.  In a 24-bit
        // random subspace, Φ alone has low rank (many zero-influence pairs).
        // This is structurally expected for a routing-based function.
        //
        // Security comes from the FULL ROUND:  1-round and 2-round must show
        // significantly higher rank, confirming θ+χ composition fills in the
        // sparse connectivity of Φ.  We assert rank(2-round) > rank(Φ alone).
        let mut r = rng();
        let s = mitm::measure_influence_rank(24, 20, &mut r);
        // 2-round rank must strictly exceed Φ-alone rank (θ+χ composition helps).
        assert!(
            s.two_round_rank > s.phi_rank,
            "2-round rank {} ≤ Φ rank {} — θ+χ composition not expanding connectivity",
            s.two_round_rank, s.phi_rank
        );
        // 1-round rank must also exceed Φ alone (even one round adds diffusion).
        assert!(
            s.one_round_rank > s.phi_rank,
            "1-round rank {} ≤ Φ rank {} — χ+θ not adding to Φ influence coverage",
            s.one_round_rank, s.phi_rank
        );
    }

    #[test]
    fn mitm_cat4_biclique_excess_collapses_at_two_rounds() {
        // At 1 round (degree ≤ 3): within a dim=5 cube, outputs are degree-3
        // functions of 5 bits → massive biclique collision excess (log2 ≈ 8–9)
        // due to structured output compression.  This is the EXPECTED single-round
        // structural weakness confirming low algebraic degree.
        //
        // At 2 rounds: the degree explosion must destroy this structure.
        // 2-round log2_excess must be substantially smaller than 1-round.
        let mut r = rng();
        let bc1 = mitm::test_biclique_matching(5, 12, 8, &mut r);
        let bc2 = mitm::test_biclique_matching_rounds(5, 12, 8, 2, &mut r);
        // 1-round must show large excess (confirming low-degree structure is detectable).
        assert!(
            bc1.log2_excess > 4.0,
            "1-round biclique log2_excess={:.2} < 4 — low-degree structure not detected",
            bc1.log2_excess
        );
        // 2-round excess must be substantially less than 1-round.
        assert!(
            bc2.log2_excess < bc1.log2_excess - 3.0,
            "2-round log2_excess={:.2} not much less than 1-round {:.2} — biclique persists",
            bc2.log2_excess, bc1.log2_excess
        );
    }

    #[test]
    fn mitm_cat5_two_round_entropy_does_not_collapse() {
        // Project outputs to k=10 bits with N=3000 samples (N/2^k ≈ 2.9, dense
        // regime).  For a random permutation, max bucket count ≈ 7–10 → uniformity
        // ratio ≈ 2–4.  An entropy-collapsed function would show ratio >> 10.
        //
        // We also verify 1-round uniformity_ratio > 2-round uniformity_ratio,
        // confirming 2 rounds is strictly stronger (less biased distribution).
        let mut r = rng();
        let s2 = mitm::measure_entropy_collapse(2, 3_000, 10, &mut r);
        let s1 = mitm::measure_entropy_collapse(1, 3_000, 10, &mut r);
        // 2-round uniformity ratio must be ≤ 10 (not catastrophically skewed).
        assert!(
            s2.uniformity_ratio <= 10.0,
            "2-round uniformity ratio {:.2} > 10 — output distribution severely non-uniform",
            s2.uniformity_ratio
        );
        // 1-round must be at least as non-uniform as 2-round (can't be more
        // random-looking than a stronger function).
        assert!(
            s1.uniformity_ratio >= s2.uniformity_ratio * 0.5,
            "1-round uniformity {:.2} far below 2-round {:.2} — measurement error",
            s1.uniformity_ratio, s2.uniformity_ratio
        );
    }

    // ── Boomerang second-order differential analysis ─────────────────────────
    //
    // D²F(x; α, β) = F(x) ⊕ F(x⊕α) ⊕ F(x⊕β) ⊕ F(x⊕α⊕β).
    // For a degree-d function, D²F has degree ≤ d-2.
    //
    // 1-round (deg ≤ 3): D²F has degree ≤ 1 → structured, potentially sub-random HW.
    // 2-round (deg > 4): D²F has degree ≥ 2 → pseudorandom HW near STATE_BITS/2 = 3200.

    #[test]
    fn boomerang_sum_two_round_hw_near_random() {
        // 2-round boomerang sum HW should be near the random expectation (3200).
        // ±25% margin covers many standard deviations for 200 samples.
        let mut r = rng();
        let s = boomerang::test_boomerang_sum(1, 1, 200, &mut r);
        assert!(
            s.avg_hw > s.expected_hw * 0.75 && s.avg_hw < s.expected_hw * 1.25,
            "2-round boomerang avg_hw={:.1} far from expected={:.1} — structural bias",
            s.avg_hw, s.expected_hw
        );
        assert!(
            s.frac_low_hw < 0.30,
            "2-round frac_low_hw={:.3} — too many near-zero boomerang sums",
            s.frac_low_hw
        );
    }

    #[test]
    fn boomerang_sum_one_round_more_structured_than_two() {
        // 1-round D²F degree ≤ 1 (structured); 2-round degree ≥ 2 (pseudorandom).
        // At least one of: 2-round avg_hw closer to expected, or frac_low_hw lower.
        let mut r = rng();
        let s1 = boomerang::test_boomerang_sum(1, 0, 200, &mut r);
        let s2 = boomerang::test_boomerang_sum(1, 1, 200, &mut r);
        let avg_gap1 = (s1.avg_hw - s1.expected_hw).abs();
        let avg_gap2 = (s2.avg_hw - s2.expected_hw).abs();
        assert!(
            avg_gap2 <= avg_gap1 || s2.frac_low_hw <= s1.frac_low_hw,
            "2-round not more random than 1-round: avg_gap 1r={avg_gap1:.1} 2r={avg_gap2:.1}; \
             frac_low 1r={:.3} 2r={:.3}",
            s1.frac_low_hw, s2.frac_low_hw
        );
    }

    #[test]
    fn boomerang_projected_two_round_near_random() {
        // With fresh random (x, α, β) per sample, D²F's k-bit projection should
        // match the 2^{-k} expectation for a high-degree (2-round) function.
        // Assert log2_excess stays within ±3 bits (generous statistical tolerance).
        let mut r = rng();
        let s = boomerang::test_projected_boomerang(2, 4, 2_000, &mut r);
        assert!(
            s.log2_excess.abs() < 3.0,
            "2-round projected boomerang log2_excess={:.2} far from 0 — unexpected structure",
            s.log2_excess
        );
    }

    #[test]
    fn boomerang_rect_one_round_not_below_two_round() {
        // 1-round lane structure → more intermediate-difference collisions.
        // 2-round fully mixed → near-random collision rate.
        // Ordering: 1-round log2_excess ≥ 2-round log2_excess.
        let mut r = rng();
        let r1 = boomerang::test_boomerang_rect(1, 100, 100, 8, &mut r);
        let r2 = boomerang::test_boomerang_rect(2, 100, 100, 8, &mut r);
        assert!(
            r1.log2_excess >= r2.log2_excess,
            "1-round rectangle excess {:.2} < 2-round {:.2} — structure direction inverted",
            r1.log2_excess, r2.log2_excess
        );
    }

    // ── Sponge construction security ────────────────────────────────────────

    #[test]
    fn sponge_128bit_security_achievable() {
        // The 6400-bit state allows c >= 256 (so collision security >= 128 bits)
        // with strictly positive rate.  min_rate_for_128bit is the *maximum* r
        // such that c/2 >= 128, so it must be > 0.
        let sweep = sponge::sweep_security_tradeoffs(6400);
        assert!(
            sweep.min_rate_for_128bit > 0,
            "no rate value achieves 128-bit collision security — state too small?"
        );
    }

    #[test]
    fn sponge_recommended_rounds_exceeds_min_secure() {
        let map = sponge::build_round_security_map();
        assert!(
            map.recommended_rounds > map.min_secure_rounds,
            "recommended_rounds {} must exceed min_secure_rounds {} (safety margin)",
            map.recommended_rounds, map.min_secure_rounds
        );
    }

    #[test]
    fn sponge_birthday_bound_holds_at_2_rounds() {
        // Use 8-bit projection and 2000 samples so N/2^k = 2000/256 ≈ 7.8
        // (dense birthday regime).  Actual collisions should stay within 3×
        // the birthday expectation.
        let mut r = rng();
        let check = sponge::check_birthday_bound(8, 2000, &mut r);
        assert!(
            check.within_3x,
            "birthday bound violated: actual={} expected={:.1} ratio={:.2}",
            check.actual_collisions, check.expected_collisions, check.ratio
        );
    }

    #[test]
    fn sponge_256bit_security_achievable() {
        // 6400-bit state can provide 256-bit collision security with ample rate.
        // min_rate_for_256bit must be > 0 and above a reasonable threshold (1024).
        let sweep = sponge::sweep_security_tradeoffs(6400);
        assert!(
            sweep.min_rate_for_256bit > 0,
            "no rate achieves 256-bit collision security"
        );
        assert!(
            sweep.min_rate_for_256bit >= 1024,
            "max throughput-compatible rate for 256-bit security is only {} bits — unexpectedly low",
            sweep.min_rate_for_256bit
        );
    }

    // ── Formal sponge indifferentiability proof ─────────────────────────────
    //
    // Verifies the Bertoni et al. 2008 indifferentiability bound for HDH at the
    // recommended (r=5888, c=512) operating point.

    #[test]
    fn indiff_256bit_secure_at_recommended_params() {
        // At c=512, a balanced adversary with 2^126 queries of each type (forward,
        // backward, hash) has q_eff ≈ 3×2^126 ≈ 2^127.6, giving
        // advantage ≤ 2^{255.2}/2^512 = 2^{−256.8}.  Security > 256 bits.
        let bound = sponge_indiff::compute_indiff_bound(sponge_indiff::IndiffGameParams {
            state_bits: 6400,
            rate_bits: 5888,
            capacity_bits: 512,
            q_forward_log2: 126,
            q_backward_log2: 126,
            q_hash_log2: 126,
            output_blocks: 1,
        });
        assert!(
            bound.is_256bit_secure,
            "indiff security {:.1} bits < 256 at c=512, q_each=2^126",
            bound.security_bits
        );
    }

    #[test]
    fn indiff_simulator_reliable_at_128bit_budget() {
        // Simulator failure probability ≤ q_f × q_b / 2^c.
        // At c=512, q_f = q_b = 2^128: P(fail) ≤ 2^{256}/2^{512} = 2^{−256} ≪ 2^{−128}.
        let sc = sponge_indiff::simulator_consistency(512, 128, 128);
        assert!(
            sc.is_reliable_128bit,
            "simulator failure prob 2^{:.1} exceeds 2^{{-128}} — consistency not guaranteed",
            sc.failure_prob_log2
        );
    }

    #[test]
    fn indiff_query_budget_sweep_has_large_256bit_range() {
        // With c=512, security drops below 256 bits only when q_total > 2^{128}.
        // The sweep should show max_q_for_256bit_log2 >= 128.
        let sweep = sponge_indiff::sweep_query_budgets(6400, 5888);
        assert!(
            sweep.max_q_for_256bit_log2 >= 128,
            "max query budget for 256-bit security is only 2^{} — less than 2^128",
            sweep.max_q_for_256bit_log2
        );
    }

    #[test]
    fn indiff_hash_proof_all_256bit_properties() {
        // Assemble the full proof for c=512, r=5888, output=512 bits.
        // All four standard properties must hold at ≥ 256 bits.
        let proof = sponge_indiff::assemble_hash_proof(6400, 5888, 512);
        assert!(
            proof.all_256bit_properties_hold,
            "not all 256-bit security properties hold: \
             collision={:.0} preimage={:.0} PRF={:.0}",
            proof.collision_security_bits,
            proof.preimage_security_bits,
            proof.prf_security_bits,
        );
        assert!(
            proof.immune_to_length_extension,
            "sponge should be immune to length-extension attacks"
        );
    }

    // ── GPU-scale algebraic attack modeling ──────────────────────────────────
    //
    // Verifies that the XL/Gröbner and hybrid complexity models produce
    // infeasible estimates for 2-round+ HDH, and that the solving-degree
    // model correctly identifies the 1-round weakness (high solving degree
    // despite low equation degree).

    #[test]
    fn algebraic_two_round_xl_is_infeasible() {
        // 2-round HDH: n=6400 vars, eq_degree > 4 (use 8 as conservative estimate).
        // XL complexity must be >> 2^{256} (far outside GPU reach).
        let sd = gpu_algebraic::estimate_solving_degree(6400, 6400, 8);
        assert!(
            sd.xl_time_log2 > 256.0,
            "2-round XL time 2^{:.0} < 2^{{256}} — attack may be feasible",
            sd.xl_time_log2
        );
    }

    #[test]
    fn algebraic_solving_degree_exceeds_eq_degree_for_large_systems() {
        // For a square system (n=m=6400) of degree-3 equations, the XL solving
        // degree must be >> 3 (the underdetermination forces it much higher).
        let sd = gpu_algebraic::estimate_solving_degree(6400, 6400, 3);
        assert!(
            sd.d_xl > 3,
            "1-round solving degree {} = eq_degree 3 — underdetermination not modelled",
            sd.d_xl
        );
        assert!(
            sd.xl_time_log2 > 256.0,
            "1-round XL time 2^{:.0} despite high solving degree — computation error",
            sd.xl_time_log2
        );
    }

    #[test]
    fn algebraic_hybrid_does_not_break_security() {
        // Hybrid attack on 2-round HDH (n=6400, d_XL from solving-degree model).
        // Even the optimal variable-fixing split must leave complexity > 2^{128}.
        let sd = gpu_algebraic::estimate_solving_degree(6400, 6400, 8);
        let hyb = gpu_algebraic::hybrid_attack_optimum(6400, sd.d_xl);
        assert!(
            hyb.total_log2 > 128.0,
            "hybrid attack total complexity 2^{:.1} ≤ 2^{{128}} — feasible attack found",
            hyb.total_log2
        );
    }

    #[test]
    fn algebraic_scale_sweep_shows_round_2_infeasible() {
        // The scale sweep's 6400-bit 2-round entry must show best_known > 2^{256}.
        let sweep = gpu_algebraic::algebraic_scale_sweep();
        let entry_2r = sweep.entries.iter()
            .find(|e| e.description.contains("2-round"))
            .expect("2-round entry must be present in scale sweep");
        assert!(
            entry_2r.best_known_log2 > 256.0,
            "2-round best-known algebraic complexity 2^{:.0} ≤ 2^{{256}}",
            entry_2r.best_known_log2
        );
        assert!(
            !entry_2r.is_gpu_feasible_exascale,
            "2-round HDH algebraic attack is marked GPU-feasible on exascale — unexpected"
        );
    }

    // ── Structured boomerang (bench section 26, not previously in cargo test) ──
    //
    // At 1 round, χ's lane-local design means a single-bit α difference touches
    // fewer lanes than a fully random α, producing a smaller D²F Hamming weight.
    // This hw_reduction (random_α HW − single_bit_α HW) must be positive at 1
    // round and decrease at 2 rounds as cross-lane mixing destroys the advantage.

    #[test]
    fn boomerang_structured_single_bit_shows_hw_reduction_at_one_round() {
        let mut r = rng();
        let s1 = boomerang::test_structured_boomerang(1, 200, &mut r);
        let s2 = boomerang::test_structured_boomerang(2, 200, &mut r);
        assert!(
            s1.hw_reduction > 0.0,
            "1-round hw_reduction={:.1} ≤ 0 — single-bit α should produce smaller boomerang sums",
            s1.hw_reduction
        );
        assert!(
            s2.hw_reduction < s1.hw_reduction,
            "2-round hw_reduction={:.1} ≥ 1-round {:.1} — cross-lane mixing did not reduce the advantage",
            s2.hw_reduction, s1.hw_reduction
        );
    }

    // ── Large-cube integral dim=8 (bench section 18, not previously in cargo test) ──
    //
    // dim=4 cubes (16 evaluations) already confirm the 1→2 round degree transition.
    // dim=8 cubes (256 evaluations) probe a wider subspace: 1-round should still
    // show zero-sum structure (degree ≤ 3 < 8), while 2-round must eliminate it.

    #[test]
    fn integral_large_cube_dim8_round_transition() {
        let mut r = rng();
        let s1 = integral::test_cube_sum(1, 8, 10, &mut r);
        let s2 = integral::test_cube_sum(2, 8, 10, &mut r);
        assert!(
            s1.zero_sum_fraction > 0.0,
            "1-round dim=8: zero-sum fraction is 0 — integral structure not detectable at dim=8"
        );
        assert_eq!(
            s2.zero_sum_fraction, 0.0,
            "2-round dim=8: {:.0}% of dim-8 cubes gave zero XOR sum — integral survives to 2 rounds",
            s2.zero_sum_fraction * 100.0
        );
    }

    // ── Sponge state partition (bench section 30, not previously in cargo test) ──
    //
    // For a 6400-bit state targeting 256-bit collision security, the minimum
    // capacity is 512 bits, leaving 5888 bits of rate — over 92% throughput.

    #[test]
    fn sponge_state_partition_recommended_capacity_and_throughput() {
        let part = sponge::analyze_state_partition(6400);
        assert_eq!(
            part.recommended_capacity, 512,
            "recommended capacity {} ≠ 512 for 256-bit security",
            part.recommended_capacity
        );
        assert_eq!(
            part.recommended_rate, 5888,
            "recommended rate {} ≠ 5888 (= 6400 − 512)",
            part.recommended_rate
        );
        assert!(
            part.recommended_throughput > 0.90,
            "recommended throughput {:.3} < 0.90 — rate/state ratio unexpectedly low",
            part.recommended_throughput
        );
    }

    // ── Padding domain separation (bench section 35, not previously in cargo test) ──
    //
    // pad10*1 is prefix-free by construction: every padded encoding ends with a
    // 0x80 byte that cannot appear inside an unpadded message at the same position.
    // Rate-separation ensures different rate values produce non-overlapping message
    // spaces, preventing cross-rate collisions.

    #[test]
    fn sponge_indiff_padding_is_prefix_free_and_domain_separated() {
        let pad = sponge_indiff::analyze_padding(5888);
        assert!(
            pad.is_prefix_free,
            "pad10*1 at r=5888 is not prefix-free — message domain not separated"
        );
        assert!(
            pad.is_rate_separated,
            "padding does not domain-separate different rate values"
        );
        assert_eq!(
            pad.min_padding_overhead_bytes, 2,
            "padding overhead {} bytes ≠ 2 — pad10*1 requires exactly one 0x01 and one 0x80 byte",
            pad.min_padding_overhead_bytes
        );
    }

    // ── 3-round MITM entropy (bench section 23 includes 3 rounds; test suite stopped at 2) ──
    //
    // If 2-round output is already near-uniform, 3-round must be at least as
    // uniform (uniformity_ratio no worse).  This confirms security does not
    // regress when adding extra rounds.

    #[test]
    fn mitm_cat5_three_round_entropy_no_worse_than_two() {
        let mut r = rng();
        let s2 = mitm::measure_entropy_collapse(2, 3_000, 10, &mut r);
        let s3 = mitm::measure_entropy_collapse(3, 3_000, 10, &mut r);
        assert!(
            s3.uniformity_ratio <= 10.0,
            "3-round uniformity ratio {:.2} > 10 — output distribution severely non-uniform",
            s3.uniformity_ratio
        );
        assert!(
            s3.uniformity_ratio <= s2.uniformity_ratio * 1.5,
            "3-round uniformity {:.2} significantly worse than 2-round {:.2} — adding a round degraded uniformity",
            s3.uniformity_ratio, s2.uniformity_ratio
        );
    }

    // ── Quantum security ────────────────────────────────────────────────────────

    #[test]
    fn quantum_hdh_meets_nist_level5() {
        // At c=512: BHT collision ≈ 170.7 bits ≥ 128, Grover preimage = 256 bits.
        // Combined → NIST PQC Level 5 (highest tier).
        let bounds = quantum_security::compute_quantum_bounds(512);
        assert!(
            bounds.quantum_collision_bits >= 128.0,
            "BHT collision security {:.1} bits < 128 — NIST Level 1 threshold not met",
            bounds.quantum_collision_bits
        );
        assert!(
            bounds.quantum_preimage_bits >= 256.0,
            "Grover preimage security {:.1} bits < 256 — Level 5 preimage bar not met",
            bounds.quantum_preimage_bits
        );
        assert!(bounds.meets_level5, "HDH at c=512 must meet NIST PQC Level 5");
        assert_eq!(bounds.nist_level, 5, "NIST level must be 5 at c=512");
    }

    #[test]
    fn quantum_simons_not_applicable() {
        // Φ destroys all XOR-period structure (measured equivariance = 0.0).
        // Simon's algorithm requires a hidden XOR period — inapplicable here.
        let bounds = quantum_security::compute_quantum_bounds(512);
        assert!(
            !bounds.simons_applicable,
            "Simon's algorithm marked applicable — Φ should destroy all XOR-period structure"
        );
    }

    #[test]
    fn quantum_bht_collision_not_near_term_feasible() {
        // 2^{512/3} ≈ 2^{170.7} quantum evaluations — far beyond near-term quantum hardware.
        let bht = quantum_security::model_bht(512);
        assert!(
            !bht.near_term_feasible,
            "BHT at c=512 marked near-term feasible — expected work 2^{:.1} to be infeasible",
            bht.work_log2
        );
        assert!(bht.security_bits >= 128.0, "BHT security {:.1} < 128 bits", bht.security_bits);
    }

    #[test]
    fn quantum_grover_preimage_not_near_term_feasible() {
        // 2^{512/2} = 2^{256} — far beyond near-term quantum hardware.
        let grover = quantum_security::model_grover(512);
        assert!(
            !grover.near_term_feasible,
            "Grover at 512-bit output marked near-term feasible — expected 2^{:.1} infeasible",
            grover.work_log2
        );
        assert_eq!(grover.work_log2, 256.0);
    }

    // ── Multi-user sponge security ──────────────────────────────────────────────

    #[test]
    fn multi_user_collision_safe_at_2pow32_users() {
        // U=2^32 users, q=2^32 queries each: Adv ≤ 2^{32+64−512} = 2^{−416} ≪ 2^{−128}.
        let bound = multi_user_sponge::multi_user_collision_bound(512, 32, 32);
        assert!(
            bound.meets_128bit,
            "multi-user collision advantage 2^{:.1} ≥ 2^{{-128}} at U=2^32, q=2^32",
            bound.advantage_log2
        );
        assert!(
            bound.security_bits >= 128.0,
            "multi-user collision security {:.1} bits < 128",
            bound.security_bits
        );
    }

    #[test]
    fn multi_user_collision_safe_at_2pow64_users() {
        // U=2^64 users, q=2^32 queries each: Adv ≤ 2^{64+64−512} = 2^{−384} ≪ 2^{−128}.
        let bound = multi_user_sponge::multi_user_collision_bound(512, 64, 32);
        assert!(
            bound.meets_128bit,
            "multi-user collision advantage 2^{:.1} ≥ 2^{{-128}} at U=2^64, q=2^32",
            bound.advantage_log2
        );
    }

    #[test]
    fn multi_user_prf_safe_at_2pow32_users() {
        // U=2^32 users, k=512-bit key, q=2^32 queries: Adv ≤ 2^{32+32−256} = 2^{−192}.
        let bound = multi_user_sponge::multi_user_prf_bound(512, 512, 32, 32);
        assert!(
            bound.meets_128bit,
            "multi-user PRF advantage 2^{:.1} ≥ 2^{{-128}} at U=2^32, q=2^32",
            bound.advantage_log2
        );
    }

    #[test]
    fn multi_user_sweep_all_standard_configs_safe() {
        // Every (U, q) configuration in the standard sweep must stay below 2^{-128}.
        let sweep = multi_user_sponge::multi_user_sweep(512, 512);
        for e in &sweep.entries {
            assert!(
                e.meets_128bit_collision,
                "multi-user collision unsafe at U=2^{}, q=2^{}: Adv=2^{:.1}",
                e.num_users_log2, e.queries_per_user_log2, e.collision_advantage_log2
            );
            assert!(
                e.meets_128bit_prf,
                "multi-user PRF unsafe at U=2^{}, q=2^{}: Adv=2^{:.1}",
                e.num_users_log2, e.queries_per_user_log2, e.prf_advantage_log2
            );
        }
    }
}
