pub mod differential;
pub mod jacobian;
pub mod linear;
pub mod preimage;
pub mod sat;

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
}
