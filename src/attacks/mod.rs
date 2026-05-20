pub mod differential;
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
        // With 20k samples across 2^64 buckets the expected max count is ~1.
        // A bias > 0.01 would mean a single ΔG appears ≥ 200 times — a
        // clear structural shortcut.
        assert!(
            bias < 0.01,
            "differential bias too high: {bias:.6} — possible shortcut"
        );
    }

    #[test]
    fn differential_nonzero_diff_is_uniform() {
        let mut r = rng();
        // All-same-component difference: only Δx1 nonzero.
        let stats =
            differential::analyze_differential((1u64, 0, 0, 0), 10_000, &mut r);
        // All observed ΔG values should be distinct (or nearly so) — the
        // differential must not collapse to a constant.
        assert!(
            stats.unique_output_diffs > 9_000,
            "differential output collapsed: only {} unique values",
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
            "linear bias too high: max={:.4} — linear approximation exists",
            stats.max_bias
        );
    }

    // ── Preimage counting (8-bit) ───────────────────────────────────────────

    #[test]
    fn preimage_count_sanity() {
        // chi8 has 32-bit input / 32-bit output.  For a near-random surjection
        // the average preimage count must be 1 and max must be bounded.
        let mut r = rng();
        let stats = preimage::analyze_preimages(2_000, &mut r);
        assert!(
            stats.avg_preimage_count > 0.5 && stats.avg_preimage_count < 2.0,
            "avg preimage count {:.3} outside expected [0.5, 2.0]",
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
        assert!(
            preimage::chi8_roundtrip_check(),
            "chi8 preimage count exceeded 256 — logic error"
        );
    }

    // ── Algebraic degree (4-bit ANF) ────────────────────────────────────────

    #[test]
    fn algebraic_degree_carries_reach_high_degree() {
        let stats = preimage::analyze_algebraic_degree();

        // The low-order product bits (bits 0, 1 of x1*x2 and x3*x4) are
        // inherently degree-2 (no carry contribution), so min_degree == 2 is
        // structurally expected, not a vulnerability.  Those degree-2 bits give
        // at most 64 equations in 256 unknowns — far too few to recover inputs.
        //
        // What matters for security is that the carry-chain pushes high-order
        // output bits to degree ≥ 4 in the 4-bit model (degree ≈ 32+ at 64 bits),
        // making the full constraint system unsolvable by XL or Gröbner at D=2.
        assert!(
            stats.max_degree >= 4,
            "max algebraic degree {} < 4 — carry chain not contributing; \
             system may be vulnerable to degree-2 algebraic attacks",
            stats.max_degree
        );

        // Verify that the majority of output bits are high-degree (> 2), confirming
        // the degree-2 subset is a small minority that cannot overdetermine the system.
        let high_degree_count = stats.degrees.iter().filter(|&&d| d > 2).count();
        let total = stats.degrees.len();
        assert!(
            high_degree_count * 2 >= total,
            "only {high_degree_count}/{total} output bits have degree > 2; \
             degree-2 bits may form an overdetermined subsystem"
        );
    }

    // ── SAT / XL model ─────────────────────────────────────────────────────

    #[test]
    fn xl_model_dimensions_are_correct() {
        let m = sat::model_xl_complexity();
        assert_eq!(m.input_vars, 256);
        assert_eq!(m.output_bits_per_pair, 256);
        // Degree-2 linearization must not be sufficient.
        assert!(!m.degree2_xl_sufficient);
        // Required degree must be at least 16 (conservative lower bound).
        assert!(
            m.required_xl_degree_lower_bound >= 16,
            "required XL degree {} too low",
            m.required_xl_degree_lower_bound
        );
    }

    #[test]
    fn dpll_search_space_is_exponential() {
        let m = sat::model_dpll_complexity();
        // After unit propagation, ≥ 32 free variables must remain,
        // giving a search tree of size ≥ 2^32.
        assert!(
            m.free_vars_after_up >= 32,
            "DPLL free vars {} after UP is too low — solver may be efficient",
            m.free_vars_after_up
        );
    }
}
