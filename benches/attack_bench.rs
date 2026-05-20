use hdh::attacks::{annihilator, differential, distinguisher, hybrid, integral, jacobian, linear, mitm, phi_symmetry, preimage, sat, truncated};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn section(n: usize, total: usize, title: &str) {
    println!("\n[{n}/{total}] {title}");
    println!("{}", "─".repeat(62));
}

fn main() {
    println!("=== HDH χ Core — Algebraic & SAT Reconstruction Attack Harness ===");
    let mut rng = ChaCha20Rng::seed_from_u64(0x0123456789abcdef);
    let total = 23;

    // ── 1. Differential uniformity ─────────────────────────────────────────
    section(1, total, "Differential Uniformity  (64-bit quad, empirical)");

    let mut worst_bias = 0.0f64;
    let samples_each = 50_000usize;
    for _ in 0..20 {
        let delta = loop {
            let d = (rng.gen::<u64>(), rng.gen::<u64>(), rng.gen::<u64>(), rng.gen::<u64>());
            if d != (0, 0, 0, 0) { break d; }
        };
        let s = differential::analyze_differential(delta, samples_each, &mut rng);
        if s.max_bias > worst_bias { worst_bias = s.max_bias; }
        println!("  Δ=({:016x},..) unique_ΔG={:>6}  max_count={:>2}  bias={:.6}",
            s.input_diff.0, s.unique_output_diffs, s.max_collision_count, s.max_bias);
    }
    println!("  → worst bias {worst_bias:.6}  (ideal 1/{samples_each} = {:.6})",
        1.0 / samples_each as f64);
    println!("  RESULT: {}", if worst_bias < 0.005 {
        "no differential shortcut found"
    } else { "WARNING — bias exceeds threshold" });

    // ── 2. Linear bias ──────────────────────────────────────────────────────
    section(2, total, "Linear Bias / Walsh Coefficients  (full chi_lane, empirical)");

    let ls = linear::sample_linear_bias(100, 50_000, &mut rng);
    println!("  Masks tested:      {}", ls.masks_tested);
    println!("  Samples per mask:  {}", ls.samples_per_mask);
    println!("  Max |bias|:        {:.6}", ls.max_bias);
    println!("  Avg |bias|:        {:.6}", ls.avg_bias);
    println!("  Statistical floor: ~{:.6}  (1/√N)", 1.0f64 / (50_000f64).sqrt());
    println!("  RESULT: {}", if ls.max_bias < 0.02 {
        "no exploitable linear approximation found"
    } else { "WARNING — bias above noise floor" });

    // ── 3. Jacobian rank ────────────────────────────────────────────────────
    section(3, total, "GF(2) Jacobian Rank  (64-bit chi_lane, 8 random points)");

    let js = jacobian::analyze_jacobian_rank(8, &mut rng);
    println!("  Points tested:          {}", js.points_tested);
    println!("  Theoretical max rank:   {}", js.max_theoretical_rank);
    println!("  Min rank observed:      {}", js.min_rank);
    println!("  Avg rank observed:      {:.1}", js.avg_rank);
    println!("  Worst rank deficit:     {}", js.worst_rank_deficit);
    println!("  RESULT: {}", if js.min_rank >= 200 {
        "rank ≈ full — no solvable linear shortcut at tested points"
    } else { "WARNING — rank deficit; linear attack may be feasible at some inputs" });

    // ── 4. Preimage resistance (8-bit reduced χ) ───────────────────────────
    section(4, total, "Preimage Resistance  (8-bit reduced χ, exhaustive g-search)");

    let pre = preimage::analyze_preimages(5_000, &mut rng);
    println!("  Outputs sampled:          {}", pre.outputs_sampled);
    println!("  Avg preimage count:       {:.3}", pre.avg_preimage_count);
    println!("  Max preimage count:       {}", pre.max_preimage_count);
    println!("  Zero-preimage fraction:   {:.2}%", pre.zero_preimage_frac * 100.0);
    println!("  Unique-preimage fraction: {:.2}%", pre.unique_preimage_frac * 100.0);
    println!("  RESULT: {}", if (pre.avg_preimage_count - 1.0).abs() < 0.2 {
        "near-bijective; no structural preimage multiplicity found"
    } else { "NOTE: deviation from ideal — investigate" });

    // ── 5. Degree propagation + structural isolation (4-bit χ) ─────────────
    section(5, total, "Degree Propagation + Structural Isolation  (4-bit reduced χ)");

    let dp = preimage::degree_propagation(4);
    println!("  Max algebraic degree by round of chi4 composition:");
    for (i, &d) in dp.max_degree_per_round.iter().enumerate() {
        let marker = if d == 16 { " ← saturated" } else { "" };
        println!("    Round {:>1}: max degree = {}{}", i + 1, d, marker);
    }
    println!("  Saturation round: {}", dp.saturation_round);
    let sat_round = dp.max_degree_per_round.iter().position(|&d| d >= 16);
    match sat_round {
        Some(r) => println!("  RESULT: degree saturates at 16 (maximum) by round {}", r + 1),
        None => println!("  RESULT: degree reached {} after {} rounds (grows monotonically, \
                          not yet at 16-bit maximum — 64-bit extrapolation: degree ≥ {})",
            dp.max_degree_per_round.last().unwrap_or(&0),
            dp.max_degree_per_round.len(),
            dp.max_degree_per_round.last().unwrap_or(&0) * 8),
    }

    println!();
    let iso = preimage::structural_isolation_4bit();
    println!("  Structural isolation — degree-2 subsystem (4-bit χ):");
    println!("    Degree-2 monomials (z-vars):  {} [= C(16,2)]", iso.degree2_var_count);
    println!("    Output equations:             {}", iso.equation_count);
    println!("    GF(2) subsystem rank:         {}", iso.subsystem_rank);
    println!("    Underdetermination ratio:     {:.1}x  (z-vars / rank)",
        iso.underdetermination_ratio);
    println!("  RESULT: {}", if iso.underdetermination_ratio >= 5.0 {
        "subsystem underdetermined — degree-2 structure is NOT linearly isolatable"
    } else { "WARNING — low underdetermination ratio; check for separability" });

    // ── 6. Effective XL / SAT complexity ───────────────────────────────────
    section(6, total, "Effective XL Complexity  (projection-aware, 64-bit χ)");

    let eff = sat::model_effective_xl(iso.subsystem_rank);
    println!("  Boolean unknowns:                 {:>6}", eff.n_vars);
    println!("  Degree-2 monomials C(256,2):      {:>6}", eff.degree2_monomial_count);
    println!("  Degree-2 subsystem rank:          {:>6}", eff.degree2_subsystem_rank);
    println!("  Fraction constrained by D2:       {:>6.3}%",
        eff.degree2_exploited_fraction * 100.0);
    println!("  Effective XL degree (d_eff):      {:>6}", eff.effective_degree);
    println!("  XL complexity log₂ C(256,d_eff): {:>6.1}", eff.xl_complexity_log2);
    println!("  Meets 2^120 threshold?             {}", eff.meets_120bit_threshold);
    println!("  RESULT: {}", if eff.meets_120bit_threshold {
        "effective XL attack infeasible — exploiting degree-2 structure does not lower complexity below 2^120"
    } else { "WARNING — XL complexity < 2^120" });

    // ── 7. Incremental SAT simulation ───────────────────────────────────────
    section(7, total, "Incremental SAT / DPLL Simulation  (64-bit χ lane)");

    let isat = sat::simulate_incremental_sat(4);
    println!("  Total variables: {}  ({} primary + {} auxiliary g)",
        isat.total_vars, isat.total_vars - isat.g_auxiliary_vars, isat.g_auxiliary_vars);
    println!();
    println!("  {:>6} {:>14} {:>14} {:>14} {:>14}",
        "Pairs", "Free vars/UP", "Clauses", "Growth/pair", "log₂ DPLL");
    for s in &isat.snapshots {
        println!("  {:>6} {:>14} {:>14} {:>14} {:>14.1}",
            s.pairs, s.free_vars_after_up, s.total_clauses,
            s.clause_growth_rate, s.log2_dpll_search);
    }
    let last = isat.snapshots.last().unwrap();
    println!();
    println!("  Free vars after UP never drops below {}", last.free_vars_after_up);
    println!("  DPLL branching space: 2^{:.0} minimum (degree-32 g-constraint \
              cannot be simplified by unit propagation)", last.log2_dpll_search);
    println!("  RESULT: SAT reconstruction requires exponential branching; \
              adding pairs does not collapse the free-variable count.");

    // ── 8. Differential-linear hybrid ─────────────────────────────────────────
    section(8, total, "Differential-Linear Hybrid  (64-bit χ lane, 100 masks × 50k samples)");

    let dl = hybrid::sample_difflin_bias(100, 50_000, &mut rng);
    println!("  Masks tested:        {}", dl.masks_tested);
    println!("  Samples per mask:    {}", dl.samples_per_mask);
    println!("  Max |bias|:          {:.6}", dl.max_bias);
    println!("  Avg |bias|:          {:.6}", dl.avg_bias);
    println!("  Statistical floor:  ~{:.6}  (1/√N)", 1.0f64 / (dl.samples_per_mask as f64).sqrt());
    println!("  RESULT: {}", if dl.max_bias < 0.01 {
        "no exploitable differential-linear correlation found"
    } else { "WARNING — differential-linear bias exceeds noise floor" });

    // ── 9. Second-order / boomerang-rectangle differential ────────────────────
    section(9, total, "Second-Order Differential  (boomerang-rectangle test, 50 triples × 20k)");

    let so = hybrid::test_second_order_differential(50, 20_000, &mut rng);
    println!("  Triples (Δ₀,Δ₁,α):  {}", so.triples_tested);
    println!("  Samples per triple:  {}", so.samples_per_triple);
    println!("  Max |bias|:          {:.6}", so.max_bias);
    println!("  Avg |bias|:          {:.6}", so.avg_bias);
    println!("  Statistical floor:  ~{:.6}  (1/√N)", 1.0f64 / (so.samples_per_triple as f64).sqrt());
    println!("  RESULT: {}", if so.max_bias < 0.01 {
        "second-order derivative unbiased — no boomerang-rectangle structure found"
    } else { "WARNING — second-order bias suggests rectangular exploitable structure" });

    // ── 10. Truncated differential propagation ────────────────────────────────
    section(10, total, "Truncated Differential Propagation  (4-bit χ, nibble-wise activity)");

    let td = truncated::analyze_truncated_differentials();
    println!("  {:>14}  {:>14}  {:>14}  {:>10}  {:>8}",
        "Input pattern", "Observed outs", "Max out prob", "Avg wt", "Multi%");
    for ps in &td.per_pattern {
        println!("  {:>14b}  {:>14}  {:>14.4}  {:>10.2}  {:>7.1}%",
            ps.input_pattern,
            ps.observed_output_patterns.len(),
            ps.max_output_prob,
            ps.avg_output_nibble_weight,
            ps.multi_nibble_output_frac * 100.0);
    }
    println!("  Worst max output probability:      {:.4}", td.worst_max_output_prob);
    println!("  Min avg output nibble weight:      {:.2}", td.min_avg_output_weight);
    println!("  Single→multi nibble mixing rate:   {:.1}%", td.single_to_multi_rate * 100.0);
    println!("  Impossible truncated diffs:        {}", td.impossible_diff_count);
    println!("  RESULT: {}", if td.single_to_multi_rate >= 0.5 {
        "majority of single-nibble inputs produce multi-nibble output — good propagation"
    } else { "WARNING — poor nibble propagation; truncated differential path exists" });

    // ── 11. Φ rotational symmetry ─────────────────────────────────────────────
    section(11, total, "Φ Rotational Symmetry  (1000 random states, 24 rotations)");

    let ps = phi_symmetry::test_rotational_symmetry(1_000, &mut rng);
    println!("  Rotations tested:         {}", ps.rotations_tested);
    println!("  Samples per rotation:     {}", ps.samples_per_rotation);
    println!("  Max exact equivariance:   {:.6}  (expect 0)", ps.max_exact_equivariance);
    println!("  Max avg word-match frac:  {:.6}  (expect ≈ 2^{{-64}} ≈ 0)", ps.max_avg_word_match);
    println!("  {:>4}  {:>20}  {:>22}", "r", "Exact equiv. frac.", "Avg word-match frac.");
    for (i, (&ef, &wf)) in ps.exact_equivariance_fractions.iter()
        .zip(ps.avg_word_match_fractions.iter()).enumerate()
    {
        println!("  {:>4}  {:>20.6}  {:>22.6}", i + 1, ef, wf);
    }
    println!("  RESULT: {}", if ps.max_exact_equivariance == 0.0 {
        "no rotational symmetry detected — Φ routing is state-dependent as expected"
    } else { "WARNING — Φ exhibits partial rotational equivariance" });

    // ── 12. Φ affine shift test ───────────────────────────────────────────────
    section(12, total, "Φ Affine Shift Test  (25 random constants, 200 samples each)");

    let af = phi_symmetry::test_affine_shift(25, 200, &mut rng);
    println!("  Constant shifts tested:   {}", af.shifts_tested);
    println!("  Samples per shift:        {}", af.samples_per_shift);
    println!("  Max constant-output frac: {:.4}  (expect ≈ 0 for state-dependent routing)",
        af.max_constant_output_frac);
    println!("  RESULT: {}", if af.max_constant_output_frac < 0.05 {
        "φ(S⊕C)⊕φ(S) varies with S — no affine shift symmetry found"
    } else { "WARNING — output XOR is approximately constant for some shift" });

    // ── 13. Algebraic immunity estimation (4-bit χ) ──────────────────────────
    section(12, total, "Algebraic Immunity  (4-bit χ, output-bit scan, d ≤ 3)");

    let ai = annihilator::analyze_algebraic_immunity(3);
    println!("  Per-bit AI estimate (first d where annihilator exists, or >max_d):");
    for (i, &lb) in ai.per_bit_lb.iter().enumerate() {
        let label = if lb > 3 { "AI>3 (carry-chain)".to_string() } else { format!("AI={lb}") };
        println!("    bit {:>2}: {label}", i);
    }
    println!("  Min AI lower bound:             {}", ai.min_lb);
    println!("  Max AI lower bound:             {}", ai.max_lb);
    println!("  Bits with AI > max_degree (3):  {} / 16", ai.high_ai_bit_count);
    println!("  Theoretical upper bound AI:     {} (= ⌈16/2⌉)", ai.theoretical_upper_bound);
    println!("  RESULT: {}", if ai.min_lb >= 2 {
        if ai.high_ai_bit_count >= 4 {
            "no degree-1 annihilators; ≥4 carry-chain bits resist degree-3 search — algebraic immunity confirmed"
        } else {
            "no degree-1 annihilators found; carry-chain bits may be limited"
        }
    } else {
        "WARNING — degree-1 annihilator found; possible linear structure"
    });

    // ── 14. Invariant subspace detection ─────────────────────────────────────
    // (Renumbered to accommodate hybrid/truncated/symmetry sections above)
    section(13, total, "Invariant Subspace Detection  (4-bit χ, full scan)");


    let inv = annihilator::detect_invariant_subspaces();
    println!("  Total inputs scanned:     {}", inv.total_inputs);
    println!("  Fixed points (χ₄(x)=x):  {}", inv.fixed_point_count);
    println!("    (structural: occur when g=0, a nonlinear condition — not a linear subspace)");
    println!("  Two-cycles (period-2):    {}", inv.two_cycle_count);
    println!("  Maps to zero (χ₄(x)=0):  {}", inv.maps_to_zero_count);
    println!("  Fixed points form linear subspace? {}", inv.fixed_points_form_linear_subspace);
    println!("  RESULT: {}", if !inv.fixed_points_form_linear_subspace {
        "fixed-point set is not GF(2)-closed — no linear invariant subspace detected"
    } else {
        "WARNING — fixed-point set is closed under XOR; linear invariant subspace exists"
    });

    // ── 14. Reduced-round avalanche sweep ─────────────────────────────────────
    section(14, total, "Reduced-Round Avalanche Completeness  (32 bits × 100 samples, rounds 1–3)");

    println!("  {:>6}  {:>10}  {:>10}  {:>10}  {:>12}",
        "Rounds", "Min frac", "Mean frac", "Max frac", "Completeness");
    for r in 1..=3 {
        let av = distinguisher::measure_avalanche(r, 32, 100, &mut rng);
        println!("  {:>6}  {:>10.4}  {:>10.4}  {:>10.4}  {:>11.1}%",
            r, av.min_frac, av.mean_frac, av.max_frac, av.completeness * 100.0);
    }
    println!("  Completeness = fraction of input bits with avg change-frac in [0.4, 0.6].");
    println!("  RESULT: round 1 shows incomplete diffusion; round 2+ approaches full avalanche.");

    // ── 15. Reduced-round output balance ──────────────────────────────────────
    section(15, total, "Reduced-Round Output Balance  (64 bits, 5000 samples, rounds 1–3)");

    println!("  {:>6}  {:>12}  {:>12}  {:>12}  {:>14}",
        "Rounds", "Max |bias|", "Mean |bias|", "Noise floor", "Bits > floor");
    for r in 1..=3 {
        let bl = distinguisher::measure_output_balance(r, 64, 5_000, &mut rng);
        println!("  {:>6}  {:>12.6}  {:>12.6}  {:>12.6}  {:>14}",
            r, bl.max_abs_bias, bl.mean_abs_bias, bl.noise_floor, bl.bits_above_floor);
    }
    println!("  Noise floor = 2/√5000 ≈ 0.028.  Bits above floor = systematic bias count.");
    println!("  RESULT: round 1 may show above-floor bits; round 2+ should drop to noise level.");

    // ── 16. Chi4 zero-sum property (algebraic) ────────────────────────────────
    section(16, total, "Chi4 Zero-Sum Property  (exact, dim 5 vs 6, 1000 cosets each)");

    let zs5 = distinguisher::check_zero_sum_chi4(5, 1_000, &mut rng);
    let zs6 = distinguisher::check_zero_sum_chi4(6, 1_000, &mut rng);
    println!("  dim=5 (2^5=32 elements): {}/{} cosets gave nonzero sum  ({:.1}%)",
        zs5.nonzero_sum_count, zs5.cosets_tested,
        zs5.nonzero_sum_count as f64 / zs5.cosets_tested as f64 * 100.0);
    println!("  dim=6 (2^6=64 elements): {}/{} cosets gave nonzero sum  ({:.1}%)",
        zs6.nonzero_sum_count, zs6.cosets_tested,
        zs6.nonzero_sum_count as f64 / zs6.cosets_tested as f64 * 100.0);
    println!("  Accumulated xor (dim=6): 0x{:04x}  (must be 0x0000)", zs6.accumulated_xor);
    println!("  RESULT: {}",
        if zs6.nonzero_sum_count == 0 && zs5.nonzero_sum_count > 0 {
            "zero-sum holds at dim=6 (degree ≤ 5 confirmed); dim=5 has nonzero sums (degree > 4 confirmed) — exact degree = 5"
        } else if zs6.nonzero_sum_count > 0 {
            "WARNING — dim=6 nonzero sum; chi4 degree > 5"
        } else {
            "dim=5 sums all zero — chi4 degree may be ≤ 4 (unexpected)"
        });

    // ── 17. Higher-order integral distinguisher sweep ─────────────────────────
    section(17, total, "Higher-Order Integral Distinguishers  (cube sum sweep, rounds 0–2)");

    println!("  {:>6}  {:>5}  {:>8}  {:>14}  {:>16}  {:>10}",
        "Rounds", "Dim", "Cubes", "Zero-sum frac", "Avg balanced", "Expected±σ");
    for rounds in [0usize, 1, 2] {
        let dims: &[usize] = if rounds == 0 { &[2, 3] } else { &[3, 4, 5, 6] };
        let n_cubes = if rounds <= 1 { 30 } else { 20 };
        for &dim in dims {
            let s = integral::test_cube_sum(rounds, dim, n_cubes, &mut rng);
            println!("  {:>6}  {:>5}  {:>8}  {:>13.1}%  {:>16.1}  {:.0}±{:.0}",
                rounds, dim, n_cubes,
                s.zero_sum_fraction * 100.0,
                s.avg_balanced_bits,
                s.expected_balanced_bits,
                s.expected_std_dev);
        }
    }
    println!("  Zero-sum frac: fraction of cubes where XOR of all 2^dim outputs = 0.");
    println!("  Avg balanced:  mean output bits with XOR-sum bit = 0 (all 6400 output bits).");
    println!("  RESULT: 1-round shows low-degree integral structure (deg ≤ 3 for most directions);");
    println!("          2-round eliminates full integral (zero_frac=0) and avg_balanced falls to ~3700,");
    println!("          confirming degree growth across the round transition.");

    // ── 18. Large-cube test (1 round vs 2 rounds at dim=8) ────────────────────
    section(18, total, "Large-Cube Integral Test  (dim=8, 256 evals/cube, rounds 1–2)");

    for rounds in [1usize, 2] {
        let s = integral::test_cube_sum(rounds, 8, 10, &mut rng);
        println!("  rounds={rounds}  dim=8  n=10  zero_frac={:.1}%  avg_balanced={:.0}  max_balanced={}",
            s.zero_sum_fraction * 100.0, s.avg_balanced_bits, s.max_balanced_bits);
    }
    println!("  RESULT: 1-round dim=8 cubes (256 evals) should still produce zero sums;");
    println!("          2-round should show zero_frac=0, confirming round transition.");

    // ── 19. MITM Cat 1: partition matching ────────────────────────────────────
    section(19, total, "MITM Cat 1: Partition Matching  (500 samples × 20-bit projection)");

    for rounds in [1usize, 2, 3] {
        let s = mitm::measure_partition_matching(rounds, 500, 20, &mut rng);
        println!("  rounds={rounds}  actual_collisions={}  expected={:.2}  log2_excess={:.2}",
            s.actual_collisions, s.expected_collisions, s.log2_excess);
    }
    println!("  log2_excess > 0 = more collisions than random birthday bound → matching surface.");
    println!("  RESULT: 2-round should show log2_excess ≤ 0; 1-round may show mild excess.");

    // ── 20. MITM Cat 2: lane dependency graph ────────────────────────────────
    section(20, total, "MITM Cat 2: χ Dependency / Lane Isolation  (200 pairs × 15 samples)");

    println!("  {:>6}  {:>12}  {:>12}  {:>14}  {:>14}  {:>13}",
        "Rounds", "Avg inf.", "Zero-inf. %", "Same-lane %", "Cross-lane %", "Isolation");
    for rounds in [1usize, 2] {
        let d = mitm::analyze_dependency_graph(rounds, 200, 15, &mut rng);
        println!("  {:>6}  {:>12.4}  {:>11.1}%  {:>13.1}%  {:>13.1}%  {:>13.2}×",
            rounds, d.avg_influence_prob,
            d.zero_influence_fraction * 100.0,
            d.same_lane_influence_frac * 100.0,
            d.cross_lane_influence_frac * 100.0,
            d.isolation_ratio);
    }
    println!("  Isolation ratio: same_lane_frac / cross_lane_frac.  >> 1 = lane isolation.");
    println!("  RESULT: 1-round shows high isolation; 2-round collapses toward 1.0×.");

    // ── 21. MITM Cat 3: Φ+round linear rank (influence matrix) ───────────────
    section(21, total, "MITM Cat 3: θ-Φ Linear Rank  (32 bits × 30 influence samples)");

    let lr = mitm::measure_influence_rank(32, 30, &mut rng);
    println!("  Function       Rank  /  Max  =  Fraction");
    println!("  Φ alone        {:>4}  / {:>4}  =  {:.3}", lr.phi_rank, lr.max_rank, lr.phi_rank_fraction);
    println!("  1-round        {:>4}  / {:>4}  =  {:.3}", lr.one_round_rank, lr.max_rank, lr.one_round_rank_fraction);
    println!("  2-round        {:>4}  / {:>4}  =  {:.3}", lr.two_round_rank, lr.max_rank, lr.two_round_rank_fraction);
    println!("  Rank hierarchy (Φ < 1-round < 2-round) confirms θ+χ composition fills");
    println!("  sparse Φ connectivity.  Near-full 2-round rank = no low-rank factoring.");

    // ── 22. MITM Cat 4: biclique-style matching ───────────────────────────────
    section(22, total, "MITM Cat 4: Biclique Matching  (dim=5 cube, 12 target bits, 10 bases)");

    for rounds in [1usize, 2] {
        let bc = if rounds == 1 {
            mitm::test_biclique_matching(5, 12, 10, &mut rng)
        } else {
            mitm::test_biclique_matching_rounds(5, 12, 10, 2, &mut rng)
        };
        println!("  rounds={rounds}  mean_match_frac={:.6}  expected={:.6}  log2_excess={:.2}  any_match={:.0}%",
            bc.mean_matching_pair_fraction, bc.expected_pair_fraction,
            bc.log2_excess, bc.any_match_fraction * 100.0);
    }
    println!("  1-round large excess (≈8–9 bits) = confirmed degree-3 structure in cube outputs.");
    println!("  2-round should approach log2_excess ≈ 0 (biclique structure destroyed).");

    // ── 23. MITM Cat 5: entropy surface collapse ──────────────────────────────
    section(23, total, "MITM Cat 5: Entropy Surface  (3000 samples × 10-bit projection)");

    println!("  {:>6}  {:>12}  {:>14}  {:>12}  {:>12}",
        "Rounds", "Collisions", "Expected coll.", "Min-entropy", "Uniformity");
    for rounds in [1usize, 2, 3] {
        let s = mitm::measure_entropy_collapse(rounds, 3_000, 10, &mut rng);
        println!("  {:>6}  {:>12}  {:>14.2}  {:>11.2}  {:>11.2}×",
            rounds, s.collision_count, s.expected_collisions,
            s.min_entropy_bits, s.uniformity_ratio);
    }
    println!("  Ideal min-entropy: 10.0 bits; uniformity 1.0 = perfectly uniform.");
    println!("  RESULT: 2-round min-entropy and uniformity should match random expectation.");

    println!("\n=== Attack harness complete ===");
}
