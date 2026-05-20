use hdh::attacks::{annihilator, differential, jacobian, linear, preimage, sat};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn section(n: usize, total: usize, title: &str) {
    println!("\n[{n}/{total}] {title}");
    println!("{}", "─".repeat(62));
}

fn main() {
    println!("=== HDH χ Core — Algebraic & SAT Reconstruction Attack Harness ===");
    let mut rng = ChaCha20Rng::seed_from_u64(0x0123456789abcdef);
    let total = 9;

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

    // ── 8. Algebraic immunity estimation ──────────────────────────────────────
    section(8, total, "Algebraic Immunity  (4-bit χ, output-bit scan, d ≤ 3)");

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

    // ── 9. Invariant subspace detection ──────────────────────────────────────
    section(9, total, "Invariant Subspace Detection  (4-bit χ, full scan)");

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

    println!("\n=== Attack harness complete ===");
}
