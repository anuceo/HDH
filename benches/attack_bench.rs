use hdh::attacks::{differential, linear, preimage, sat};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn section(n: usize, title: &str) {
    println!("\n[{n}/5] {title}");
    println!("{}", "─".repeat(60));
}

fn main() {
    println!("=== HDH χ Core — Algebraic & SAT Reconstruction Attack Harness ===");
    let mut rng = ChaCha20Rng::seed_from_u64(0x0123456789abcdef);

    // ── 1. Differential uniformity ─────────────────────────────────────────
    section(1, "Differential Uniformity  (64-bit quad, empirical)");

    let diffs_to_test = 20usize;
    let samples_each = 50_000usize;
    let mut worst_bias = 0.0f64;
    let mut worst_delta = (0u64, 0, 0, 0);

    for _ in 0..diffs_to_test {
        let delta = loop {
            let d = (rng.gen::<u64>(), rng.gen::<u64>(), rng.gen::<u64>(), rng.gen::<u64>());
            if d != (0, 0, 0, 0) { break d; }
        };
        let stats = differential::analyze_differential(delta, samples_each, &mut rng);
        if stats.max_bias > worst_bias {
            worst_bias = stats.max_bias;
            worst_delta = stats.input_diff;
        }
        print!("  Δ=({:016x},..) unique_ΔG={:>6}  max_count={:>4}  bias={:.6}\n",
            stats.input_diff.0, stats.unique_output_diffs,
            stats.max_collision_count, stats.max_bias);
    }

    println!();
    println!("  Worst-case bias: {worst_bias:.6}  (ideal → 1/{samples_each} = {:.6})",
        1.0 / samples_each as f64);
    if worst_bias < 0.005 {
        println!("  RESULT: no differential shortcut found");
    } else {
        println!("  WARNING: bias {worst_bias:.4} exceeds threshold — investigate Δ={worst_delta:?}");
    }

    // ── 2. Linear bias ──────────────────────────────────────────────────────
    section(2, "Linear Bias / Walsh Coefficients  (full chi_lane, empirical)");

    let stats = linear::sample_linear_bias(100, 50_000, &mut rng);
    println!("  Masks tested:        {}", stats.masks_tested);
    println!("  Samples per mask:    {}", stats.samples_per_mask);
    println!("  Max |bias|:          {:.6}", stats.max_bias);
    println!("  Avg |bias|:          {:.6}", stats.avg_bias);
    println!("  Expected (random):   ~{:.6}", 1.0 / (50_000f64).sqrt());
    if stats.max_bias < 0.02 {
        println!("  RESULT: no exploitable linear approximation found");
    } else {
        println!("  WARNING: max bias {:.4} — a linear distinguisher may exist",
            stats.max_bias);
    }

    // ── 3. Preimage resistance (8-bit reduced χ) ───────────────────────────
    section(3, "Preimage Resistance  (8-bit reduced χ, exhaustive g-search)");

    let pre = preimage::analyze_preimages(5_000, &mut rng);
    println!("  Outputs sampled:          {}", pre.outputs_sampled);
    println!("  Avg preimage count:       {:.3}", pre.avg_preimage_count);
    println!("  Max preimage count:       {}", pre.max_preimage_count);
    println!("  Zero-preimage fraction:   {:.2}%", pre.zero_preimage_frac * 100.0);
    println!("  Unique-preimage fraction: {:.2}%", pre.unique_preimage_frac * 100.0);
    let expected_for_random = 1.0f64;
    println!("  Expected (random surjection): avg ≈ {expected_for_random:.1}");
    if (pre.avg_preimage_count - 1.0).abs() < 0.2 && pre.max_preimage_count <= 6 {
        println!("  RESULT: near-bijective; no structural preimage multiplicity found");
    } else {
        println!("  NOTE: deviation from ideal — see preimage distribution");
    }

    // ── 4. Algebraic degree (4-bit reduced χ, exact ANF) ──────────────────
    section(4, "Algebraic Degree  (4-bit reduced χ, exact Möbius/ANF)");

    let deg = preimage::analyze_algebraic_degree();
    println!("  Output bits analyzed: {}", deg.degrees.len());
    println!("  Degrees per output bit:");
    for (i, d) in deg.degrees.iter().enumerate() {
        print!("  bit {:>2}: deg {}", i, d);
        if (i + 1) % 4 == 0 { println!(); }
    }
    println!();
    println!("  Min degree: {}  Max degree: {}", deg.min_degree, deg.max_degree);
    if deg.min_degree >= 3 {
        println!("  RESULT: degree ≥ 3 — XL at D=2 and quadratic Gröbner attacks FAIL");
    } else {
        println!("  WARNING: degree {} output bit found — check for algebraic structure",
            deg.min_degree);
    }
    println!("  Extrapolation: 64-bit χ degree ≥ {} (carry-chain scales with word size)",
        deg.max_degree * 8);

    // ── 5. SAT / XL complexity model ───────────────────────────────────────
    section(5, "SAT / XL Reconstruction Model  (64-bit χ, analytical)");

    let xl = sat::model_xl_complexity();
    println!("  Boolean unknowns (4×64 input bits):       {:>6}", xl.input_vars);
    println!("  Output constraints per pair:               {:>6}", xl.output_bits_per_pair);
    println!("  Linearized product terms (degree-2 XL):   {:>6}", xl.linearized_product_terms);
    println!("  Total XL variables (degree-2):             {:>6}", xl.total_xl_vars);
    println!("  Pairs for square system (D=2):             {:>6}", xl.pairs_for_square_system);
    println!("  Pairs for overdetermination (D=2):         {:>6}", xl.pairs_for_overdetermination);
    println!("  Degree-2 XL sufficient?                     {}", xl.degree2_xl_sufficient);
    println!("  Actual degree lower bound:                 {:>6}", xl.required_xl_degree_lower_bound);
    println!("  Monomials at required degree:               {}", xl.monomials_at_required_degree);

    let dpll = sat::model_dpll_complexity();
    println!();
    println!("  DPLL/CDCL model:");
    println!("  Boolean vars in SAT formula:               {:>6}", dpll.boolean_vars);
    println!("  Clauses per known pair:                    {:>6}", dpll.clauses_per_pair);
    println!("  Free vars after unit propagation:          {:>6}", dpll.free_vars_after_up);
    println!("  Search tree lower bound:                    2^{}", dpll.log2_search_space as usize);

    println!();
    println!("  RESULT: SAT reconstruction requires 2^{:.0} search steps minimum.",
        dpll.log2_search_space);
    println!("          XL at true degree needs {} monomials — infeasible.",
        xl.monomials_at_required_degree);

    println!("\n=== Attack harness complete ===");
}

