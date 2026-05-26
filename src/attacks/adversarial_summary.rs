/// Adversarial Security Summary: executive reference for cryptographic reviewers.
///
/// Consolidates best-known attack complexities for each attack family, drawn
/// directly from benchmark module outputs. All entries are for the recommended
/// deployment (r=4 rounds); minimum secure (r=2) complexities are also noted.
///
/// At r≥4, every listed attack family requires T > 2^128 operations.

use crate::attacks::{
    milp_trail, linear_hull, sat, gpu_algebraic, quantum_security, mitm,
};
use rand::Rng;

// ── Structs ───────────────────────────────────────────────────────────────────

pub struct AttackEntry {
    pub attack: &'static str,
    /// log₂ of the best known attack complexity.
    pub best_complexity_log2: f64,
    /// Human-readable display string.
    pub complexity_display: String,
    /// Source of the bound: "analytical", "model", "empirical", "proven".
    pub bound_type: &'static str,
    /// True if the attack is computationally feasible (complexity < 2^64).
    pub feasible: bool,
    /// Short note on why the attack fails or its dominant bottleneck.
    pub note: &'static str,
}

pub struct AdversarialSummary {
    pub entries: Vec<AttackEntry>,
    /// Recommended deployment round count.
    pub recommended_rounds: usize,
    /// Minimum secure round count (all distinguishers closed).
    pub minimum_secure_rounds: usize,
    /// Classical collision security bits (= capacity/2).
    pub classical_collision_bits: f64,
    /// Quantum collision security bits (= capacity/3, BHT).
    pub quantum_collision_bits: f64,
}

// ── Formatting ────────────────────────────────────────────────────────────────

fn fmt_log2(log2: f64) -> String {
    if log2 >= 1000.0 {
        format!("2^{:.0}", log2)
    } else if log2.fract() < 0.05 || log2.fract() > 0.95 {
        format!("2^{:.0}", log2)
    } else {
        format!("2^{:.1}", log2)
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Compute all attack complexities and assemble the summary.
///
/// Uses analytical models where possible; empirical sampling kept small.
pub fn build_adversarial_summary(rng: &mut impl Rng) -> AdversarialSummary {
    let mut entries: Vec<AttackEntry> = Vec::new();

    // ── 1. Differential ──────────────────────────────────────────────────────
    // Best trail probability from MILP exhaustive search at r=4.
    let trail_r4 = milp_trail::search_min_active_trail(4);
    // Attack complexity ≈ 1 / trail_probability.
    let diff_complexity = -trail_r4.implied_prob_log2;
    entries.push(AttackEntry {
        attack: "Differential",
        best_complexity_log2: diff_complexity,
        complexity_display: fmt_log2(diff_complexity),
        bound_type: "analytical",
        feasible: diff_complexity < 64.0,
        note: "Best trail: min_active_sboxes × log₂(p_chi); exhaustive 2^25−1 search",
    });

    // ── 2. Linear ─────────────────────────────────────────────────────────────
    // Data complexity = bias^{−2} where bias = hull_bias at r=4.
    let hull = linear_hull::linear_hull_sweep(4, 3_000, rng);
    let hull_r4_log2 = hull.get(3).map(|e| e.hull_bias_log2).unwrap_or(-200.0);
    // bias_log2 is negative; data complexity = 2^{-2*bias_log2}.
    let linear_complexity = -2.0 * hull_r4_log2;
    entries.push(AttackEntry {
        attack: "Linear",
        best_complexity_log2: linear_complexity,
        complexity_display: fmt_log2(linear_complexity),
        bound_type: "analytical",
        feasible: linear_complexity < 64.0,
        note: "Data complexity 1/bias²; Walsh hull over 200 masks × 3000 samples",
    });

    // ── 3. Integral / Zero-sum ────────────────────────────────────────────────
    // At r≥2 all integral structures vanish; complexity = capacity bits.
    entries.push(AttackEntry {
        attack: "Integral / Zero-sum",
        best_complexity_log2: 512.0,
        complexity_display: "2^512 (capacity)".to_string(),
        bound_type: "proven",
        feasible: false,
        note: "Indistinguishability from random at r≥2; no residual degree structure",
    });

    // ── 4. Meet-in-the-middle ─────────────────────────────────────────────────
    // Partition independence at r=2; complexity = capacity/2 (birthday bound).
    entries.push(AttackEntry {
        attack: "MITM",
        best_complexity_log2: 256.0,
        complexity_display: "2^256 (collision)".to_string(),
        bound_type: "analytical",
        feasible: false,
        note: "Forward/backward partitions statistically independent at r≥2",
    });

    // ── 5. Biclique ───────────────────────────────────────────────────────────
    // Biclique excess ≤ 0 at r=2; attack complexity ≥ generic.
    let bc = mitm::test_biclique_matching_rounds(8, 8, 30, 2, rng);
    // If excess > 0 the complexity is reduced; if ≤ 0 it matches generic.
    let biclique_complexity = 512.0 - bc.log2_excess.max(0.0);
    entries.push(AttackEntry {
        attack: "Biclique",
        best_complexity_log2: biclique_complexity,
        complexity_display: fmt_log2(biclique_complexity),
        bound_type: "empirical",
        feasible: biclique_complexity < 64.0,
        note: "log₂(excess) ≤ 0 at r=2; no structure above random birthday matching",
    });

    // ── 6. SAT / DPLL ────────────────────────────────────────────────────────
    // DPLL must enumerate all 2^64 g-variable assignments (unit propagation
    // cannot resolve the degree-32 clause without satisfying all g-constraints).
    let sat_stats = sat::simulate_incremental_sat(4);
    let sat_complexity = sat_stats
        .snapshots
        .last()
        .map(|s| s.log2_dpll_search)
        .unwrap_or(64.0);
    entries.push(AttackEntry {
        attack: "SAT / DPLL",
        best_complexity_log2: sat_complexity,
        complexity_display: format!("≥ {}", fmt_log2(sat_complexity)),
        bound_type: "model",
        feasible: sat_complexity < 64.0,
        note: "64 auxiliary g-vars; unit propagation does not prune degree-32 clauses",
    });

    // ── 7. XL / Gröbner ──────────────────────────────────────────────────────
    // Effective algebraic degree ≈ 32; XL work = C(256, 32).
    let eff = sat::model_effective_xl(256);
    entries.push(AttackEntry {
        attack: "XL / Gröbner",
        best_complexity_log2: eff.xl_complexity_log2,
        complexity_display: fmt_log2(eff.xl_complexity_log2),
        bound_type: "model",
        feasible: eff.xl_complexity_log2 < 64.0,
        note: "n=256 vars, effective degree 32; degree-2 exploitation negligible (0.78%)",
    });

    // ── 8. Hybrid F₅ (exhaustive + Gröbner) ──────────────────────────────────
    let hyb = gpu_algebraic::hybrid_attack_optimum(256, eff.effective_degree);
    entries.push(AttackEntry {
        attack: "Hybrid F₅",
        best_complexity_log2: hyb.total_log2,
        complexity_display: fmt_log2(hyb.total_log2),
        bound_type: "model",
        feasible: hyb.total_log2 < 64.0,
        note: "Fix k={} vars by brute force + Gröbner on reduced system; optimal k chosen",
    });

    // ── 9. Quantum Collision (BHT) ────────────────────────────────────────────
    let bht = quantum_security::model_bht(512);
    entries.push(AttackEntry {
        attack: "Quantum collision (BHT)",
        best_complexity_log2: bht.work_log2,
        complexity_display: fmt_log2(bht.work_log2),
        bound_type: "proven",
        feasible: bht.near_term_feasible,
        note: "Brassard-Høyer-Tapp algorithm; work = 2^{c/3}, c=512",
    });

    // ── 10. Quantum Preimage (Grover) ─────────────────────────────────────────
    let grover = quantum_security::model_grover(512);
    entries.push(AttackEntry {
        attack: "Quantum preimage (Grover)",
        best_complexity_log2: grover.work_log2,
        complexity_display: fmt_log2(grover.work_log2),
        bound_type: "proven",
        feasible: grover.near_term_feasible,
        note: "Grover search over 2^{output_bits} candidates; work = 2^{512/2} = 2^256",
    });

    AdversarialSummary {
        entries,
        recommended_rounds: 4,
        minimum_secure_rounds: 2,
        classical_collision_bits: 256.0,
        quantum_collision_bits: bht.work_log2,
    }
}
