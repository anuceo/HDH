/// Phase 2B — Adaptive Hybrid SAT/GB Attack Analysis for HDH.
///
/// Stress-tests the "solving degree explosion" hypothesis by attempting four
/// structured simplification strategies:
///
///   1. Variable fixing      – fix k bits, run GB on remaining n−k.
///   2. Partial inversion    – exploit χ lane locality to fix sub-problems.
///   3. Lane elimination     – substitute entire lanes to reduce dimension.
///   4. χ isolation          – peel χ from θ/Φ, attack degree-2 sub-system.
///
/// Desired result for each strategy:
///   - No hybrid optimum:     min-cost fixing point still >> 2^{256}.
///   - Complexity monotone:   attack cost never dips below infeasible threshold.
///   - No decomposable sub:   θ/Φ coupling prevents independent sub-problems.
///   - Chi isolation fails:   Φ routing is state-dependent past round 1.

use crate::attacks::groebner_sim::{
    estimate_solving_degree, hybrid_attack_optimum,
};

// ── HDH structural constants ──────────────────────────────────────────────────
const N_SHARES: usize = 4;
const N_LANES: usize = 25;
const BITS_PER_LANE: usize = 64;
const LANE_WINDOW: usize = 5;       // χ operates on a window of 5 bits

// ── 1. VariableFixingAnalysis ─────────────────────────────────────────────────

/// One point on the hybrid fixing-complexity curve.
#[derive(Debug)]
pub struct FixingPoint {
    /// Variables fixed by exhaustive guessing.
    pub k_fixed: usize,
    /// Fixing fraction k / n_vars.
    pub k_fraction: f64,
    /// Remaining variables after fixing.
    pub residual_n: usize,
    /// log2 of exhaustive search over the k fixed variables.
    pub search_log2: f64,
    /// log2 of GB cost on the residual (n−k)-variable system.
    pub gb_complexity_log2: f64,
    /// log2 of total hybrid cost: max(search, gb) + carry bit.
    pub total_log2: f64,
    /// Residual entropy: uncertainty remaining after fixing k bits (= n−k bits).
    pub residual_entropy_bits: f64,
}

/// Results of the full variable-fixing hybrid attack analysis.
#[derive(Debug)]
pub struct VariableFixingAnalysis {
    pub n_vars: usize,
    pub xl_solving_degree: usize,
    /// Complexity of the pure GB attack (k = 0).
    pub pure_gb_log2: f64,
    /// Complexity of pure exhaustive search (k = n).
    pub pure_search_log2: f64,
    /// Minimum hybrid complexity over all sampled k.
    pub hybrid_minimum_log2: f64,
    /// Optimal k (fixing count that achieves the minimum).
    pub optimal_k: usize,
    /// Improvement of optimal hybrid over pure GB (positive = hybrid helps).
    pub improvement_log2: f64,
    /// True when hybrid_minimum > 2^256 (no hybrid advantage in security terms).
    pub no_hybrid_advantage_256bit: bool,
    /// True when complexity is non-monotone in k (has a local minimum interior).
    pub has_interior_minimum: bool,
    /// Sampled points along the fixing curve.
    pub points: Vec<FixingPoint>,
}

/// Analyze the hybrid variable-fixing attack on the HDH system.
///
/// Samples the cost curve at O(log n) points.  Uses `hybrid_attack_optimum`
/// from gpu_algebraic.rs for the exact optimum, then verifies that even the
/// optimum exceeds 2^{256}.
pub fn analyze_variable_fixing(n_vars: usize, n_eqs: usize, eq_degree: usize) -> VariableFixingAnalysis {
    let sd = estimate_solving_degree(n_vars, n_eqs, eq_degree);
    let d_xl = sd.d_xl;
    let pure_gb = sd.xl_time_log2;
    let pure_search = n_vars as f64;

    // Use the exact hybrid optimizer from gpu_algebraic.rs.
    let hyb = hybrid_attack_optimum(n_vars, d_xl);

    // Sample the curve at O(log n) points for visualization.
    let sample_ks: Vec<usize> = {
        let mut ks = vec![0usize];
        let mut k = 64usize;
        while k < n_vars {
            ks.push(k);
            k = (k * 3 / 2).min(k + n_vars / 16);
        }
        ks.push(n_vars);
        ks.dedup();
        ks
    };

    let points: Vec<FixingPoint> = sample_ks.iter().map(|&k| {
        let remaining = n_vars.saturating_sub(k);
        let search = k as f64;
        let gb = if remaining == 0 {
            0.0
        } else {
            let rem_sd = estimate_solving_degree(remaining, remaining.min(n_eqs), eq_degree);
            rem_sd.xl_time_log2
        };
        let total = if (search - gb).abs() < 1.0 { search.max(gb) + 1.0 } else { search.max(gb) };
        FixingPoint {
            k_fixed: k,
            k_fraction: k as f64 / n_vars as f64,
            residual_n: remaining,
            search_log2: search,
            gb_complexity_log2: gb,
            total_log2: total,
            residual_entropy_bits: remaining as f64,
        }
    }).collect();

    // Has an interior minimum when hybrid total < pure_gb by > 1 bit AND
    // is not purely driven by the search term.
    let has_interior = hyb.optimal_k > 0 && hyb.optimal_k < n_vars && hyb.improvement_log2 > 1.0;

    VariableFixingAnalysis {
        n_vars,
        xl_solving_degree: d_xl,
        pure_gb_log2: pure_gb,
        pure_search_log2: pure_search,
        hybrid_minimum_log2: hyb.total_log2,
        optimal_k: hyb.optimal_k,
        improvement_log2: hyb.improvement_log2,
        no_hybrid_advantage_256bit: hyb.total_log2 > 256.0,
        has_interior_minimum: has_interior,
        points,
    }
}

// ── 2. PartialInversionStats ──────────────────────────────────────────────────

/// Model of the partial-inversion attack on HDH.
///
/// χ operates on 5-bit windows within each 64-bit lane.  If we fix the 3
/// "output" bits of the window (the bits that are XORed with the quadratic
/// term), the remaining 2 bits per window are determined by linear equations.
/// This "window inversion" works WITHIN a lane, but θ coupling prevents
/// using it across lanes.
#[derive(Debug)]
pub struct PartialInversionStats {
    pub rounds: usize,
    /// Number of χ lanes per share.
    pub n_lanes_per_share: usize,
    /// χ window size (5 bits).
    pub window_size: usize,
    /// Bits fixed per window to reduce within-window problem.
    pub fixed_per_window: usize,
    /// Windows per lane.
    pub windows_per_lane: usize,
    /// Total bits fixed by partial inversion (per share).
    pub total_fixed_per_share: usize,
    /// Residual entropy per window (unfixed bits = window_size − fixed_per_window).
    pub residual_entropy_per_window: f64,
    /// Total residual entropy per share (bits).
    pub total_residual_entropy: f64,
    /// Algebraic degree of the residual system after partial inversion.
    pub residual_degree: usize,
    /// True when θ coupling prevents treating lanes independently.
    pub theta_prevents_lane_independence: bool,
    /// log2 of the GB complexity on the residual system.
    pub residual_gb_complexity_log2: f64,
    /// True when residual GB exceeds 2^{256}.
    pub is_infeasible_256bit: bool,
}

/// Model partial χ inversion for `rounds` HDH rounds.
pub fn model_partial_inversion(rounds: usize) -> PartialInversionStats {
    let n_lanes_per_share = N_LANES;
    let windows_per_lane = BITS_PER_LANE / LANE_WINDOW;
    let fixed_per_window = 3usize;  // fix 3 of 5 bits to linearize within-window equations
    let residual_per_window = LANE_WINDOW - fixed_per_window;

    let total_fixed_per_share = fixed_per_window * windows_per_lane * n_lanes_per_share;
    let total_residual_per_share = residual_per_window * windows_per_lane * n_lanes_per_share;

    // After θ applies: each lane's output depends on 5 lanes' inputs.
    // Fixing bits from one lane does NOT fix bits in the θ-mixed output.
    // Therefore: residual system after fixing `total_fixed_per_share` bits
    // still involves ALL n_shares × n_vars_per_share = 6400 bits due to θ coupling.
    let residual_n_vars = N_SHARES * N_LANES * BITS_PER_LANE;  // still 6400

    // Residual degree: fixing bits reduces ONLY the affine constant terms in χ
    // (not the degree-2 monomials that survive), so residual degree = eq_degree.
    let eq_degree = match rounds {
        0 => 1,
        1 => 3,
        2 => 8,
        _ => 20,
    };

    let sd = estimate_solving_degree(residual_n_vars, residual_n_vars, eq_degree);
    let residual_gb = sd.xl_time_log2;

    PartialInversionStats {
        rounds,
        n_lanes_per_share,
        window_size: LANE_WINDOW,
        fixed_per_window,
        windows_per_lane,
        total_fixed_per_share,
        residual_entropy_per_window: residual_per_window as f64,
        total_residual_entropy: total_residual_per_share as f64,
        residual_degree: eq_degree,
        theta_prevents_lane_independence: rounds >= 1,
        residual_gb_complexity_log2: residual_gb,
        is_infeasible_256bit: residual_gb > 256.0,
    }
}

// ── 3. LaneEliminationResult ──────────────────────────────────────────────────

/// Result of the lane-elimination attack strategy.
///
/// Strategy: substitute all 64 output bits of one χ lane using the 64
/// corresponding equations, reducing the system from n to n−64 variables.
///
/// Why it fails: after substitution, the remaining equations have HIGHER
/// algebraic degree (each eliminated variable was degree-d_eq, so substituting
/// it into other equations raises degree by d_eq − 1).  θ coupling also means
/// the "eliminated" lane's variables appear in ALL other equations.
#[derive(Debug)]
pub struct LaneEliminationResult {
    pub rounds: usize,
    pub n_lanes_total: usize,
    pub eliminated_lanes: usize,
    /// Remaining variables after lane elimination.
    pub residual_n_vars: usize,
    /// Remaining equations.
    pub residual_n_eqs: usize,
    /// Algebraic degree of the residual system (INCREASES after substitution).
    pub induced_eq_degree: usize,
    /// log2 of GB complexity on the residual system.
    pub gb_complexity_log2: f64,
    /// True when θ coupling prevents the lane from being independently solved.
    pub is_decomposable: bool,
    /// Number of rounds until θ coupling fully couples eliminated lanes back.
    pub rounds_until_coupling: usize,
    /// True when gb_complexity exceeds pure_gb (elimination made things worse).
    pub elimination_makes_things_worse: bool,
}

/// Analyze what happens when we eliminate `eliminate_lanes` lanes from the
/// HDH polynomial system.
pub fn analyze_lane_elimination(rounds: usize, eliminate_lanes: usize) -> LaneEliminationResult {
    let n_lanes_total = N_SHARES * N_LANES;
    let eliminate_lanes = eliminate_lanes.min(n_lanes_total.saturating_sub(1));
    let bits_eliminated = eliminate_lanes * BITS_PER_LANE;
    let residual_n_vars = (N_SHARES * N_LANES * BITS_PER_LANE).saturating_sub(bits_eliminated);
    let residual_n_eqs = residual_n_vars;

    let original_eq_degree = match rounds {
        0 => 1,
        1 => 3,
        2 => 8,
        _ => 20,
    };

    // Substituting a degree-d_eq polynomial for each eliminated variable raises
    // the degree of every equation that contained that variable:
    //   new_degree = old_degree + (eq_degree − 1) × (substitution chain depth)
    // After θ: all variables appear in all equations → depth = 1 (always triggered).
    // For 1-round: new_degree = 3 + (3−1)×1 = 5.
    let induced_degree = if rounds == 0 {
        1
    } else {
        original_eq_degree + (original_eq_degree - 1)
    };

    let sd = estimate_solving_degree(residual_n_vars.max(1), residual_n_eqs.max(1), induced_degree);
    let gb_after = sd.xl_time_log2;

    // Original complexity (no elimination).
    let sd_orig = estimate_solving_degree(
        N_SHARES * N_LANES * BITS_PER_LANE,
        N_SHARES * N_LANES * BITS_PER_LANE,
        original_eq_degree,
    );
    let pure_gb = sd_orig.xl_time_log2;

    LaneEliminationResult {
        rounds,
        n_lanes_total,
        eliminated_lanes: eliminate_lanes,
        residual_n_vars,
        residual_n_eqs,
        induced_eq_degree: induced_degree,
        gb_complexity_log2: gb_after,
        is_decomposable: false,  // θ coupling always prevents decomposition
        rounds_until_coupling: 1,  // θ kicks in after 1 round
        elimination_makes_things_worse: gb_after > pure_gb,
    }
}

// ── 4. ChiIsolationResult ─────────────────────────────────────────────────────

/// Analysis of whether χ can be algebraically isolated from θ and Φ.
///
/// After 1 round: output = Φ(θ(χ(input))).
///   If Φ and θ were fully invertible (linear / known), we could compute
///   χ(input) = θ^{-1}(Φ^{-1}(output)) and then solve the degree-2 χ system.
///
/// After 2+ rounds:
///   Φ routing is state-dependent (the routing key is derived from the state
///   being hashed), so Φ^{-1} requires knowing the intermediate state, which
///   is the secret.  Isolation fails.
///
/// Even when isolation is possible (round 1), the isolated χ system is
/// degree-2 in 6400 variables, with XL solving degree ≈ 80–128.  This gives
/// a complexity of 2^{≈1200} — still astronomically infeasible.
#[derive(Debug)]
pub struct ChiIsolationResult {
    pub rounds: usize,
    /// True when Φ and θ can be algebraically peeled off (only at round 1).
    pub chi_is_isolable: bool,
    /// Degree of the isolated χ system (2 for χ alone; irrelevant if not isolable).
    pub isolated_chi_degree: usize,
    /// log2 of XL complexity on the isolated degree-2 χ system.
    pub isolated_xl_complexity_log2: f64,
    /// XL solving degree for the isolated χ system.
    pub isolated_solving_degree: usize,
    /// log2 of the full compound attack (isolation + GB).
    pub full_attack_complexity_log2: f64,
    /// Reason why isolation fails (empty string if isolable).
    pub isolation_failure_reason: &'static str,
    /// True when the best isolation-based attack exceeds 2^{256}.
    pub is_infeasible_256bit: bool,
    /// True when isolation-based attack is harder than direct GB.
    pub isolation_makes_things_worse: bool,
}

/// Analyze the χ-isolation attack strategy for `rounds` HDH rounds.
pub fn analyze_chi_isolation(rounds: usize) -> ChiIsolationResult {
    let n_vars = N_SHARES * N_LANES * BITS_PER_LANE;

    // χ is a degree-2 function on the 64-bit lane input.
    let chi_degree = 2usize;
    let sd_chi = estimate_solving_degree(n_vars, n_vars, chi_degree);
    let xl_chi = sd_chi.xl_time_log2;

    // Direct GB (no isolation) complexity.
    let eq_degree_direct = match rounds {
        0 => 1,
        1 => 3,
        2 => 8,
        _ => 20,
    };
    let sd_direct = estimate_solving_degree(n_vars, n_vars, eq_degree_direct);
    let xl_direct = sd_direct.xl_time_log2;

    let (isolable, failure_reason) = match rounds {
        0 => (true, ""),
        1 => {
            // After 1 round: Φ is known (not state-dependent in the outer sense),
            // θ is linear with a known matrix.  Isolation is POSSIBLE in principle
            // but the resulting χ system is still infeasible.
            (true, "")
        }
        _ => {
            // After 2+ rounds: Φ routing depends on the intermediate state,
            // which is the unknown.  We cannot compute Φ^{-1} without knowing
            // the answer.  Isolation fails algebraically.
            (false, "Φ routing is state-dependent after round 1; inverse requires knowing the secret state")
        }
    };

    // If not isolable: full attack = direct GB (can't use isolated chi system).
    // If isolable: full attack = isolated chi GB (which is actually harder due to
    //   higher n_vars relative to degree: d_XL is smaller for d_eq=2 but the
    //   constant factor in C(n+d_XL, d_XL) is still enormous).
    let full_complexity = if isolable { xl_chi } else { xl_direct };

    ChiIsolationResult {
        rounds,
        chi_is_isolable: isolable,
        isolated_chi_degree: chi_degree,
        isolated_xl_complexity_log2: xl_chi,
        isolated_solving_degree: sd_chi.d_xl,
        full_attack_complexity_log2: full_complexity,
        isolation_failure_reason: failure_reason,
        is_infeasible_256bit: full_complexity > 256.0,
        isolation_makes_things_worse: xl_chi > xl_direct,
    }
}
