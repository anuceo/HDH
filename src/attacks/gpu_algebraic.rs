/// GPU-scale algebraic attack complexity modeling for HDH.
///
/// Extrapolates from empirical 4-bit and 64-bit measurements to the full
/// 6400-bit HDH state, computing F4/F5 Gröbner basis, hybrid attack, and
/// XL algorithm complexity bounds.
///
/// Key insight: algebraic EQUATION degree (≤ 3 at 1 round) is not the same as
/// the XL SOLVING degree.  For a square system with n = m = N equations of
/// degree d_eq, XL must extend to solving degree d_XL ≈ N^{(d_eq−1)/d_eq},
/// making the attack astronomically infeasible even when d_eq is small.
///
/// Empirical inputs drawn from sat.rs / preimage.rs:
///   - 64-bit χ lane: effective XL degree ≥ 32, complexity ≥ 2^{120} (sat.rs).
///   - 4-bit χ toy: ANF degree = 5, AI ≥ 2 (preimage.rs / annihilator.rs).
///   - GF(2) Jacobian rank ≈ 256 at every tested point (jacobian.rs).
///
/// Five components:
/// 1. XLComplexityModel   – XL/Gröbner complexity for parameterised systems.
/// 2. HybridAttackModel   – optimal variable-fixing split.
/// 3. SolvingDegreeModel  – estimates XL solving degree for square systems.
/// 4. GpuFeasibility      – concrete GPU operations-per-second and wall time.
/// 5. AlgebraicScaleSweep – sweep from 4-bit toy to 6400-bit HDH.

// ── helpers ──────────────────────────────────────────────────────────────────

/// log2(C(n, k)) via Stirling-style addition, numerically stable.
pub(crate) fn log2_binom(n: usize, k: usize) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    let k = k.min(n - k);
    (0..k).map(|i| ((n - i) as f64 / (i + 1) as f64).log2()).sum()
}

/// Matrix multiplication exponent (Coppersmith-Winograd).
const OMEGA: f64 = 2.37;

// ── 1. XLComplexityModel ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct MQSystemParams {
    /// Number of boolean variables.
    pub n_vars: usize,
    /// Number of equations.
    pub n_eqs: usize,
    /// Algebraic degree of the equations.
    pub eq_degree: usize,
    /// XL solving degree (may be >> eq_degree for square / underdetermined systems).
    pub xl_solving_degree: usize,
}

#[derive(Debug)]
pub struct XLComplexity {
    pub params: MQSystemParams,
    /// log2 of C(n + d_XL, d_XL): Macaulay column dimension.
    pub macaulay_log2: f64,
    /// log2 of XL complexity: macaulay_log2 × ω.
    pub xl_complexity_log2: f64,
    /// log2 of memory required: macaulay_log2 × 2 (matrix storage in bits).
    pub memory_log2: f64,
    pub is_infeasible_128bit: bool,
    pub is_infeasible_256bit: bool,
}

pub fn xl_complexity(params: MQSystemParams) -> XLComplexity {
    let mac = log2_binom(params.n_vars + params.xl_solving_degree, params.xl_solving_degree);
    let xl = mac * OMEGA;
    let mem = mac * 2.0;   // rows × cols in bits ≈ C(n+d,d)² bits
    XLComplexity {
        macaulay_log2: mac,
        xl_complexity_log2: xl,
        memory_log2: mem,
        is_infeasible_128bit: xl.min(mem) > 128.0,
        is_infeasible_256bit: xl.min(mem) > 256.0,
        params,
    }
}

// ── 2. HybridAttackModel ─────────────────────────────────────────────────────

/// Hybrid Gröbner attack: fix k variables exhaustively, then solve the remaining
/// (n−k)-variable system.  Optimal k minimises:
///   cost(k) = log2(2^k + C(n−k + d_XL, d_XL)^ω)
///           ≈ max(k, GB(n−k, d_XL))   when the two terms differ by > 1 bit.
///
/// Note: d_XL(n−k) < d_XL(n) because removing variables reduces the solving degree.
/// We conservatively assume d_XL stays constant (worst-case estimate for the attacker).
#[derive(Debug)]
pub struct HybridResult {
    pub n_vars: usize,
    pub xl_solving_degree: usize,
    /// Optimal number of variables to fix by exhaustive search.
    pub optimal_k: usize,
    pub search_log2: f64,
    pub groebner_log2: f64,
    pub total_log2: f64,
    /// Improvement over pure-Gröbner (positive = hybrid helps).
    pub improvement_log2: f64,
}

pub fn hybrid_attack_optimum(n_vars: usize, xl_solving_degree: usize) -> HybridResult {
    let pure_gb = log2_binom(n_vars + xl_solving_degree, xl_solving_degree) * OMEGA;

    let mut best = pure_gb;
    let mut best_k = 0usize;
    let mut best_search = 0.0f64;
    let mut best_gb_k = pure_gb;

    for k in 1..n_vars {
        let remaining = n_vars - k;
        let gb = log2_binom(remaining + xl_solving_degree, xl_solving_degree) * OMEGA;
        let search = k as f64;
        let total = if (search - gb).abs() < 1.0 {
            search.max(gb) + 1.0
        } else {
            search.max(gb)
        };
        if total < best {
            best = total;
            best_k = k;
            best_search = search;
            best_gb_k = gb;
        }
        // Early exit: once search dominates and GB is negligible, no further improvement.
        if search > pure_gb + 2.0 { break; }
    }

    HybridResult {
        n_vars,
        xl_solving_degree,
        optimal_k: best_k,
        search_log2: best_search,
        groebner_log2: best_gb_k,
        total_log2: best,
        improvement_log2: pure_gb - best,
    }
}

// ── 3. SolvingDegreeModel ────────────────────────────────────────────────────

/// Estimates the XL solving degree d_XL for a square system (n = m = N) of
/// degree-d_eq equations over GF(2).
///
/// Derivation: XL succeeds when the number of distinct degree-d_XL equations
/// (original equations × all degree-(d_XL − d_eq) multipliers) exceeds the
/// number of degree-d_XL monomials:
///
///   N × C(N, d_XL − d_eq)  ≥  C(N + d_XL, d_XL)
///
/// Rearranging for the dominant terms:
///
///   N^{d_eq} / C(d_XL, d_eq)  ≈  1
///   d_XL ≈ N^{(d_eq−1)/d_eq}   (rough leading-order estimate)
///
/// For small systems (N ≤ 32) we bound d_XL ≤ N (exhaustive ANF degree).
#[derive(Debug)]
pub struct SolvingDegreeEstimate {
    pub n_vars: usize,
    pub n_eqs: usize,
    pub eq_degree: usize,
    /// Estimated XL solving degree.
    pub d_xl: usize,
    /// log2 of C(N + d_XL, d_XL): Macaulay matrix column count.
    pub macaulay_log2: f64,
    /// log2 of XL time complexity: macaulay × ω.
    pub xl_time_log2: f64,
    /// log2 of Macaulay matrix memory (bits): macaulay × 2.
    pub xl_memory_log2: f64,
}

pub fn estimate_solving_degree(n_vars: usize, n_eqs: usize, eq_degree: usize) -> SolvingDegreeEstimate {
    // Hard ceiling: for Boolean systems with N vars, degree ≤ N.
    let d_xl_cap = n_vars;

    // Binary search for the smallest d where n_eqs × C(n,d−d_eq) ≥ C(n+d, d).
    // Work entirely in log2 to avoid overflow.
    let mut lo = eq_degree;
    let mut hi = d_xl_cap;
    let log2_m = (n_eqs as f64).log2();

    let span_log2 = |d: usize| -> f64 {
        if d < eq_degree { return f64::NEG_INFINITY; }
        log2_m + log2_binom(n_vars, d - eq_degree)
    };
    let mono_log2 = |d: usize| -> f64 { log2_binom(n_vars + d, d) };

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if span_log2(mid) >= mono_log2(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    let d_xl = lo.min(d_xl_cap);
    let mac = log2_binom(n_vars + d_xl, d_xl);
    SolvingDegreeEstimate {
        n_vars,
        n_eqs,
        eq_degree,
        d_xl,
        macaulay_log2: mac,
        xl_time_log2: mac * OMEGA,
        xl_memory_log2: mac * 2.0,
    }
}

// ── 4. GpuFeasibility ────────────────────────────────────────────────────────

/// Models concrete GPU throughput and wall-clock feasibility.
///
/// GF(2) operations per second on modern GPU hardware (bit-parallel):
///   A100 (80 GB): ~300 TFLOPS FP32 → ~10^16 GF(2) ops/s ≈ 2^{53}
///   H100 (80 GB): ~1 PFLOPS FP64    → ~10^17 GF(2) ops/s ≈ 2^{57}
///   1000-GPU A100 cluster:           → 2^{63} GF(2) ops/s
///   1M-GPU exascale:                 → 2^{73} GF(2) ops/s (speculative)
///
/// Feasibility thresholds (wall-clock):
///   Seconds per year: ~2^{25}
///   Age of universe:  ~2^{58} seconds
#[derive(Debug)]
pub struct GpuFeasibility {
    pub complexity_log2: f64,
    /// log2 of GF(2) operations per second on the target hardware.
    pub gpu_ops_per_sec_log2: f64,
    pub hardware_description: &'static str,
    /// log2 of wall-clock seconds: complexity − throughput.
    pub wall_time_log2: f64,
    pub is_feasible_in_one_year: bool,
    pub is_feasible_in_universe_age: bool,
}

#[derive(Debug)]
pub struct GpuFeasibilityTable {
    pub complexity_log2: f64,
    pub entries: Vec<GpuFeasibility>,
}

const GPU_CONFIGS: &[(&'static str, f64)] = &[
    ("single A100",          53.0),
    ("single H100",          57.0),
    ("cluster 1 000× A100",  63.0),
    ("cluster 1M× (speculative exascale)", 73.0),
];

const LOG2_SECONDS_PER_YEAR: f64 = 24.98;  // log2(365.25 × 24 × 3600) ≈ 24.98
const LOG2_SECONDS_UNIVERSE: f64 = 57.6;   // log2(4.35×10^17 s) ≈ 57.6

pub fn gpu_feasibility_table(complexity_log2: f64) -> GpuFeasibilityTable {
    let entries = GPU_CONFIGS.iter().map(|&(desc, tput)| {
        let t = complexity_log2 - tput;
        GpuFeasibility {
            complexity_log2,
            gpu_ops_per_sec_log2: tput,
            hardware_description: desc,
            wall_time_log2: t,
            is_feasible_in_one_year: t < LOG2_SECONDS_PER_YEAR,
            is_feasible_in_universe_age: t < LOG2_SECONDS_UNIVERSE,
        }
    }).collect();
    GpuFeasibilityTable { complexity_log2, entries }
}

// ── 5. AlgebraicScaleSweep ───────────────────────────────────────────────────

/// Sweep of algebraic attack complexity from 4-bit toy to full 6400-bit HDH.
///
/// Equation degrees from empirical analysis:
///   4-bit chi (toy):          d_eq = 5 (exact ANF from preimage.rs)
///   8-bit chi (estimated):    d_eq = 7 (extrapolated)
///   64-bit chi (1 lane):      d_eff = 32 (sat.rs free-variable bound)
///   6400-bit HDH 1-round:     d_eq ≤ 3 (integral distinguisher result)
///   6400-bit HDH 2-round:     d_eq > 4 (integral distinguisher result)
///   6400-bit HDH 4-round:     d_eq ≈ 81 (3^4 round-degree compound estimate)
///
/// For small systems (n ≤ 32): exhaustive search in 2^n is dominant.
/// For 64-bit chi (n=256, d_eff=32): XL uses the SAT-derived effective degree,
///   not the theoretical equation degree (which is unknown exactly).
/// For 6400-bit HDH: solving degree >> equation degree due to underdetermination.
#[derive(Debug)]
pub struct ScaleEntry {
    pub description: &'static str,
    pub n_vars: usize,
    pub eq_degree: usize,
    pub xl_solving_degree: usize,
    pub macaulay_log2: f64,
    pub xl_time_log2: f64,
    pub xl_memory_log2: f64,
    pub hybrid_time_log2: f64,
    pub exhaustive_log2: f64,
    /// Best known attack complexity (min of XL, hybrid, exhaustive).
    pub best_known_log2: f64,
    pub is_gpu_feasible_exascale: bool,
}

#[derive(Debug)]
pub struct AlgebraicScaleSweep {
    pub entries: Vec<ScaleEntry>,
}

pub fn algebraic_scale_sweep() -> AlgebraicScaleSweep {
    // (description, n_vars, eq_degree, use_exact_d_XL_or_estimate)
    // For each entry: (description, n, eq_degree, override_dxl: Option<usize>)
    let configs: &[(&'static str, usize, usize, Option<usize>)] = &[
        ("4-bit χ toy (n=16)",          16,   5, Some(5)),   // toy: d_eq = d_XL = 5
        ("8-bit χ reduced (n=32)",      32,   7, Some(7)),   // still small
        ("64-bit χ lane (n=256)",       256, 32, Some(32)),  // d_eff from sat.rs
        ("6400-bit HDH 1-round",       6400,  3, None),      // estimate d_XL
        ("6400-bit HDH 2-round (d>4)", 6400,  8, None),      // d_eq > 4 → use 8
        ("6400-bit HDH 4-round (est.)",6400, 81, None),      // 3^4 round composition
    ];

    let entries = configs.iter().map(|&(desc, n, d_eq, override_dxl)| {
        let sd = match override_dxl {
            Some(d) => SolvingDegreeEstimate {
                n_vars: n, n_eqs: n, eq_degree: d_eq, d_xl: d,
                macaulay_log2: log2_binom(n + d, d),
                xl_time_log2:  log2_binom(n + d, d) * OMEGA,
                xl_memory_log2: log2_binom(n + d, d) * 2.0,
            },
            None => estimate_solving_degree(n, n, d_eq),
        };

        let hyb = hybrid_attack_optimum(n, sd.d_xl);
        let exh = n as f64;  // exhaustive: 2^n
        let best = sd.xl_time_log2.min(hyb.total_log2).min(exh);

        ScaleEntry {
            description: desc,
            n_vars: n,
            eq_degree: d_eq,
            xl_solving_degree: sd.d_xl,
            macaulay_log2: sd.macaulay_log2,
            xl_time_log2: sd.xl_time_log2,
            xl_memory_log2: sd.xl_memory_log2,
            hybrid_time_log2: hyb.total_log2,
            exhaustive_log2: exh,
            best_known_log2: best,
            // Exascale GPU feasible in 1 year: complexity < 2^{73+25} = 2^{98}
            is_gpu_feasible_exascale: best < 73.0 + LOG2_SECONDS_PER_YEAR,
        }
    }).collect();

    AlgebraicScaleSweep { entries }
}
