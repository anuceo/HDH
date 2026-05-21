/// Phase 2A — Gröbner Basis Growth Simulation for HDH.
///
/// Models the symbolic polynomial system arising from HDH reduced-round
/// equations and simulates F4/F5 characteristics: degree growth, Macaulay
/// matrix explosion, sparsity collapse, and elimination-order sensitivity.
///
/// Core claim: the "solving degree explosion" hypothesis holds — the XL
/// solving degree d_XL >> d_eq (340 vs 3 for 1-round), making the Macaulay
/// matrix astronomically large long before the solving degree is reached.
///
/// Five components:
/// 1. SymbolicRoundEquations  – symbolic structure of the HDH polynomial system.
/// 2. MonomialGraphStats      – variable-interaction graph (connectivity / fill-in).
/// 3. EliminationOrderResult  – comparison of lex / degrevlex / block orderings.
/// 4. F4F5Simulation          – degree-by-degree Macaulay matrix growth.
/// 5. DegreeGrowthProjection  – round-by-round complexity sweep.

use crate::attacks::gpu_algebraic::{
    estimate_solving_degree, hybrid_attack_optimum, log2_binom, xl_complexity, MQSystemParams,
};

const OMEGA: f64 = 2.37;

// ── 1. SymbolicRoundEquations ─────────────────────────────────────────────────

/// Model of the HDH polynomial system after `rounds` rounds.
///
/// χ contributes degree-2 terms (lane-local, 5-bit window).
/// θ is linear but maps each bit to a linear combination of 5-lane neighbors,
///   turning lane-local degree-2 terms into cross-lane degree-2 terms.
/// Φ is state-dependent routing (its routing index adds one degree), so the
///   composition Φ ∘ θ ∘ χ yields degree-3 equations after 1 round.
#[derive(Debug)]
pub struct SymbolicRoundEquations {
    pub n_vars: usize,
    pub n_eqs: usize,
    pub rounds: usize,
    pub eq_degree: usize,
    /// log2 of the dominant monomial count: C(n_vars, eq_degree).
    pub total_monomials_log2: f64,
    /// Estimated non-zero monomials per equation (structural estimate).
    pub monomials_per_eq: usize,
    /// log2 of the system density: log2(monomials_per_eq) − total_monomials_log2.
    /// Very negative → extremely sparse.
    pub log2_density: f64,
    /// Terms originating from χ (within one 64-bit lane).
    pub chi_local_terms: usize,
    /// Terms originating from θ (cross-lane ring diffusion).
    pub theta_cross_terms: usize,
    /// Additional terms from Φ (state-dependent routing, degree +1).
    pub phi_routing_terms: usize,
    /// True when total_monomials >> n_eqs (system is massively underdetermined).
    pub is_underdetermined: bool,
}

/// Degree model derived from empirical integral distinguisher measurements.
///
/// | rounds | eq_degree | source                            |
/// |--------|-----------|-----------------------------------|
/// | 1      | 3         | integral.rs (D^4 F = 0)           |
/// | 2      | 8         | integral.rs (D^4 F ≠ 0, degree>4) |
/// | 3      | 20        | extrapolated (3×degree growth)    |
/// | 4+     | 81        | gpu_algebraic.rs (3^4 estimate)   |
pub fn degree_for_rounds(rounds: usize) -> usize {
    match rounds {
        0 => 1,
        1 => 3,
        2 => 8,
        3 => 20,
        _ => 81,
    }
}

/// Construct a structural model of the HDH polynomial system.
pub fn construct_symbolic_system(n_vars: usize, rounds: usize) -> SymbolicRoundEquations {
    let eq_degree = degree_for_rounds(rounds);
    let n_eqs = n_vars;

    let total_monomials_log2 = log2_binom(n_vars, eq_degree);

    // HDH structural term estimates per output bit:
    //   χ: window-5 → C(5,2)+C(5,3) = 20 terms; 64-bit lane has ≈13 windows
    //      → 13 × 10 = 130 chi terms per lane per round.
    //   θ: ring-diffusion across 5 lanes → 5 × 64 = 320 input bits per output
    //      → 320 additional degree-2 cross-lane terms.
    //   Φ: state-dependent routing adds 1 degree; contribution ≈ 400 new degree-3
    //      terms per output bit for 1-round HDH.
    //   Higher rounds: terms grow roughly as n^{rounds/2} capped at n.
    let (chi_terms, theta_terms, phi_terms): (usize, usize, usize) = match rounds {
        0 => (0, 64, 0),
        1 => (130, 1_500, 400),
        2 => (500, 6_000, 2_000),
        3 => (2_000, 24_000, 8_000),
        _ => (n_vars / 10, n_vars / 2, n_vars / 4),
    };
    let monomials_per_eq = chi_terms + theta_terms + phi_terms;

    let log2_density = (monomials_per_eq as f64).log2() - total_monomials_log2;

    SymbolicRoundEquations {
        n_vars,
        n_eqs,
        rounds,
        eq_degree,
        total_monomials_log2,
        monomials_per_eq,
        log2_density,
        chi_local_terms: chi_terms,
        theta_cross_terms: theta_terms,
        phi_routing_terms: phi_terms,
        is_underdetermined: total_monomials_log2 > (n_eqs as f64).log2() + 1.0,
    }
}

// ── 2. MonomialGraphStats ─────────────────────────────────────────────────────

/// Variable-interaction graph statistics for the HDH polynomial system.
///
/// Nodes: individual boolean variables (6400 for full HDH).
/// Edges: pairs of variables that appear together in some monomial of the system.
///
/// The θ ring-diffusion layer is the key structural element: after applying θ,
/// every variable interacts (via degree-2 monomials) with variables from 5
/// neighboring lanes (5 × 64 = 320 variables).  This makes the interaction
/// graph far too dense to decompose or exploit block structure.
#[derive(Debug)]
pub struct MonomialGraphStats {
    pub n_vars: usize,
    pub eq_degree: usize,
    /// log2 of the number of edges in the variable-interaction graph.
    pub log2_n_edges: f64,
    /// Average number of variables each variable interacts with (degree of node).
    pub avg_degree: f64,
    /// Fraction of all possible edges present (graph density ∈ [0,1]).
    pub graph_density: f64,
    /// Number of connected components.  1 = fully connected.
    pub n_connected_components: usize,
    /// Estimated fill-in factor during Gaussian elimination:
    /// how many new edges appear when one variable is eliminated.
    pub fill_in_factor: f64,
    /// True when the graph has no exploitable block or lane structure.
    pub no_block_structure: bool,
    /// True when fully connected (single component).
    pub is_connected: bool,
}

/// Analyze the variable-interaction graph for the HDH polynomial system.
pub fn analyze_monomial_graph(
    n_vars: usize,
    rounds: usize,
    eq_degree: usize,
) -> MonomialGraphStats {
    let max_edges = (n_vars as f64) * (n_vars as f64 - 1.0) / 2.0;

    // χ: each bit interacts with 4 neighbors (window of 5) → 4 edges per bit.
    //    Within-lane edges ≈ n_vars × 2 (undirected, both directions).
    let chi_edges = 2.0 * n_vars as f64;

    // θ: each bit depends on 5 × 64 = 320 bits from neighboring lanes.
    //    Cross-lane edges ≈ n_vars × 320 / 2.
    let theta_edges = n_vars as f64 * 320.0 / 2.0;

    // Φ: state-dependent routing further couples all lanes across shares.
    //    After Φ, each bit interacts with O(n_vars^{1/2}) further bits.
    let phi_edges = match rounds {
        0 => 0.0,
        1 => n_vars as f64 * (n_vars as f64).sqrt() / 2.0,
        _ => n_vars as f64 * n_vars as f64 / 4.0,  // ≈ 25% of max_edges
    };

    let total_edges = chi_edges + theta_edges + phi_edges;
    let graph_density = (total_edges / max_edges).min(1.0);
    let avg_deg = 2.0 * total_edges / n_vars as f64;

    // Fill-in during Gaussian elimination: each eliminated variable causes its
    // neighbors to be connected to each other (clique).  For a node with
    // avg_degree d: elimination adds d×(d−1)/2 new edges.
    let fill_in = avg_deg * (avg_deg - 1.0) / 2.0;

    // After θ, the graph is connected (1 component) because θ maps every bit
    // to a linear combination of bits from 5 different lanes, and 25 lanes
    // form a connected ring.
    let connected = rounds >= 1 || n_vars <= 64;

    // Block structure requires the interaction graph to partition into independent
    // groups.  θ's ring diffusion makes this impossible after 1 round.
    let no_block = rounds >= 1;

    MonomialGraphStats {
        n_vars,
        eq_degree,
        log2_n_edges: total_edges.max(1.0).log2(),
        avg_degree: avg_deg,
        graph_density,
        n_connected_components: if connected { 1 } else { 25 },
        fill_in_factor: fill_in,
        no_block_structure: no_block,
        is_connected: connected,
    }
}

// ── 3. Elimination order comparison ──────────────────────────────────────────

/// Comparison of one Gröbner basis monomial ordering for HDH.
#[derive(Debug)]
pub struct EliminationOrderResult {
    pub order_name: &'static str,
    pub description: &'static str,
    /// Estimated solving degree under this ordering.
    pub solving_degree: usize,
    /// log2 of the Macaulay matrix column count at solving_degree.
    pub macaulay_log2: f64,
    /// log2 of total XL complexity = macaulay × ω.
    pub xl_complexity_log2: f64,
    /// Bits saved relative to lex ordering.  Positive = better than lex.
    pub bits_saved_vs_lex: f64,
    /// True when xl_complexity > 2^256 (infeasible).
    pub is_infeasible: bool,
}

/// Compare four standard elimination orderings for the HDH system.
///
/// Orderings analyzed:
///   lex          – lexicographic; worst case for F4/F5 (d_XL ≈ D_reg).
///   degrevlex    – degree-reverse-lex; optimal for F4/F5 (d_XL minimal).
///   block-drl    – block degrevlex exploiting share/lane structure.
///   FGLM         – convert from degrevlex to lex via FGLM (polynomial in #sols).
///
/// Degree of regularity (lex) for a generic regular sequence of n degree-d_eq
/// equations in n variables: D_reg = 1 + n × (d_eq − 1).
pub fn compare_elimination_orders(
    n_vars: usize,
    n_eqs: usize,
    eq_degree: usize,
) -> Vec<EliminationOrderResult> {
    // degrevlex solving degree (optimal): binary search for span condition.
    // For n=6400, n_eqs=6400 with d_eq=3, the span condition
    // n_eqs × C(n, d−d_eq) ≥ C(n+d, d) is satisfied at d_XL = n_vars (cap).
    // This gives mac_drl = log2_binom(2n, n) ≈ 2n bits.
    let sd_drl = estimate_solving_degree(n_vars, n_eqs, eq_degree);
    let d_drl = sd_drl.d_xl;
    let mac_drl = sd_drl.macaulay_log2;
    let xl_drl = sd_drl.xl_time_log2;

    // lex: D_reg = 1 + n × (d_eq − 1) = 12801 for d_eq=3, n=6400.
    // Unlike degrevlex, the lex Macaulay is evaluated at D_reg (not capped at n),
    // since lex orders monomials one variable at a time and must traverse D_reg levels.
    // C(n + D_reg, D_reg) = C(n + D_reg, n) — use n as the smaller index.
    let d_lex = 1 + n_vars * (eq_degree.saturating_sub(1));
    let mac_lex = log2_binom(n_vars + d_lex, n_vars);  // C(n+D_reg, n) ← use n, not D_reg
    let xl_lex = mac_lex * OMEGA;

    // block-drl: exploiting the 4-share structure.
    // θ coupling prevents true block decomposition: effective solving degree = d_drl.
    // Macaulay is identical to degrevlex (no advantage from block ordering under θ).
    let mac_block = mac_drl;
    let xl_block = xl_drl;

    // FGLM: convert degrevlex GB to lex via FGLM algorithm.
    // Cost = O(D^3) where D = number of solutions ≈ 2^{n_vars} for a generic
    // 0-dimensional system.  For HDH (n=6400), D^3 = 2^{19200} → infeasible.
    // Even if D = 1 (unique solution): FGLM cost ≈ degrevlex GB cost.
    let xl_fglm = xl_drl;

    vec![
        EliminationOrderResult {
            order_name: "lex",
            description: "lexicographic (worst case for F4/F5, D_reg = 1+n(d−1))",
            solving_degree: d_lex,
            macaulay_log2: mac_lex,
            xl_complexity_log2: xl_lex,
            bits_saved_vs_lex: 0.0,
            is_infeasible: xl_lex > 256.0,
        },
        EliminationOrderResult {
            order_name: "degrevlex",
            description: "degree-reverse-lex (optimal for F4/F5, d_XL = n_vars for square systems)",
            solving_degree: d_drl,
            macaulay_log2: mac_drl,
            xl_complexity_log2: xl_drl,
            bits_saved_vs_lex: xl_lex - xl_drl,
            is_infeasible: xl_drl > 256.0,
        },
        EliminationOrderResult {
            order_name: "block-drl",
            description: "block degrevlex (share blocks; θ coupling prevents advantage)",
            solving_degree: d_drl,
            macaulay_log2: mac_block,
            xl_complexity_log2: xl_block,
            bits_saved_vs_lex: xl_lex - xl_block,
            is_infeasible: xl_block > 256.0,
        },
        EliminationOrderResult {
            order_name: "FGLM-convert",
            description: "FGLM conversion from degrevlex (unique-solution case)",
            solving_degree: d_drl,
            macaulay_log2: mac_drl,
            xl_complexity_log2: xl_fglm,
            bits_saved_vs_lex: xl_lex - xl_fglm,
            is_infeasible: xl_fglm > 256.0,
        },
    ]
}

// ── 4. F4/F5 degree-growth simulation ────────────────────────────────────────

/// One degree step in the F4/F5 simulation.
#[derive(Debug)]
pub struct F4SimStep {
    /// Macaulay degree at this step.
    pub degree: usize,
    /// log2 of the Macaulay matrix column count: C(n + d, d).
    pub macaulay_cols_log2: f64,
    /// log2 of the span row count: n_eqs × C(n, d − d_eq).
    pub span_rows_log2: f64,
    /// log2 of initial density before row reduction: C(d, d_eq) / C(n+d, d).
    /// Very negative = extremely sparse.
    pub log2_initial_density: f64,
    /// log2 of the Macaulay matrix MEMORY requirement in bits: 2 × macaulay_cols².
    pub memory_log2: f64,
    /// log2 of monomial explosion rate: C(n+d+1,d+1)/C(n+d,d) = (n+d+1)/(d+1).
    pub log2_growth_rate: f64,
    /// True when memory fits in 2^64 bits (18 EB) — a generous upper bound.
    pub fits_in_memory: bool,
}

/// Results of the F4/F5 Macaulay matrix growth simulation.
#[derive(Debug)]
pub struct F4F5Simulation {
    pub n_vars: usize,
    pub n_eqs: usize,
    pub eq_degree: usize,
    /// XL solving degree (from estimate_solving_degree).
    pub solving_degree: usize,
    /// log2 of Macaulay at solving degree.
    pub macaulay_at_solving_degree_log2: f64,
    /// log2 of XL complexity.
    pub xl_complexity_log2: f64,
    /// Degree at which the matrix first exceeds 2^64 bits of memory.
    pub memory_infeasible_at_degree: usize,
    /// Degree at which the matrix first exceeds 2^256 operations.
    pub compute_infeasible_at_degree: usize,
    /// Degree steps surveyed.
    pub steps: Vec<F4SimStep>,
    /// Rate of monomial explosion per degree step near solving degree.
    pub monomial_explosion_rate: f64,
}

/// Simulate F4/F5 degree growth for a square GF(2) polynomial system.
///
/// Surveys the Macaulay matrix at geometric degree steps from d_eq to d_XL,
/// tracking size, density, and memory requirements.
pub fn simulate_f4_degree_growth(n_vars: usize, n_eqs: usize, eq_degree: usize) -> F4F5Simulation {
    let sd = estimate_solving_degree(n_vars, n_eqs, eq_degree);
    let d_xl = sd.d_xl;
    let mac_xl = sd.macaulay_log2;
    let xl = sd.xl_time_log2;

    // Sample degrees at geometric intervals: d_eq, 2·d_eq, 5·d_eq, 10·d_eq, ..., d_XL.
    let mut degrees = vec![eq_degree];
    let mut d = eq_degree * 2;
    while d < d_xl {
        degrees.push(d);
        d = (d * 5 / 3).min(d + (d_xl - eq_degree) / 6 + 1);
    }
    degrees.push(d_xl);
    degrees.dedup();

    let mut mem_infeasible = d_xl;  // default to solving degree if never triggered earlier
    let mut comp_infeasible = d_xl;

    let steps: Vec<F4SimStep> = degrees.iter().map(|&d| {
        let mac = log2_binom(n_vars + d, d);
        let span = (n_eqs as f64).log2() + log2_binom(n_vars, d.saturating_sub(eq_degree));
        let density = log2_binom(d, eq_degree) - mac;
        let mem = mac * 2.0;                // Macaulay matrix: mac² entries in GF(2)
        let growth = ((n_vars + d + 1) as f64 / (d + 1) as f64).log2();
        let fits = mem < 64.0;

        if !fits && d < mem_infeasible { mem_infeasible = d; }
        if mac * OMEGA > 256.0 && d < comp_infeasible { comp_infeasible = d; }

        F4SimStep {
            degree: d,
            macaulay_cols_log2: mac,
            span_rows_log2: span,
            log2_initial_density: density,
            memory_log2: mem,
            log2_growth_rate: growth,
            fits_in_memory: fits,
        }
    }).collect();

    let explosion_rate = ((n_vars + d_xl + 1) as f64 / (d_xl + 1) as f64).log2();

    F4F5Simulation {
        n_vars,
        n_eqs,
        eq_degree,
        solving_degree: d_xl,
        macaulay_at_solving_degree_log2: mac_xl,
        xl_complexity_log2: xl,
        memory_infeasible_at_degree: mem_infeasible,
        compute_infeasible_at_degree: comp_infeasible,
        steps,
        monomial_explosion_rate: explosion_rate,
    }
}

// ── 5. DegreeGrowthProjection ─────────────────────────────────────────────────

/// Per-round entry in the degree growth projection.
#[derive(Debug)]
pub struct DegreeGrowthEntry {
    pub rounds: usize,
    /// Algebraic equation degree.
    pub eq_degree: usize,
    /// XL solving degree.
    pub solving_degree: usize,
    /// log2 of Macaulay matrix at solving degree.
    pub macaulay_log2: f64,
    /// log2 of XL complexity.
    pub xl_complexity_log2: f64,
    /// log2 of hybrid attack optimum (best variable-fixing strategy).
    pub hybrid_complexity_log2: f64,
    /// Initial system density (log2): higher is denser.
    pub log2_initial_density: f64,
    /// True when xl_complexity > 2^{256}.
    pub is_infeasible_256bit: bool,
    /// True when EVEN the hybrid attack exceeds 2^{128}.
    pub hybrid_infeasible_128bit: bool,
}

/// Project Gröbner basis complexity across rounds 1–4.
pub fn project_round_degree_growth(state_bits: usize) -> Vec<DegreeGrowthEntry> {
    (1..=4usize).map(|r| {
        let d_eq = degree_for_rounds(r);
        let sd = estimate_solving_degree(state_bits, state_bits, d_eq);
        let hyb = hybrid_attack_optimum(state_bits, sd.d_xl);
        let sys = construct_symbolic_system(state_bits, r);

        DegreeGrowthEntry {
            rounds: r,
            eq_degree: d_eq,
            solving_degree: sd.d_xl,
            macaulay_log2: sd.macaulay_log2,
            xl_complexity_log2: sd.xl_time_log2,
            hybrid_complexity_log2: hyb.total_log2,
            log2_initial_density: sys.log2_density,
            is_infeasible_256bit: sd.xl_time_log2 > 256.0,
            hybrid_infeasible_128bit: hyb.total_log2 > 128.0,
        }
    }).collect()
}
