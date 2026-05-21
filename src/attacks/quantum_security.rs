/// Quantum security analysis for the HDH sponge construction.
///
/// Models quantum attacks using:
///   - Grover's algorithm (preimage):          O(2^{n/2}) quantum queries.
///   - BHT collision finding (Brassard–Høyer–Tapp 1998): O(2^{n/3}) quantum queries.
///   - NIST PQC security level mapping.
///   - Quantum-hybrid XL (Grover on the search component).
///   - Simon's algorithm inapplicability analysis.
///
/// Key finding at c=512, output=512:
///   Classical collision:  256 bits.
///   Quantum BHT collision: ⌊512/3⌋ = 170 bits  (exceeds NIST Level 5 = 128 bits).
///   Quantum Grover preimage: 256 bits.
///   For 256-bit quantum collision: c ≥ 768 required.
///
/// Five components:
/// 1. QuantumHashBounds      – Grover + BHT security bounds per (c, n) pair.
/// 2. QuantumCapacitySweep   – sweep over candidate capacities.
/// 3. QuantumXLModel         – quantum-hybrid algebraic attack complexity.
/// 4. NistLevelMapping       – NIST PQC security level classification.
/// 5. QuantumSecuritySummary – assembled claims for HDH at recommended params.

// ── 1. QuantumHashBounds ────────────────────────────────────────────────────

/// Classical and quantum security bounds for a sponge hash with given parameters.
///
/// Classical bounds (Bertoni et al. 2011):
///   Collision:       min(c/2, n/2) bits.   (birthday on inner or output)
///   Preimage:        min(c,   n)   bits.   (brute-force inner or output)
///   2nd preimage:    min(c,   n)   bits.
///
/// Quantum bounds:
///   Collision (BHT): min(c/3, n/3) bits.   (quantum birthday / quantum walk)
///   Preimage (Grover): min(c/2, n/2) bits. (Grover on inner or output search)
///   2nd preimage:    same as preimage.
///
/// Note: QROM (quantum random oracle model) indifferentiability requires a
/// separate analysis; the above bounds apply under the classical oracle model
/// with quantum adversary queries to the construction (QCCA2 setting).
#[derive(Debug)]
pub struct QuantumHashBounds {
    pub state_bits: usize,
    pub capacity_bits: usize,
    pub output_bits: usize,

    // ── Classical ──────────────────────────────────────────────────────────
    /// Classical collision: min(c/2, n/2).
    pub classical_collision_bits: f64,
    /// Classical preimage: min(c, n).
    pub classical_preimage_bits: f64,
    /// Classical 2nd-preimage: min(c, n).
    pub classical_second_preimage_bits: f64,

    // ── Quantum ────────────────────────────────────────────────────────────
    /// Quantum collision (BHT): min(c/3, n/3).
    pub quantum_collision_bits: f64,
    /// Quantum preimage (Grover): min(c/2, n/2).
    pub quantum_preimage_bits: f64,
    /// Quantum 2nd-preimage: min(c/2, n/2).
    pub quantum_second_preimage_bits: f64,

    // ── NIST level flags ───────────────────────────────────────────────────
    /// NIST PQC Level 1: quantum collision ≥ 64, quantum preimage ≥ 128.
    pub meets_nist_level1: bool,
    /// NIST PQC Level 3: quantum collision ≥ 96, quantum preimage ≥ 192.
    pub meets_nist_level3: bool,
    /// NIST PQC Level 5: quantum collision ≥ 128, quantum preimage ≥ 256.
    pub meets_nist_level5: bool,
}

pub fn quantum_hash_bounds(
    state_bits: usize,
    capacity_bits: usize,
    output_bits: usize,
) -> QuantumHashBounds {
    let c = capacity_bits as f64;
    let n = output_bits as f64;

    let cl_col = (c / 2.0).min(n / 2.0);
    let cl_pre = c.min(n);
    let q_col  = (c / 3.0).min(n / 3.0);
    let q_pre  = (c / 2.0).min(n / 2.0);

    QuantumHashBounds {
        state_bits,
        capacity_bits,
        output_bits,
        classical_collision_bits: cl_col,
        classical_preimage_bits: cl_pre,
        classical_second_preimage_bits: cl_pre,
        quantum_collision_bits: q_col,
        quantum_preimage_bits: q_pre,
        quantum_second_preimage_bits: q_pre,
        meets_nist_level1: q_col >= 64.0  && q_pre >= 128.0,
        meets_nist_level3: q_col >= 96.0  && q_pre >= 192.0,
        meets_nist_level5: q_col >= 128.0 && q_pre >= 256.0,
    }
}

// ── 2. QuantumCapacitySweep ──────────────────────────────────────────────────

#[derive(Debug)]
pub struct QuantumCapacityEntry {
    pub capacity_bits: usize,
    pub rate_bits: usize,
    pub throughput_fraction: f64,
    pub bounds: QuantumHashBounds,
}

#[derive(Debug)]
pub struct QuantumCapacitySweep {
    pub state_bits: usize,
    pub output_bits: usize,
    pub entries: Vec<QuantumCapacityEntry>,
    /// Minimum capacity for 128-bit quantum collision resistance (BHT).
    pub min_capacity_quantum_128_collision: usize,
    /// Minimum capacity for 256-bit quantum collision resistance.
    pub min_capacity_quantum_256_collision: usize,
    /// Minimum capacity for NIST PQC Level 5 (q_col ≥ 128, q_pre ≥ 256).
    pub min_capacity_nist_level5: usize,
}

const SWEEP_CAPACITIES: &[usize] = &[
    128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1280, 1536,
];

pub fn quantum_capacity_sweep(state_bits: usize, output_bits: usize) -> QuantumCapacitySweep {
    let mut entries = Vec::new();
    let mut min_q128 = usize::MAX;
    let mut min_q256 = usize::MAX;
    let mut min_l5 = usize::MAX;

    for &c in SWEEP_CAPACITIES {
        if c >= state_bits { continue; }
        let r = state_bits - c;
        let b = quantum_hash_bounds(state_bits, c, output_bits);

        if b.quantum_collision_bits >= 128.0 && c < min_q128 { min_q128 = c; }
        if b.quantum_collision_bits >= 256.0 && c < min_q256 { min_q256 = c; }
        if b.meets_nist_level5       && c < min_l5  { min_l5  = c; }

        entries.push(QuantumCapacityEntry {
            capacity_bits: c,
            rate_bits: r,
            throughput_fraction: r as f64 / state_bits as f64,
            bounds: b,
        });
    }

    QuantumCapacitySweep {
        state_bits,
        output_bits,
        entries,
        min_capacity_quantum_128_collision: min_q128,
        min_capacity_quantum_256_collision: min_q256,
        min_capacity_nist_level5: min_l5,
    }
}

// ── 3. QuantumXLModel ────────────────────────────────────────────────────────

/// Quantum-hybrid XL complexity model.
///
/// Classical hybrid: fix k vars exhaustively (2^k), Gröbner on remaining n-k.
/// Quantum hybrid:   fix k vars with Grover (2^{k/2}), Gröbner classically.
///
/// Quantum Gröbner speedup: in general, quantum linear algebra (HHL) does not
/// help for Gaussian elimination over GF(2) because:
///   1. HHL solves Ax=b over ℝ (wrong field).
///   2. The quantum speedup is conditional on well-conditioned matrices.
///   3. Read-out of the full solution requires exponential samples.
/// Conservative model: Gröbner complexity is identical classically and quantumly.
///
/// Aggressive model (BBBV / Grover on solution-checking):
///   Quantum XL ≈ √(classical XL) when XL is viewed as a search over candidate
///   solutions.  This requires oracle access to a compact check function —
///   realistic for small systems, unrealistic for large ones.
#[derive(Debug)]
pub struct QuantumXLEntry {
    pub description: &'static str,
    pub n_vars: usize,
    pub xl_solving_degree: usize,
    /// log2 of classical XL time.
    pub classical_xl_log2: f64,
    /// log2 of quantum-hybrid time (Grover on k-bit search, classical Gröbner).
    pub quantum_hybrid_log2: f64,
    /// log2 of aggressive quantum (√ of classical).
    pub quantum_aggressive_log2: f64,
    /// Best quantum complexity (minimum of hybrid and aggressive models).
    pub best_quantum_log2: f64,
    /// Classical hybrid improvement over pure Gröbner (log2 bits).
    pub classical_hybrid_improvement: f64,
}

#[derive(Debug)]
pub struct QuantumXLModel {
    pub entries: Vec<QuantumXLEntry>,
}

fn log2_binom_q(n: usize, k: usize) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    let k = k.min(n - k);
    (0..k).map(|i| ((n - i) as f64 / (i + 1) as f64).log2()).sum()
}
const OMEGA_Q: f64 = 2.37;

fn classical_xl(n: usize, d: usize) -> f64 {
    log2_binom_q(n + d, d) * OMEGA_Q
}

fn quantum_hybrid_xl(n: usize, d: usize) -> f64 {
    // For each k, quantum cost = 2^{k/2} + Gröbner(n-k, d).
    // minimize over k.
    let pure_gb = classical_xl(n, d);
    let mut best = pure_gb;
    for k in 1..n {
        let gb = classical_xl(n - k, d);
        // Grover on k bits: 2^{k/2}.
        let grover = k as f64 / 2.0;
        let total = if (grover - gb).abs() < 1.0 {
            grover.max(gb) + 1.0
        } else {
            grover.max(gb)
        };
        if total < best { best = total; }
        if grover > pure_gb + 2.0 { break; }
    }
    best
}

// Configs: (description, n, eq_degree, d_XL)
const XL_CONFIGS: &[(&str, usize, usize, usize)] = &[
    ("4-bit χ (n=16, d=5)",          16,   5,  5),
    ("64-bit χ (n=256, d_eff=32)",   256,  32, 32),
    ("6400b HDH 1-round (d_eq=3)",  6400,   3, 340),  // estimate_solving_degree
    ("6400b HDH 2-round (d_eq>4)",  6400,   8, 700),  // higher solving degree
    ("6400b HDH 4-round (est.)",    6400,  81, 2000),
];

pub fn quantum_xl_model() -> QuantumXLModel {
    let entries = XL_CONFIGS.iter().map(|&(desc, n, _d_eq, d_xl)| {
        let cl = classical_xl(n, d_xl);
        let q_hyb = quantum_hybrid_xl(n, d_xl);
        // Aggressive: sqrt of classical
        let q_agg = cl / 2.0;
        let best_q = q_hyb.min(q_agg);
        // Classical hybrid improvement
        let cl_hyb_k = (0..n).map(|k| {
            let gb = classical_xl(n - k, d_xl);
            let s = k as f64;
            if (s - gb).abs() < 1.0 { s.max(gb) + 1.0 } else { s.max(gb) }
        }).fold(cl, f64::min);
        QuantumXLEntry {
            description: desc,
            n_vars: n,
            xl_solving_degree: d_xl,
            classical_xl_log2: cl,
            quantum_hybrid_log2: q_hyb,
            quantum_aggressive_log2: q_agg,
            best_quantum_log2: best_q,
            classical_hybrid_improvement: cl - cl_hyb_k,
        }
    }).collect();
    QuantumXLModel { entries }
}

// ── 4. NistLevelMapping ──────────────────────────────────────────────────────

/// NIST PQC security level definitions (from NIST SP 800-208 and PQC evaluation):
///   Level 1: at least as hard to break as AES-128 (key search: 2^{128} classical,
///            2^{64} quantum via Grover).
///   Level 3: at least as hard as AES-192 (2^{192} classical, 2^{96} quantum).
///   Level 5: at least as hard as AES-256 (2^{256} classical, 2^{128} quantum).
///
/// For hash functions:
///   Level 1: collision resistance ≥ 2^{64} quantum, preimage ≥ 2^{128} quantum.
///   Level 3: collision ≥ 2^{96} quantum, preimage ≥ 2^{192} quantum.
///   Level 5: collision ≥ 2^{128} quantum, preimage ≥ 2^{256} quantum.
#[derive(Debug)]
pub struct NistLevelAssignment {
    pub state_bits: usize,
    pub capacity_bits: usize,
    pub output_bits: usize,
    pub quantum_collision_bits: f64,
    pub quantum_preimage_bits: f64,
    pub nist_level: u8,   // 0=none, 1, 3, or 5
    pub nist_level_description: &'static str,
    /// Bits of quantum collision margin above the assigned level.
    pub collision_margin_bits: f64,
}

pub fn map_to_nist_level(
    state_bits: usize,
    capacity_bits: usize,
    output_bits: usize,
) -> NistLevelAssignment {
    let b = quantum_hash_bounds(state_bits, capacity_bits, output_bits);
    let q_col = b.quantum_collision_bits;
    let q_pre = b.quantum_preimage_bits;

    let (level, desc, threshold) = if q_col >= 128.0 && q_pre >= 256.0 {
        (5u8, "NIST PQC Level 5 (≥ AES-256 quantum)", 128.0f64)
    } else if q_col >= 96.0 && q_pre >= 192.0 {
        (3u8, "NIST PQC Level 3 (≥ AES-192 quantum)", 96.0f64)
    } else if q_col >= 64.0 && q_pre >= 128.0 {
        (1u8, "NIST PQC Level 1 (≥ AES-128 quantum)", 64.0f64)
    } else {
        (0u8, "below NIST PQC Level 1", 0.0f64)
    };

    NistLevelAssignment {
        state_bits,
        capacity_bits,
        output_bits,
        quantum_collision_bits: q_col,
        quantum_preimage_bits: q_pre,
        nist_level: level,
        nist_level_description: desc,
        collision_margin_bits: q_col - threshold,
    }
}

/// Sweep NIST level assignments across output lengths at c=512.
#[derive(Debug)]
pub struct NistLevelSweep {
    pub state_bits: usize,
    pub capacity_bits: usize,
    pub entries: Vec<(usize, NistLevelAssignment)>,  // (output_bits, assignment)
}

pub fn nist_level_sweep(state_bits: usize, capacity_bits: usize) -> NistLevelSweep {
    let output_sizes = [128usize, 224, 256, 384, 512, 1024];
    let entries = output_sizes.iter()
        .map(|&n| (n, map_to_nist_level(state_bits, capacity_bits, n)))
        .collect();
    NistLevelSweep { state_bits, capacity_bits, entries }
}

// ── 5. QuantumSecuritySummary ────────────────────────────────────────────────

/// Assembled quantum security summary for HDH at recommended parameters.
#[derive(Debug)]
pub struct QuantumSecuritySummary {
    pub state_bits: usize,
    pub rate_bits: usize,
    pub capacity_bits: usize,
    pub output_bits: usize,

    // ── Classical ──────────────────────────────────────────────────────────
    pub classical_collision_bits: f64,
    pub classical_preimage_bits: f64,

    // ── Quantum ────────────────────────────────────────────────────────────
    pub quantum_collision_bits: f64,
    pub quantum_preimage_bits: f64,
    pub nist_level: u8,
    pub nist_level_description: &'static str,

    // ── Capacity notes ─────────────────────────────────────────────────────
    /// Minimum capacity for 128-bit quantum collision (BHT) = 384 bits.
    pub min_capacity_quantum_128_collision: usize,
    /// Minimum capacity for 256-bit quantum collision (BHT) = 768 bits.
    pub min_capacity_quantum_256_collision: usize,
    /// Whether current c meets 128-bit quantum collision.
    pub meets_quantum_128_collision: bool,

    // ── Algebraic quantum ──────────────────────────────────────────────────
    /// log2 of best quantum algebraic attack on 2-round HDH.
    pub best_quantum_algebraic_log2: f64,
    /// Whether quantum algebraic attack is infeasible (> 2^{128}).
    pub quantum_algebraic_infeasible: bool,

    // ── Simon's algorithm ──────────────────────────────────────────────────
    /// Simon's algorithm applies to functions with a hidden XOR period.
    /// HDH's Φ routing is state-dependent (non-affine) → no hidden period.
    pub simon_applicable: bool,
    pub simon_inapplicability_reason: &'static str,
}

pub fn quantum_security_summary(
    state_bits: usize,
    rate_bits: usize,
    output_bits: usize,
) -> QuantumSecuritySummary {
    let c = state_bits - rate_bits;
    let bounds = quantum_hash_bounds(state_bits, c, output_bits);
    let nist = map_to_nist_level(state_bits, c, output_bits);

    // Quantum algebraic: best attack from QuantumXLModel on 2-round HDH entry.
    // Using d_XL ≈ 700 (conservative estimate for 2-round, d_eq=8).
    let q_alg = quantum_xl_model()
        .entries.iter()
        .find(|e| e.description.contains("2-round"))
        .map(|e| e.best_quantum_log2)
        .unwrap_or(f64::INFINITY);

    // Capacity sweep to find minimums.
    let sweep = quantum_capacity_sweep(state_bits, output_bits);

    QuantumSecuritySummary {
        state_bits,
        rate_bits,
        capacity_bits: c,
        output_bits,
        classical_collision_bits: bounds.classical_collision_bits,
        classical_preimage_bits: bounds.classical_preimage_bits,
        quantum_collision_bits: bounds.quantum_collision_bits,
        quantum_preimage_bits: bounds.quantum_preimage_bits,
        nist_level: nist.nist_level,
        nist_level_description: nist.nist_level_description,
        min_capacity_quantum_128_collision: sweep.min_capacity_quantum_128_collision,
        min_capacity_quantum_256_collision: sweep.min_capacity_quantum_256_collision,
        meets_quantum_128_collision: bounds.quantum_collision_bits >= 128.0,
        best_quantum_algebraic_log2: q_alg,
        quantum_algebraic_infeasible: q_alg > 128.0,
        simon_applicable: false,
        simon_inapplicability_reason:
            "Φ routing j=(s1⊕s2)%25 is a nonlinear function of state; \
             no hidden XOR period exists in the permutation orbit structure.",
    }
}
