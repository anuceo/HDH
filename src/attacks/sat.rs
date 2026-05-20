/// SAT/XL (eXtended Linearization) complexity model for one χ lane.
///
/// The XL attack replaces each product term aᵢ·bⱼ with a fresh variable zᵢⱼ,
/// turning the nonlinear constraint system into a linear one.  We model:
///   - how many variables the linearized system has
///   - how many input-output pairs are needed to overdetermine it
///   - why the attack still fails: the consistency constraints zᵢⱼ = aᵢ·bⱼ are
///     themselves nonlinear, and the true algebraic degree of wrapping_mul
///     (≈ 32 for 64-bit words) makes exhaustive Gröbner / XL infeasible.
///
/// The DPLL/CDCL model estimates the branch-and-bound search tree size when
/// a SAT solver attacks the per-lane Boolean constraint system directly.

pub struct XlModel {
    /// Primary Boolean unknowns: 4 shares × 64 bits.
    pub input_vars: usize,
    /// Constraint bits produced by one input-output pair.
    pub output_bits_per_pair: usize,
    /// Product monomials introduced by linearizing wrapping_mul × 2.
    /// Each 64×64 multiplication contributes ≤ 64·65/2 = 2080 distinct cross-products
    /// (upper-triangular, due to carry truncation at bit 63).
    pub linearized_product_terms: usize,
    /// Total unknowns in the degree-2 linearized system.
    pub total_xl_vars: usize,
    /// Pairs needed for the linearized system to be square (equations = unknowns).
    pub pairs_for_square_system: usize,
    /// Pairs needed for overdetermination (equations > unknowns).
    pub pairs_for_overdetermination: usize,
    /// Whether degree-2 XL is sufficient to solve — it is NOT, because
    /// the actual algebraic degree of 64-bit wrapping_mul is ≈ 32.
    pub degree2_xl_sufficient: bool,
    /// Lower bound on the XL degree needed to capture wrapping_mul's true degree.
    pub required_xl_degree_lower_bound: usize,
    /// Estimated number of monomials at the required XL degree (C(256, 32)).
    pub monomials_at_required_degree: String,
}

/// Conservative estimate: log2 of C(n, k).
fn log2_binom(n: usize, k: usize) -> f64 {
    // Use Stirling / log-sum to avoid overflow.
    let k = k.min(n - k);
    (0..k).map(|i| ((n - i) as f64).log2() - ((i + 1) as f64).log2()).sum()
}

pub fn model_xl_complexity() -> XlModel {
    let input_vars = 4 * 64; // 256
    let output_bits_per_pair = 4 * 64; // 256

    // Two wrapping_mul calls in quad; each 64×64 cross-product contributes
    // at most 64·65/2 = 2080 distinct aᵢ·bⱼ monomials (i ≤ j due to symmetry).
    let linearized_product_terms = 2 * (64 * 65 / 2); // 4160

    let total_xl_vars = input_vars + linearized_product_terms; // 4416

    let pairs_for_square = (total_xl_vars + output_bits_per_pair - 1) / output_bits_per_pair;
    let pairs_for_overdetermination = pairs_for_square + 1;

    // True algebraic degree of 64-bit wrapping_mul over GF(2):
    // Bit k of a·b mod 2^64 involves carries from all lower bits; the carry
    // chain raises the degree exponentially.  Empirical / theoretical results
    // show degree ≥ n/2 for the high-order bits of n-bit integer multiplication.
    let required_degree: usize = 32;

    let monomials_desc = format!("C(256,32) ≈ 2^{:.1}", log2_binom(256, 32));

    XlModel {
        input_vars,
        output_bits_per_pair,
        linearized_product_terms,
        total_xl_vars,
        pairs_for_square_system: pairs_for_square,
        pairs_for_overdetermination,
        degree2_xl_sufficient: false,
        required_xl_degree_lower_bound: required_degree,
        monomials_at_required_degree: monomials_desc,
    }
}

pub struct DpllModel {
    /// Boolean variables in the per-lane SAT formula.
    pub boolean_vars: usize,
    /// Clauses from one known input-output pair (2-CNF lower bound per bit).
    pub clauses_per_pair: usize,
    /// After unit propagation, free variables remaining (rough estimate).
    pub free_vars_after_up: usize,
    /// Lower bound on DPLL branch-and-bound nodes (2^free_vars).
    pub log2_search_space: f64,
}

pub fn model_dpll_complexity() -> DpllModel {
    let boolean_vars = 4 * 64; // 256

    // Each output bit produces one non-trivial clause; wrapping_mul introduces
    // O(n^2) additional clauses encoding the multiplication structure.
    // Lower bound: 256 output equalities + n^2 product clauses per mul.
    let clauses_per_pair = 256 + 2 * 64 * 64; // ≈ 8448

    // Unit propagation on the output equality clauses can force at most one
    // share's value once g is fixed.  But g itself is constrained by a degree-32+
    // equation — UP cannot resolve it without branching on ≈ 64 free bits.
    let free_vars_after_up = 64usize;

    DpllModel {
        boolean_vars,
        clauses_per_pair,
        free_vars_after_up,
        log2_search_space: free_vars_after_up as f64,
    }
}
