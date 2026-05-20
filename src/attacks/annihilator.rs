/// Algebraic immunity and invariant subspace analysis for 4-bit reduced χ.
///
/// **Annihilator search**: For each output bit of chi4, find the minimum
/// degree d such that no nonzero g of degree ≤ d satisfies f·g = 0 (or
/// (1⊕f)·g = 0).  This gives a lower bound on algebraic immunity (AI).
///
/// The search works by building a matrix whose rows are the evaluations of
/// all degree-≤d monomials restricted to the support of f (or 1⊕f).  If
/// the null space is trivial (rank = number of monomials), no annihilator
/// of that degree exists.
///
/// **Invariant subspace detection**: Scan all 65536 inputs to chi4 for
/// fixed points (chi4(x)=x), 2-cycles (chi4(chi4(x))=x, x≠chi4(x)), and
/// inputs that map to zero.  For a "random-looking" permutation on 2^16
/// points the expected fixed-point count is 1; large deviations indicate
/// algebraic structure.
///
/// **Algebraic immunity lower bound**: AI(f) ≥ min d where no annihilator
/// of degree ≤ d exists for any output bit.  Theoretical upper bound:
/// ⌈16/2⌉ = 8.  Any AI < 3 would indicate a practically exploitable
/// algebraic weakness.

// ── 4-bit χ (local copy — same as preimage.rs but private here) ────────────

fn chi4(packed: u16) -> u16 {
    let x1 = (packed & 0xF) as u8;
    let x2 = ((packed >> 4) & 0xF) as u8;
    let x3 = ((packed >> 8) & 0xF) as u8;
    let x4 = ((packed >> 12) & 0xF) as u8;

    let g = (x1.wrapping_mul(x2) ^ x3.wrapping_mul(x4)) & 0xF;
    let o1 = (x1 ^ g) & 0xF;
    let o2 = (x2 ^ ((g << 1) | (g >> 3)) & 0xF) & 0xF;
    let o3 = (x3 ^ ((g << 2) | (g >> 2)) & 0xF) & 0xF;
    let o4 = (x4 ^ ((g << 3) | (g >> 1)) & 0xF) & 0xF;
    (o1 as u16) | ((o2 as u16) << 4) | ((o3 as u16) << 8) | ((o4 as u16) << 12)
}

/// Truth table for output `bit` of chi4: index = 16-bit input, value ∈ {0,1}.
fn chi4_truth_table(bit: usize) -> Vec<u8> {
    (0u16..=0xFFFF)
        .map(|x| ((chi4(x) >> bit) & 1) as u8)
        .collect()
}

// ── Annihilator search ─────────────────────────────────────────────────────

/// Enumerate all monomials of degree exactly `d` over 16 variables.
/// Each monomial is represented as a bitmask of variable indices.
fn monomials_of_degree(n_vars: usize, d: usize) -> Vec<u32> {
    let mut result = Vec::new();
    // Iterate over all subsets of size d from {0..n_vars-1}
    // Use lexicographic combination enumeration.
    if d == 0 {
        result.push(0u32);
        return result;
    }
    let mut combo = vec![0usize; d];
    for i in 0..d {
        combo[i] = i;
    }
    loop {
        let mask: u32 = combo.iter().fold(0u32, |acc, &i| acc | (1 << i));
        result.push(mask);
        // Next combination in lexicographic order
        let mut i = d - 1;
        loop {
            combo[i] += 1;
            if combo[i] <= n_vars - (d - i) {
                for j in (i + 1)..d {
                    combo[j] = combo[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return result;
            }
            i -= 1;
        }
    }
}

/// All monomials of degree ≤ `d` over 16 variables.
fn monomials_up_to_degree(n_vars: usize, d: usize) -> Vec<u32> {
    (0..=d).flat_map(|k| monomials_of_degree(n_vars, k)).collect()
}

/// Evaluate monomial `mask` at input `x` (bit i of x gives variable i).
fn eval_monomial(mask: u32, x: u16) -> u8 {
    // Product of all x_i where bit i is set in mask (over GF(2), this is AND of bits)
    if mask == 0 {
        return 1; // constant 1
    }
    let selected = (x as u32) & mask;
    // GF(2) product = 1 iff all selected bits are 1
    if selected == mask { 1 } else { 0 }
}

/// GF(2) row reduction (in-place) over a matrix with `n_cols` columns packed
/// into u64 words.  Returns the rank.
fn gf2_rank(mut mat: Vec<Vec<u64>>, n_cols: usize) -> usize {
    let n_words = (n_cols + 63) / 64;
    let n_rows = mat.len();
    let mut pivot_row = 0usize;
    for col in 0..n_cols {
        let word = col / 64;
        let bit = col % 64;
        let found = (pivot_row..n_rows).find(|&r| (mat[r][word] >> bit) & 1 == 1);
        if let Some(r) = found {
            mat.swap(pivot_row, r);
            let prow: Vec<u64> = mat[pivot_row].clone();
            for r2 in 0..n_rows {
                if r2 != pivot_row && (mat[r2][word] >> bit) & 1 == 1 {
                    for w in 0..n_words {
                        mat[r2][w] ^= prow[w];
                    }
                }
            }
            pivot_row += 1;
        }
    }
    pivot_row
}

/// Check whether a nonzero annihilator of degree ≤ `d` exists for `tt`
/// (a Boolean truth table of length 2^16).
///
/// An annihilator g satisfies: for all x where tt[x]=1, g(x)=0.
/// We look for g ≠ 0 of degree ≤ d.
///
/// Method: build a matrix M where each row is a monomial m_i evaluated on the
/// support of tt.  If rank(M) < number of monomials, the null space is nontrivial
/// and an annihilator exists.  We check both f and 1⊕f.
pub fn has_annihilator_deg(tt: &[u8], d: usize) -> bool {
    has_ann_for_fn(tt, d) || has_ann_for_fn(&tt.iter().map(|&b| b ^ 1).collect::<Vec<_>>(), d)
}

fn has_ann_for_fn(tt: &[u8], d: usize) -> bool {
    const N_VARS: usize = 16;
    let monomials = monomials_up_to_degree(N_VARS, d);
    let n_mono = monomials.len();

    // Support of f: indices where tt[x] = 1.
    // Use 0u16..=u16::MAX to avoid overflow when tt.len() == 65536.
    let support: Vec<u16> = (0u16..=u16::MAX).filter(|&x| tt[x as usize] == 1).collect();
    let support_size = support.len();

    if support_size == 0 {
        // f is identically 0 → the constant 1 is an annihilator of degree 0
        return true;
    }

    // Build the evaluation matrix: rows = monomials, cols = support points.
    // Entry [i][j] = m_i(support[j]).
    // We need rank < n_mono to have a nontrivial kernel.
    // Equivalently, rank of the transpose (rows=support, cols=monomials) < n_mono.
    // We work with the transpose for efficient row-reduction.

    let n_words = (n_mono + 63) / 64;
    let mut mat: Vec<Vec<u64>> = vec![vec![0u64; n_words]; support_size];

    for (row_idx, &x) in support.iter().enumerate() {
        for (col_idx, &mono) in monomials.iter().enumerate() {
            if eval_monomial(mono, x) == 1 {
                mat[row_idx][col_idx / 64] |= 1u64 << (col_idx % 64);
            }
        }
    }

    let rank = gf2_rank(mat, n_mono);
    rank < n_mono
}

// ── Public API ─────────────────────────────────────────────────────────────

pub struct AiStats {
    /// AI estimate per output bit: the first d where an annihilator of degree ≤ d
    /// was found, or `max_degree + 1` if none was found in [1, max_degree].
    /// Interpretation: per_bit_lb[i] = AI(f_i) if AI < max_degree+1, else AI > max_degree.
    pub per_bit_lb: Vec<usize>,
    /// Minimum value across all output bits.
    pub min_lb: usize,
    /// Maximum value across all output bits.
    pub max_lb: usize,
    /// Theoretical upper bound: ⌈16/2⌉ = 8.
    pub theoretical_upper_bound: usize,
    /// Number of output bits for which no annihilator was found within the search range
    /// (i.e., AI > max_degree for these bits).  These are the carry-chain bits.
    pub high_ai_bit_count: usize,
}

/// Estimate algebraic immunity for each output bit of chi4.
///
/// Returns AI(f_i) = min degree where annihilator exists, for each output bit.
/// If no annihilator of degree ≤ max_degree is found, the estimate is max_degree+1
/// (a lower bound: AI > max_degree).
///
/// Note: for the degree-2 output bits of chi4 (where g's contribution is degree-2),
/// AI = 2 is expected and correct — f is itself an annihilator of 1⊕f at degree 2.
/// The cryptographically interesting bits are those with AI ≥ 3 (carry-chain bits).
pub fn analyze_algebraic_immunity(max_degree: usize) -> AiStats {
    let mut per_bit_lb = Vec::with_capacity(16);

    for bit in 0..16 {
        let tt = chi4_truth_table(bit);
        // Find the minimum d ∈ [1, max_degree] where an annihilator of degree ≤ d exists.
        // Annihilators of degree ≤ d include those of all lower degrees, so once TRUE
        // for degree d₀, it stays TRUE for all d ≥ d₀.
        // If no annihilator found, return max_degree+1 (AI is strictly greater).
        let ai_estimate = (1..=max_degree)
            .find(|&d| has_annihilator_deg(&tt, d))
            .unwrap_or(max_degree + 1);
        per_bit_lb.push(ai_estimate);
    }

    let min_lb = *per_bit_lb.iter().min().unwrap_or(&0);
    let max_lb = *per_bit_lb.iter().max().unwrap_or(&0);
    let high_ai_bit_count = per_bit_lb.iter().filter(|&&lb| lb > max_degree).count();

    AiStats {
        per_bit_lb,
        min_lb,
        max_lb,
        theoretical_upper_bound: 8,
        high_ai_bit_count,
    }
}

// ── Invariant subspace detection ───────────────────────────────────────────

pub struct InvariantStats {
    /// Total inputs tested (65536 for 16-bit chi4).
    pub total_inputs: usize,
    /// Inputs x with chi4(x) = x.
    pub fixed_point_count: usize,
    /// Inputs x with chi4(chi4(x)) = x and chi4(x) ≠ x (each pair counted once).
    pub two_cycle_count: usize,
    /// Inputs x with chi4(x) = 0.
    pub maps_to_zero_count: usize,
    /// Whether the fixed-point set is closed under XOR (i.e., a GF(2)-linear subspace).
    /// TRUE would indicate a cryptographically dangerous linear invariant subspace.
    pub fixed_points_form_linear_subspace: bool,
}

/// Scan all 65536 inputs of chi4 for fixed points, 2-cycles, and maps-to-zero.
/// Also checks whether the fixed-point set is a GF(2)-linear subspace.
pub fn detect_invariant_subspaces() -> InvariantStats {
    const SIZE: usize = 1 << 16;
    let mut fixed_pts: Vec<u16> = Vec::new();
    let mut two_cycle = 0usize;
    let mut to_zero = 0usize;

    for x in 0u16..=0xFFFFu16 {
        let fx = chi4(x);
        if fx == x {
            fixed_pts.push(x);
        }
        // Count each 2-cycle (x, fx) once: require fx > x to avoid double-counting.
        if fx != x && chi4(fx) == x && fx > x {
            two_cycle += 1;
        }
        if fx == 0 {
            to_zero += 1;
        }
    }

    // A GF(2)-linear subspace must: (a) have power-of-2 cardinality, (b) contain 0,
    // (c) be closed under XOR.  Condition (a) is a fast necessary check.
    let fixed_count = fixed_pts.len();
    let is_linear = fixed_count.is_power_of_two()
        && fixed_pts.binary_search(&0).is_ok()
        && fixed_pts.iter().all(|&a| {
            fixed_pts.iter().all(|&b| fixed_pts.binary_search(&(a ^ b)).is_ok())
        });

    InvariantStats {
        total_inputs: SIZE,
        fixed_point_count: fixed_count,
        two_cycle_count: two_cycle,
        maps_to_zero_count: to_zero,
        fixed_points_form_linear_subspace: is_linear,
    }
}
