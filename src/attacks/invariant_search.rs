/// Hidden invariant search for chi4 using exhaustive enumeration and GF(2) algebra.
///
/// A degree-d invariant is a multivariate polynomial f: GF(2)^16 → GF(2) of degree ≤ d
/// such that f(x) = f(chi4(x)) for all x ∈ {0,1}^16.
///
/// For degree-1 (affine): exhaustively test all 65535 nonzero linear masks.
/// For degree-2 (quadratic): test all 136 monomials individually, then compute
///   null space dimension via GF(2) Gaussian elimination on the 136 × 65536 matrix.

use rand::Rng;
use crate::attacks::distinguisher::random_state;
use crate::chi::chi_lane;

// ── Internal chi4 ─────────────────────────────────────────────────────────────

/// Chi4: 4×4-bit chi mapping on a 16-bit packed state (same as orbit.rs).
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

// ── Structs ───────────────────────────────────────────────────────────────────

pub struct InvariantSearchResult {
    pub degree: usize,
    pub space_bits: usize,
    pub monomials_tested: usize,
    pub trivial_invariants: usize,
    pub nontrivial_found: usize,
    pub null_space_dim: usize,
}

// ── Affine invariant search ───────────────────────────────────────────────────

/// Search for degree-1 (affine/linear) invariants of chi4.
///
/// A nonzero v ∈ GF(2)^16 is a linear invariant iff:
///   for ALL x ∈ {0..65535}: inner_product(v, x) = inner_product(v, chi4(x))
/// i.e., v·x = v·chi4(x) for every x — not just on average.
///
/// Equivalently: for ALL x, parity(v & x) = parity(v & chi4(x))
/// i.e., parity(v & (x XOR chi4(x))) = 0 for every single x individually.
///
/// This test checks all 65535 nonzero v values.
pub fn search_affine_invariants_chi4() -> InvariantSearchResult {
    const N: usize = 65536;
    const WORDS: usize = N / 64; // 1024 u64s

    // Precompute perm table
    let perm: Vec<u16> = (0u16..=u16::MAX).map(|x| chi4(x)).collect();

    // For each bit position i (0..16), build a 65536-bit column vector
    // col_i[x] = 1 iff bit i of (x XOR chi4(x)) is 1
    let mut cols: Vec<Vec<u64>> = vec![vec![0u64; WORDS]; 16];
    for x in 0u32..N as u32 {
        let diff = (x as u16) ^ perm[x as usize];
        for i in 0..16 {
            if (diff >> i) & 1 == 1 {
                cols[i][x as usize / 64] ^= 1u64 << (x % 64);
            }
        }
    }

    // A mask v is a linear invariant iff XOR of cols[i] for all set bits i in v = zero vector
    // Use Gray code traversal: v and v+1 differ by exactly one bit (Gray code step).
    // Gray code: G(k) = k XOR (k >> 1)
    // Maintain combined = XOR of cols[i] for set bits in current Gray code value.
    // Transition from G(k) to G(k+1): find the changed bit and XOR in/out that column.
    let mut nontrivial_found = 0usize;

    let mut combined = vec![0u64; WORDS];
    let mut gray_prev = 0u32;

    for k in 1u32..=65535u32 {
        let gray_curr = k ^ (k >> 1);
        let changed_bit = (gray_prev ^ gray_curr).trailing_zeros() as usize;
        // XOR in/out the changed column
        for w in 0..WORDS {
            combined[w] ^= cols[changed_bit][w];
        }
        if combined.iter().all(|&w| w == 0) {
            nontrivial_found += 1;
        }
        gray_prev = gray_curr;
    }

    // The null space dimension
    let null_space_dim = if nontrivial_found == 0 {
        0
    } else {
        let count = nontrivial_found + 1; // including zero vector
        let mut dim = 0;
        let mut c = count;
        while c > 1 {
            c >>= 1;
            dim += 1;
        }
        dim
    };

    InvariantSearchResult {
        degree: 1,
        space_bits: 16,
        monomials_tested: 65535,
        trivial_invariants: 0,
        nontrivial_found,
        null_space_dim,
    }
}

// ── Quadratic invariant search ────────────────────────────────────────────────

/// Search for degree-2 (quadratic) invariants of chi4.
///
/// Enumerates all 136 monomials (16 linear + 120 quadratic).
/// For each monomial m_k, checks: m_k(x) == m_k(chi4(x)) for all x.
/// Also computes null space dimension of the 136 × 65536 indicator matrix via
/// GF(2) Gaussian elimination.
pub fn search_quadratic_invariants_chi4() -> InvariantSearchResult {
    const N: usize = 65536;
    const WORDS: usize = N / 64; // 1024 u64s per column

    // Precompute perm table
    let perm: Vec<u16> = (0u16..=u16::MAX).map(|x| chi4(x)).collect();

    // Build monomials: 16 linear (bit i) + 120 quadratic (bit i * bit j, i<j)
    // Total: 136 monomials

    // For each monomial k, compute col_k[x] = m_k(x) XOR m_k(perm[x])
    // Store as a bitvector of 65536 bits = 1024 u64s

    // Total storage: 136 × 1024 u64s = 139264 u64s ≈ 1 MB
    let mut matrix: Vec<Vec<u64>> = Vec::with_capacity(136);

    // Helper: extract bit b from u16 value
    let bit = |x: u16, b: usize| -> u64 { ((x >> b) & 1) as u64 };

    // Linear monomials: x_i for i in 0..16
    for i in 0..16 {
        let mut col = vec![0u64; WORDS];
        for x in 0u16..=u16::MAX {
            let mx = bit(x, i);
            let mpx = bit(perm[x as usize], i);
            let diff = mx ^ mpx;
            if diff == 1 {
                let idx = x as usize;
                col[idx / 64] ^= 1u64 << (idx % 64);
            }
        }
        matrix.push(col);
    }

    // Quadratic monomials: x_i * x_j for i < j
    for i in 0..16 {
        for j in (i + 1)..16 {
            let mut col = vec![0u64; WORDS];
            for x in 0u16..=u16::MAX {
                let mx = bit(x, i) & bit(x, j);
                let mpx = bit(perm[x as usize], i) & bit(perm[x as usize], j);
                let diff = mx ^ mpx;
                if diff == 1 {
                    let idx = x as usize;
                    col[idx / 64] ^= 1u64 << (idx % 64);
                }
            }
            matrix.push(col);
        }
    }

    let num_monomials = matrix.len(); // should be 136

    // Count individual monomial invariants (all-zero columns = monomial is invariant)
    let mut individual_invariants = 0usize;
    for col in &matrix {
        if col.iter().all(|&w| w == 0) {
            individual_invariants += 1;
        }
    }

    // GF(2) Gaussian elimination to find null space dimension
    // We have num_monomials rows, each of width WORDS u64s
    // We want to find the rank (pivot count), then null_space_dim = num_monomials - rank
    let mut rank = 0usize;

    // Work on a mutable copy
    let mut mat = matrix.clone();

    // We do column-pivoted Gaussian elimination:
    // For each row i (monomial), find a pivot bit position in the 65536-bit vector
    // Actually: standard row reduction — find first nonzero bit in current row and use as pivot

    // We iterate over "columns" of the 65536-bit space as potential pivots
    // But with 65536 columns that's expensive. Instead:
    // Row-reduce the num_monomials × 65536 bit matrix.
    // For each of the 136 rows, find the leading nonzero bit position.

    let mut used_pivot_bits: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for i in 0..num_monomials {
        // Find first nonzero word in mat[pivot_row..] that isn't already used
        // Actually: find the first nonzero bit in row i
        let mut found_pivot = false;
        let mut pivot_bit = 0usize;

        'outer: for word_idx in 0..WORDS {
            let w = mat[i][word_idx];
            if w != 0 {
                for bit_idx in 0..64 {
                    let candidate = word_idx * 64 + bit_idx;
                    if (w >> bit_idx) & 1 == 1 && !used_pivot_bits.contains(&candidate) {
                        pivot_bit = candidate;
                        found_pivot = true;
                        break 'outer;
                    }
                }
            }
        }

        if !found_pivot {
            continue; // row is zero or dependent
        }

        used_pivot_bits.insert(pivot_bit);
        rank += 1;

        // Eliminate this pivot from all other rows
        let pivot_word = pivot_bit / 64;
        let pivot_offset = pivot_bit % 64;

        for j in 0..num_monomials {
            if j == i {
                continue;
            }
            if (mat[j][pivot_word] >> pivot_offset) & 1 == 1 {
                // XOR row i into row j
                let row_i = mat[i].clone();
                for w in 0..WORDS {
                    mat[j][w] ^= row_i[w];
                }
            }
        }
    }

    let null_space_dim = num_monomials - rank;

    // nontrivial_found: null space vectors that aren't constant (i.e. null_space_dim minus
    // contribution from the constant-function invariant if it exists).
    // The constant function f(x)=0 is always trivial; if f(x)=1 is invariant, count it.
    // For simplicity: nontrivial = null_space_dim (each null vector = degree-2 polynomial invariant)
    // subtract 1 if the all-zero monomial combination (trivial) is counted.
    // The null space only contains non-trivial combinations (the zero vector is always in null space
    // but represents f=0 which is trivial). null_space_dim counts the dimensionality.
    // Non-trivial: any null vector with at least one nonzero coefficient.
    // We report null_space_dim as the number of independent non-trivial invariants.
    let nontrivial_found = if null_space_dim > 0 {
        // Check if any column is genuinely all-zero (meaning a lone monomial is an invariant)
        individual_invariants
    } else {
        0
    };

    InvariantSearchResult {
        degree: 2,
        space_bits: 16,
        monomials_tested: num_monomials,
        trivial_invariants: individual_invariants,
        nontrivial_found,
        null_space_dim,
    }
}

// ── Sampled linear invariant search for chi_lane ──────────────────────────────

/// Sample random linear masks v ∈ GF(2)^256 and check if v·x = v·chi_lane(x) empirically.
///
/// Returns the maximum observed bias (expected ≈ 0 if no linear invariants).
pub fn search_linear_invariants_chi_lane_sampled(samples: usize, rng: &mut impl Rng) -> f64 {
    let num_masks = samples.min(10_000);
    let eval_samples = 1000usize;
    let mut max_bias = 0.0f64;

    for _ in 0..num_masks {
        // Random 256-bit mask v = (v1, v2, v3, v4)
        let v1: u64 = rng.gen();
        let v2: u64 = rng.gen();
        let v3: u64 = rng.gen();
        let v4: u64 = rng.gen();

        let mut agree = 0usize;

        for _ in 0..eval_samples {
            let s = random_state(rng);
            let x1 = s.s1[0];
            let x2 = s.s2[0];
            let x3 = s.s3[0];
            let x4 = s.s4[0];

            let (o1, o2, o3, o4) = chi_lane(x1, x2, x3, x4);

            // v · x
            let px = ((v1 & x1).count_ones()
                + (v2 & x2).count_ones()
                + (v3 & x3).count_ones()
                + (v4 & x4).count_ones()) & 1;

            // v · chi_lane(x)
            let po = ((v1 & o1).count_ones()
                + (v2 & o2).count_ones()
                + (v3 & o3).count_ones()
                + (v4 & o4).count_ones()) & 1;

            if px == po {
                agree += 1;
            }
        }

        let bias = (agree as f64 / eval_samples as f64 - 0.5).abs();
        if bias > max_bias {
            max_bias = bias;
        }
    }

    max_bias
}
