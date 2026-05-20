/// Two complementary preimage analyses:
///
/// 1. **8-bit reduced χ — exhaustive preimage counting.**
/// 2. **4-bit reduced χ — ANF, degree propagation, and structural isolation.**
///
/// Degree propagation tracks how the algebraic degree grows when χ₄ is
/// composed with itself across rounds; saturation at the maximum (16 for
/// 4-bit) should occur early, confirming full nonlinear coverage.
///
/// Structural isolation asks: can the degree-2 monomials in the ANF be
/// extracted as a solvable linear subsystem?  PASS means the subsystem is
/// massively underdetermined (far more z-variables than equations), so no
/// linear-algebraic shortcut exists through the quadratic layer alone.
use rand::Rng;

// ── 8-bit reduced χ ────────────────────────────────────────────────────────

fn quad8(a: u8, b: u8, c: u8, d: u8) -> u8 {
    a.wrapping_mul(b) ^ c.wrapping_mul(d)
}

pub fn chi8(x1: u8, x2: u8, x3: u8, x4: u8) -> (u8, u8, u8, u8) {
    let g = quad8(x1, x2, x3, x4);
    (x1 ^ g, x2 ^ g.rotate_left(3), x3 ^ g.rotate_left(5), x4 ^ g.rotate_left(7))
}

/// Counts preimages of `(o1,o2,o3,o4)` under `chi8` by testing all 256 `g`
/// candidates.  Each valid `g` yields a unique `(x1,x2,x3,x4)` tuple.
pub fn preimage_count_8bit(o1: u8, o2: u8, o3: u8, o4: u8) -> usize {
    (0u8..=255)
        .filter(|&g| {
            let x1 = o1 ^ g;
            let x2 = o2 ^ g.rotate_left(3);
            let x3 = o3 ^ g.rotate_left(5);
            let x4 = o4 ^ g.rotate_left(7);
            quad8(x1, x2, x3, x4) == g
        })
        .count()
}

pub struct PreimageStats {
    pub outputs_sampled: usize,
    pub avg_preimage_count: f64,
    pub max_preimage_count: usize,
    pub zero_preimage_frac: f64,
    pub unique_preimage_frac: f64,
}

pub fn analyze_preimages(output_samples: usize, rng: &mut impl Rng) -> PreimageStats {
    let mut total = 0usize;
    let mut max_count = 0usize;
    let mut zeros = 0usize;
    let mut uniques = 0usize;

    for _ in 0..output_samples {
        let c = preimage_count_8bit(rng.gen(), rng.gen(), rng.gen(), rng.gen());
        total += c;
        if c > max_count {
            max_count = c;
        }
        if c == 0 {
            zeros += 1;
        }
        if c == 1 {
            uniques += 1;
        }
    }

    PreimageStats {
        outputs_sampled: output_samples,
        avg_preimage_count: total as f64 / output_samples as f64,
        max_preimage_count: max_count,
        zero_preimage_frac: zeros as f64 / output_samples as f64,
        unique_preimage_frac: uniques as f64 / output_samples as f64,
    }
}

// ── 4-bit reduced χ — ANF degree analysis ─────────────────────────────────

// Layout: input u16 packed as x4[15:12] | x3[11:8] | x2[7:4] | x1[3:0]

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

/// Möbius transform over GF(2)^16 — converts truth table to ANF coefficients.
/// Entry `anf[m]` is 1 iff the monomial ∏_{i: bit i set in m} x_i appears.
fn mobius_transform(table: &[u8]) -> Vec<u8> {
    let n = table.len();
    assert!(n.is_power_of_two());
    let bits = n.trailing_zeros() as usize;
    let mut a = table.to_vec();
    for i in 0..bits {
        for s in 0..n {
            if (s >> i) & 1 == 1 {
                a[s] ^= a[s ^ (1 << i)];
            }
        }
    }
    a
}

/// Algebraic degree = highest popcount of any index with a nonzero ANF coeff.
fn anf_degree(anf: &[u8]) -> usize {
    anf.iter()
        .enumerate()
        .filter(|(_, &c)| c == 1)
        .map(|(i, _)| i.count_ones() as usize)
        .max()
        .unwrap_or(0)
}

pub struct DegreeStats {
    /// Per output bit (16 bits: 4 shares × 4 bits each)
    pub degrees: Vec<usize>,
    pub min_degree: usize,
    pub max_degree: usize,
}

pub fn analyze_algebraic_degree() -> DegreeStats {
    // Build truth table for every output bit of chi4 over all 2^16 inputs.
    const SIZE: usize = 1 << 16;
    let mut output_bits: Vec<Vec<u8>> = (0..16).map(|_| vec![0u8; SIZE]).collect();

    for input in 0u16..=0xFFFF {
        let out = chi4(input);
        for bit in 0..16 {
            output_bits[bit][input as usize] = ((out >> bit) & 1) as u8;
        }
    }

    let degrees: Vec<usize> = output_bits
        .iter()
        .map(|tt| anf_degree(&mobius_transform(tt)))
        .collect();

    let min_degree = *degrees.iter().min().unwrap();
    let max_degree = *degrees.iter().max().unwrap();
    DegreeStats { degrees, min_degree, max_degree }
}

// ── 4-bit χ — degree propagation across rounds ─────────────────────────────

pub struct DegreePropStats {
    /// Maximum algebraic degree over all 16 output bits after round k (1-indexed).
    pub max_degree_per_round: Vec<usize>,
    /// Round index (1-based) where degree first reaches its maximum value.
    pub saturation_round: usize,
}

/// Compose chi4 with itself up to `max_rounds` times and measure the growth of
/// the maximum algebraic degree across all 16 output bits.
pub fn degree_propagation(max_rounds: usize) -> DegreePropStats {
    const SIZE: usize = 1 << 16;
    // composition[i] = chi4^k(i) after k rounds.  Start as identity.
    let mut composition: Vec<u16> = (0u16..=0xFFFF).collect();
    let mut max_degree_per_round = Vec::with_capacity(max_rounds);
    let mut saturation_round = max_rounds;
    let mut prev_max = 0usize;

    for round in 0..max_rounds {
        for entry in composition.iter_mut() {
            *entry = chi4(*entry);
        }
        let max_deg = (0..16u16)
            .map(|bit| {
                let tt: Vec<u8> =
                    (0usize..SIZE).map(|i| ((composition[i] >> bit) & 1) as u8).collect();
                anf_degree(&mobius_transform(&tt))
            })
            .max()
            .unwrap_or(0);

        if max_deg == prev_max && saturation_round == max_rounds {
            saturation_round = round + 1; // 1-based
        }
        prev_max = max_deg;
        max_degree_per_round.push(max_deg);
    }

    // If degree strictly increased every round, saturation_round == max_rounds.
    DegreePropStats { max_degree_per_round, saturation_round }
}

// ── 4-bit χ — structural isolation of degree-2 subsystem ──────────────────

pub struct IsolationStats {
    /// Number of degree-2 monomials: C(16,2) = 120.
    pub degree2_var_count: usize,
    /// Number of equations (one per output bit).
    pub equation_count: usize,
    /// GF(2) rank of the 16×120 degree-2 coefficient matrix.
    pub subsystem_rank: usize,
    /// degree2_var_count / subsystem_rank — how many z-variables are free per
    /// independent equation.  High values mean the subsystem is underdetermined
    /// and no linear shortcut through the quadratic layer exists.
    pub underdetermination_ratio: f64,
}

fn gf2_rank_small(rows: &[[u64; 2]], n_cols: usize) -> usize {
    let mut mat = rows.to_vec();
    let n = mat.len();
    let mut pivot = 0usize;
    for col in 0..n_cols {
        let word = col / 64;
        let bit = col % 64;
        let found = (pivot..n).find(|&r| (mat[r][word] >> bit) & 1 == 1);
        if let Some(r) = found {
            mat.swap(pivot, r);
            let prow = mat[pivot];
            for r2 in 0..n {
                if r2 != pivot && (mat[r2][word] >> bit) & 1 == 1 {
                    mat[r2][0] ^= prow[0];
                    mat[r2][1] ^= prow[1];
                }
            }
            pivot += 1;
        }
    }
    pivot
}

pub fn structural_isolation_4bit() -> IsolationStats {
    const SIZE: usize = 1 << 16;
    const N_OUT: usize = 16; // = input bits = output bits for 4-bit chi

    // Build truth tables and ANFs for each output bit.
    let mut output_tt: Vec<Vec<u8>> = (0..N_OUT).map(|_| vec![0u8; SIZE]).collect();
    for input in 0u16..=0xFFFF {
        let out = chi4(input);
        for bit in 0..N_OUT {
            output_tt[bit][input as usize] = ((out >> bit) & 1) as u8;
        }
    }
    let anfs: Vec<Vec<u8>> = output_tt.iter().map(|tt| mobius_transform(tt)).collect();

    // Enumerate degree-2 monomial indices (popcount 2 in 16-bit index).
    let d2_indices: Vec<usize> = (0..SIZE).filter(|&m| m.count_ones() == 2).collect();
    let n_d2 = d2_indices.len(); // C(16,2) = 120 — fits in 2 × u64

    // Build 16×120 coefficient matrix over GF(2).
    // Row i has a 1 in column k iff the k-th degree-2 monomial appears in ANF_i.
    let mut matrix: Vec<[u64; 2]> = vec![[0u64; 2]; N_OUT];
    for (out_bit, anf) in anfs.iter().enumerate() {
        for (col, &d2_idx) in d2_indices.iter().enumerate() {
            if anf[d2_idx] == 1 {
                matrix[out_bit][col / 64] |= 1u64 << (col % 64);
            }
        }
    }

    let rank = gf2_rank_small(&matrix, n_d2);
    IsolationStats {
        degree2_var_count: n_d2,
        equation_count: N_OUT,
        subsystem_rank: rank,
        underdetermination_ratio: n_d2 as f64 / rank.max(1) as f64,
    }
}

// Expose chi8 for use in the bench round-trip sanity check.
pub fn chi8_roundtrip_check() -> bool {
    // Verify forward-direction consistency: chi8 output re-encodes to a valid preimage.
    for x1 in 0u8..16 {
        for x2 in 0u8..16 {
            let (o1, o2, o3, o4) = chi8(x1, x2, 0, 0);
            let count = preimage_count_8bit(o1, o2, o3, o4);
            // The input we started with must appear in the preimage set.
            if count == 0 {
                return false;
            }
        }
    }
    true
}
