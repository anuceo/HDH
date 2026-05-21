/// GF(2) Jacobian rank of the full 64-bit χ lane.
///
/// The Jacobian J of a vector Boolean function f: GF(2)^n → GF(2)^m at a
/// point x is the m×n matrix where J[i,j] = f_i(x⊕eⱼ) ⊕ f_i(x).  For
/// chi_lane this is a 256×256 matrix over GF(2).
///
/// rank(J) ≈ n means the local linearization covers all output dimensions —
/// no solvable linear shortcut exists at that point.
/// rank(J) << n means an attacker could exploit the rank deficiency to recover
/// inputs from outputs with linear-algebraic work.
use crate::algorithm::chi::chi_lane;
use rand::Rng;

/// 256-bit row (4 × u64 words).
type Row256 = [u64; 4];

fn bit_is_set(row: &Row256, col: usize) -> bool {
    (row[col / 64] >> (col % 64)) & 1 == 1
}

fn gf2_rank_256(mut rows: Vec<Row256>) -> usize {
    let n = rows.len();
    let mut pivot = 0usize;
    for col in 0..256 {
        let found = (pivot..n).find(|&r| bit_is_set(&rows[r], col));
        if let Some(r) = found {
            rows.swap(pivot, r);
            let prow: Row256 = rows[pivot]; // Copy (Row256 is Copy)
            for r2 in 0..n {
                if r2 != pivot && bit_is_set(&rows[r2], col) {
                    rows[r2][0] ^= prow[0];
                    rows[r2][1] ^= prow[1];
                    rows[r2][2] ^= prow[2];
                    rows[r2][3] ^= prow[3];
                }
            }
            pivot += 1;
        }
    }
    pivot
}

/// Compute rank(J_chi(x)) for a specific input point.
/// Requires 257 evaluations of chi_lane.
pub fn jacobian_rank_at_point(x1: u64, x2: u64, x3: u64, x4: u64) -> usize {
    let (b0, b1, b2, b3) = chi_lane(x1, x2, x3, x4);

    let rows: Vec<Row256> = (0..256)
        .map(|bit| {
            let mask = 1u64 << (bit % 64);
            let (nx1, nx2, nx3, nx4) = match bit / 64 {
                0 => (x1 ^ mask, x2, x3, x4),
                1 => (x1, x2 ^ mask, x3, x4),
                2 => (x1, x2, x3 ^ mask, x4),
                _ => (x1, x2, x3, x4 ^ mask),
            };
            let (f0, f1, f2, f3) = chi_lane(nx1, nx2, nx3, nx4);
            [b0 ^ f0, b1 ^ f1, b2 ^ f2, b3 ^ f3]
        })
        .collect();

    gf2_rank_256(rows)
}

pub struct JacobianStats {
    pub points_tested: usize,
    pub min_rank: usize,
    pub avg_rank: f64,
    pub max_theoretical_rank: usize,
    /// max_theoretical_rank − min_rank.  Zero = full rank at every tested point.
    pub worst_rank_deficit: usize,
}

pub fn analyze_jacobian_rank(points: usize, rng: &mut impl Rng) -> JacobianStats {
    let ranks: Vec<usize> = (0..points)
        .map(|_| jacobian_rank_at_point(rng.gen(), rng.gen(), rng.gen(), rng.gen()))
        .collect();

    let min_rank = *ranks.iter().min().unwrap();
    let avg_rank = ranks.iter().sum::<usize>() as f64 / ranks.len() as f64;

    JacobianStats {
        points_tested: points,
        min_rank,
        avg_rank,
        max_theoretical_rank: 256,
        worst_rank_deficit: 256 - min_rank,
    }
}
