/// Structured meet-in-the-middle analysis for HDH.
///
/// Five independent categories probe whether 2-round HDH maintains
/// exploitable forward/backward separability after the degree-inflection
/// transition identified by the integral distinguisher analysis.
///
/// ## Design context
///
/// Round order: entropy-inject → χ → θ → Φ
///
/// - **χ**: fully lane-local; output at lane i depends only on s1..s4 at lane i.
/// - **θ**: ring XOR; each share output at lane i = input_i ⊕ input_{i-1} ⊕ input_{i+1}.
/// - **Φ**: state-dependent routing: j = (s1[i]⊕s2[i]) % 25, k = (s3[i]⊕s4[i]) % 25;
///   out.s1[i] ^= s2[j], out.s2[i] ^= s3[k], out.s3[i] ^= s4[j], out.s4[i] ^= s1[k].
///
/// After 1 round:
///   Lane i output depends on lanes {i-1, i, i+1} (via χ+θ for the routing index)
///   plus lanes {j-1, j, j+1} and {k-1, k, k+1} (for the fetched values).
///   On average ~6 out of 25 lanes affect each output lane.  Lane isolation persists.
///
/// After 2 rounds: the 6-lane footprint expands to cover all 25 lanes.
///
/// ## Expected results (healthy behavior)
///
/// - Cat 1: 1-round shows moderate collision excess; 2-round drops to near-zero.
/// - Cat 2: 1-round high isolation ratio (same-lane >> cross-lane influence);
///           2-round isolation ratio → 1.0 (uniform dependence).
/// - Cat 3: Φ near-full linear rank (≥ 0.9 × n_bits); no low-rank factoring.
/// - Cat 4: 1-round biclique excess visible; 2-round within random noise.
/// - Cat 5: entropy collapse rate equals random expectation at 2 rounds.

use crate::attacks::distinguisher::{apply_rounds, random_state, STATE_BITS};
use crate::algorithm::chi::phi;
use crate::algorithm::state::State;
use rand::Rng;
use std::collections::HashMap;

// ── Local helpers ────────────────────────────────────────────────────────────

fn flip_bit(s: &mut State, idx: usize) {
    let (share, rem) = (idx / 1600, idx % 1600);
    let (lane, bp) = (rem / 64, rem % 64);
    let mask = 1u64 << bp;
    match share {
        0 => s.s1[lane] ^= mask,
        1 => s.s2[lane] ^= mask,
        2 => s.s3[lane] ^= mask,
        _ => s.s4[lane] ^= mask,
    }
    s.parity[lane] = s.s1[lane] ^ s.s2[lane] ^ s.s3[lane] ^ s.s4[lane];
}

fn get_bit(s: &State, idx: usize) -> u8 {
    let (share, rem) = (idx / 1600, idx % 1600);
    let (lane, bp) = (rem / 64, rem % 64);
    let w = match share { 0 => s.s1[lane], 1 => s.s2[lane], 2 => s.s3[lane], _ => s.s4[lane] };
    ((w >> bp) & 1) as u8
}

/// Lane index (0..25) of a STATE_BITS position.
fn lane_of(idx: usize) -> usize { (idx % 1600) / 64 }

/// Project a state to `k` specific bit positions, packed into a u64.
fn project(s: &State, positions: &[usize]) -> u64 {
    debug_assert!(positions.len() <= 64);
    positions.iter().enumerate().fold(0u64, |acc, (bit, &pos)| {
        acc | ((get_bit(s, pos) as u64) << bit)
    })
}

/// GF(2) rank of an n×n binary matrix stored as n rows of u64 (low n_cols bits).
fn gf2_rank(mut rows: Vec<u64>, n_cols: usize) -> usize {
    let n_rows = rows.len();
    let mut rank = 0usize;
    for col in 0..n_cols {
        if let Some(pivot) = (rank..n_rows).find(|&r| (rows[r] >> col) & 1 == 1) {
            rows.swap(rank, pivot);
            let pv = rows[rank];
            for r in 0..n_rows {
                if r != rank && (rows[r] >> col) & 1 == 1 {
                    rows[r] ^= pv;
                }
            }
            rank += 1;
        }
    }
    rank
}

// ── Category 1: Forward/backward partition matching ──────────────────────────

pub struct PartitionMatchStats {
    pub rounds: usize,
    pub samples: usize,
    pub projection_bits: usize,
    /// Observed collisions in the k-bit projected output.
    pub actual_collisions: usize,
    /// Birthday expectation: samples*(samples-1) / 2^(k+1).
    pub expected_collisions: f64,
    /// log2(actual+1 / expected+1); > 1.0 = excess collisions → matching surface.
    pub log2_excess: f64,
}

/// Project `rounds`-round outputs to `k_bits` random positions; count
/// actual vs. birthday-expected collisions.  Excess indicates exploitable
/// intermediate state compression.
pub fn measure_partition_matching(
    rounds: usize,
    samples: usize,
    k_bits: usize,
    rng: &mut impl Rng,
) -> PartitionMatchStats {
    assert!(k_bits <= 32);
    let proj_positions: Vec<usize> = (0..k_bits).map(|_| rng.gen_range(0..STATE_BITS)).collect();

    let mut counts: HashMap<u64, usize> = HashMap::new();
    for _ in 0..samples {
        let s = random_state(rng);
        let out = apply_rounds(s, rounds);
        *counts.entry(project(&out, &proj_positions)).or_insert(0) += 1;
    }

    let actual_collisions: usize = counts.values().map(|&c| c * c.saturating_sub(1) / 2).sum();
    let expected = samples as f64 * (samples - 1) as f64 / (2u64.pow(k_bits as u32 + 1) as f64);
    let log2_excess = ((actual_collisions + 1) as f64).log2() - (expected + 1.0).log2();

    PartitionMatchStats { rounds, samples, projection_bits: k_bits, actual_collisions, expected_collisions: expected, log2_excess }
}

// ── Category 2: χ dependency graph / lane isolation ──────────────────────────

pub struct DependencyStats {
    pub rounds: usize,
    pub pairs_tested: usize,
    pub influence_samples: usize,
    /// Pr[flip random input bit changes random output bit].
    pub avg_influence_prob: f64,
    /// Fraction of (in, out) pairs with zero observed influence (strong isolation).
    pub zero_influence_fraction: f64,
    /// Pr[influence > 0] for pairs where input and output share the same lane.
    pub same_lane_influence_frac: f64,
    /// Pr[influence > 0] for pairs where |lane_in - lane_out| > 1 (cross-lane).
    pub cross_lane_influence_frac: f64,
    /// same_lane / cross_lane; >> 1 → lane isolation; ≈ 1 → full diffusion.
    pub isolation_ratio: f64,
}

/// For `n_pairs` random (input_bit, output_bit) pairs, estimate influence
/// probability over `influence_samples` random states.
pub fn analyze_dependency_graph(
    rounds: usize,
    n_pairs: usize,
    influence_samples: usize,
    rng: &mut impl Rng,
) -> DependencyStats {
    let mut total_influence = 0u64;
    let mut zero_count = 0usize;
    let mut same_lane_hit = 0usize; let mut same_lane_total = 0usize;
    let mut cross_lane_hit = 0usize; let mut cross_lane_total = 0usize;

    for _ in 0..n_pairs {
        let in_bit = rng.gen_range(0..STATE_BITS);
        let out_bit = rng.gen_range(0..STATE_BITS);
        let same_lane = lane_of(in_bit) == lane_of(out_bit);

        let mut hits = 0u64;
        for _ in 0..influence_samples {
            let base = random_state(rng);
            let mut flipped = base.clone();
            flip_bit(&mut flipped, in_bit);
            let ob = apply_rounds(base, rounds);
            let of = apply_rounds(flipped, rounds);
            if get_bit(&ob, out_bit) != get_bit(&of, out_bit) {
                hits += 1;
            }
        }

        total_influence += hits;
        if hits == 0 { zero_count += 1; }

        // Lane distance > 1 (ignoring wrap) = true cross-lane
        let dist = (lane_of(in_bit) as isize - lane_of(out_bit) as isize).unsigned_abs();
        let is_cross = dist > 1 && dist < 24; // exclude wrap-around neighbors

        if same_lane { same_lane_total += 1; if hits > 0 { same_lane_hit += 1; } }
        else if is_cross { cross_lane_total += 1; if hits > 0 { cross_lane_hit += 1; } }
    }

    let avg_inf = total_influence as f64 / (n_pairs * influence_samples) as f64;
    let zero_frac = zero_count as f64 / n_pairs as f64;
    let same_frac = if same_lane_total > 0 { same_lane_hit as f64 / same_lane_total as f64 } else { 0.0 };
    let cross_frac = if cross_lane_total > 0 { cross_lane_hit as f64 / cross_lane_total as f64 } else { 0.0 };
    let iso_ratio = if cross_frac > 0.0 { same_frac / cross_frac } else { f64::INFINITY };

    DependencyStats { rounds, pairs_tested: n_pairs, influence_samples, avg_influence_prob: avg_inf,
        zero_influence_fraction: zero_frac, same_lane_influence_frac: same_frac,
        cross_lane_influence_frac: cross_frac, isolation_ratio: iso_ratio }
}

// ── Category 3: Φ and 1-round linear rank (influence matrix rank) ─────────────

pub struct LinearRankStats {
    pub n_bits: usize,
    pub influence_samples: usize,
    /// GF(2) rank of Φ's n_bits × n_bits influence matrix (threshold: any hit).
    pub phi_rank: usize,
    /// GF(2) rank of 1-round influence matrix.
    pub one_round_rank: usize,
    /// GF(2) rank of 2-round influence matrix.
    pub two_round_rank: usize,
    pub max_rank: usize,
    pub phi_rank_fraction: f64,
    pub one_round_rank_fraction: f64,
    pub two_round_rank_fraction: f64,
}

/// Estimate GF(2) rank of the n×n influence matrix for Φ, 1-round, and 2-round
/// functions, using `influence_samples` states per cell.
pub fn measure_influence_rank(
    n_bits: usize,
    influence_samples: usize,
    rng: &mut impl Rng,
) -> LinearRankStats {
    assert!(n_bits <= 64, "n_bits must be ≤ 64");

    let in_bits: Vec<usize> = (0..n_bits).map(|_| rng.gen_range(0..STATE_BITS)).collect();
    let out_bits: Vec<usize> = (0..n_bits).map(|_| rng.gen_range(0..STATE_BITS)).collect();

    // Build influence matrices for Φ alone, 1-round, and 2-round.
    // mat_fn[row] = bitmask over in_bits for out_bits[row].
    //
    // Threshold: entry (out_i, in_j) = 1 iff observed influence > HALF the
    // samples.  This avoids the all-ones trap: a fully-connected function has
    // true influence prob ≈ 0.5, giving ~50% ones per row → near-full rank.
    // A sparse function (like Φ alone) has many zero-influence pairs → few ones
    // per row → lower rank.  Uses strictly-majority vote (hits*2 > samples).
    let mut build_mat = |apply: &dyn Fn(&State) -> State| -> Vec<u64> {
        (0..n_bits).map(|oi| {
            let out_pos = out_bits[oi];
            let mut row = 0u64;
            for (ii, &in_pos) in in_bits.iter().enumerate() {
                let mut hits = 0u32;
                for _ in 0..influence_samples {
                    let base = random_state(rng);
                    let mut fl = base.clone();
                    flip_bit(&mut fl, in_pos);
                    let b_out = apply(&base);
                    let f_out = apply(&fl);
                    if get_bit(&b_out, out_pos) != get_bit(&f_out, out_pos) { hits += 1; }
                }
                // Majority-vote threshold: entry = 1 iff influence observed > half the samples.
                if hits * 2 > influence_samples as u32 { row |= 1u64 << ii; }
            }
            row
        }).collect()
    };

    // Phi alone (state → phi(state), no chi/theta).
    let phi_mat = build_mat(&|s: &State| phi(s));
    // 1-round and 2-round full permutation.
    let r1_mat = build_mat(&|s: &State| apply_rounds(s.clone(), 1));
    let r2_mat = build_mat(&|s: &State| apply_rounds(s.clone(), 2));

    let phi_rank = gf2_rank(phi_mat, n_bits);
    let one_round_rank = gf2_rank(r1_mat, n_bits);
    let two_round_rank = gf2_rank(r2_mat, n_bits);

    LinearRankStats {
        n_bits, influence_samples,
        phi_rank, one_round_rank, two_round_rank,
        max_rank: n_bits,
        phi_rank_fraction: phi_rank as f64 / n_bits as f64,
        one_round_rank_fraction: one_round_rank as f64 / n_bits as f64,
        two_round_rank_fraction: two_round_rank as f64 / n_bits as f64,
    }
}

// ── Category 4: Biclique-style partial matching ───────────────────────────────

pub struct BiCliqueStats {
    /// Cube dimension: 2^cube_dim input variants per base.
    pub cube_dim: usize,
    /// Number of output bits matched on per pair.
    pub target_bits: usize,
    pub bases_tested: usize,
    /// Mean fraction of cube-pair (i,j) combinations that share all target bits in 1-round output.
    pub mean_matching_pair_fraction: f64,
    /// Expected fraction for a random function: 1 / 2^target_bits.
    pub expected_pair_fraction: f64,
    /// log2(mean_match_frac / expected_frac); near 0 = random, > 0 = biclique excess.
    pub log2_excess: f64,
    /// Fraction of tested bases where at least one matching pair was found.
    pub any_match_fraction: f64,
}

/// For each of `n_bases` random base states, generate 2^`cube_dim` variants by
/// varying `cube_dim` random input bit positions; apply 1 round; count pairs
/// sharing all `target_bits` projected output bits.
pub fn test_biclique_matching(
    cube_dim: usize,
    target_bits: usize,
    n_bases: usize,
    rng: &mut impl Rng,
) -> BiCliqueStats {
    assert!(cube_dim <= 16 && target_bits <= 32);
    let cube_size = 1usize << cube_dim;
    let expected_pair_fraction = 2f64.powi(-(target_bits as i32));

    let mut total_match_frac = 0.0f64;
    let mut any_match_count = 0usize;

    for _ in 0..n_bases {
        let base = random_state(rng);
        let proj: Vec<usize> = (0..target_bits).map(|_| rng.gen_range(0..STATE_BITS)).collect();
        let mut cube_positions: Vec<usize> = Vec::with_capacity(cube_dim);
        while cube_positions.len() < cube_dim {
            let p = rng.gen_range(0..STATE_BITS);
            if !cube_positions.contains(&p) { cube_positions.push(p); }
        }

        // Compute 2^cube_dim projected outputs.
        let projected: Vec<u64> = (0..cube_size).map(|mask| {
            let mut s = base.clone();
            for (bi, &pos) in cube_positions.iter().enumerate() {
                if (mask >> bi) & 1 == 1 { flip_bit(&mut s, pos); }
            }
            project(&apply_rounds(s, 1), &proj)
        }).collect();

        // Count pairs with identical projection (collisions = matching pairs).
        let mut freq: HashMap<u64, usize> = HashMap::new();
        for &p in &projected { *freq.entry(p).or_insert(0) += 1; }
        let matching_pairs: usize = freq.values().map(|&c| c * c.saturating_sub(1) / 2).sum();
        let total_pairs = cube_size * (cube_size - 1) / 2;
        let frac = matching_pairs as f64 / total_pairs as f64;
        total_match_frac += frac;
        if matching_pairs > 0 { any_match_count += 1; }
    }

    let mean_frac = total_match_frac / n_bases as f64;
    let log2_excess = (mean_frac + 1e-30).log2() - expected_pair_fraction.log2();

    BiCliqueStats {
        cube_dim, target_bits, bases_tested: n_bases,
        mean_matching_pair_fraction: mean_frac,
        expected_pair_fraction,
        log2_excess,
        any_match_fraction: any_match_count as f64 / n_bases as f64,
    }
}

/// Same as `test_biclique_matching` but applies `rounds` rounds instead of 1.
pub fn test_biclique_matching_rounds(
    cube_dim: usize,
    target_bits: usize,
    n_bases: usize,
    rounds: usize,
    rng: &mut impl Rng,
) -> BiCliqueStats {
    assert!(cube_dim <= 16 && target_bits <= 32);
    let cube_size = 1usize << cube_dim;
    let expected_pair_fraction = 2f64.powi(-(target_bits as i32));

    let mut total_match_frac = 0.0f64;
    let mut any_match_count = 0usize;

    for _ in 0..n_bases {
        let base = random_state(rng);
        let proj: Vec<usize> = (0..target_bits).map(|_| rng.gen_range(0..STATE_BITS)).collect();
        let mut cube_positions: Vec<usize> = Vec::with_capacity(cube_dim);
        while cube_positions.len() < cube_dim {
            let p = rng.gen_range(0..STATE_BITS);
            if !cube_positions.contains(&p) { cube_positions.push(p); }
        }

        let projected: Vec<u64> = (0..cube_size).map(|mask| {
            let mut s = base.clone();
            for (bi, &pos) in cube_positions.iter().enumerate() {
                if (mask >> bi) & 1 == 1 { flip_bit(&mut s, pos); }
            }
            project(&apply_rounds(s, rounds), &proj)
        }).collect();

        let mut freq: HashMap<u64, usize> = HashMap::new();
        for &p in &projected { *freq.entry(p).or_insert(0) += 1; }
        let matching_pairs: usize = freq.values().map(|&c| c * c.saturating_sub(1) / 2).sum();
        let total_pairs = cube_size * (cube_size - 1) / 2;
        let frac = matching_pairs as f64 / total_pairs as f64;
        total_match_frac += frac;
        if matching_pairs > 0 { any_match_count += 1; }
    }

    let mean_frac = total_match_frac / n_bases as f64;
    let log2_excess = (mean_frac + 1e-30).log2() - expected_pair_fraction.log2();

    BiCliqueStats {
        cube_dim, target_bits, bases_tested: n_bases,
        mean_matching_pair_fraction: mean_frac,
        expected_pair_fraction,
        log2_excess,
        any_match_fraction: any_match_count as f64 / n_bases as f64,
    }
}

// ── Category 5: Entropy-surface collapse ─────────────────────────────────────

pub struct EntropyCollapseStats {
    pub rounds: usize,
    pub samples: usize,
    pub projection_bits: usize,
    pub collision_count: usize,
    pub expected_collisions: f64,
    /// max bucket count / (samples / 2^k); 1.0 = perfectly uniform.
    pub uniformity_ratio: f64,
    /// -log2(max_bucket_count / samples).
    pub min_entropy_bits: f64,
    /// Ideal: projection_bits (= log2(2^k) = k).
    pub ideal_entropy_bits: f64,
}

/// Project `rounds`-round outputs to `k_bits` random positions; measure
/// entropy collapse via collision rate and distribution uniformity.
pub fn measure_entropy_collapse(
    rounds: usize,
    samples: usize,
    k_bits: usize,
    rng: &mut impl Rng,
) -> EntropyCollapseStats {
    assert!(k_bits <= 32);
    let proj: Vec<usize> = (0..k_bits).map(|_| rng.gen_range(0..STATE_BITS)).collect();

    let mut freq: HashMap<u64, usize> = HashMap::new();
    for _ in 0..samples {
        let s = random_state(rng);
        let out = apply_rounds(s, rounds);
        *freq.entry(project(&out, &proj)).or_insert(0) += 1;
    }

    let collisions: usize = freq.values().map(|&c| c * c.saturating_sub(1) / 2).sum();
    let expected = samples as f64 * (samples - 1) as f64 / (2u64.pow(k_bits as u32 + 1) as f64);
    let max_count = *freq.values().max().unwrap_or(&0);
    let ideal_bucket = samples as f64 / (2u64.pow(k_bits as u32) as f64);
    let uniformity = max_count as f64 / ideal_bucket;
    let min_entropy = -(max_count as f64 / samples as f64).log2();

    EntropyCollapseStats {
        rounds, samples, projection_bits: k_bits,
        collision_count: collisions, expected_collisions: expected,
        uniformity_ratio: uniformity, min_entropy_bits: min_entropy,
        ideal_entropy_bits: k_bits as f64,
    }
}
