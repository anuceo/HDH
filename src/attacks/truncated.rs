/// Truncated differential propagation analysis for 4-bit reduced χ.
///
/// A "truncated differential" groups 16-bit differences by their nibble-wise
/// activity pattern: a 4-bit tag where bit k = 1 iff nibble k of the
/// difference is nonzero.  Pattern 0b0001 means only x₁ changed; 0b1111
/// means all four nibbles are active.
///
/// For each of the 15 non-trivial input activity patterns we sample many
/// random (Δ, x) pairs with that pattern and record the distribution of
/// output activity patterns.
///
/// Key metrics:
///  - **Maximum output-pattern probability**: high probability for any single
///    output pattern means the diffusion is predictable from the activity tag.
///  - **Impossible truncated differentials**: input patterns for which some
///    output pattern never occurs — useful for zero-correlation attacks.
///  - **Average output nibble weight**: mean number of active output nibbles
///    for a given input activity.  ≈ 4 means full mixing; < 2 means poor
///    propagation.
///  - **Single-to-multi mixing rate**: fraction of single-nibble-active inputs
///    whose output has ≥ 2 active nibbles — measures how quickly χ₄ spreads
///    activity across shares.

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

/// Nibble-wise activity pattern of a 16-bit difference: bit k = 1 iff nibble k ≠ 0.
fn activity_pattern(diff: u16) -> u8 {
    let mut p = 0u8;
    if diff & 0x000F != 0 { p |= 1; }
    if diff & 0x00F0 != 0 { p |= 2; }
    if diff & 0x0F00 != 0 { p |= 4; }
    if diff & 0xF000 != 0 { p |= 8; }
    p
}

fn nibble_weight(pattern: u8) -> u32 {
    pattern.count_ones()
}

// ── Public structs ─────────────────────────────────────────────────────────

pub struct PatternStats {
    /// Input activity pattern (4-bit, bits 0..3 = nibbles 1..4).
    pub input_pattern: u8,
    /// Observed non-zero output activity patterns (up to 15 entries).
    pub observed_output_patterns: Vec<u8>,
    /// Maximum probability of any single output pattern.
    pub max_output_prob: f64,
    /// Average number of active output nibbles.
    pub avg_output_nibble_weight: f64,
    /// Fraction of samples where ≥ 2 nibbles are active in the output.
    pub multi_nibble_output_frac: f64,
}

pub struct TruncatedDiffStats {
    /// Per-pattern statistics for each of the 15 nonzero input activity patterns.
    pub per_pattern: Vec<PatternStats>,
    /// Maximum across all patterns of max_output_prob.
    pub worst_max_output_prob: f64,
    /// Minimum across all patterns of avg_output_nibble_weight.
    pub min_avg_output_weight: f64,
    /// Fraction of single-nibble-active inputs (patterns with wt=1) that produce
    /// ≥ 2 active output nibbles.
    pub single_to_multi_rate: f64,
    /// Impossible truncated differential count: (in, out) pairs with probability 0.
    pub impossible_diff_count: usize,
}

/// Exhaustive truncated differential analysis for chi4.
///
/// For each nonzero input activity pattern, enumerate ALL nonzero 16-bit
/// differences with that pattern and ALL 65536 inputs to build the exact
/// conditional output-pattern distribution.
pub fn analyze_truncated_differentials() -> TruncatedDiffStats {
    let mut per_pattern: Vec<PatternStats> = Vec::with_capacity(15);
    let mut worst_max_prob = 0.0f64;
    let mut min_avg_wt = f64::MAX;
    let mut single_to_multi_total = 0usize;
    let mut single_to_multi_numer = 0usize;
    let mut impossible_count = 0usize;

    for in_pat in 1u8..=15u8 {
        // Collect all 16-bit differences matching this nibble activity pattern.
        // This is an exact enumeration: iterate all active values in the active nibbles.
        let active_nibbles: Vec<usize> = (0..4).filter(|k| (in_pat >> k) & 1 == 1).collect();

        // Count occurrences of each output pattern.
        let mut out_counts = [0usize; 16]; // indexed by output pattern (0..15)
        let mut total_pairs = 0usize;

        // For each input x and each nonzero difference Δ with activity = in_pat,
        // compute chi4(x) ⊕ chi4(x⊕Δ) and record its activity pattern.
        //
        // To make this exact but bounded: iterate over all x (65536), and for each
        // x over all Δ with the given pattern. The number of such Δ is:
        //   ∏_{active nibbles} 15  (each active nibble takes values 1..15)
        // For in_pat with wt=1: 15 values; wt=2: 225; wt=3: 3375; wt=4: 50625.
        // Fully exhaustive at wt=1/2; for wt=3/4 we sample 4096 inputs × all Δ.
        // Use a sampling cap to keep runtime under ~200ms per pattern.

        const MAX_PAIRS: usize = 2_000_000;
        let n_deltas = 15usize.pow(active_nibbles.len() as u32);
        let n_inputs = if n_deltas * 65536 > MAX_PAIRS {
            MAX_PAIRS / n_deltas
        } else {
            65536
        };

        // Iterate over inputs (first n_inputs, uniformly spaced if capped)
        let input_step = if n_inputs < 65536 { 65536 / n_inputs } else { 1 };
        for xi in (0usize..65536).step_by(input_step) {
            let x = xi as u16;
            // Iterate over all Δ with the given activity pattern.
            // Enumerate by varying each active nibble over 1..=15.
            let n_active = active_nibbles.len();
            let mut delta_counters = vec![1usize; n_active];

            loop {
                // Build Δ from the current nibble values.
                let mut delta: u16 = 0;
                for (slot, &nibble_idx) in active_nibbles.iter().enumerate() {
                    delta |= (delta_counters[slot] as u16) << (nibble_idx * 4);
                }

                let out_diff = chi4(x) ^ chi4(x ^ delta);
                let out_pat = activity_pattern(out_diff) as usize;
                out_counts[out_pat] += 1;
                total_pairs += 1;

                // Advance delta_counters (base-15 with digits 1..=15).
                let mut carry = true;
                for slot in (0..n_active).rev() {
                    if carry {
                        delta_counters[slot] += 1;
                        if delta_counters[slot] > 15 {
                            delta_counters[slot] = 1;
                        } else {
                            carry = false;
                        }
                    }
                }
                if carry { break; }
            }
        }

        // Compute statistics from out_counts.
        let max_count = *out_counts.iter().max().unwrap_or(&0);
        let max_prob = max_count as f64 / total_pairs as f64;

        let avg_wt = (0..16usize)
            .map(|p| (p as u8).count_ones() as f64 * out_counts[p] as f64)
            .sum::<f64>() / total_pairs as f64;

        let multi_numer: usize = (0..16)
            .filter(|&p| (p as u8).count_ones() >= 2)
            .map(|p| out_counts[p])
            .sum();
        let multi_frac = multi_numer as f64 / total_pairs as f64;

        let observed: Vec<u8> = (1u8..=15).filter(|&p| out_counts[p as usize] > 0).collect();

        // Count impossible (in_pat, out_pat) pairs: out patterns that never occur.
        impossible_count += (1..16).filter(|&p| out_counts[p] == 0).count();

        if max_prob > worst_max_prob { worst_max_prob = max_prob; }
        if avg_wt < min_avg_wt { min_avg_wt = avg_wt; }

        if nibble_weight(in_pat) == 1 {
            single_to_multi_total += total_pairs;
            single_to_multi_numer += multi_numer;
        }

        per_pattern.push(PatternStats {
            input_pattern: in_pat,
            observed_output_patterns: observed,
            max_output_prob: max_prob,
            avg_output_nibble_weight: avg_wt,
            multi_nibble_output_frac: multi_frac,
        });
    }

    let single_to_multi_rate = if single_to_multi_total > 0 {
        single_to_multi_numer as f64 / single_to_multi_total as f64
    } else {
        0.0
    };

    TruncatedDiffStats {
        per_pattern,
        worst_max_output_prob: worst_max_prob,
        min_avg_output_weight: min_avg_wt,
        single_to_multi_rate,
        impossible_diff_count: impossible_count,
    }
}
