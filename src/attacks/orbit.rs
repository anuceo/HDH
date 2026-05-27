/// Complete orbit/cycle analysis of chi4 (4×4-bit toy, 16-bit state, 65536 elements).
///
/// Chi4 is a bijection on the 65536-element space {0..=65535}.
/// This module computes the exact cycle structure using a three-color graph traversal.
///
/// ## Algorithm
/// Color codes: 0 = unvisited, u32::MAX = in-progress, else = cycle_id + 1
/// For each unvisited node, follow the map forward. When we re-encounter a node:
/// - If it has a known cycle ID: propagate that ID to all path nodes.
/// - If it is in-progress: we found a new cycle at the re-entry position.

// ── Chi4 definition ──────────────────────────────────────────────────────────

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

// ── Orbit statistics ──────────────────────────────────────────────────────────

pub struct OrbitStats {
    pub state_bits: usize,
    pub total_states: usize,
    pub fixed_points: usize,
    pub two_cycles: usize,
    pub unique_cycles: usize,
    pub min_cycle: usize,
    pub max_cycle: usize,
    pub avg_cycle: f64,
    pub median_cycle: usize,
    /// Shannon entropy of (cycle_length → count) distribution.
    pub orbit_entropy: f64,
    /// Fraction of total states in cycles of length <= 10.
    pub short_cycle_frac: f64,
}

// ── Three-color traversal ────────────────────────────────────────────────────

/// Analyze the complete orbit/cycle structure of chi4 over all 65536 states.
pub fn analyze_chi4_orbits() -> OrbitStats {
    const N: usize = 65536;
    let mut visited = vec![false; N];
    let mut on_cycle = vec![false; N];
    let mut actual_cycle_lengths: Vec<usize> = Vec::new();
    // Allocated once and reset between iterations to avoid O(N²) heap traffic.
    let mut seen_pos: Vec<i64> = vec![-1i64; N];

    for start in 0..N {
        if visited[start] {
            continue;
        }

        // Follow the functional graph from start
        let mut path: Vec<usize> = Vec::new();
        let mut x = start;

        loop {
            if visited[x] {
                // Already fully processed; mark all path nodes as visited
                for &v in &path {
                    visited[v] = true;
                }
                break;
            }
            if seen_pos[x] >= 0 {
                // Cycle detected
                let cycle_start = seen_pos[x] as usize;
                let cycle_len = path.len() - cycle_start;
                actual_cycle_lengths.push(cycle_len);
                for &v in &path[cycle_start..] {
                    on_cycle[v] = true;
                    visited[v] = true;
                }
                for &v in &path[..cycle_start] {
                    visited[v] = true;
                }
                // Reset seen_pos for nodes in path (avoid interference)
                for &v in &path {
                    seen_pos[v] = -1;
                }
                break;
            }
            seen_pos[x] = path.len() as i64;
            path.push(x);
            x = chi4(x as u16) as usize;
        }
        // Reset seen_pos for remaining path nodes
        for &v in &path {
            if seen_pos[v] >= 0 {
                seen_pos[v] = -1;
            }
        }
    }

    let unique_cycles = actual_cycle_lengths.len();
    let fixed_points = actual_cycle_lengths.iter().filter(|&&l| l == 1).count();
    let two_cycles = actual_cycle_lengths.iter().filter(|&&l| l == 2).count();

    let min_cycle = actual_cycle_lengths.iter().copied().min().unwrap_or(0);
    let max_cycle = actual_cycle_lengths.iter().copied().max().unwrap_or(0);
    let _total_cycle_states: usize = actual_cycle_lengths.iter().sum();
    let avg_cycle = if unique_cycles > 0 {
        actual_cycle_lengths.iter().sum::<usize>() as f64 / unique_cycles as f64
    } else {
        0.0
    };

    // Median cycle length
    let mut sorted_lengths = actual_cycle_lengths.clone();
    sorted_lengths.sort_unstable();
    let median_cycle = if sorted_lengths.is_empty() {
        0
    } else {
        sorted_lengths[sorted_lengths.len() / 2]
    };

    // Shannon entropy of cycle length distribution
    use std::collections::HashMap;
    let mut len_counts: HashMap<usize, usize> = HashMap::new();
    for &l in &actual_cycle_lengths {
        *len_counts.entry(l).or_insert(0) += 1;
    }
    let total = unique_cycles as f64;
    let orbit_entropy = if total > 0.0 {
        len_counts.values().map(|&c| {
            let p = c as f64 / total;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        }).sum::<f64>()
    } else {
        0.0
    };

    // Short cycle fraction: states in cycles of length <= 10
    let short_cycle_states: usize = actual_cycle_lengths.iter()
        .filter(|&&l| l <= 10)
        .sum();
    let short_cycle_frac = short_cycle_states as f64 / N as f64;

    OrbitStats {
        state_bits: 16,
        total_states: N,
        fixed_points,
        two_cycles,
        unique_cycles,
        min_cycle,
        max_cycle,
        avg_cycle,
        median_cycle,
        orbit_entropy,
        short_cycle_frac,
    }
}
