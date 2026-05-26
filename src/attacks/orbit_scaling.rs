/// Multi-precision orbit analysis for chi at 8-bit, 16-bit, 32-bit, and 64-bit.
///
/// 8-bit chi (4×2-bit, 256 states): exact enumeration
/// 16-bit chi4 (4×4-bit, 65536 states): from orbit::analyze_chi4_orbits()
/// 32-bit chi (4×8-bit, 2^32 states): Floyd's cycle detection, sampled
/// 64-bit chi_lane (4×64-bit, 2^256 state space): Floyd's cycle detection, sampled

use rand::Rng;
use std::collections::HashMap;
use crate::chi::chi_lane;

// ── Structs ───────────────────────────────────────────────────────────────────

pub struct OrbitScalingEntry {
    pub bits_per_share: usize,
    pub total_state_bits: usize,
    pub method: &'static str,
    pub sample_count: usize,
    pub fixed_point_frac: f64,
    pub avg_cycle_len: f64,
    pub max_cycle_len: u64,
    pub entropy_bits: f64,
}

// ── 8-bit chi (4×2-bit) ───────────────────────────────────────────────────────

/// Chi on 4×2-bit shares packed into a u8.
/// Uses same rotation pattern as chi4 but mod 2.
fn chi_2bit(packed: u8) -> u8 {
    let x1 = packed & 0x3;
    let x2 = (packed >> 2) & 0x3;
    let x3 = (packed >> 4) & 0x3;
    let x4 = (packed >> 6) & 0x3;
    let g = (x1.wrapping_mul(x2) ^ x3.wrapping_mul(x4)) & 0x3;
    // Rotations mod 2: rot_0 = 0, rot_1 = 1, rot_2 = 0 (mod 2), rot_3 = 1 (mod 2)
    let o1 = x1 ^ g;
    let o2 = x2 ^ ((g << 1) | (g >> 1)) & 0x3;  // 1-bit rotation within 2 bits
    let o3 = x3 ^ g;                              // 2-bit rotation = identity mod 2
    let o4 = x4 ^ ((g << 1) | (g >> 1)) & 0x3;  // 3-bit rotation = 1-bit rotation mod 2
    (o1 & 0x3) | ((o2 & 0x3) << 2) | ((o3 & 0x3) << 4) | ((o4 & 0x3) << 6)
}

/// Exact orbit analysis for 8-bit chi (256 states).
pub fn analyze_chi_2bit_exact() -> OrbitScalingEntry {
    const N: usize = 256;
    let mut visited = vec![false; N];
    let mut on_cycle = vec![false; N];
    let mut cycle_lengths: Vec<u64> = Vec::new();

    for start in 0..N {
        if visited[start] {
            continue;
        }
        let mut path: Vec<usize> = Vec::new();
        let mut seen_pos = vec![-1i32; N];
        let mut x = start;

        loop {
            if visited[x] {
                for &v in &path {
                    visited[v] = true;
                }
                break;
            }
            if seen_pos[x] >= 0 {
                let cycle_start = seen_pos[x] as usize;
                let cycle_len = path.len() - cycle_start;
                cycle_lengths.push(cycle_len as u64);
                for &v in &path[cycle_start..] {
                    on_cycle[v] = true;
                    visited[v] = true;
                }
                for &v in &path[..cycle_start] {
                    visited[v] = true;
                }
                for &v in &path {
                    seen_pos[v] = -1;
                }
                break;
            }
            seen_pos[x] = path.len() as i32;
            path.push(x);
            x = chi_2bit(x as u8) as usize;
        }
        for &v in &path {
            if seen_pos[v] >= 0 {
                seen_pos[v] = -1;
            }
        }
    }

    compute_entry_from_cycles(2, 8, "exact", N, cycle_lengths)
}

// ── 32-bit chi (4×8-bit) ──────────────────────────────────────────────────────

/// Chi on 4×8-bit shares packed into a u32.
/// Uses rotation constants matching chi_lane mod 8: 0, 7, 13%8=5, 31%8=7.
fn chi_8bit(packed: u32) -> u32 {
    let x1 = (packed & 0xFF) as u8;
    let x2 = ((packed >> 8) & 0xFF) as u8;
    let x3 = ((packed >> 16) & 0xFF) as u8;
    let x4 = ((packed >> 24) & 0xFF) as u8;
    let g = x1.wrapping_mul(x2) ^ x3.wrapping_mul(x4);
    let o1 = x1 ^ g;
    let o2 = x2 ^ g.rotate_left(7);
    let o3 = x3 ^ g.rotate_left(5); // 13 mod 8 = 5
    let o4 = x4 ^ g.rotate_left(7); // 31 mod 8 = 7
    (o1 as u32) | ((o2 as u32) << 8) | ((o3 as u32) << 16) | ((o4 as u32) << 24)
}


/// Sampled orbit analysis for 32-bit chi via bounded hash-based cycle detection.
/// Iterates up to max_walk steps from each starting point and checks for a cycle.
pub fn analyze_chi_8bit_sampled(samples: usize, rng: &mut impl Rng) -> OrbitScalingEntry {
    let mut cycle_lengths: Vec<u64> = Vec::with_capacity(samples);
    let mut fixed_points = 0usize;
    const MAX_WALK: usize = 2000; // check up to 2000 steps

    for _ in 0..samples {
        let start: u32 = rng.gen();
        let mut seen: HashMap<u32, usize> = HashMap::new();
        let mut x = start;
        let mut found_cycle = false;

        for step in 0..MAX_WALK {
            if let Some(&prev_pos) = seen.get(&x) {
                let len = (step - prev_pos) as u64;
                if len == 1 && x == start { fixed_points += 1; }
                cycle_lengths.push(len);
                found_cycle = true;
                break;
            }
            seen.insert(x, step);
            x = chi_8bit(x);
        }

        if !found_cycle {
            // No cycle found in MAX_WALK steps; record a large value
            cycle_lengths.push(MAX_WALK as u64);
        }
    }

    let fixed_point_frac = fixed_points as f64 / samples as f64;
    let avg = cycle_lengths.iter().sum::<u64>() as f64 / samples as f64;
    let max = *cycle_lengths.iter().max().unwrap_or(&0);
    let entropy = entropy_from_samples(&cycle_lengths);

    OrbitScalingEntry {
        bits_per_share: 8,
        total_state_bits: 32,
        method: "sampled",
        sample_count: samples,
        fixed_point_frac,
        avg_cycle_len: avg,
        max_cycle_len: max,
        entropy_bits: entropy,
    }
}

// ── 64-bit chi_lane (4×64-bit) ────────────────────────────────────────────────

type ChiLaneState = (u64, u64, u64, u64);

fn chi_lane_step(x: ChiLaneState) -> ChiLaneState {
    let (x1, x2, x3, x4) = x;
    chi_lane(x1, x2, x3, x4)
}


/// Sampled orbit analysis for 64-bit chi_lane via bounded iteration + hash detection.
/// Iterates up to max_walk steps from each starting point to check for short cycles.
pub fn analyze_chi_lane_64bit_sampled(samples: usize, rng: &mut impl Rng) -> OrbitScalingEntry {
    let mut cycle_lengths: Vec<u64> = Vec::with_capacity(samples);
    let mut fixed_points = 0usize;
    const MAX_WALK: usize = 1000; // bounded walk

    for _ in 0..samples {
        let start: ChiLaneState = (rng.gen(), rng.gen(), rng.gen(), rng.gen());
        let mut seen: HashMap<ChiLaneState, usize> = HashMap::new();
        let mut x = start;
        let mut found_cycle = false;

        for step in 0..MAX_WALK {
            if let Some(&prev_pos) = seen.get(&x) {
                let len = (step - prev_pos) as u64;
                if len == 1 && x == start { fixed_points += 1; }
                cycle_lengths.push(len);
                found_cycle = true;
                break;
            }
            seen.insert(x, step);
            x = chi_lane_step(x);
        }

        if !found_cycle {
            // No short cycle found; record large value indicating long/no cycle
            cycle_lengths.push(MAX_WALK as u64);
        }
    }

    let fixed_point_frac = fixed_points as f64 / samples as f64;
    let avg = cycle_lengths.iter().sum::<u64>() as f64 / samples as f64;
    let max = *cycle_lengths.iter().max().unwrap_or(&0);
    let entropy = entropy_from_samples(&cycle_lengths);

    OrbitScalingEntry {
        bits_per_share: 64,
        total_state_bits: 256,
        method: "sampled",
        sample_count: samples,
        fixed_point_frac,
        avg_cycle_len: avg,
        max_cycle_len: max,
        entropy_bits: entropy,
    }
}

// ── Full scaling table ─────────────────────────────────────────────────────────

/// Fast exact orbit analysis for chi4 (16-bit, 65536 states).
/// Uses a two-pass algorithm with an index array for O(N) cycle detection.
fn analyze_chi4_exact_fast() -> OrbitScalingEntry {
    const N: usize = 65536;
    // Build permutation table (chi4)
    let perm: Vec<u16> = (0u16..=u16::MAX).map(|x| {
        let x1 = (x & 0xF) as u8;
        let x2 = ((x >> 4) & 0xF) as u8;
        let x3 = ((x >> 8) & 0xF) as u8;
        let x4 = ((x >> 12) & 0xF) as u8;
        let g = (x1.wrapping_mul(x2) ^ x3.wrapping_mul(x4)) & 0xF;
        let o1 = (x1 ^ g) & 0xF;
        let o2 = (x2 ^ ((g << 1) | (g >> 3)) & 0xF) & 0xF;
        let o3 = (x3 ^ ((g << 2) | (g >> 2)) & 0xF) & 0xF;
        let o4 = (x4 ^ ((g << 3) | (g >> 1)) & 0xF) & 0xF;
        (o1 as u16) | ((o2 as u16) << 4) | ((o3 as u16) << 8) | ((o4 as u16) << 12)
    }).collect();

    // O(N) cycle detection:
    // state: 0 = unvisited, u32::MAX = in current path, else = resolved (cycle_id)
    let mut state = vec![0u32; N];
    let mut path_pos = vec![0u32; N]; // position in current path (valid when state=MAX)
    let mut cycle_lengths: Vec<u64> = Vec::new();
    let mut next_id = 1u32;

    for start in 0..N {
        if state[start] != 0 {
            continue;
        }
        let mut path: Vec<u32> = Vec::new();
        let mut x = start;
        loop {
            if state[x] != 0 && state[x] != u32::MAX {
                // Already resolved; mark all path nodes with same id
                let id = state[x];
                for &v in &path { state[v as usize] = id; }
                break;
            }
            if state[x] == u32::MAX {
                // Back edge: found a cycle
                let cycle_start = path_pos[x] as usize;
                let len = path.len() - cycle_start;
                let id = next_id; next_id += 1;
                cycle_lengths.push(len as u64);
                for &v in &path { state[v as usize] = id; }
                break;
            }
            state[x] = u32::MAX;
            path_pos[x] = path.len() as u32;
            path.push(x as u32);
            x = perm[x] as usize;
        }
        // Reset path state for cleanup (already set to resolved id above)
    }

    compute_entry_from_cycles(4, 16, "exact", N, cycle_lengths)
}

/// Returns orbit entries for 8-bit exact, 16-bit exact, 32-bit sampled, 64-bit sampled.
pub fn orbit_scaling_table(rng: &mut impl Rng) -> Vec<OrbitScalingEntry> {
    let e8 = analyze_chi_2bit_exact();

    // 16-bit: fast exact method (avoids the slow O(N^2) orbit.rs approach)
    let e16 = analyze_chi4_exact_fast();

    let e32 = analyze_chi_8bit_sampled(100, rng);
    let e64 = analyze_chi_lane_64bit_sampled(50, rng);

    vec![e8, e16, e32, e64]
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn compute_entry_from_cycles(
    bits_per_share: usize,
    total_state_bits: usize,
    method: &'static str,
    total_states: usize,
    cycle_lengths: Vec<u64>,
) -> OrbitScalingEntry {
    let sample_count = cycle_lengths.len();
    let fixed_points = cycle_lengths.iter().filter(|&&l| l == 1).count();
    let fixed_point_frac = if total_states > 0 {
        // For exact: use total_states as denominator
        // Actually fixed_point_frac = fixed_point_cycles * 1 / total_states
        // But we want fraction of states that are fixed points
        // Each fixed-point cycle has length 1, so contributes 1 state
        fixed_points as f64 / total_states as f64
    } else {
        fixed_points as f64 / sample_count.max(1) as f64
    };

    let avg = if sample_count > 0 {
        cycle_lengths.iter().sum::<u64>() as f64 / sample_count as f64
    } else {
        0.0
    };
    let max = cycle_lengths.iter().copied().max().unwrap_or(0);
    let entropy = entropy_from_samples(&cycle_lengths);

    OrbitScalingEntry {
        bits_per_share,
        total_state_bits,
        method,
        sample_count,
        fixed_point_frac,
        avg_cycle_len: avg,
        max_cycle_len: max,
        entropy_bits: entropy,
    }
}

fn entropy_from_samples(cycle_lens: &[u64]) -> f64 {
    let mut freq_map: HashMap<u64, u64> = HashMap::new();
    for &l in cycle_lens {
        *freq_map.entry(l).or_insert(0) += 1;
    }
    let n = cycle_lens.len() as f64;
    freq_map.values().map(|&c| {
        let p = c as f64 / n;
        if p > 0.0 { -p * p.log2() } else { 0.0 }
    }).sum()
}
