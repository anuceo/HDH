/// MILP-inspired exhaustive differential trail search over all 2^25 − 1 activity patterns.
///
/// Under the simplified model where φ = identity (conservative lower bound),
/// each round's differential follows:
///   Input activity Δ₀ → S-box cost = popcount(Δ₀) → after θ: Δ₁ = theta_activity(Δ₀)
///
/// For r rounds, total active S-boxes = Σ_{k=0}^{r-1} popcount(theta_activity^k(Δ₀)).
/// This search is exhaustive over all 2^25−1 starting patterns.

// ── Fast theta_activity for 25-bit words ──────────────────────────────────────

/// Rotate a 25-bit value (stored in u32) left by `k` positions (mod 25).
#[inline(always)]
fn rot25(x: u32, k: usize) -> u32 {
    let k = k % 25;
    ((x << k) | (x >> (25 - k))) & 0x1FFFFFF
}

/// Fast inlined version of theta_activity for the 25-bit activity vector.
/// out[i] = in[i] ^ in[(i+24)%25] ^ in[(i+1)%25] ^ in[(i+7)%25] ^ in[(i+18)%25]
/// = x ^ rot25(x, 24) ^ rot25(x, 1) ^ rot25(x, 7) ^ rot25(x, 18)
#[inline(always)]
fn theta_fast(x: u32) -> u32 {
    x ^ rot25(x, 24) ^ rot25(x, 1) ^ rot25(x, 7) ^ rot25(x, 18)
}

/// Maximum log2 differential probability for a single active chi lane.
/// From differential uniformity bench measurement.
pub const LOG2_P_MAX_CHI: f64 = -15.6;

// ── Structs ───────────────────────────────────────────────────────────────────

pub struct TrailSearchResult {
    pub rounds: usize,
    pub min_active_sboxes: usize,
    pub best_start_pattern: u32,
    pub best_start_weight: usize,
    pub implied_prob_log2: f64,
    pub patterns_searched: u64,
}

pub struct TrailSweepResult {
    pub entries: Vec<TrailSearchResult>,
    /// Analytical lower bounds from branch number: (rounds, analytical_min)
    pub analytical_bounds: Vec<(usize, usize)>,
}

// ── Exhaustive trail search ───────────────────────────────────────────────────

/// Exhaustively search over all 2^25 − 1 nonzero 25-bit activity patterns
/// to find the minimum total active S-boxes over `rounds` rounds.
pub fn search_min_active_trail(rounds: usize) -> TrailSearchResult {
    let patterns_searched: u64 = (1u64 << 25) - 1;
    let mut min_cost = usize::MAX;
    let mut best_start: u32 = 1;

    for delta in 1u32..(1u32 << 25) {
        let mut d = delta;
        let mut cost = 0usize;
        for _ in 0..rounds {
            cost += d.count_ones() as usize;
            d = theta_fast(d);
        }
        if cost < min_cost {
            min_cost = cost;
            best_start = delta;
        }
    }

    let best_start_weight = best_start.count_ones() as usize;
    let implied_prob_log2 = min_cost as f64 * LOG2_P_MAX_CHI;

    TrailSearchResult {
        rounds,
        min_active_sboxes: min_cost,
        best_start_pattern: best_start,
        best_start_weight,
        implied_prob_log2,
        patterns_searched,
    }
}

/// Sweep trail search for r = 1..=max_rounds.
pub fn trail_sweep(max_rounds: usize) -> TrailSweepResult {
    let entries: Vec<TrailSearchResult> = (1..=max_rounds)
        .map(|r| search_min_active_trail(r))
        .collect();

    // Analytical lower bounds: based on branch number B=6 (fan-out = 5)
    // r=1: 1 (single active lane)
    // r=2: 1+5 = 6 (branch number B=6)
    // r=3: 1+5+5 = 11 (conservative; exhaustive may find better)
    // r=4: saturates at 25 (full state)
    let analytical_bounds: Vec<(usize, usize)> = vec![
        (1, 1),
        (2, 6),
        (3, 11),
        (4, 25),
    ]
    .into_iter()
    .filter(|&(r, _)| r <= max_rounds)
    .collect();

    TrailSweepResult {
        entries,
        analytical_bounds,
    }
}
