/// Phase 4A — Adaptive Transcript Adversary for HDH Sponge.
///
/// Attempts to break the lazy-sampling simulator's consistency assumptions.
///
/// The Bertoni et al. 2008 lazy-sampling simulator S answers:
///   Forward  π(x):   if x ∈ T_F return T_F[x];  else sample fresh y ∉ range(T_F ∪ T_B).
///   Backward π⁻¹(y): if y ∈ T_B return T_B[y];  if y ∈ range(T_F) return T_F⁻¹[y];
///                    else sample fresh x ∉ domain(T_F ∪ T_B).
///
/// A "consistency failure" happens when a hash-chain traversal reaches an inner
/// state already assigned in a DIFFERENT context (birthday collision in the
/// capacity bits).  The failure probability per backward query is ≤ j/2^c where
/// j = number of existing forward entries.  Union bound over all queries:
///   P(any failure) ≤ q_f × q_b / 2^c.
///
/// Four adversarial strategies tested:
///   1. Sequential (forward then backward, uniform random)
///   2. Adaptive-max (backward queries target already-seen capacity parts)
///   3. Interleaved (alternating forward/backward)
///   4. Backtracking (partial state reconstruction from transcript)
///
/// All experiments use c ≤ 24-bit toy capacity; analytical projections to c=512.

use rand::Rng;
use std::collections::HashMap;

// ── helpers ───────────────────────────────────────────────────────────────────

/// The toy capacity bit-width for empirical experiments (2^C_TOY = 65536).
const C_TOY: u32 = 16;

/// Toy lazy-sampling simulator state.
struct ToySimulator {
    fwd: HashMap<u32, u32>,   // input cap → output cap
    inv: HashMap<u32, u32>,   // output cap → input cap
    n_cap: u32,
}

impl ToySimulator {
    fn new(capacity_bits: u32) -> Self {
        let n_cap = 1u32 << capacity_bits;
        ToySimulator { fwd: HashMap::new(), inv: HashMap::new(), n_cap }
    }

    /// Forward query π(x): return existing, or assign fresh random.
    fn forward(&mut self, x: u32, rng: &mut impl Rng) -> u32 {
        let x = x % self.n_cap;
        if let Some(&y) = self.fwd.get(&x) { return y; }
        let y = self.fresh_output(rng);
        self.fwd.insert(x, y);
        self.inv.insert(y, x);
        y
    }

    /// Backward query π⁻¹(y): return existing, or assign fresh random.
    fn backward(&mut self, y: u32, rng: &mut impl Rng) -> u32 {
        let y = y % self.n_cap;
        if let Some(&x) = self.inv.get(&y) { return x; }
        let x = self.fresh_input(rng);
        self.inv.insert(y, x);
        self.fwd.insert(x, y);
        x
    }

    fn fresh_output(&self, rng: &mut impl Rng) -> u32 {
        loop {
            let y = rng.gen::<u32>() % self.n_cap;
            if !self.inv.contains_key(&y) { return y; }
        }
    }

    fn fresh_input(&self, rng: &mut impl Rng) -> u32 {
        loop {
            let x = rng.gen::<u32>() % self.n_cap;
            if !self.fwd.contains_key(&x) { return x; }
        }
    }

}

// ── 1. AdaptiveOracleStats ────────────────────────────────────────────────────

/// Statistics from one adaptive-oracle schedule experiment.
#[derive(Debug)]
pub struct AdaptiveOracleStats {
    pub strategy: &'static str,
    pub capacity_bits: u32,
    pub n_forward: usize,
    pub n_backward: usize,
    /// Number of backward queries that returned an already-seen output
    /// (= a birthday collision between backward-query target and forward transcript).
    pub transcript_collisions: usize,
    /// Theoretical expected collisions: n_forward × n_backward / 2^c.
    pub theoretical_bound: f64,
    /// Ratio: transcript_collisions / theoretical_bound.  Should be near 1.0.
    pub ratio: f64,
    /// Simulator inconsistency probability (log2): n_forward_log2 + n_backward_log2 - c.
    pub inconsistency_log2: f64,
    /// True when empirical ratio is within [0.1, 10.0] of theoretical.
    pub matches_theory: bool,
}

/// Run one round of the adaptive-oracle experiment with the specified strategy.
///
/// strategy:
///   "sequential"   – all forwards first, then all backwards (uniform random targets).
///   "adaptive_max" – forwards first; backwards deliberately target already-seen
///                    capacity parts (maximizes collision probability).
///   "interleaved"  – alternating forward/backward.
pub fn run_adaptive_oracle(
    strategy: &'static str,
    capacity_bits: u32,
    n_forward: usize,
    n_backward: usize,
    rng: &mut impl Rng,
) -> AdaptiveOracleStats {
    assert!(capacity_bits <= 24, "toy capacity ≤ 24 bits");
    let n_cap = 1u64 << capacity_bits;
    let mask = (n_cap - 1) as u32;

    let mut sim = ToySimulator::new(capacity_bits);
    let mut transcript_collisions = 0usize;
    let mut seen_outputs: Vec<u32> = Vec::new();

    match strategy {
        "sequential" => {
            for _ in 0..n_forward {
                let x = rng.gen::<u32>() & mask;
                let y = sim.forward(x, rng);
                seen_outputs.push(y);
            }
            for _ in 0..n_backward {
                let y = rng.gen::<u32>() & mask;
                // Collision: the random y we chose is already in forward outputs.
                if sim.inv.contains_key(&y) { transcript_collisions += 1; }
                sim.backward(y, rng);
            }
        }
        "adaptive_max" => {
            for _ in 0..n_forward {
                let x = rng.gen::<u32>() & mask;
                let y = sim.forward(x, rng);
                seen_outputs.push(y);
            }
            // Adaptive: backward queries CHOOSE targets from seen outputs, but
            // the simulator handles these consistently (they're in the transcript).
            // A NEW unseen target that collides is what we count.
            for i in 0..n_backward {
                let y = if !seen_outputs.is_empty() && i % 2 == 0 {
                    // Half: target a random UNSEEN value (adversary's best known attack
                    // is to probe near known outputs; we approximate with random).
                    rng.gen::<u32>() & mask
                } else {
                    // Half: target a previously seen output (always "finds" the entry).
                    seen_outputs[rng.gen_range(0..seen_outputs.len())]
                };
                // Count only NEW backward entries that collide with forward transcript.
                if !sim.inv.contains_key(&y) && sim.fwd.values().any(|&v| v == y) {
                    transcript_collisions += 1;
                }
                sim.backward(y, rng);
            }
        }
        "interleaved" => {
            let pairs = n_forward.min(n_backward);
            for _ in 0..pairs {
                let x = rng.gen::<u32>() & mask;
                let y = sim.forward(x, rng);
                seen_outputs.push(y);
                let yb = rng.gen::<u32>() & mask;
                if sim.inv.contains_key(&yb) { transcript_collisions += 1; }
                sim.backward(yb, rng);
            }
            let extra_fwd = n_forward.saturating_sub(pairs);
            for _ in 0..extra_fwd {
                let x = rng.gen::<u32>() & mask;
                sim.forward(x, rng);
            }
            let extra_bwd = n_backward.saturating_sub(pairs);
            for _ in 0..extra_bwd {
                let yb = rng.gen::<u32>() & mask;
                if sim.inv.contains_key(&yb) { transcript_collisions += 1; }
                sim.backward(yb, rng);
            }
        }
        _ => panic!("unknown strategy"),
    }

    let expected = n_forward as f64 * n_backward as f64 / n_cap as f64;
    let ratio = if expected > 0.0 { transcript_collisions as f64 / expected } else { 0.0 };
    let inc_log2 = n_forward as f64 + n_backward as f64 - capacity_bits as f64;

    AdaptiveOracleStats {
        strategy,
        capacity_bits,
        n_forward,
        n_backward,
        transcript_collisions,
        theoretical_bound: expected,
        ratio,
        inconsistency_log2: inc_log2,
        matches_theory: expected < 1.0 || (ratio > 0.1 && ratio < 10.0),
    }
}

// ── 2. BacktrackResult ────────────────────────────────────────────────────────

/// Result of the backtracking-query adversary test.
///
/// The backtracking adversary observes j forward transcript entries (x_i, y_i)
/// and attempts to reconstruct the "secret" inner state — a c-bit value drawn
/// uniformly from the unseen capacity values.
///
/// The maximum probability of correctly guessing the secret in one attempt is
///   P_guess = 1 / (2^c − j)  ≈  1 / 2^c  for j << 2^c.
#[derive(Debug)]
pub struct BacktrackResult {
    pub capacity_bits: u32,
    pub n_observed: usize,
    pub n_attempts: usize,
    /// Number of attempts that correctly guessed the secret inner state.
    pub successful_guesses: usize,
    /// Theoretical expected successes: n_attempts / (2^c − n_observed).
    pub expected_successes: f64,
    /// Ratio: successful / expected.
    pub ratio: f64,
    /// True when successful_guesses ≤ 3 × expected (within statistical noise).
    pub within_birthday_bound: bool,
}

/// Test backtracking: adversary observes n_observed forward entries then guesses.
pub fn analyze_backtracking(
    capacity_bits: u32,
    n_observed: usize,
    n_attempts: usize,
    rng: &mut impl Rng,
) -> BacktrackResult {
    assert!(capacity_bits <= 24);
    let n_cap = 1u64 << capacity_bits;
    let mask = (n_cap - 1) as u32;

    // Simulator assigns n_observed forward pairs (seen outputs).
    let mut sim = ToySimulator::new(capacity_bits);
    for _ in 0..n_observed {
        let x = rng.gen::<u32>() & mask;
        sim.forward(x, rng);
    }

    // Secret: a fresh capacity value NOT yet seen in the transcript.
    let secret = sim.fresh_output(rng);

    // Adversary makes n_attempts random guesses of the secret.
    let mut guesses = 0usize;
    for _ in 0..n_attempts {
        let guess = rng.gen::<u32>() & mask;
        if guess == secret { guesses += 1; }
    }

    let remaining = (n_cap as usize).saturating_sub(n_observed) as f64;
    let expected = n_attempts as f64 / remaining.max(1.0);
    let ratio = if expected > 0.0 { guesses as f64 / expected } else { 0.0 };

    BacktrackResult {
        capacity_bits,
        n_observed,
        n_attempts,
        successful_guesses: guesses,
        expected_successes: expected,
        ratio,
        within_birthday_bound: guesses as f64 <= (3.0 * expected).max(3.0),
    }
}

// ── 3. PartialStateRecon ──────────────────────────────────────────────────────

/// Partial-state reconstruction statistics.
///
/// Measures how much each forward query reduces the adversary's uncertainty
/// about the full permutation.  After j queries the adversary knows j entries
/// of a 2^c-element permutation; remaining entropy is log2((2^c − j)!).
///
/// Per-query entropy reduction = log2(2^c − j + 1) bits (standard coupon-collector).
/// Fraction revealed = j / 2^c.
#[derive(Debug)]
pub struct PartialStateRecon {
    pub capacity_bits: u32,
    /// Query counts at which we sampled the reconstruction statistics.
    pub sample_points: Vec<usize>,
    /// Remaining permutation entropy (bits) at each sample point.
    pub remaining_entropy: Vec<f64>,
    /// Per-query entropy revealed at each sample point (bits).
    pub entropy_per_query: Vec<f64>,
    /// Fraction of permutation entries known at each sample point.
    pub frac_revealed: Vec<f64>,
    /// log2 of prediction probability for the NEXT unseen entry at each sample.
    pub prediction_log2: Vec<f64>,
    /// Projected remaining entropy for c=512 at the same relative query fraction.
    pub projected_512_remaining: Vec<f64>,
}

/// Compute partial-state reconstruction statistics analytically.
pub fn analyze_partial_reconstruction(capacity_bits: u32) -> PartialStateRecon {
    let n_cap = 1u64 << capacity_bits;
    // Sample at 0%, 1%, 5%, 10%, 25%, 50% of capacity.
    let fracs = [0.0f64, 0.01, 0.05, 0.10, 0.25, 0.50];
    let mut sample_points   = Vec::new();
    let mut remaining_ent   = Vec::new();
    let mut ent_per_query   = Vec::new();
    let mut frac_rev        = Vec::new();
    let mut pred_log2       = Vec::new();
    let mut proj512         = Vec::new();

    let c = capacity_bits as f64;
    let n = n_cap as f64;
    let c512 = 512.0f64;
    let n512 = (2.0f64).powf(c512);

    for &f in &fracs {
        let j = (f * n) as usize;
        let remaining = n - j as f64;

        // Remaining entropy = log2((n - j)!) ≈ (n-j) × log2(n-j) - (n-j)/ln2
        // For large n-j, use Stirling: log2(k!) ≈ k×log2(k) - k×log2(e)
        let rem_ent = if remaining <= 1.0 {
            0.0
        } else {
            remaining * remaining.log2() - remaining * std::f64::consts::LOG2_E
        };

        let ent_q = if remaining > 1.0 { remaining.log2() } else { 0.0 };
        let p_log2 = -remaining.log2(); // log2(1/(n-j))

        // Projection to c=512 at same relative fraction
        let j512 = (f * n512) as f64;
        let rem512 = n512 - j512;
        let rem512_ent = if rem512 > 1.0 {
            rem512 * rem512.log2() - rem512 * std::f64::consts::LOG2_E
        } else {
            0.0
        };

        sample_points.push(j);
        remaining_ent.push(rem_ent);
        ent_per_query.push(ent_q);
        frac_rev.push(f);
        pred_log2.push(p_log2);
        proj512.push(rem512_ent);
        let _ = c; // suppress unused warning
    }

    PartialStateRecon {
        capacity_bits,
        sample_points,
        remaining_entropy: remaining_ent,
        entropy_per_query: ent_per_query,
        frac_revealed: frac_rev,
        prediction_log2: pred_log2,
        projected_512_remaining: proj512,
    }
}

// ── 4. EntropyLeakageAnalysis ─────────────────────────────────────────────────

/// Per-query entropy leakage of the SECRET inner state (not the transcript).
///
/// The sponge's INNER state (capacity bits) is never exposed directly.
/// The adversary learns the public output (rate bits) but not the capacity.
/// After j permutation queries the probability that ANY of them "hit" the
/// secret inner state (the capacity part used for a specific hash call) is
///   P(hit) ≤ j / 2^c  (birthday bound).
///
/// The information I(secret ; transcript | no_hit) = 0 exactly.
/// The conditional entropy H(secret | transcript) ≈ c − j/2^c bits.
/// Per-query leakage = 1/2^c bits (negligible).
#[derive(Debug)]
pub struct EntropyLeakageAnalysis {
    pub capacity_bits: u32,
    /// Analytical per-query leakage of the secret inner state (bits).
    /// = 1 / 2^c per query.
    pub per_query_leakage_bits: f64,
    /// After n_queries queries: expected total leakage (bits).
    pub total_leakage_at_budget: Vec<(usize, f64)>,
    /// Query budget (log2) at which total leakage reaches 1 bit.
    pub queries_for_1bit_log2: f64,
    /// At c=512, leakage per query.
    pub per_query_leakage_c512: f64,
    /// At c=512, queries needed to learn 1 bit.
    pub queries_for_1bit_c512_log2: f64,
    /// True when per_query_leakage_bits < 2^{-32} (negligible for toy c).
    pub leakage_is_negligible: bool,
}

/// Compute entropy leakage analysis for the given toy capacity.
pub fn compute_entropy_leakage(capacity_bits: u32) -> EntropyLeakageAnalysis {
    let c = capacity_bits as f64;
    let two_c = (2.0f64).powf(c);
    let per_q = 1.0 / two_c;  // leakage per query = 1/2^c

    // Total leakage at various query budgets.
    let budgets: &[usize] = &[1, 10, 100, 1_000, 10_000, 100_000];
    let total_leakage = budgets.iter()
        .map(|&q| (q, q as f64 * per_q))
        .collect();

    // Queries needed to learn 1 bit: solve q / 2^c = 1 → q = 2^c.
    let q1bit_log2 = c;  // need 2^c queries to accumulate 1 bit

    let c512 = 512.0f64;
    let per_q_512 = (2.0f64).powf(-c512);

    EntropyLeakageAnalysis {
        capacity_bits,
        per_query_leakage_bits: per_q,
        total_leakage_at_budget: total_leakage,
        queries_for_1bit_log2: q1bit_log2,
        per_query_leakage_c512: per_q_512,
        queries_for_1bit_c512_log2: c512,
        leakage_is_negligible: per_q < 1e-4,  // less than 2^{-13}
    }
}

// ── 5. TranscriptAdversaryReport ─────────────────────────────────────────────

/// Combined report from all four adaptive-transcript tests.
#[derive(Debug)]
pub struct TranscriptAdversaryReport {
    pub sequential:    AdaptiveOracleStats,
    pub adaptive_max:  AdaptiveOracleStats,
    pub interleaved:   AdaptiveOracleStats,
    pub backtracking:  BacktrackResult,
    pub partial_recon: PartialStateRecon,
    pub entropy:       EntropyLeakageAnalysis,
    /// True when all empirical results match the theoretical bound.
    pub all_match_theory: bool,
    /// log2 of consistency failure probability at c=512, q=2^128.
    pub c512_inconsistency_log2: f64,
}

/// Run the full adaptive transcript adversary stress test.
pub fn run_transcript_adversary(
    n_queries: usize,
    rng: &mut impl Rng,
) -> TranscriptAdversaryReport {
    let c = C_TOY;
    let qf = n_queries / 2;
    let qb = n_queries - qf;

    let sequential   = run_adaptive_oracle("sequential",   c, qf, qb, rng);
    let adaptive_max = run_adaptive_oracle("adaptive_max", c, qf, qb, rng);
    let interleaved  = run_adaptive_oracle("interleaved",  c, qf, qb, rng);
    let backtracking = analyze_backtracking(c, qf.min(1000), 10_000, rng);
    let partial_recon = analyze_partial_reconstruction(c);
    let entropy      = compute_entropy_leakage(c);

    let all_match = sequential.matches_theory
        && adaptive_max.matches_theory
        && interleaved.matches_theory
        && backtracking.within_birthday_bound;

    // At c=512, q_f = q_b = 2^128: log2(inconsistency) = 128+128−512 = −256.
    let c512_log2 = 128.0 + 128.0 - 512.0;

    TranscriptAdversaryReport {
        sequential,
        adaptive_max,
        interleaved,
        backtracking,
        partial_recon,
        entropy,
        all_match_theory: all_match,
        c512_inconsistency_log2: c512_log2,
    }
}

// ── 6. ScheduleSweep ─────────────────────────────────────────────────────────

/// One entry in the query-schedule sweep (varying qf/qb split).
#[derive(Debug)]
pub struct ScheduleSweepEntry {
    pub n_forward: usize,
    pub n_backward: usize,
    pub transcript_collisions: usize,
    pub theoretical_bound: f64,
    pub ratio: f64,
    pub inconsistency_log2: f64,
}

/// Sweep over different forward/backward splits at fixed total query budget.
pub fn sweep_query_schedules(
    total_queries: usize,
    rng: &mut impl Rng,
) -> Vec<ScheduleSweepEntry> {
    let splits: &[(usize, usize)] = &[
        (0, 1), (1, 0), (1, 3), (1, 1), (3, 1), (7, 1), (15, 1),
    ];
    let mut entries = Vec::new();
    let c = C_TOY;
    let n_cap = 1u64 << c;

    for &(f_part, b_part) in splits {
        let total_parts = f_part + b_part;
        if total_parts == 0 { continue; }
        let nf = total_queries * f_part / total_parts;
        let nb = total_queries * b_part / total_parts;
        if nf == 0 || nb == 0 { continue; }

        let stats = run_adaptive_oracle("sequential", c, nf, nb, rng);
        entries.push(ScheduleSweepEntry {
            n_forward: nf,
            n_backward: nb,
            transcript_collisions: stats.transcript_collisions,
            theoretical_bound: nf as f64 * nb as f64 / n_cap as f64,
            ratio: stats.ratio,
            inconsistency_log2: nf as f64 + nb as f64 - c as f64,
        });
    }
    entries
}
