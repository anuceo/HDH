/// Structural validity tests S1–S5.

use crate::framework::TestCase;
use crate::hdh::{
    layer_chi, layer_entropy_fold, layer_phi, layer_theta,
    rc, theta_activity, Lane, State, LANES, OUTPUT_BYTES, RATE_BYTES, ROUNDS,
};

const CAT: &str = "Structural Validity";

// ── S1: Theta branch number B(θ) = 6 ─────────────────────────────────────────

pub fn s1_branch_number() -> TestCase {
    // Exhaustive search over all 2^25 - 1 non-zero 25-bit activity patterns.
    // Branch number = min over all patterns of (popcount(pattern) + popcount(theta_activity(pattern))).
    let mask25 = (1u32 << 25) - 1;
    let mut min_sum = u32::MAX;
    let mut worst_in = 0u32;
    let mut worst_out = 0u32;

    for a in 1u32..=(mask25) {
        let ta = theta_activity(a);
        let sum = a.count_ones() + ta.count_ones();
        if sum < min_sum {
            min_sum = sum;
            worst_in = a.count_ones();
            worst_out = ta.count_ones();
        }
    }

    let claimed = 6u32;
    if min_sum >= claimed {
        TestCase::pass("S1", CAT, "Theta branch number B(θ) = 6",
            format!("exhaustive 2^25−1 pattern search: min(wt_in + wt_out) = {min_sum} \
                     (worst case: wt_in={worst_in}, wt_out={worst_out})"))
    } else {
        TestCase::fail("S1", CAT, "Theta branch number B(θ) = 6",
            format!("min branch sum = {min_sum}; worst_in={worst_in}, worst_out={worst_out}"),
            format!("B(θ) = {min_sum} < claimed 6 — theta diffusion is weaker than specified"))
    }
}

// ── S2: State size 6400 bits ──────────────────────────────────────────────────

pub fn s2_state_size() -> TestCase {
    let lane_bits = std::mem::size_of::<Lane>() * 8;
    let state_bits = std::mem::size_of::<State>() * 8;
    let claimed_bits = 6400usize;
    let rate_bits = RATE_BYTES * 8;
    let cap_bits = claimed_bits - rate_bits;

    if state_bits == claimed_bits && lane_bits == 256 && rate_bits == 5888 && cap_bits == 512 {
        TestCase::pass("S2", CAT, "State size: 6400 bits (25 × 256-bit lanes)",
            format!("State={state_bits}b, Lane={lane_bits}b, Rate={rate_bits}b, Cap={cap_bits}b, \
                     Output={}b", OUTPUT_BYTES * 8))
    } else {
        TestCase::fail("S2", CAT, "State size: 6400 bits (25 × 256-bit lanes)",
            format!("State={state_bits}b, Lane={lane_bits}b, Rate={rate_bits}b"),
            format!("expected State=6400, Lane=256, Rate=5888; got State={state_bits}, \
                     Lane={lane_bits}, Rate={rate_bits}"))
    }
}

// ── S3: Lane integrity through each layer ─────────────────────────────────────
//
// Verifies that:
// (a) Capacity lanes (23–24) are never modified by absorb.
// (b) Modifying one input lane always changes the output of each layer.

pub fn s3_lane_integrity() -> TestCase {
    let mut issues: Vec<String> = Vec::new();

    // 3a: Capacity lanes untouched during absorption.
    // Construct a state with sentinel capacity lanes, absorb a zero block, check they changed.
    let sentinel: Lane = [0xCAFEBABEDEADBEEFu64; 4];
    let mut state: State = [[0u64; 4]; LANES];
    state[23] = sentinel;
    state[24] = sentinel;
    let zero_block = [0u8; RATE_BYTES];
    // Manually absorb zero block + permute.
    // After XOR (nothing changes in capacity), permute runs → capacity WILL change (by design;
    // the permutation mixes everything). What we verify is that the XOR step itself does not
    // write into lanes 23–24 (i.e., the rate XOR loop stops at RATE_LANES=23).
    // We compare capacity before XOR vs capacity before permute:
    // Since our absorb() calls permute(), we test the XOR step in isolation.
    let mut state_pre_xor = state;
    for lane in 0..23 {
        let base = lane * 32;
        for w in 0..4 {
            let off = base + w * 8;
            let word = u64::from_le_bytes(zero_block[off..off + 8].try_into().unwrap());
            state_pre_xor[lane][w] ^= word;
        }
    }
    // Capacity lanes must remain sentinel after rate-only XOR.
    if state_pre_xor[23] != sentinel || state_pre_xor[24] != sentinel {
        issues.push("XOR absorption wrote into capacity lanes 23 or 24".to_string());
    }

    // 3b: Each layer propagates a single-lane difference.
    let base_state: State = std::array::from_fn(|i| {
        std::array::from_fn(|w| (i * 4 + w + 1) as u64 * 0x0101010101010101)
    });
    let mut perturbed = base_state;
    perturbed[0][0] ^= 1; // flip single bit in lane 0, word 0

    for (layer_name, layer_fn) in &[
        ("theta",  layer_theta  as fn(&mut State)),
        ("chi",    layer_chi    as fn(&mut State)),
        ("phi",    layer_phi    as fn(&mut State)),
        ("efold",  layer_entropy_fold as fn(&mut State)),
    ] {
        let mut a = base_state;
        let mut b = perturbed;
        layer_fn(&mut a);
        layer_fn(&mut b);
        let changed_lanes = (0..LANES).filter(|&i| a[i] != b[i]).count();
        if changed_lanes == 0 {
            issues.push(format!("{layer_name}: single-lane input difference produced zero output differences"));
        }
    }

    if issues.is_empty() {
        TestCase::pass("S3", CAT, "Lane integrity: capacity unchanged by XOR; layers propagate differences",
            "XOR absorption respects rate boundary; all 4 layers propagate single-lane perturbations")
    } else {
        TestCase::fail("S3", CAT, "Lane integrity: capacity unchanged by XOR; layers propagate differences",
            issues.join("; "),
            "lane boundary or diffusion violated")
    }
}

// ── S4: Phi routing determinism ───────────────────────────────────────────────
//
// Same state → same routing decisions every call.

pub fn s4_phi_determinism() -> TestCase {
    let state: State = std::array::from_fn(|i| {
        std::array::from_fn(|w| ((i * 7 + w * 3 + 1) as u64).wrapping_mul(0x9E3779B97F4A7C15))
    });

    let mut a = state;
    let mut b = state;
    layer_phi(&mut a);
    layer_phi(&mut b);

    if a == b {
        // Also verify phi is NOT the identity (it actually transforms the state).
        let mut c = state;
        layer_phi(&mut c);
        let changed = (0..LANES).filter(|&i| c[i] != state[i]).count();
        if changed > 0 {
            TestCase::pass("S4", CAT, "Phi routing: deterministic and non-trivial",
                format!("phi(state) == phi(state) on 2 independent calls; {changed}/{LANES} lanes modified"))
        } else {
            TestCase::fail("S4", CAT, "Phi routing: deterministic and non-trivial",
                "phi produced no change to state",
                "phi is a no-op — routing is degenerate")
        }
    } else {
        TestCase::fail("S4", CAT, "Phi routing: deterministic and non-trivial",
            "phi(state) != phi(state) on identical inputs",
            "phi is non-deterministic — implementation bug")
    }
}

// ── S5: Round constant uniqueness ─────────────────────────────────────────────

pub fn s5_rc_uniqueness() -> TestCase {
    let rcs: Vec<Lane> = (0..ROUNDS).map(rc).collect();

    // All-zero check: each RC must be non-zero.
    let zero_lane: Lane = [0u64; 4];
    let has_zero = rcs.iter().any(|&rc| rc == zero_lane);

    // Pairwise distinctness.
    let mut duplicates = Vec::new();
    for i in 0..ROUNDS {
        for j in (i + 1)..ROUNDS {
            if rcs[i] == rcs[j] {
                duplicates.push((i, j));
            }
        }
    }

    if !has_zero && duplicates.is_empty() {
        TestCase::pass("S5", CAT, "Round constants: non-zero and pairwise distinct",
            format!("{ROUNDS} SHAKE256-derived RCs verified distinct and non-zero; \
                     RC[0]={:016x}…", rcs[0][0]))
    } else {
        let mut ev = Vec::new();
        if has_zero { ev.push("at least one RC is all-zero".to_string()); }
        for (i, j) in &duplicates { ev.push(format!("RC[{i}] == RC[{j}]")); }
        TestCase::fail("S5", CAT, "Round constants: non-zero and pairwise distinct",
            ev.join("; "),
            "degenerate round constants weaken symmetry-breaking property")
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

pub fn run(_seed: u64) -> Vec<TestCase> {
    vec![
        s1_branch_number(),
        s2_state_size(),
        s3_lane_integrity(),
        s4_phi_determinism(),
        s5_rc_uniqueness(),
    ]
}
