/// HDH algorithm module.
///
/// Contains the complete DFA-hardened χ-core permutation and the sponge hash
/// construction built on top of it.
///
/// # Module layout
///
/// | Sub-module  | Purpose                                           |
/// |-------------|---------------------------------------------------|
/// | `state`     | 6400-bit state type, fault injection, avalanche   |
/// | `chi`       | Non-linear χ layer, θ diffusion, Φ routing        |
/// | `hash`      | Sponge hash construction and public API           |

pub mod chi;
pub mod hash;
pub mod state;

// Re-export the most commonly used types and functions at this module level.
pub use state::State;
pub use hash::{
    hdh_hash, hdh_hash_256, hdh_hash_512, hdh_hash_1024, hdh_prf,
    HdhSponge, RATE_BYTES, CAPACITY_BYTES, ROUNDS,
};

// ── HDH round function ────────────────────────────────────────────────────────

/// Apply one round of the HDH permutation to `s`.
///
/// Round order:
///   1. Entropy inject   — per-share BLAKE3-derived constant XOR
///   2. χ (chi)          — non-linear quad mixing within each lane
///   3. θ (theta)        — ring-diffusion across 25 lanes
///   4. Φ (phi)          — state-dependent lane routing
pub fn round(mut s: State, round_idx: u64) -> State {
    let r = blake3::hash(&round_idx.to_le_bytes()).as_bytes()[0] as u64;

    for i in 0..25 {
        s.s1[i] ^= r;
        s.s2[i] ^= r.rotate_left(11);
        s.s3[i] ^= r.rotate_left(23);
        s.s4[i] ^= r.rotate_left(37);
    }

    s = chi::chi(&s);
    s = chi::theta(&s);
    s = chi::phi(&s);
    s
}

// ── Unit tests for permutation primitives ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use state::{inject_bit_fault, avalanche_ratio, hamming_distance};

    fn fixed_state(seed: u64) -> State {
        use std::num::Wrapping;
        let mut v = Wrapping(seed);
        let mut fill = || {
            v = v * Wrapping(6364136223846793005) + Wrapping(1442695040888963407);
            v.0
        };
        let mut arr = || std::array::from_fn::<u64, 25, _>(|_| fill());
        State {
            s1: arr(),
            s2: arr(),
            s3: arr(),
            s4: arr(),
            parity: [0u64; 25],
        }
    }

    #[test]
    fn chi_deterministic() {
        let s = fixed_state(42);
        let a = chi::chi(&s);
        let b = chi::chi(&s);
        for i in 0..25 {
            assert_eq!(a.s1[i], b.s1[i]);
            assert_eq!(a.s2[i], b.s2[i]);
            assert_eq!(a.s3[i], b.s3[i]);
            assert_eq!(a.s4[i], b.s4[i]);
        }
    }

    #[test]
    fn chi_parity_binding() {
        let s = fixed_state(7);
        let out = chi::chi(&s);
        for i in 0..25 {
            let expected = out.s1[i] ^ out.s2[i] ^ out.s3[i] ^ out.s4[i];
            assert_eq!(out.parity[i], expected, "parity mismatch at lane {i}");
        }
    }

    #[test]
    fn theta_parity_invariant() {
        let s = chi::chi(&fixed_state(31415));
        let out = chi::theta(&s);
        for i in 0..25 {
            let prev = (i + 24) % 25;
            let next = (i + 1) % 25;
            let expected = s.parity[prev] ^ s.parity[i] ^ s.parity[next];
            assert_eq!(out.parity[i], expected, "theta parity mismatch at lane {i}");
            let actual = out.s1[i] ^ out.s2[i] ^ out.s3[i] ^ out.s4[i];
            assert_eq!(out.parity[i], actual, "theta share parity broken at lane {i}");
        }
    }

    #[test]
    fn theta_cross_lane_diffusion() {
        let s = fixed_state(271828);
        let mut perturbed = s.clone();
        perturbed.s1[12] ^= 1;
        let a = chi::theta(&s);
        let b = chi::theta(&perturbed);
        for i in 0..25 {
            let changed = (a.s1[i] ^ b.s1[i]) != 0;
            if i == 11 || i == 12 || i == 13 {
                assert!(changed, "theta failed to propagate to lane {i}");
            } else {
                assert!(!changed, "theta unexpectedly changed lane {i}");
            }
        }
    }

    #[test]
    fn phi_deterministic() {
        let s = fixed_state(99);
        let a = chi::phi(&s);
        let b = chi::phi(&s);
        for i in 0..25 {
            assert_eq!(a.s1[i], b.s1[i]);
        }
    }

    #[test]
    fn round_avalanche_single_bit_flip() {
        let s = fixed_state(1337);
        let mut s2 = s.clone();
        s2.s1[0] ^= 1;
        let r1 = round(s.clone(), 1);
        let r2 = round(s2.clone(), 1);
        let dist = hamming_distance(&r1, &r2);
        let ratio = avalanche_ratio(dist);
        assert!(
            ratio > 0.08,
            "single-round avalanche too weak: {:.2}% bits flipped",
            ratio * 100.0
        );
    }

    #[test]
    fn round_avalanche_two_rounds() {
        let s = fixed_state(1337);
        let mut s2 = s.clone();
        s2.s1[0] ^= 1;
        let mut r1 = s.clone();
        let mut r2 = s2.clone();
        for i in 0..2 {
            r1 = round(r1, i);
            r2 = round(r2, i);
        }
        let dist = hamming_distance(&r1, &r2);
        let ratio = avalanche_ratio(dist);
        assert!(
            ratio > 0.40,
            "two-round avalanche too weak: {:.2}% bits flipped",
            ratio * 100.0
        );
    }

    #[test]
    fn round_avalanche_four_rounds() {
        let s = fixed_state(1337);
        let mut s2 = s.clone();
        s2.s1[0] ^= 1;
        let mut r1 = s;
        let mut r2 = s2;
        for i in 0..4 {
            r1 = round(r1, i);
            r2 = round(r2, i);
        }
        let dist = hamming_distance(&r1, &r2);
        let ratio = avalanche_ratio(dist);
        assert!(
            ratio > 0.45,
            "four-round avalanche too weak: {:.2}% bits flipped",
            ratio * 100.0
        );
    }

    #[test]
    fn dfa_diffusion_detectable() {
        let s = fixed_state(555);
        let clean = round(s.clone(), 1);
        let mut faulty_s = s;
        inject_bit_fault(&mut faulty_s, 3, 12);
        let faulty = round(faulty_s, 1);
        let dist = hamming_distance(&clean, &faulty);
        assert!(dist > 0, "fault produced zero diffusion — DFA hardening not functioning");
    }

    #[test]
    fn round_differs_from_input() {
        let s = fixed_state(0xdeadbeef);
        let out = round(s.clone(), 0);
        let identical = (0..25).all(|i| out.s1[i] == s.s1[i] && out.s2[i] == s.s2[i]);
        assert!(!identical, "round produced no change");
    }
}
