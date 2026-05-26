#[cfg(test)]
mod tests {
    use crate::dfa::inject_bit_fault;
    use crate::state::State;
    use crate::stats::{avalanche_ratio, hamming_distance};
    use crate::{chi, phi, theta, round};

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
        // θ is linear and per-share; parity must satisfy
        // parity_out[i] = parity[i] ^ parity[prev] ^ parity[next].
        let s = chi::chi(&fixed_state(31415));
        let out = theta::theta(&s);
        for i in 0..25 {
            let prev = (i + 24) % 25;
            let next = (i + 1) % 25;
            let expected = s.parity[prev] ^ s.parity[i] ^ s.parity[next];
            assert_eq!(out.parity[i], expected, "theta parity mismatch at lane {i}");
            // also verify parity matches actual share XOR
            let actual = out.s1[i] ^ out.s2[i] ^ out.s3[i] ^ out.s4[i];
            assert_eq!(out.parity[i], actual, "theta share parity broken at lane {i}");
        }
    }

    #[test]
    fn theta_cross_lane_diffusion() {
        // a single-lane perturbation must reach its two neighbours after θ
        let s = fixed_state(271828);
        let mut perturbed = s.clone();
        perturbed.s1[12] ^= 1;
        let a = theta::theta(&s);
        let b = theta::theta(&perturbed);
        // lanes 11, 12, 13 must differ; all others must be identical
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
        let a = phi::phi(&s);
        let b = phi::phi(&s);
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
        // θ spreads each chi-output perturbation to 3 lanes before φ routes it
        // further; a single round achieves ~10-16% flipped bits in practice.
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
        // θ expands each round's reach to 3 lanes; after two rounds the ring
        // diffusion covers the full 25-lane state and φ completes the mixing.
        // Two rounds reliably reach full (~50%) avalanche.
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
        // four rounds of χ→θ→φ converge near the 50% ideal; tight bound
        // confirms sustained saturation rather than a one-round spike.
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
        // a single injected bit must produce measurable diffusion
        assert!(dist > 0, "fault produced zero diffusion — DFA hardening not functioning");
    }

    #[test]
    fn round_differs_from_input() {
        let s = fixed_state(0xdeadbeef);
        let out = round(s.clone(), 0);
        // output should not be identical to input across all shares
        let identical = (0..25).all(|i| out.s1[i] == s.s1[i] && out.s2[i] == s.s2[i]);
        assert!(!identical, "round produced no change");
    }

    // ── Known Answer Tests ──────────────────────────────────────────────────────
    //
    // Fixed input → exact expected output. Any change to a rotation constant,
    // round constant derivation, or lane ordering will break these immediately,
    // unlike statistical tests which only check aggregate distributions.
    //
    // Values computed from fixed_state(0x0123456789abcdef) via gen_kat.

    #[test]
    fn chi_known_answer() {
        let s = fixed_state(0x0123456789abcdef);
        let out = chi::chi(&s);
        assert_eq!(out.s1[0],     0xafac5c8c8eff0c8c, "chi s1[0] mismatch");
        assert_eq!(out.s2[0],     0xcac68eb543f99550, "chi s2[0] mismatch");
        assert_eq!(out.s3[0],     0x6902af9049b860ad, "chi s3[0] mismatch");
        assert_eq!(out.s4[0],     0x02bb7b48a924d40c, "chi s4[0] mismatch");
        assert_eq!(out.parity[0], 0x0ed306e12d9a2d7d, "chi parity[0] mismatch");
        assert_eq!(out.s1[12],    0xe5c5d1b277bb1438, "chi s1[12] mismatch");
        assert_eq!(out.s2[24],    0x435bb73d238da941, "chi s2[24] mismatch");
    }

    #[test]
    fn round_known_answer_r0() {
        let s = fixed_state(0x0123456789abcdef);
        let out = round(s, 0);
        assert_eq!(out.s1[0],     0x1677231c61220912, "round0 s1[0] mismatch");
        assert_eq!(out.s2[0],     0x9a3529301850d22a, "round0 s2[0] mismatch");
        assert_eq!(out.s3[0],     0x4c9de3100f71e878, "round0 s3[0] mismatch");
        assert_eq!(out.s4[0],     0x7c35d6bca531526e, "round0 s4[0] mismatch");
        assert_eq!(out.parity[0], 0x4dc8300ceb4a9aa1, "round0 parity[0] mismatch");
        assert_eq!(out.s1[12],    0xc08c64643eb87574, "round0 s1[12] mismatch");
        assert_eq!(out.s2[24],    0x465e4d18b806efa6, "round0 s2[24] mismatch");
    }

    #[test]
    fn round_known_answer_r1() {
        let s = fixed_state(0x0123456789abcdef);
        let out = round(s, 1);
        assert_eq!(out.s1[0],     0xda8da8fffaf8424a, "round1 s1[0] mismatch");
        assert_eq!(out.s2[0],     0x96b8010a521c801c, "round1 s2[0] mismatch");
        assert_eq!(out.s3[0],     0xc6747804a4011c1d, "round1 s3[0] mismatch");
        assert_eq!(out.s4[0],     0x36623a323e218c6b, "round1 s4[0] mismatch");
        assert_eq!(out.parity[0], 0xd33525d3b6cb576b, "round1 parity[0] mismatch");
        assert_eq!(out.s1[12],    0xfc0e1c205bfdfd6a, "round1 s1[12] mismatch");
        assert_eq!(out.s2[24],    0xab78b015265214dd, "round1 s2[24] mismatch");
    }

    #[test]
    fn four_round_chain_known_answer() {
        let s = fixed_state(0x0123456789abcdef);
        let mut out = s;
        for i in 0..4 {
            out = round(out, i);
        }
        assert_eq!(out.s1[0],     0xf2ab96e88584de3b, "4-round s1[0] mismatch");
        assert_eq!(out.s2[0],     0xb56d7205d067a37e, "4-round s2[0] mismatch");
        assert_eq!(out.s3[0],     0x9ba69bca39458315, "4-round s3[0] mismatch");
        assert_eq!(out.s4[0],     0x44d6043b4a98c338, "4-round s4[0] mismatch");
        assert_eq!(out.parity[0], 0x72f0e6ed239398f0, "4-round parity[0] mismatch");
        assert_eq!(out.s1[12],    0x2a9342d68bd31ac0, "4-round s1[12] mismatch");
        assert_eq!(out.s2[24],    0x98aa4ebcfb57e8d3, "4-round s2[24] mismatch");
    }

    #[test]
    fn round_idx_produces_distinct_outputs() {
        // Different round indices derive different round constants via BLAKE3,
        // so the same input state must produce distinct outputs for each index.
        let s = fixed_state(0x0123456789abcdef);
        let r0 = round(s.clone(), 0);
        let r1 = round(s.clone(), 1);
        let r2 = round(s.clone(), 2);
        assert_ne!(r0.s1[0], r1.s1[0], "round 0 and 1 produced identical s1[0]");
        assert_ne!(r1.s1[0], r2.s1[0], "round 1 and 2 produced identical s1[0]");
        assert_ne!(r0.s1[0], r2.s1[0], "round 0 and 2 produced identical s1[0]");
    }
}
