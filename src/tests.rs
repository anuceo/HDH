#[cfg(test)]
mod tests {
    use crate::dfa::inject_bit_fault;
    use crate::state::State;
    use crate::stats::{avalanche_ratio, hamming_distance};
    use crate::{chi, phi, round};

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
        // chi operates per-lane; without a dedicated theta-like diffusion layer
        // a single round diffuses within the lane across shares (~3-5% total).
        // This threshold confirms non-trivial diffusion begins in round 1.
        assert!(
            ratio > 0.01,
            "single-round avalanche absent: {:.2}% bits flipped",
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
        // phi propagates the perturbation to ~1/25 new lanes per round via
        // state-dependent indexing; two rounds yields measurably more diffusion
        // than one round but not yet full avalanche.
        assert!(
            ratio > 0.05,
            "two-round avalanche absent: {:.2}% bits flipped",
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
        // exponential lane contamination through phi reaches global diffusion
        // (~8+ lanes affected) by round 4; full avalanche expected here.
        assert!(
            ratio > 0.20,
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
}
