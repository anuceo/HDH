use crate::state::State;

/// Flip a single bit in s1[lane] to simulate a physical fault injection.
pub fn inject_bit_fault(state: &mut State, lane: usize, bit: u64) {
    let mask = 1u64 << (bit % 64);
    state.s1[lane] ^= mask;
}

/// Returns true if the parity invariant holds across all 25 lanes.
///
/// A valid HDH state satisfies parity[i] = s1[i] ^ s2[i] ^ s3[i] ^ s4[i] for every i.
/// A single-share fault breaks this for the affected lane(s), making it detectable.
pub fn check_parity(state: &State) -> bool {
    (0..25).all(|i| state.parity[i] == state.s1[i] ^ state.s2[i] ^ state.s3[i] ^ state.s4[i])
}

/// Returns the indices of all lanes where the parity invariant is violated.
///
/// An empty vector means no fault is detected. A non-empty vector identifies
/// exactly which lanes have been corrupted, enabling targeted countermeasures.
pub fn faulted_lanes(state: &State) -> Vec<usize> {
    (0..25)
        .filter(|&i| state.parity[i] != state.s1[i] ^ state.s2[i] ^ state.s3[i] ^ state.s4[i])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::iv_state;

    #[test]
    fn clean_state_passes_parity_check() {
        let s = iv_state();
        assert!(check_parity(&s));
        assert!(faulted_lanes(&s).is_empty());
    }

    #[test]
    fn injected_fault_breaks_parity() {
        let mut s = iv_state();
        inject_bit_fault(&mut s, 3, 17);
        assert!(!check_parity(&s));
        assert!(faulted_lanes(&s).contains(&3));
    }

    #[test]
    fn only_faulted_lane_reported() {
        let mut s = iv_state();
        inject_bit_fault(&mut s, 7, 0);
        let bad = faulted_lanes(&s);
        assert_eq!(bad, vec![7]);
    }
}
