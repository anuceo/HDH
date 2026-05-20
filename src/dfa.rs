use crate::state::State;

pub fn inject_bit_fault(state: &mut State, lane: usize, bit: u64) {
    let mask = 1u64 << (bit % 64);
    state.s1[lane] ^= mask;
}
