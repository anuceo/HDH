use crate::state::State;

pub fn phi(state: &State) -> State {
    let mut out = state.clone();

    for i in 0..25 {
        let j = (state.s1[i] ^ state.s2[i]) as usize % 25;
        let k = (state.s3[i] ^ state.s4[i]) as usize % 25;

        out.s1[i] ^= state.s2[j];
        out.s2[i] ^= state.s3[k];
        out.s3[i] ^= state.s4[j];
        out.s4[i] ^= state.s1[k];
    }

    out
}
