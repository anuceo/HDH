use crate::state::State;

/// Linear ring-diffusion layer.
///
/// Each lane XORs with its two immediate neighbours in a 25-lane ring,
/// applied independently to every share.  Because θ is a per-share linear
/// map, it does not couple shares (masking is preserved) and the parity
/// invariant transforms as parity_out[i] = parity[i]^parity[prev]^parity[next].
pub fn theta(state: &State) -> State {
    let mut out = state.clone();

    for i in 0..25 {
        let prev = (i + 24) % 25;
        let next = (i + 1) % 25;

        out.s1[i] ^= state.s1[prev] ^ state.s1[next];
        out.s2[i] ^= state.s2[prev] ^ state.s2[next];
        out.s3[i] ^= state.s3[prev] ^ state.s3[next];
        out.s4[i] ^= state.s4[prev] ^ state.s4[next];

        out.parity[i] = state.parity[prev] ^ state.parity[i] ^ state.parity[next];
    }

    out
}
