use super::state::State;

pub fn quad(a: u64, b: u64, c: u64, d: u64) -> u64 {
    a.wrapping_mul(b) ^ c.wrapping_mul(d)
}

/// Single-lane χ: returns (out1, out2, out3, out4) for direct attack analysis.
pub fn chi_lane(x1: u64, x2: u64, x3: u64, x4: u64) -> (u64, u64, u64, u64) {
    let g = quad(x1, x2, x3, x4);
    (x1 ^ g, x2 ^ g.rotate_left(7), x3 ^ g.rotate_left(13), x4 ^ g.rotate_left(31))
}

pub fn chi(state: &State) -> State {
    let mut out = state.clone();

    for i in 0..25 {
        let x1 = state.s1[i];
        let x2 = state.s2[i];
        let x3 = state.s3[i];
        let x4 = state.s4[i];

        let g = quad(x1, x2, x3, x4);

        out.s1[i] ^= g;
        out.s2[i] ^= g.rotate_left(7);
        out.s3[i] ^= g.rotate_left(13);
        out.s4[i] ^= g.rotate_left(31);

        // parity binding for DFA resistance: any single-share fault flips parity
        out.parity[i] = out.s1[i] ^ out.s2[i] ^ out.s3[i] ^ out.s4[i];
    }

    out
}

/// Linear ring-diffusion layer.
///
/// Each lane XORs with its two immediate neighbors in a 25-lane ring,
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
