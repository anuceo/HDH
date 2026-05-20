use crate::state::State;

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
