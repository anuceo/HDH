use crate::state::State;

/// Linear ring-diffusion layer.
///
/// Each lane XORs with its four neighbours at offsets ±1 and ±7 (mod 25).
/// gcd(7, 25) = 1, so the ±7 stride generates the full ring.  Using two
/// independent generators reduces the diffusion diameter from 12 (±1 only)
/// to ≈7, meaning a single-lane perturbation reaches all 25 lanes in ≈7
/// rounds instead of 12.  θ remains a per-share linear map so masking is
/// preserved, and the parity invariant becomes:
///   parity_out[i] = parity[i] ^ parity[prev] ^ parity[next] ^ parity[far_a] ^ parity[far_b]
pub fn theta(state: &State) -> State {
    let mut out = state.clone();

    for i in 0..25 {
        let prev  = (i + 24) % 25;   // i - 1
        let next  = (i + 1)  % 25;   // i + 1
        let far_a = (i + 7)  % 25;   // i + 7
        let far_b = (i + 18) % 25;   // i - 7

        out.s1[i] ^= state.s1[prev] ^ state.s1[next] ^ state.s1[far_a] ^ state.s1[far_b];
        out.s2[i] ^= state.s2[prev] ^ state.s2[next] ^ state.s2[far_a] ^ state.s2[far_b];
        out.s3[i] ^= state.s3[prev] ^ state.s3[next] ^ state.s3[far_a] ^ state.s3[far_b];
        out.s4[i] ^= state.s4[prev] ^ state.s4[next] ^ state.s4[far_a] ^ state.s4[far_b];

        out.parity[i] = state.parity[i] ^ state.parity[prev] ^ state.parity[next]
                      ^ state.parity[far_a] ^ state.parity[far_b];
    }

    out
}
