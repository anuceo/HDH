pub mod attacks;
pub mod chi;
pub mod dfa;
pub mod entropy;
pub mod hash;
pub mod mask;
pub mod phi;
pub mod state;
pub mod stats;
pub mod theta;

#[cfg(test)]
mod tests;

use blake3::hash;
use state::State;

pub fn round(mut s: State, round_idx: u64) -> State {
    let digest = hash(&round_idx.to_le_bytes());
    let b = digest.as_bytes();
    let r0 = u64::from_le_bytes(b[ 0.. 8].try_into().unwrap());
    let r1 = u64::from_le_bytes(b[ 8..16].try_into().unwrap());
    let r2 = u64::from_le_bytes(b[16..24].try_into().unwrap());
    let r3 = u64::from_le_bytes(b[24..32].try_into().unwrap());

    for i in 0..25 {
        s.s1[i] ^= r0;
        s.s2[i] ^= r1;
        s.s3[i] ^= r2;
        s.s4[i] ^= r3;
        s.parity[i] = s.s1[i] ^ s.s2[i] ^ s.s3[i] ^ s.s4[i];
    }

    s = chi::chi(&s);
    s = theta::theta(&s);
    s = phi::phi(&s);

    s
}
