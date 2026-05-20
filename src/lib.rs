pub mod attacks;
pub mod chi;
pub mod dfa;
pub mod entropy;
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
    let r = hash(&round_idx.to_le_bytes()).as_bytes()[0] as u64;

    for i in 0..25 {
        s.s1[i] ^= r;
        s.s2[i] ^= r.rotate_left(11);
        s.s3[i] ^= r.rotate_left(23);
        s.s4[i] ^= r.rotate_left(37);
    }

    s = chi::chi(&s);
    s = theta::theta(&s);
    s = phi::phi(&s);

    s
}
