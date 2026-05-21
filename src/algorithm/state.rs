use blake3::hash;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand::RngCore;

pub const N: usize = 1600;

#[derive(Clone)]
pub struct State {
    pub s1: [u64; 25],
    pub s2: [u64; 25],
    pub s3: [u64; 25],
    pub s4: [u64; 25],
    pub parity: [u64; 25],
}

impl State {
    pub fn zero() -> Self {
        State {
            s1: [0u64; 25],
            s2: [0u64; 25],
            s3: [0u64; 25],
            s4: [0u64; 25],
            parity: [0u64; 25],
        }
    }
}

pub fn hamming_distance(a: &State, b: &State) -> u32 {
    let mut d = 0u32;

    for i in 0..25 {
        d += (a.s1[i] ^ b.s1[i]).count_ones();
        d += (a.s2[i] ^ b.s2[i]).count_ones();
        d += (a.s3[i] ^ b.s3[i]).count_ones();
        d += (a.s4[i] ^ b.s4[i]).count_ones();
    }

    d
}

pub fn avalanche_ratio(distance: u32) -> f64 {
    // total bits across 4 shares × 25 lanes × 64 bits
    let total_bits = (4 * 25 * 64) as f64;
    distance as f64 / total_bits
}

pub fn inject_bit_fault(state: &mut State, lane: usize, bit: u64) {
    let mask = 1u64 << (bit % 64);
    state.s1[lane] ^= mask;
}

pub fn entropy_mix(data: &[u8]) -> u64 {
    let h = hash(data);
    let bytes = h.as_bytes();
    u64::from_le_bytes(bytes[0..8].try_into().unwrap())
}

pub fn fresh_mask(seed: u64) -> u64 {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    rng.next_u64()
}
