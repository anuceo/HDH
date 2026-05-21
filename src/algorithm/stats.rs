use super::state::State;

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
