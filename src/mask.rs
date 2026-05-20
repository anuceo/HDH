use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand::RngCore;

pub fn fresh_mask(seed: u64) -> u64 {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    rng.next_u64()
}
