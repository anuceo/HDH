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
