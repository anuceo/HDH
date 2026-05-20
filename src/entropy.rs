use blake3::hash;

pub fn entropy_mix(data: &[u8]) -> u64 {
    let h = hash(data);
    let bytes = h.as_bytes();
    u64::from_le_bytes(bytes[0..8].try_into().unwrap())
}
