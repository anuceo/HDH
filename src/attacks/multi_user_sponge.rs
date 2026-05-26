/// Multi-user security analysis for the HDH sponge hash.
///
/// In the multi-user setting an adversary targets any one of U independent
/// hash instances (users) rather than a single target.  The advantage of
/// breaking at least one instance scales linearly in U:
///
/// Collision (generic sponge bound, Bertoni et al.):
///   Adv^{coll}_{U}(q) ≤ U · q² / 2^c
///   → log₂(Adv) = log₂(U) + 2·log₂(q) − c
///
/// PRF security (keyed sponge, key length k = c = 512 bits):
///   Adv^{PRF}_{U}(q) ≤ U · q / 2^{min(k, c/2)}
///   → log₂(Adv) = log₂(U) + log₂(q) − min(k, c/2)
///
/// At c = 512, U = 2^{32}, q = 2^{32} per user:
///   collision advantage ≤ 2^{32+64−512} = 2^{−416}   (negligible)
///   PRF advantage      ≤ 2^{32+32−256} = 2^{−192}    (negligible)

// ── Multi-user collision ──────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MultiUserCollisionBound {
    pub capacity_bits: usize,
    pub num_users_log2: u32,
    pub queries_per_user_log2: u32,
    /// log₂ of the upper bound on collision advantage.
    pub advantage_log2: f64,
    /// Effective security bits = −advantage_log2.
    pub security_bits: f64,
    /// Advantage below 2^{−128}.
    pub meets_128bit: bool,
}

pub fn multi_user_collision_bound(
    capacity_bits: usize,
    num_users_log2: u32,
    queries_per_user_log2: u32,
) -> MultiUserCollisionBound {
    let advantage_log2 =
        num_users_log2 as f64 + 2.0 * queries_per_user_log2 as f64 - capacity_bits as f64;
    let security_bits = -advantage_log2;
    MultiUserCollisionBound {
        capacity_bits,
        num_users_log2,
        queries_per_user_log2,
        advantage_log2,
        security_bits,
        meets_128bit: advantage_log2 <= -128.0,
    }
}

// ── Multi-user PRF ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MultiUserPrfBound {
    pub capacity_bits: usize,
    pub key_bits: usize,
    pub num_users_log2: u32,
    pub queries_per_user_log2: u32,
    /// log₂ of the upper bound on PRF advantage.
    pub advantage_log2: f64,
    pub security_bits: f64,
    pub meets_128bit: bool,
}

pub fn multi_user_prf_bound(
    capacity_bits: usize,
    key_bits: usize,
    num_users_log2: u32,
    queries_per_user_log2: u32,
) -> MultiUserPrfBound {
    // Effective key security = min(k, c/2).
    let eff = (key_bits as f64).min(capacity_bits as f64 / 2.0);
    let advantage_log2 =
        num_users_log2 as f64 + queries_per_user_log2 as f64 - eff;
    let security_bits = -advantage_log2;
    MultiUserPrfBound {
        capacity_bits,
        key_bits,
        num_users_log2,
        queries_per_user_log2,
        advantage_log2,
        security_bits,
        meets_128bit: advantage_log2 <= -128.0,
    }
}

// ── Advantage sweep ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MultiUserEntry {
    pub num_users_log2: u32,
    pub queries_per_user_log2: u32,
    pub collision_advantage_log2: f64,
    pub prf_advantage_log2: f64,
    pub meets_128bit_collision: bool,
    pub meets_128bit_prf: bool,
}

pub struct MultiUserSweep {
    pub capacity_bits: usize,
    pub key_bits: usize,
    pub entries: Vec<MultiUserEntry>,
}

pub fn multi_user_sweep(capacity_bits: usize, key_bits: usize) -> MultiUserSweep {
    let configs: &[(u32, u32)] = &[
        (0, 32),
        (16, 32),
        (32, 32),
        (32, 64),
        (64, 32),
        (64, 64),
    ];
    let entries = configs
        .iter()
        .map(|&(u, q)| {
            let col = multi_user_collision_bound(capacity_bits, u, q);
            let prf = multi_user_prf_bound(capacity_bits, key_bits, u, q);
            MultiUserEntry {
                num_users_log2: u,
                queries_per_user_log2: q,
                collision_advantage_log2: col.advantage_log2,
                prf_advantage_log2: prf.advantage_log2,
                meets_128bit_collision: col.meets_128bit,
                meets_128bit_prf: prf.meets_128bit,
            }
        })
        .collect();
    MultiUserSweep { capacity_bits, key_bits, entries }
}
