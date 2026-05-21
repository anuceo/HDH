/// HDH — DFA-Hardened Hash Function library.
///
/// Two top-level modules:
///
/// - [`algorithm`] — the permutation and hash construction.
/// - [`attacks`]   — cryptanalysis and security-property verification.
///
/// # Quick start
///
/// ```rust
/// use hdh::algorithm::hdh_hash_256;
///
/// let digest = hdh_hash_256(b"hello world");
/// assert_eq!(digest.len(), 32);
/// ```

pub mod algorithm;
pub mod attacks;
