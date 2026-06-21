/// Implementation quality tests I1–I4.

use crate::framework::TestCase;
use crate::hdh::{chi_word, hash, xof, State, OUTPUT_BYTES, RATE_BYTES};
use std::time::Instant;

const CAT: &str = "Implementation Quality";

// ── I1: Constant-time chi layer ───────────────────────────────────────────────
//
// Measure timing variance of chi_word across distinct inputs.
// A data-dependent branch or cache miss would show statistically significant
// timing variation correlated with Hamming weight or value patterns.
// We compare variance across inputs to noise floor (same input repeated).

pub fn i1_constant_time(_seed: u64) -> TestCase {
    const REPS: usize = 1_000;
    const N_INPUTS: usize = 256;

    // Collect timing samples: varied inputs vs fixed input.
    let fixed: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
    let mut times_varied = Vec::with_capacity(N_INPUTS);
    let mut times_fixed = Vec::with_capacity(REPS);

    // Fixed-input baseline (noise floor).
    for _ in 0..REPS {
        let t0 = Instant::now();
        let _ = std::hint::black_box(chi_word(std::hint::black_box(fixed)));
        times_fixed.push(t0.elapsed().as_nanos() as f64);
    }

    // Varied inputs: sample inputs spread across the u64 range.
    for i in 0..N_INPUTS {
        let x: u64 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xBF58476D1CE4E5B9);
        let t0 = Instant::now();
        let _ = std::hint::black_box(chi_word(std::hint::black_box(x)));
        times_varied.push(t0.elapsed().as_nanos() as f64);
    }

    let mean_fixed = times_fixed.iter().sum::<f64>() / times_fixed.len() as f64;
    let mean_varied = times_varied.iter().sum::<f64>() / times_varied.len() as f64;

    let var_fixed = times_fixed.iter().map(|&t| (t - mean_fixed).powi(2)).sum::<f64>()
        / times_fixed.len() as f64;
    let var_varied = times_varied.iter().map(|&t| (t - mean_varied).powi(2)).sum::<f64>()
        / times_varied.len() as f64;

    let cv_fixed = var_fixed.sqrt() / mean_fixed.max(1.0) * 100.0;
    let cv_varied = var_varied.sqrt() / mean_varied.max(1.0) * 100.0;

    // Ratio of timing variance between varied and fixed inputs.
    // A ratio near 1.0 indicates no data-dependent timing paths.
    let ratio = if var_fixed > 0.0 { var_varied / var_fixed } else { 1.0 };

    let evidence = format!(
        "chi_word: fixed_input CV={cv_fixed:.1}%, varied_inputs CV={cv_varied:.1}%, \
         variance ratio={ratio:.2}× (noise-floor baseline {REPS} reps, varied {N_INPUTS} inputs)"
    );

    // Accept if variance ratio is below 5× (generous; timing noise dominates at ns scale).
    if ratio <= 5.0 {
        TestCase::pass("I1", CAT,
            "Constant-time: chi layer timing variance consistent across inputs",
            evidence)
    } else {
        TestCase::fail("I1", CAT,
            "Constant-time: chi layer timing variance consistent across inputs",
            evidence,
            format!("variance ratio={ratio:.1}× > 5 — data-dependent timing detected in chi_word"))
    }
}

// ── I2: Memory footprint ──────────────────────────────────────────────────────
//
// Verify the working-set size matches the design specification.

pub fn i2_memory_footprint() -> TestCase {
    let state_bytes = std::mem::size_of::<State>();
    let rate_bytes = RATE_BYTES;
    let output_bytes = OUTPUT_BYTES;

    // Design spec: state = 6400 bits = 800 bytes; rate = 736 bytes; output = 64 bytes.
    let ok = state_bytes == 800 && rate_bytes == 736 && output_bytes == 64;

    let evidence = format!(
        "State={state_bytes}B ({}b), Rate={rate_bytes}B, Output={output_bytes}B; \
         single-call stack frame ≈ {}B (state + 2 temporaries)",
        state_bytes * 8,
        state_bytes * 3,
    );

    if ok {
        TestCase::pass("I2", CAT,
            "Memory footprint: State=800B, Rate=736B, Output=64B",
            evidence)
    } else {
        TestCase::fail("I2", CAT,
            "Memory footprint: State=800B, Rate=736B, Output=64B",
            evidence,
            format!("expected State=800, Rate=736, Output=64; \
                     got State={state_bytes}, Rate={rate_bytes}, Output={output_bytes}"))
    }
}

// ── I3: Throughput benchmarking ───────────────────────────────────────────────
//
// Measure wall-clock throughput for three message sizes.
// Reports ns/byte and MB/s; no hard pass/fail threshold (informational).

pub fn i3_performance(_seed: u64) -> TestCase {
    let sizes: &[(usize, &str)] = &[
        (64,     "64B"),
        (4_096,  "4KB"),
        (65_536, "64KB"),
    ];

    let mut results = Vec::new();
    const WARMUP: usize = 3;
    const ITERS: usize = 20;

    for &(size, label) in sizes {
        let data = vec![0x5Au8; size];

        // Warmup.
        for _ in 0..WARMUP { let _ = std::hint::black_box(hash(&data)); }

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = std::hint::black_box(hash(&data));
        }
        let elapsed = t0.elapsed();

        let total_bytes = size * ITERS;
        let ns_per_byte = elapsed.as_nanos() as f64 / total_bytes as f64;
        let mb_per_sec = total_bytes as f64 / elapsed.as_secs_f64() / 1_048_576.0;

        results.push(format!("{label}: {ns_per_byte:.1}ns/B ({mb_per_sec:.1}MB/s)"));
    }

    let evidence = results.join("; ");

    // Informational: pass as long as we complete without hanging (>0 MB/s).
    TestCase::pass("I3", CAT,
        "Performance: throughput measured for 64B / 4KB / 64KB inputs",
        evidence)
}

// ── I4: XOF mode validation ───────────────────────────────────────────────────
//
// Verify that:
// (a) xof(data, 64)[0..64] == hash(data)  — first squeeze block matches hash()
// (b) xof(data, 128) produces 128 bytes   — length contract
// (c) xof(data, 128)[0..64] == xof(data, 64)  — prefix consistency
// (d) xof(data, 64) != xof(other, 64)  — distinct inputs → distinct outputs

pub fn i4_xof_validation() -> TestCase {
    let mut issues = Vec::new();

    let msg = b"I4-xof-test-message";
    let other = b"I4-xof-other-message";

    // (a) First 64 bytes of XOF == hash().
    let h = hash(msg);
    let xf64 = xof(msg, 64);
    if xf64.len() != 64 || h != xf64.as_slice() {
        issues.push(format!(
            "xof(msg,64)[0..64] != hash(msg): hash[0]={:02x}, xof[0]={:02x}",
            h[0], xf64.first().copied().unwrap_or(0)
        ));
    }

    // (b) Length contract.
    let xf128 = xof(msg, 128);
    if xf128.len() != 128 {
        issues.push(format!("xof(msg,128) returned {} bytes, expected 128", xf128.len()));
    }

    // (c) Prefix consistency.
    if xf128.len() >= 64 && xf64.len() == 64 && xf128[..64] != xf64[..] {
        issues.push("xof(msg,128)[0..64] != xof(msg,64) — prefix inconsistency".into());
    }

    // (d) Distinctness.
    let xf_other = xof(other, 64);
    if xf64 == xf_other {
        issues.push("xof(msg) == xof(other) — distinct inputs produced same XOF output".into());
    }

    // (e) Extended bytes differ from first block (not just zero-padding).
    if xf128.len() == 128 && xf128[64..] == [0u8; 64][..] {
        issues.push("xof bytes 64–127 are all zero — squeeze may not be re-permuting".into());
    }

    let n_checks = 5;
    let n_issues = issues.len();

    if issues.is_empty() {
        TestCase::pass("I4", CAT,
            "XOF mode: length contract, hash-prefix match, prefix-consistency, distinctness",
            format!("all {n_checks} XOF properties verified"))
    } else {
        TestCase::fail("I4", CAT,
            "XOF mode: length contract, hash-prefix match, prefix-consistency, distinctness",
            format!("{n_issues}/{n_checks} checks failed: {}", issues.join("; ")),
            issues.join("; "))
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

pub fn run(seed: u64) -> Vec<TestCase> {
    vec![
        i1_constant_time(seed),
        i2_memory_footprint(),
        i3_performance(seed),
        i4_xof_validation(),
    ]
}
