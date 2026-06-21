/// Testing framework: result types, test-case builder, and report printer.

// ── Result type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Pass,
    Fail,
    Inconclusive,
}

impl Verdict {
    fn label(&self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Inconclusive => "INCONCLUSIVE",
        }
    }
    fn icon(&self) -> &'static str {
        match self {
            Verdict::Pass => "✓",
            Verdict::Fail => "✗",
            Verdict::Inconclusive => "~",
        }
    }
}

// ── Test case ─────────────────────────────────────────────────────────────────

pub struct TestCase {
    pub id: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub verdict: Verdict,
    /// Measured data, computed values, or logical proof supporting the verdict.
    pub evidence: String,
    /// Non-None only on FAIL: describes how the measurement deviates from the claim.
    pub deviation: Option<String>,
}

impl TestCase {
    pub fn pass(
        id: &'static str,
        cat: &'static str,
        desc: &'static str,
        evidence: impl Into<String>,
    ) -> Self {
        TestCase {
            id, category: cat, description: desc,
            verdict: Verdict::Pass,
            evidence: evidence.into(),
            deviation: None,
        }
    }

    pub fn fail(
        id: &'static str,
        cat: &'static str,
        desc: &'static str,
        evidence: impl Into<String>,
        deviation: impl Into<String>,
    ) -> Self {
        TestCase {
            id, category: cat, description: desc,
            verdict: Verdict::Fail,
            evidence: evidence.into(),
            deviation: Some(deviation.into()),
        }
    }

    pub fn inconclusive(
        id: &'static str,
        cat: &'static str,
        desc: &'static str,
        evidence: impl Into<String>,
    ) -> Self {
        TestCase {
            id, category: cat, description: desc,
            verdict: Verdict::Inconclusive,
            evidence: evidence.into(),
            deviation: None,
        }
    }
}

// ── Report ────────────────────────────────────────────────────────────────────

pub struct TestReport {
    pub cases: Vec<TestCase>,
    pub seed: u64,
}

impl TestReport {
    pub fn new(seed: u64) -> Self {
        TestReport { cases: Vec::new(), seed }
    }

    pub fn add(&mut self, c: TestCase) {
        self.cases.push(c);
    }

    pub fn extend(&mut self, cs: Vec<TestCase>) {
        self.cases.extend(cs);
    }

    pub fn print(&self) {
        let total = self.cases.len();
        let passed = self.cases.iter().filter(|c| c.verdict == Verdict::Pass).count();
        let failed = self.cases.iter().filter(|c| c.verdict == Verdict::Fail).count();
        let inco = total - passed - failed;

        let w = 74usize;
        let bar: String = std::iter::repeat('═').take(w).collect();
        let thin: String = std::iter::repeat('─').take(w).collect();

        println!("╔{bar}╗");
        println!("║{:^width$}║", "HDH Algorithm Testing Framework — Results Report", width = w);
        println!("║{:^width$}║", format!("Deterministic seed: 0x{:016x}", self.seed), width = w);
        println!("╠{bar}╣");
        println!(
            "║  Total: {:2}   ✓ Passed: {:2}   ✗ Failed: {:2}   ~ Inconclusive: {:2}{:>pad$}║",
            total, passed, failed, inco,
            "",
            pad = w - 52
        );
        println!("╚{bar}╝");

        for cat in &[
            "Correctness",
            "Structural Validity",
            "Security Claim Validation",
            "Implementation Quality",
        ] {
            let cat_tests: Vec<_> = self.cases.iter().filter(|c| c.category == *cat).collect();
            if cat_tests.is_empty() {
                continue;
            }
            println!();
            println!("┌─ {cat} {}", std::iter::repeat('─').take(w.saturating_sub(cat.len() + 3)).collect::<String>());
            for t in &cat_tests {
                println!(
                    "│  {} [{:<4}] {:12} {}",
                    t.verdict.icon(), t.id, t.verdict.label(), t.description
                );
                // wrap evidence at ~68 chars (char-boundary safe)
                let ev = &t.evidence;
                let prefix = "│              ";
                if ev.chars().count() <= 68 {
                    println!("│              Evidence: {ev}");
                } else {
                    let mut pos = 0;
                    let mut first = true;
                    while pos < ev.len() {
                        let want = pos + 68;
                        let end = if want >= ev.len() {
                            ev.len()
                        } else {
                            let mut e = want;
                            while e > pos && !ev.is_char_boundary(e) { e -= 1; }
                            e
                        };
                        if end == pos { break; }
                        if first {
                            println!("│              Evidence: {}", &ev[pos..end]);
                            first = false;
                        } else {
                            println!("{prefix}         {}", &ev[pos..end]);
                        }
                        pos = end;
                    }
                }
                if let Some(dev) = &t.deviation {
                    println!("│              ! Deviation: {dev}");
                }
            }
            println!("└{thin}");
        }

        println!();
        println!("┌─ Security Assessment {}", std::iter::repeat('─').take(52).collect::<String>());

        // Summarise each design claim
        let claims: &[(&str, &str, &[&str])] = &[
            ("Algebraic degree growth ~3^r",  "SC1", &[]),
            ("B(θ) = 6 (differential bound)", "S1",  &["SC2"]),
            ("Classical collision: 256 bits",  "SC3", &["SC5"]),
            ("Classical preimage:  256 bits",  "SC4", &["SC5"]),
            ("Quantum collision: ~170.7 bits", "SC3", &[]),
            ("Bias / uniformity",              "SC5", &[]),
            ("Rotational resistance",          "SC6", &[]),
        ];

        for (claim, primary, extra) in claims {
            let verdicts: Vec<_> = std::iter::once(*primary)
                .chain(extra.iter().copied())
                .filter_map(|id| self.cases.iter().find(|c| c.id == id))
                .map(|c| &c.verdict)
                .collect();

            let icon = if verdicts.iter().any(|v| **v == Verdict::Pass) {
                "✓"
            } else if verdicts.iter().any(|v| **v == Verdict::Fail) {
                "✗"
            } else {
                "~"
            };

            let ids: Vec<_> = std::iter::once(*primary).chain(extra.iter().copied()).collect();
            println!("│  {icon}  {claim:<42} [{}]", ids.join(", "));
        }

        println!("│");
        println!("│  Recommended further analysis:");
        println!("│  • Full MILP trail search for tighter differential bounds (SC2)");
        println!("│  • NIST STS-800-22 suite on ≥1 Mbit of output (SC5)");
        println!("│  • Hardware timing counters for confirmed constant-time (I1)");
        println!("│  • Algebraic degree verification on full 64-bit model (SC1)");
        println!("└{thin}");
    }
}

// ── Statistical utilities ─────────────────────────────────────────────────────

/// Approximate complementary error function (Abramowitz & Stegun 7.1.26).
pub fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let p = t * (0.254_829_592
        + t * (-0.284_496_736
        + t * (1.421_413_741
        + t * (-1.453_152_027
        + t * 1.061_405_429))));
    let r = p * (-x * x).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// p-value for the NIST SP 800-22 monobit (frequency) test.
/// p < 0.01 → reject uniformity at α = 0.01.
pub fn monobit_pvalue(ones: u64, total_bits: u64) -> f64 {
    let n = total_bits as f64;
    let z = ((ones as f64 - n / 2.0) / (n / 4.0).sqrt()).abs();
    erfc(z / std::f64::consts::SQRT_2)
}

/// χ² statistic for a uniform distribution over `counts`.
pub fn chi2_stat(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    let k = counts.len() as f64;
    let expected = total as f64 / k;
    counts.iter().map(|&c| {
        let d = c as f64 - expected;
        d * d / expected
    }).sum()
}

/// Approximate p-value for χ² with `df` degrees of freedom
/// via Wilson–Hilferty normal approximation.
/// Returns P(X > chi2), so p < 0.01 → reject at α = 0.01.
pub fn chi2_pvalue(chi2: f64, df: f64) -> f64 {
    let mu = 1.0 - 2.0 / (9.0 * df);
    let sigma = (2.0 / (9.0 * df)).sqrt();
    let z = ((chi2 / df).powf(1.0 / 3.0) - mu) / sigma;
    if z > 0.0 {
        0.5 * erfc(z / std::f64::consts::SQRT_2)
    } else {
        1.0 - 0.5 * erfc(-z / std::f64::consts::SQRT_2)
    }
}

/// Möbius/butterfly transform over GF(2)^8: converts truth-table → ANF.
pub fn anf_transform(tt: &[u8; 256]) -> [u8; 256] {
    let mut f = *tt;
    for i in 0..8usize {
        let bit = 1usize << i;
        for x in 0..256usize {
            if x & bit != 0 {
                f[x] ^= f[x ^ bit];
            }
        }
    }
    f
}

/// Maximum algebraic degree of a Boolean function given its ANF coefficients.
pub fn max_anf_degree(anf: &[u8; 256]) -> usize {
    anf.iter()
        .enumerate()
        .filter(|(_, &c)| c != 0)
        .map(|(x, _)| x.count_ones() as usize)
        .max()
        .unwrap_or(0)
}
