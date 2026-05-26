/// Differential ML distinguisher (Gohr-style) for reduced-round HDH.
///
/// Trains a logistic regression model to distinguish:
/// - Positive class: output XOR of (HDH-r(x), HDH-r(x ⊕ Δ)) with fixed Δ = {s1[0] ^= 1}
/// - Negative class: XOR of two independent random outputs
///
/// Features: 64 bits from s1[0] of the output XOR state, cast to f32 bits.
///
/// Uses mini-batch SGD with L2 regularization (Tikhonov).

use crate::attacks::distinguisher::{apply_rounds, random_state};
use crate::attacks::diff_bounds::xor_states;
use rand::Rng;

const N_FEATURES: usize = 64;
const EPOCHS: usize = 100;
const LR: f32 = 0.1;
const L2: f32 = 0.001;

// ── Logistic model ────────────────────────────────────────────────────────────

pub struct LogisticModel {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl LogisticModel {
    pub fn new(n_features: usize) -> Self {
        LogisticModel {
            weights: vec![0.0f32; n_features],
            bias: 0.0,
        }
    }

    pub fn predict(&self, x: &[f32]) -> f32 {
        let dot: f32 = x.iter().zip(self.weights.iter()).map(|(&xi, &wi)| xi * wi).sum();
        sigmoid(dot + self.bias)
    }

    /// Mini-batch SGD training.
    pub fn train_sgd(
        &mut self,
        features: &[Vec<f32>],
        labels: &[f32],
        epochs: usize,
        lr: f32,
        l2: f32,
    ) {
        let n = features.len();
        if n == 0 {
            return;
        }

        for _epoch in 0..epochs {
            let mut grad_w = vec![0.0f32; self.weights.len()];
            let mut grad_b = 0.0f32;

            for (x, &y) in features.iter().zip(labels.iter()) {
                let pred = self.predict(x);
                let err = pred - y;
                for (j, &xj) in x.iter().enumerate() {
                    grad_w[j] += err * xj;
                }
                grad_b += err;
            }

            let scale = lr / n as f32;
            for (j, w) in self.weights.iter_mut().enumerate() {
                *w -= scale * (grad_w[j] + l2 * *w);
            }
            self.bias -= scale * grad_b;
        }
    }

    pub fn accuracy(&self, features: &[Vec<f32>], labels: &[f32]) -> f64 {
        if features.is_empty() {
            return 0.0;
        }
        let correct = features.iter().zip(labels.iter())
            .filter(|(x, &y)| {
                let pred = self.predict(x);
                (pred >= 0.5) == (y >= 0.5)
            })
            .count();
        correct as f64 / features.len() as f64
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Feature extraction ────────────────────────────────────────────────────────

/// Extract N_FEATURES bits from s1[0] of the output XOR state as f32 values (0.0 or 1.0).
fn extract_features(xor_s1_0: u64) -> Vec<f32> {
    (0..N_FEATURES)
        .map(|bit| ((xor_s1_0 >> bit) & 1) as f32)
        .collect()
}

// ── Distinguisher result ─────────────────────────────────────────────────────

pub struct DistResult {
    pub rounds: usize,
    pub train_accuracy: f64,
    pub test_accuracy: f64,
    pub n_train: usize,
    pub n_test: usize,
    /// Distinguishable if test_accuracy > 0.55
    pub distinguishable: bool,
}

// ── Core distinguisher ────────────────────────────────────────────────────────

/// Run a logistic regression distinguisher for `rounds`-round HDH.
/// `n_train` samples per class for training, `n_test` samples per class for testing.
pub fn run_distinguisher(
    rounds: usize,
    n_train: usize,
    n_test: usize,
    rng: &mut impl Rng,
) -> DistResult {
    let total_train = n_train * 2;
    let total_test = n_test * 2;

    let mut train_features: Vec<Vec<f32>> = Vec::with_capacity(total_train);
    let mut train_labels: Vec<f32> = Vec::with_capacity(total_train);
    let mut test_features: Vec<Vec<f32>> = Vec::with_capacity(total_test);
    let mut test_labels: Vec<f32> = Vec::with_capacity(total_test);

    // Generate training data
    for _ in 0..n_train {
        // Positive: HDH pair with fixed Δ
        let feat = gen_positive_sample(rounds, rng);
        train_features.push(feat);
        train_labels.push(1.0);
    }
    for _ in 0..n_train {
        // Negative: two independent random outputs XOR'd
        let feat = gen_negative_sample(rounds, rng);
        train_features.push(feat);
        train_labels.push(0.0);
    }

    // Generate test data
    for _ in 0..n_test {
        let feat = gen_positive_sample(rounds, rng);
        test_features.push(feat);
        test_labels.push(1.0);
    }
    for _ in 0..n_test {
        let feat = gen_negative_sample(rounds, rng);
        test_features.push(feat);
        test_labels.push(0.0);
    }

    // Train model
    let mut model = LogisticModel::new(N_FEATURES);
    model.train_sgd(&train_features, &train_labels, EPOCHS, LR, L2);

    let train_accuracy = model.accuracy(&train_features, &train_labels);
    let test_accuracy = model.accuracy(&test_features, &test_labels);

    DistResult {
        rounds,
        train_accuracy,
        test_accuracy,
        n_train: total_train,
        n_test: total_test,
        distinguishable: test_accuracy > 0.55,
    }
}

fn gen_positive_sample(rounds: usize, rng: &mut impl Rng) -> Vec<f32> {
    let base = random_state(rng);
    let mut flipped = base.clone();
    flipped.s1[0] ^= 1u64;
    flipped.parity[0] = flipped.s1[0] ^ flipped.s2[0] ^ flipped.s3[0] ^ flipped.s4[0];

    let out_base = apply_rounds(base, rounds);
    let out_flipped = apply_rounds(flipped, rounds);
    let xor_out = xor_states(&out_base, &out_flipped);
    extract_features(xor_out.s1[0])
}

fn gen_negative_sample(rounds: usize, rng: &mut impl Rng) -> Vec<f32> {
    let out_a = apply_rounds(random_state(rng), rounds);
    let out_b = apply_rounds(random_state(rng), rounds);
    let xor_out = xor_states(&out_a, &out_b);
    extract_features(xor_out.s1[0])
}

// ── Sweep ──────────────────────────────────────────────────────────────────────

/// Run the distinguisher for rounds 1..=max_rounds.
pub fn distinguisher_sweep(
    max_rounds: usize,
    n_per_class: usize,
    rng: &mut impl Rng,
) -> Vec<DistResult> {
    (1..=max_rounds)
        .map(|r| {
            let n_test = (n_per_class / 4).max(100);
            run_distinguisher(r, n_per_class, n_test, rng)
        })
        .collect()
}
