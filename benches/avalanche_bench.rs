use hdh::dfa::inject_bit_fault;
use hdh::state::State;
use hdh::stats::{avalanche_ratio, hamming_distance};
use hdh::round;

fn random_state() -> State {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    State {
        s1: rng.gen(),
        s2: rng.gen(),
        s3: rng.gen(),
        s4: rng.gen(),
        parity: [0u64; 25],
    }
}

fn avalanche_test() {
    let s = random_state();
    let mut s2 = s.clone();
    s2.s1[0] ^= 1;

    let r1 = round(s, 1);
    let r2 = round(s2, 1);

    let dist = hamming_distance(&r1, &r2);
    let ratio = avalanche_ratio(dist);
    println!(
        "Avalanche — Hamming distance: {dist}  ({:.1}% of bits flipped)",
        ratio * 100.0
    );
}

fn dfa_test() {
    let s = random_state();
    let clean = round(s.clone(), 1);

    let mut faulty_s = s;
    inject_bit_fault(&mut faulty_s, 3, 12);
    let faulty = round(faulty_s, 1);

    let dist = hamming_distance(&clean, &faulty);
    let ratio = avalanche_ratio(dist);
    println!(
        "DFA diffusion — Hamming distance: {dist}  ({:.1}% of bits flipped)",
        ratio * 100.0
    );
}

fn multi_round_avalanche(rounds: u64) {
    let s = random_state();
    let mut s2 = s.clone();
    s2.s1[0] ^= 1;

    let mut r1 = s;
    let mut r2 = s2;
    for i in 0..rounds {
        r1 = round(r1, i);
        r2 = round(r2, i);
    }

    let dist = hamming_distance(&r1, &r2);
    let ratio = avalanche_ratio(dist);
    println!(
        "{rounds}-round avalanche — Hamming distance: {dist}  ({:.1}% of bits flipped)",
        ratio * 100.0
    );
}

fn main() {
    println!("=== HDH DFA-Hardened χ Core — Validation Harness ===\n");

    for _ in 0..5 {
        avalanche_test();
    }

    println!();
    for _ in 0..5 {
        dfa_test();
    }

    println!();
    for r in [1, 2, 4, 8] {
        multi_round_avalanche(r);
    }
}
