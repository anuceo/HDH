use hdh::{chi, round};
use hdh::hash::{hash, prf, CAPACITY_BYTES, OUTPUT_BYTES};

fn fixed_state(seed: u64) -> hdh::state::State {
    use std::num::Wrapping;
    let mut v = Wrapping(seed);
    let mut fill = || {
        v = v * Wrapping(6364136223846793005) + Wrapping(1442695040888963407);
        v.0
    };
    let mut arr = || std::array::from_fn::<u64, 25, _>(|_| fill());
    hdh::state::State {
        s1: arr(),
        s2: arr(),
        s3: arr(),
        s4: arr(),
        parity: [0u64; 25],
    }
}

fn print_u64_hex(label: &str, v: u64) {
    println!("    // {label}: 0x{v:016x}");
}

fn print_bytes(label: &str, data: &[u8]) {
    print!("    // {label}:");
    for (i, b) in data.iter().enumerate() {
        if i % 8 == 0 { print!("\n    //   "); }
        print!("0x{b:02x}, ");
    }
    println!();
}

fn main() {
    let s = fixed_state(0x0123456789abcdef);

    // --- chi KAT (unchanged, just verify) ---
    let chi_out = chi::chi(&s);
    println!("=== chi KAT ===");
    print_u64_hex("chi s1[0]",     chi_out.s1[0]);
    print_u64_hex("chi s2[0]",     chi_out.s2[0]);
    print_u64_hex("chi s3[0]",     chi_out.s3[0]);
    print_u64_hex("chi s4[0]",     chi_out.s4[0]);
    print_u64_hex("chi parity[0]", chi_out.parity[0]);
    print_u64_hex("chi s1[12]",    chi_out.s1[12]);
    print_u64_hex("chi s2[24]",    chi_out.s2[24]);

    // --- round 0 KAT ---
    let r0 = round(s.clone(), 0);
    println!("\n=== round 0 KAT ===");
    print_u64_hex("r0 s1[0]",     r0.s1[0]);
    print_u64_hex("r0 s2[0]",     r0.s2[0]);
    print_u64_hex("r0 s3[0]",     r0.s3[0]);
    print_u64_hex("r0 s4[0]",     r0.s4[0]);
    print_u64_hex("r0 parity[0]", r0.parity[0]);
    print_u64_hex("r0 s1[12]",    r0.s1[12]);
    print_u64_hex("r0 s2[24]",    r0.s2[24]);

    // --- round 1 KAT ---
    let r1 = round(s.clone(), 1);
    println!("\n=== round 1 KAT ===");
    print_u64_hex("r1 s1[0]",     r1.s1[0]);
    print_u64_hex("r1 s2[0]",     r1.s2[0]);
    print_u64_hex("r1 s3[0]",     r1.s3[0]);
    print_u64_hex("r1 s4[0]",     r1.s4[0]);
    print_u64_hex("r1 parity[0]", r1.parity[0]);
    print_u64_hex("r1 s1[12]",    r1.s1[12]);
    print_u64_hex("r1 s2[24]",    r1.s2[24]);

    // --- 4-round chain KAT ---
    let mut chain = s.clone();
    for i in 0..4 {
        chain = round(chain, i);
    }
    println!("\n=== 4-round chain KAT ===");
    print_u64_hex("4r s1[0]",     chain.s1[0]);
    print_u64_hex("4r s2[0]",     chain.s2[0]);
    print_u64_hex("4r s3[0]",     chain.s3[0]);
    print_u64_hex("4r s4[0]",     chain.s4[0]);
    print_u64_hex("4r parity[0]", chain.parity[0]);
    print_u64_hex("4r s1[12]",    chain.s1[12]);
    print_u64_hex("4r s2[24]",    chain.s2[24]);

    // --- hash KATs ---
    let empty_hash = hash(b"");
    println!("\n=== hash(empty) KAT ===");
    print_bytes("hash(b\"\")", &empty_hash);

    let abc_hash = hash(b"abc");
    println!("\n=== hash(\"abc\") KAT ===");
    print_bytes("hash(b\"abc\")", &abc_hash);

    // --- PRF KAT ---
    let mut key = [0u8; CAPACITY_BYTES];
    key[0] = 0x01;
    let prf_out = prf(&key, b"abc");
    println!("\n=== prf KAT ===");
    print_bytes("prf([0x01,0..], b\"abc\")", &prf_out);

    // Print in array literal form for easy copy-paste
    println!("\n=== COPY-PASTE ARRAYS ===");
    println!("hash(empty):");
    print_array_literal(&empty_hash);
    println!("hash(abc):");
    print_array_literal(&abc_hash);
    println!("prf([0x01,...],abc):");
    print_array_literal(&prf_out);
}

fn print_array_literal(data: &[u8]) {
    print!("    [");
    for (i, b) in data.iter().enumerate() {
        if i % 8 == 0 { print!("\n        "); }
        print!("0x{b:02x}, ");
    }
    println!("\n    ]");
}
