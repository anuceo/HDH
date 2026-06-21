mod framework;
mod hdh;
mod tests;

fn main() {
    let seed: u64 = 0xDEADBEEF_CAFEBABE;

    let mut report = framework::TestReport::new(seed);
    report.extend(tests::correctness::run(seed));
    report.extend(tests::structural::run(seed));
    report.extend(tests::security::run(seed));
    report.extend(tests::implementation::run(seed));
    report.print();
}
