//! Benchmarks for network derived properties
//!
//! Tests performance of passivity, reciprocity, and other derived calculations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::Network;
use std::f64::consts::PI;

/// Create a test network with random-ish S-parameters
fn create_test_network(nfreq: usize, nports: usize) -> Network {
    let freq = Frequency::new(1.0, 10.0, nfreq, FrequencyUnit::GHz, SweepType::Linear);

    let mut s = Array3::<Complex64>::zeros((nfreq, nports, nports));
    for f in 0..nfreq {
        for i in 0..nports {
            for j in 0..nports {
                // Create realistic S-parameter values
                let phase = 2.0 * PI * f as f64 / nfreq as f64;
                let mag = if i == j { 0.1 } else { 0.9 };
                s[[f, i, j]] = Complex64::from_polar(mag, phase * (i + j + 1) as f64);
            }
        }
    }

    let z0 = Array1::from_elem(nports, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0)
}

fn bench_passivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("passivity");

    for nfreq in [10, 100, 500, 1000].iter() {
        for nports in [2, 4, 8].iter() {
            let network = create_test_network(*nfreq, *nports);
            let id = BenchmarkId::new(format!("{}ports", nports), nfreq);

            group.bench_with_input(id, nfreq, |b, _| b.iter(|| black_box(network.passivity())));
        }
    }

    group.finish();
}

fn bench_reciprocity(c: &mut Criterion) {
    let mut group = c.benchmark_group("reciprocity");

    for nfreq in [10, 100, 500, 1000].iter() {
        for nports in [2, 4, 8].iter() {
            let network = create_test_network(*nfreq, *nports);
            let id = BenchmarkId::new(format!("{}ports", nports), nfreq);

            group.bench_with_input(id, nfreq, |b, _| {
                b.iter(|| black_box(network.reciprocity()))
            });
        }
    }

    group.finish();
}

fn bench_s_db(c: &mut Criterion) {
    let mut group = c.benchmark_group("s_db");

    for nfreq in [100, 500, 1000, 5000].iter() {
        let network = create_test_network(*nfreq, 4);
        let id = BenchmarkId::from_parameter(nfreq);

        group.bench_with_input(id, nfreq, |b, _| b.iter(|| black_box(network.s_db())));
    }

    group.finish();
}

fn bench_vswr(c: &mut Criterion) {
    let mut group = c.benchmark_group("vswr");

    for nfreq in [100, 500, 1000, 5000].iter() {
        let network = create_test_network(*nfreq, 4);
        let id = BenchmarkId::from_parameter(nfreq);

        group.bench_with_input(id, nfreq, |b, _| b.iter(|| black_box(network.vswr())));
    }

    group.finish();
}

fn bench_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability");

    for nfreq in [100, 500, 1000, 5000].iter() {
        // Stability only works for 2-port networks
        let network = create_test_network(*nfreq, 2);
        let id = BenchmarkId::from_parameter(nfreq);

        group.bench_with_input(id, nfreq, |b, _| b.iter(|| black_box(network.stability())));
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_passivity,
    bench_reciprocity,
    bench_s_db,
    bench_vswr,
    bench_stability,
);
criterion_main!(benches);
