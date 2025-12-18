//! Benchmarks for Vector Fitting algorithms
//!
//! Tests performance of pole relocation and residue fitting.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::vector_fitting::{InitPoleSpacing, VectorFitting};
use skrf_core::Network;
use std::f64::consts::PI;

/// Create a synthetic network with known frequency response
fn create_test_network(nfreq: usize, nports: usize) -> Network {
    let freq = Frequency::new(1.0, 10.0, nfreq, FrequencyUnit::GHz, SweepType::Linear);

    let mut s = Array3::<Complex64>::zeros((nfreq, nports, nports));
    let f_vec = freq.f();

    for (f_idx, f) in f_vec.iter().enumerate() {
        let s_val = Complex64::new(0.0, 2.0 * PI * f * 1e9);

        // Create realistic S-parameters with known poles
        let pole1 = Complex64::new(-1e9, 5e9);
        let pole2 = Complex64::new(-0.5e9, 8e9);

        for i in 0..nports {
            for j in 0..nports {
                let residue = Complex64::new(1e9 * (i + 1) as f64, 0.5e9 * (j + 1) as f64);
                let mut val = Complex64::new(0.1, 0.0); // DC offset

                // Add pole contributions
                val += residue / (s_val - pole1) + residue.conj() / (s_val - pole1.conj());
                val += residue * 0.5 / (s_val - pole2)
                    + (residue * 0.5).conj() / (s_val - pole2.conj());

                // Normalize to physical S-param range
                s[[f_idx, i, j]] = val / (val.norm() + 1.0) * 0.9;
            }
        }
    }

    let z0 = Array1::from_elem(nports, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0).unwrap()
}

fn bench_vector_fit_1port(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_fit_1port");
    group.sample_size(10); // Reduce sample size for slower benchmarks

    for nfreq in [50, 100, 200].iter() {
        let network = create_test_network(*nfreq, 1);

        for n_poles in [2, 4, 8].iter() {
            let id = BenchmarkId::new(format!("{}freqs", nfreq), n_poles);

            group.bench_with_input(id, n_poles, |b, &n| {
                b.iter(|| {
                    let mut vf = VectorFitting::new();
                    vf.max_iterations = 5; // Limit iterations for benchmark
                    black_box(vf.vector_fit(
                        &network,
                        0,
                        n,
                        InitPoleSpacing::Logarithmic,
                        true,
                        false,
                    ))
                })
            });
        }
    }

    group.finish();
}

fn bench_vector_fit_2port(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_fit_2port");
    group.sample_size(10);

    for nfreq in [50, 100].iter() {
        let network = create_test_network(*nfreq, 2);

        for n_poles in [2, 4].iter() {
            let id = BenchmarkId::new(format!("{}freqs", nfreq), n_poles);

            group.bench_with_input(id, n_poles, |b, &n| {
                b.iter(|| {
                    let mut vf = VectorFitting::new();
                    vf.max_iterations = 5;
                    black_box(vf.vector_fit(
                        &network,
                        0,
                        n,
                        InitPoleSpacing::Logarithmic,
                        true,
                        false,
                    ))
                })
            });
        }
    }

    group.finish();
}

fn bench_pole_spacing_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("pole_spacing");
    group.sample_size(10);

    let network = create_test_network(100, 1);

    for spacing in [InitPoleSpacing::Linear, InitPoleSpacing::Logarithmic].iter() {
        let id = BenchmarkId::from_parameter(format!("{:?}", spacing));

        group.bench_with_input(id, spacing, |b, s| {
            b.iter(|| {
                let mut vf = VectorFitting::new();
                vf.max_iterations = 5;
                black_box(vf.vector_fit(&network, 0, 4, *s, true, false))
            })
        });
    }

    group.finish();
}

fn bench_iteration_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_scaling");
    group.sample_size(10);

    let network = create_test_network(100, 1);

    for max_iters in [1, 3, 5, 10].iter() {
        let id = BenchmarkId::from_parameter(max_iters);

        group.bench_with_input(id, max_iters, |b, &n| {
            b.iter(|| {
                let mut vf = VectorFitting::new();
                vf.max_iterations = n;
                black_box(vf.vector_fit(&network, 0, 4, InitPoleSpacing::Logarithmic, true, false))
            })
        });
    }

    group.finish();
}

fn bench_model_response_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_response_eval");

    // First, fit a model
    let network = create_test_network(100, 2);
    let mut vf = VectorFitting::new();
    vf.max_iterations = 10;
    let _ = vf.vector_fit(&network, 0, 4, InitPoleSpacing::Logarithmic, true, false);

    for n_eval_freqs in [100, 500, 1000, 5000].iter() {
        let eval_freqs: Vec<f64> = (0..*n_eval_freqs)
            .map(|i| 1e9 + 9e9 * i as f64 / (*n_eval_freqs - 1) as f64)
            .collect();

        let id = BenchmarkId::from_parameter(n_eval_freqs);

        group.bench_with_input(id, &eval_freqs, |b, freqs| {
            b.iter(|| black_box(vf.get_model_response(0, 0, freqs)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_fit_1port,
    bench_vector_fit_2port,
    bench_pole_spacing_comparison,
    bench_iteration_scaling,
    bench_model_response_evaluation,
);
criterion_main!(benches);
