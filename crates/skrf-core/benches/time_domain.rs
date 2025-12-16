//! Benchmarks for time-domain analysis
//!
//! Tests performance of impulse response and step response calculations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::network::WindowType;
use skrf_core::Network;
use std::f64::consts::PI;

/// Create a test network with linear phase (simulating a delay line)
fn create_delay_network(nfreq: usize) -> Network {
    let freq = Frequency::new(0.0, 10.0, nfreq, FrequencyUnit::GHz, SweepType::Linear);

    let mut s = Array3::<Complex64>::zeros((nfreq, 1, 1));
    for f in 0..nfreq {
        // Unity magnitude, linear phase (simulating 1ns delay)
        let phase = -2.0 * PI * f as f64 * 0.1;
        s[[f, 0, 0]] = Complex64::from_polar(0.9, phase);
    }

    let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0)
}

fn create_2port_network(nfreq: usize) -> Network {
    let freq = Frequency::new(0.0, 10.0, nfreq, FrequencyUnit::GHz, SweepType::Linear);

    let mut s = Array3::<Complex64>::zeros((nfreq, 2, 2));
    for f in 0..nfreq {
        let phase = -2.0 * PI * f as f64 * 0.1;
        // Typical 2-port with some reflection and transmission
        s[[f, 0, 0]] = Complex64::from_polar(0.1, phase);
        s[[f, 0, 1]] = Complex64::from_polar(0.9, phase * 1.5);
        s[[f, 1, 0]] = Complex64::from_polar(0.9, phase * 1.5);
        s[[f, 1, 1]] = Complex64::from_polar(0.1, phase * 2.0);
    }

    let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0)
}

fn bench_impulse_response_1port(c: &mut Criterion) {
    let mut group = c.benchmark_group("impulse_response_1port");

    for nfreq in [64, 128, 256, 512, 1024].iter() {
        let network = create_delay_network(*nfreq);
        let id = BenchmarkId::from_parameter(nfreq);

        group.bench_with_input(id, nfreq, |b, _| {
            b.iter(|| black_box(network.impulse_response(WindowType::Hamming, 0)))
        });
    }

    group.finish();
}

fn bench_impulse_response_2port(c: &mut Criterion) {
    let mut group = c.benchmark_group("impulse_response_2port");

    for nfreq in [64, 128, 256, 512].iter() {
        let network = create_2port_network(*nfreq);
        let id = BenchmarkId::from_parameter(nfreq);

        group.bench_with_input(id, nfreq, |b, _| {
            b.iter(|| black_box(network.impulse_response(WindowType::Hamming, 0)))
        });
    }

    group.finish();
}

fn bench_step_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_response");

    for nfreq in [64, 128, 256, 512].iter() {
        let network = create_delay_network(*nfreq);
        let id = BenchmarkId::from_parameter(nfreq);

        group.bench_with_input(id, nfreq, |b, _| {
            b.iter(|| black_box(network.step_response(WindowType::Hamming, 0)))
        });
    }

    group.finish();
}

fn bench_window_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_comparison");

    let network = create_delay_network(256);

    for window in [
        WindowType::None,
        WindowType::Hamming,
        WindowType::Hanning,
        WindowType::Blackman,
    ]
    .iter()
    {
        let id = BenchmarkId::from_parameter(format!("{:?}", window));

        group.bench_with_input(id, window, |b, w| {
            b.iter(|| black_box(network.impulse_response(*w, 0)))
        });
    }

    group.finish();
}

fn bench_with_padding(c: &mut Criterion) {
    let mut group = c.benchmark_group("impulse_with_padding");

    let network = create_delay_network(128);

    for pad in [0, 64, 128, 256, 512].iter() {
        let id = BenchmarkId::from_parameter(pad);

        group.bench_with_input(id, pad, |b, p| {
            b.iter(|| black_box(network.impulse_response(WindowType::Hamming, *p)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_impulse_response_1port,
    bench_impulse_response_2port,
    bench_step_response,
    bench_window_types,
    bench_with_padding,
);
criterion_main!(benches);
