//! Time Domain Tests
//!
//! Tests for time-domain analysis functions (impulse response, step response).
//! Translated from Python scikit-rf test_network.py (lines 127-290)

use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::network::Network;
use skrf_core::network::WindowType;
use std::f64::consts::PI;

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

/// Helper to create a network with linear phase (delay line)
fn create_delay_line(delay_ns: f64, nfreq: usize) -> Network {
    let freq = Frequency::new(0.0, 10.0, nfreq, FrequencyUnit::GHz, SweepType::Linear);
    let mut s = Array3::<Complex64>::zeros((nfreq, 1, 1));

    for f in 0..nfreq {
        // Unity magnitude, linear phase (delay)
        let f_hz = f as f64 * 1e9; // GHz to Hz
        let phase = -2.0 * PI * f_hz * delay_ns * 1e-9;
        s[[f, 0, 0]] = Complex64::from_polar(1.0, phase);
    }

    let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0)
}

/// Helper to create a 2-port through line
fn create_thru_line(nfreq: usize) -> Network {
    let freq = Frequency::new(0.0, 10.0, nfreq, FrequencyUnit::GHz, SweepType::Linear);
    let mut s = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        // Perfect thru: S21 = S12 = 1, S11 = S22 = 0
        s[[f, 0, 1]] = Complex64::new(1.0, 0.0);
        s[[f, 1, 0]] = Complex64::new(1.0, 0.0);
    }

    let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0)
}

// ============================================================================
// Impulse Response Tests (Python line 168-206)
// ============================================================================

#[test]
fn test_impulse_response_basic() {
    let ntwk = create_delay_line(0.1, 21);

    let result = ntwk.impulse_response(WindowType::None, 0);
    assert!(result.is_some(), "Impulse response should succeed");

    let (t, ir) = result.unwrap();
    assert_eq!(t.len(), 21);
    assert_eq!(ir.shape(), &[21, 1, 1]);
}

#[test]
fn test_impulse_response_with_hamming() {
    let ntwk = create_delay_line(0.1, 21);

    let result = ntwk.impulse_response(WindowType::Hamming, 0);
    assert!(
        result.is_some(),
        "Impulse response with Hamming should succeed"
    );

    let (t, ir) = result.unwrap();
    assert_eq!(t.len(), 21);
    assert_eq!(ir.shape(), &[21, 1, 1]);
}

#[test]
fn test_impulse_response_with_hanning() {
    let ntwk = create_delay_line(0.1, 21);

    let result = ntwk.impulse_response(WindowType::Hanning, 0);
    assert!(
        result.is_some(),
        "Impulse response with Hanning should succeed"
    );
}

#[test]
fn test_impulse_response_with_blackman() {
    let ntwk = create_delay_line(0.1, 21);

    let result = ntwk.impulse_response(WindowType::Blackman, 0);
    assert!(
        result.is_some(),
        "Impulse response with Blackman should succeed"
    );
}

#[test]
fn test_impulse_response_with_padding() {
    let ntwk = create_delay_line(0.1, 21);
    let pad = 10;

    let result = ntwk.impulse_response(WindowType::Hamming, pad);
    assert!(
        result.is_some(),
        "Impulse response with padding should succeed"
    );

    let (t, ir) = result.unwrap();
    // Total points should include padding
    assert_eq!(t.len(), 21 + pad);
    assert_eq!(ir.shape(), &[21 + pad, 1, 1]);
}

#[test]
fn test_impulse_response_multiport() {
    let ntwk = create_thru_line(21);

    let result = ntwk.impulse_response(WindowType::Hamming, 0);
    assert!(result.is_some(), "2-port impulse response should succeed");

    let (t, ir) = result.unwrap();
    assert_eq!(t.len(), 21);
    assert_eq!(ir.shape(), &[21, 2, 2]);
}

// ============================================================================
// Step Response Tests (Python line 182-192)
// ============================================================================

#[test]
fn test_step_response_basic() {
    let ntwk = create_delay_line(0.1, 21);

    let result = ntwk.step_response(WindowType::Hamming, 0);
    assert!(result.is_some(), "Step response should succeed");

    let (t, sr) = result.unwrap();
    assert_eq!(t.len(), 21);
    assert_eq!(sr.shape(), &[21, 1, 1]);
}

#[test]
fn test_step_response_multiport() {
    let ntwk = create_thru_line(21);

    let result = ntwk.step_response(WindowType::Hamming, 0);
    assert!(result.is_some(), "2-port step response should succeed");

    let (t, sr) = result.unwrap();
    assert_eq!(t.len(), 21);
    assert_eq!(sr.shape(), &[21, 2, 2]);
}

#[test]
fn test_step_response_with_padding() {
    let ntwk = create_delay_line(0.1, 21);
    let pad = 32;

    let result = ntwk.step_response(WindowType::Hamming, pad);
    assert!(result.is_some());

    let (t, _) = result.unwrap();
    assert_eq!(t.len(), 21 + pad);
}

// ============================================================================
// Time Vector Tests
// ============================================================================

#[test]
fn test_time_vector_centered() {
    let ntwk = create_delay_line(0.0, 11);

    let (t, _) = ntwk.impulse_response(WindowType::None, 0).unwrap();

    // Time vector should include negative and positive times
    let has_negative = t.iter().any(|&x| x < 0.0);
    let has_positive = t.iter().any(|&x| x > 0.0);

    assert!(has_negative, "Time vector should have negative values");
    assert!(has_positive, "Time vector should have positive values");
}

#[test]
fn test_time_vector_monotonic() {
    let ntwk = create_delay_line(0.0, 21);

    let (t, _) = ntwk.impulse_response(WindowType::Hamming, 0).unwrap();

    // Time vector should be monotonically increasing
    for i in 1..t.len() {
        assert!(t[i] > t[i - 1], "Time vector should be monotonic");
    }
}

// ============================================================================
// Real Network Tests
// ============================================================================

#[test]
fn test_impulse_response_from_file() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let result = ntwk.impulse_response(WindowType::Hamming, 32);
    assert!(result.is_some(), "Impulse response from file should work");

    let (t, ir) = result.unwrap();
    assert!(t.len() > 0);
    assert_eq!(ir.shape()[1], 2);
    assert_eq!(ir.shape()[2], 2);
}

#[test]
fn test_step_response_from_file() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let result = ntwk.step_response(WindowType::Hamming, 32);
    assert!(result.is_some(), "Step response from file should work");

    let (t, sr) = result.unwrap();
    assert!(t.len() > 0);
    assert_eq!(sr.shape()[1], 2);
    assert_eq!(sr.shape()[2], 2);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_single_frequency_returns_none() {
    // Single frequency point shouldn't support time domain
    let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
    let s = Array3::<Complex64>::zeros((1, 1, 1));
    let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
    let ntwk = Network::new(freq, s, z0);

    let result = ntwk.impulse_response(WindowType::None, 0);
    assert!(result.is_none(), "Single freq should return None");
}
