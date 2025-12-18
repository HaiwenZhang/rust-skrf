//! Network Operator Tests
//!
//! Tests for arithmetic operators (+, -, *, /) on Networks.
//! Translated from Python scikit-rf test_network.py (lines 1502-1537)
//!
//! Note: Rust implementation supports:
//! - Network + Network (element-wise addition)
//! - Network - Network (element-wise subtraction)
//! - Network * scalar (scalar multiplication)
//! - Network / scalar (scalar division)
//!
//! Network * Network and Network / Network are NOT implemented in Rust.

use approx::assert_relative_eq;
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::network::Network;

/// Helper to create 1-port test networks
fn create_1port_network(s_values: &[Complex64]) -> Network {
    let nfreq = s_values.len();
    let freq = Frequency::new(
        1.0,
        nfreq as f64,
        nfreq,
        FrequencyUnit::GHz,
        SweepType::Linear,
    );
    let mut s = Array3::<Complex64>::zeros((nfreq, 1, 1));
    for (f, &val) in s_values.iter().enumerate() {
        s[[f, 0, 0]] = val;
    }
    let z0 = Array1::from_elem(1, Complex64::new(1.0, 0.0));
    Network::new(freq, s, z0)
}

// ============================================================================
// Scalar Multiplication Tests (Python line 1502-1510)
// ============================================================================

#[test]
fn test_mul_scalar_f64() {
    let a = create_1port_network(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);

    let result = &a * 2.0;

    // 2 * (1+2j) = 2+4j
    assert_relative_eq!(result.s()[[0, 0, 0]].re, 2.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, 4.0, epsilon = 1e-10);

    // 2 * (3+4j) = 6+8j
    assert_relative_eq!(result.s()[[1, 0, 0]].re, 6.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[1, 0, 0]].im, 8.0, epsilon = 1e-10);
}

#[test]
fn test_mul_scalar_complex() {
    let a = create_1port_network(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);

    let scalar = Complex64::new(2.0, 0.0);
    let result = &a * scalar;

    // Same as multiplying by 2.0
    assert_relative_eq!(result.s()[[0, 0, 0]].re, 2.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, 4.0, epsilon = 1e-10);
}

#[test]
fn test_mul_scalar_complex_imaginary() {
    let a = create_1port_network(&[Complex64::new(1.0, 0.0)]);

    // Multiply by j
    let scalar = Complex64::new(0.0, 1.0);
    let result = &a * scalar;

    // 1 * j = j
    assert_relative_eq!(result.s()[[0, 0, 0]].re, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, 1.0, epsilon = 1e-10);
}

// ============================================================================
// Subtraction Tests (Python line 1512-1519)
// ============================================================================

#[test]
fn test_sub_networks() {
    // a - a should give zero
    let a = create_1port_network(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);

    let result = (&a - &a).expect("Subtraction failed");

    assert_relative_eq!(result.s()[[0, 0, 0]].re, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[1, 0, 0]].re, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[1, 0, 0]].im, 0.0, epsilon = 1e-10);
}

#[test]
fn test_sub_networks_different() {
    let a = create_1port_network(&[Complex64::new(3.0, 4.0)]);
    let b = create_1port_network(&[Complex64::new(1.0, 1.0)]);

    let result = (&a - &b).expect("Subtraction failed");

    // (3+4j) - (1+1j) = 2+3j
    assert_relative_eq!(result.s()[[0, 0, 0]].re, 2.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, 3.0, epsilon = 1e-10);
}

// ============================================================================
// Scalar Division Tests (Python line 1521-1528)
// ============================================================================

#[test]
fn test_div_scalar_f64() {
    let a = create_1port_network(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);

    let result = &a / 2.0;

    // (1+2j) / 2 = 0.5 + 1j
    assert_relative_eq!(result.s()[[0, 0, 0]].re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, 1.0, epsilon = 1e-10);

    // (3+4j) / 2 = 1.5 + 2j
    assert_relative_eq!(result.s()[[1, 0, 0]].re, 1.5, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[1, 0, 0]].im, 2.0, epsilon = 1e-10);
}

#[test]
fn test_div_scalar_complex() {
    let a = create_1port_network(&[Complex64::new(2.0, 0.0)]);

    // Divide by (1+1j)
    let scalar = Complex64::new(1.0, 1.0);
    let result = &a / scalar;

    // 2 / (1+j) = 2(1-j)/2 = 1-j
    assert_relative_eq!(result.s()[[0, 0, 0]].re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, -1.0, epsilon = 1e-10);
}

// ============================================================================
// Addition Tests (Python line 1530-1537)
// ============================================================================

#[test]
fn test_add_networks() {
    // a + a should give 2*a
    let a = create_1port_network(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);

    let result = (&a + &a).expect("Addition failed");

    assert_relative_eq!(result.s()[[0, 0, 0]].re, 2.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, 4.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[1, 0, 0]].re, 6.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[1, 0, 0]].im, 8.0, epsilon = 1e-10);
}

#[test]
fn test_add_networks_different() {
    let a = create_1port_network(&[Complex64::new(1.0, 2.0)]);
    let b = create_1port_network(&[Complex64::new(3.0, 4.0)]);

    let result = (&a + &b).expect("Addition failed");

    // (1+2j) + (3+4j) = 4+6j
    assert_relative_eq!(result.s()[[0, 0, 0]].re, 4.0, epsilon = 1e-10);
    assert_relative_eq!(result.s()[[0, 0, 0]].im, 6.0, epsilon = 1e-10);
}

// ============================================================================
// Mixed Operator Tests
// ============================================================================

#[test]
fn test_operators_preserve_z0() {
    let a = create_1port_network(&[Complex64::new(1.0, 2.0)]);
    let b = create_1port_network(&[Complex64::new(3.0, 4.0)]);

    // Add/Sub should preserve z0
    let result_add = (&a + &b).expect("Add failed");
    assert_relative_eq!(result_add.z0()[0].re, a.z0()[0].re, epsilon = 1e-10);

    let result_sub = (&a - &b).expect("Sub failed");
    assert_relative_eq!(result_sub.z0()[0].re, a.z0()[0].re, epsilon = 1e-10);

    // Scalar mul/div should preserve z0
    let result_mul = &a * 2.0;
    assert_relative_eq!(result_mul.z0()[0].re, a.z0()[0].re, epsilon = 1e-10);

    let result_div = &a / 2.0;
    assert_relative_eq!(result_div.z0()[0].re, a.z0()[0].re, epsilon = 1e-10);
}

#[test]
fn test_operators_preserve_frequency() {
    let a = create_1port_network(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
    let b = create_1port_network(&[Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)]);

    let result = (&a + &b).expect("Add failed");
    assert_eq!(result.nfreq(), a.nfreq());
}

#[test]
fn test_combined_operations() {
    // Test: (a + b) * 0.5 = (a + b) / 2
    let a = create_1port_network(&[Complex64::new(2.0, 4.0)]);
    let b = create_1port_network(&[Complex64::new(4.0, 2.0)]);

    let sum = (&a + &b).expect("Add failed");
    let result1 = &sum * 0.5;
    let result2 = &sum / 2.0;

    assert_relative_eq!(
        result1.s()[[0, 0, 0]].re,
        result2.s()[[0, 0, 0]].re,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        result1.s()[[0, 0, 0]].im,
        result2.s()[[0, 0, 0]].im,
        epsilon = 1e-10
    );

    // (2+4j) + (4+2j) = 6+6j, divided by 2 = 3+3j
    assert_relative_eq!(result1.s()[[0, 0, 0]].re, 3.0, epsilon = 1e-10);
    assert_relative_eq!(result1.s()[[0, 0, 0]].im, 3.0, epsilon = 1e-10);
}
