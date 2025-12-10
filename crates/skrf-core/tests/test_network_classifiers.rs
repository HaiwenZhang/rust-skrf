//! Network Classifier Tests
//!
//! Tests for network classification properties (reciprocal, symmetric, passive, lossless).
//! Translated from Python scikit-rf test_network.py (lines 1593-1719)

use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::network::Network;

/// Helper to create a network with specific S-parameters
fn create_network_from_s(s_matrix: Vec<Vec<Complex64>>) -> Network {
    let nports = s_matrix.len();
    let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
    let mut s = Array3::<Complex64>::zeros((1, nports, nports));
    for i in 0..nports {
        for j in 0..nports {
            s[[0, i, j]] = s_matrix[i][j];
        }
    }
    let z0 = Array1::from_elem(nports, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0)
}

// ============================================================================
// Reciprocity Tests (Python line 1593-1606)
// ============================================================================

#[test]
fn test_is_reciprocal_circulator() {
    // A circulator is NOT reciprocal
    // S = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    let s_matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(!ntwk.is_reciprocal(None), "A circulator is not reciprocal");
}

#[test]
fn test_is_reciprocal_power_divider() {
    // A reciprocal power divider: S_ij = S_ji
    // S = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    let s_matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(ntwk.is_reciprocal(None), "This power divider is reciprocal");
}

// ============================================================================
// Passivity Tests (Python line 1692-1704)
// ============================================================================

#[test]
fn test_is_passive_power_divider() {
    // A passive power divider: |S_ij| <= 1 for all i,j
    let s_matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(ntwk.is_passive(None), "This power divider is passive");
}

#[test]
fn test_is_passive_amplifier() {
    // A unilateral amplifier is NOT passive: |S21| > 1
    let s_matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(10.0, 0.0), Complex64::new(0.0, 0.0)],  // S21 = 10 (20dB gain)
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(!ntwk.is_passive(None), "A unilateral amplifier is not passive");
}

// ============================================================================
// Lossless Tests (Python line 1706-1719)
// ============================================================================

#[test]
fn test_is_lossless_resistive_divider() {
    // A resistive power divider is lossy (not lossless)
    let s_matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(!ntwk.is_lossless(None), "A resistive power divider is lossy");
}

#[test]
fn test_is_lossless_ideal_divider() {
    // An ideal lossless power divider (Wilkinson without resistor)
    // S^H * S = I (unitary matrix)
    let sqrt2 = (2.0_f64).sqrt();
    let s_matrix = vec![
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0 / sqrt2),
            Complex64::new(0.0, -1.0 / sqrt2),
        ],
        vec![
            Complex64::new(0.0, -1.0 / sqrt2),
            Complex64::new(0.5, 0.0),
            Complex64::new(-0.5, 0.0),
        ],
        vec![
            Complex64::new(0.0, -1.0 / sqrt2),
            Complex64::new(-0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(ntwk.is_lossless(None), "This unmatched power divider is lossless");
}

// ============================================================================
// Symmetry Tests (Python line 1608-1690)
// ============================================================================

#[test]
fn test_is_symmetric_short() {
    // A short circuit is symmetric: S11 = S22
    let s_matrix = vec![
        vec![Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(ntwk.is_symmetric(None), "A short is symmetric");
}

#[test]
fn test_is_symmetric_asymmetric() {
    // An asymmetric network: S11 != S22
    let s_matrix = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(!ntwk.is_symmetric(None), "This network is not symmetric");
}

// ============================================================================
// Combined Tests
// ============================================================================

#[test]
fn test_ideal_thru() {
    // An ideal thru connection
    let s_matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(ntwk.is_reciprocal(None), "Thru is reciprocal");
    assert!(ntwk.is_passive(None), "Thru is passive");
    assert!(ntwk.is_lossless(None), "Thru is lossless");
    assert!(ntwk.is_symmetric(None), "Thru is symmetric");
}

#[test]
fn test_attenuator() {
    // A 6dB attenuator (0.5 transmission)
    let s_matrix = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let ntwk = create_network_from_s(s_matrix);

    assert!(ntwk.is_reciprocal(None), "Attenuator is reciprocal");
    assert!(ntwk.is_passive(None), "Attenuator is passive");
    assert!(!ntwk.is_lossless(None), "Attenuator is lossy");
    assert!(ntwk.is_symmetric(None), "Attenuator is symmetric");
}
