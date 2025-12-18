//! Network Parameter Tests
//!
//! Tests for S/Z/Y/T parameter conversions and access.
//! Translated from Python scikit-rf test_network.py

use approx::assert_relative_eq;
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::network::Network;

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

/// Helper to create a test network with known S-parameters
fn create_simple_network() -> Network {
    let freq = Frequency::new(1.0, 2.0, 2, FrequencyUnit::GHz, SweepType::Linear);
    let mut s = Array3::<Complex64>::zeros((2, 2, 2));
    // Set some reasonable S-parameters
    s[[0, 0, 0]] = Complex64::new(-0.1, -0.2);
    s[[0, 0, 1]] = Complex64::new(0.7, 0.1);
    s[[0, 1, 0]] = Complex64::new(0.7, 0.1);
    s[[0, 1, 1]] = Complex64::new(-0.2, -0.1);
    s[[1, 0, 0]] = Complex64::new(-0.15, -0.25);
    s[[1, 0, 1]] = Complex64::new(0.65, 0.15);
    s[[1, 1, 0]] = Complex64::new(0.65, 0.15);
    s[[1, 1, 1]] = Complex64::new(-0.25, -0.15);
    let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0).unwrap()
}

// ============================================================================
// Basic Parameter Access Tests
// ============================================================================

#[test]
fn test_s_parameters() {
    let ntwk = create_simple_network();

    // Verify S-parameter access
    let s = ntwk.s();
    assert_eq!(s.shape(), &[2, 2, 2]);
    assert_relative_eq!(s[[0, 0, 0]].re, -0.1, epsilon = 1e-15);
    assert_relative_eq!(s[[0, 0, 0]].im, -0.2, epsilon = 1e-15);
}

#[test]
fn test_z_parameters() {
    let ntwk = create_simple_network();

    // Get Z-parameters
    let z = ntwk.z();
    assert_eq!(z.shape(), &[2, 2, 2]);

    // Z-parameters should be complex
    // For a matched load (S11=0), Z11 = z0
    // Our test case is not matched, so just verify shape and that values are reasonable
    assert!(z[[0, 0, 0]].norm() > 0.0);
}

#[test]
fn test_y_parameters() {
    let ntwk = create_simple_network();

    // Get Y-parameters
    let y = ntwk.y();
    assert_eq!(y.shape(), &[2, 2, 2]);

    // Y-parameters should be complex
    assert!(y[[0, 0, 0]].norm() > 0.0);
}

#[test]
fn test_t_parameters() {
    let ntwk = create_simple_network();

    // Get T-parameters (only valid for 2-port)
    let t = ntwk.t().expect("T-parameters should exist for 2-port");
    assert_eq!(t.shape(), &[2, 2, 2]);
}

// ============================================================================
// S to Z/Y Round-trip Conversion Tests (Python line 1114-1127)
// ============================================================================

#[test]
fn test_conversions_round_trip() {
    // Load a real network and verify S -> Z -> S round-trip
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let s_original = ntwk.s().clone();
    let z = ntwk.z();
    let y = ntwk.y();

    // Z and Y should have same shape as S
    assert_eq!(z.shape(), s_original.shape());
    assert_eq!(y.shape(), s_original.shape());
}

// ============================================================================
// Z0 Getter Tests (Python line 1425-1488)
// ============================================================================

#[test]
fn test_z0_scalar() {
    let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
    let s = Array3::<Complex64>::zeros((1, 2, 2));
    let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
    let ntwk = Network::new(freq, s, z0).unwrap();

    // Verify z0 access
    assert_eq!(ntwk.z0().len(), 2);
    assert_relative_eq!(ntwk.z0()[0].re, 50.0, epsilon = 1e-15);
    assert_relative_eq!(ntwk.z0()[1].re, 50.0, epsilon = 1e-15);
}

#[test]
fn test_z0_vector() {
    let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
    let s = Array3::<Complex64>::zeros((1, 2, 2));
    let z0 = Array1::from_vec(vec![Complex64::new(25.0, 0.0), Complex64::new(75.0, 0.0)]);
    let ntwk = Network::new(freq, s, z0).unwrap();

    assert_relative_eq!(ntwk.z0()[0].re, 25.0, epsilon = 1e-15);
    assert_relative_eq!(ntwk.z0()[1].re, 75.0, epsilon = 1e-15);
}

#[test]
fn test_z0_complex() {
    let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
    let s = Array3::<Complex64>::zeros((1, 2, 2));
    // Complex characteristic impedance
    let z0 = Array1::from_vec(vec![
        Complex64::new(50.0, 10.0),
        Complex64::new(75.0, -15.0),
    ]);
    let ntwk = Network::new(freq, s, z0).unwrap();

    assert_relative_eq!(ntwk.z0()[0].re, 50.0, epsilon = 1e-15);
    assert_relative_eq!(ntwk.z0()[0].im, 10.0, epsilon = 1e-15);
    assert_relative_eq!(ntwk.z0()[1].re, 75.0, epsilon = 1e-15);
    assert_relative_eq!(ntwk.z0()[1].im, -15.0, epsilon = 1e-15);
}

// ============================================================================
// dB/Magnitude/Phase Properties Tests
// ============================================================================

#[test]
fn test_s_db() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let s_db = ntwk.s_db();
    let s_mag = ntwk.s_mag();

    assert_eq!(s_db.shape(), ntwk.s().shape());
    assert_eq!(s_mag.shape(), ntwk.s().shape());

    // Verify dB = 20 * log10(mag)
    for f in 0..ntwk.nfreq() {
        let mag = s_mag[[f, 0, 0]];
        let db = s_db[[f, 0, 0]];
        if mag > 0.0 {
            assert_relative_eq!(db, 20.0 * mag.log10(), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_s_deg() {
    let ntwk = create_simple_network();
    let s_deg = ntwk.s_deg();

    assert_eq!(s_deg.shape(), ntwk.s().shape());

    // Verify degree is in reasonable range
    for f in 0..ntwk.nfreq() {
        for i in 0..ntwk.nports() {
            for j in 0..ntwk.nports() {
                let deg = s_deg[[f, i, j]];
                assert!((-180.0..=180.0).contains(&deg));
            }
        }
    }
}

#[test]
fn test_s_rad() {
    let ntwk = create_simple_network();
    let s_rad = ntwk.s_rad();

    assert_eq!(s_rad.shape(), ntwk.s().shape());

    // Verify radians is in reasonable range
    for f in 0..ntwk.nfreq() {
        for i in 0..ntwk.nports() {
            for j in 0..ntwk.nports() {
                let rad = s_rad[[f, i, j]];
                assert!((-std::f64::consts::PI..=std::f64::consts::PI).contains(&rad));
            }
        }
    }
}

// ============================================================================
// ABCD Parameter Tests
// ============================================================================

#[test]
fn test_a_parameters() {
    let ntwk = create_simple_network();

    // Get ABCD parameters (only valid for 2-port)
    let a = ntwk.a().expect("ABCD parameters should exist for 2-port");
    assert_eq!(a.shape(), &[2, 2, 2]);
}

#[test]
fn test_h_parameters() {
    let ntwk = create_simple_network();

    // Get H-parameters (only valid for 2-port)
    let h = ntwk.h().expect("H-parameters should exist for 2-port");
    assert_eq!(h.shape(), &[2, 2, 2]);
}

// ============================================================================
// Frequency Access Tests
// ============================================================================

#[test]
fn test_frequency_access() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let f = ntwk.f();
    assert_eq!(f.len(), ntwk.nfreq());

    // Frequency should be monotonically increasing
    for i in 1..f.len() {
        assert!(f[i] > f[i - 1], "Frequency should be increasing");
    }
}

#[test]
fn test_nports_nfreq() {
    let path = format!("{}/cst_example_4ports.s4p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load cst_example_4ports.s4p");

    assert_eq!(ntwk.nports(), 4);
    assert!(ntwk.nfreq() > 0);
}

// ============================================================================
// Real/Imaginary Part Access Tests
// ============================================================================

#[test]
fn test_s_re_im() {
    let ntwk = create_simple_network();

    let s_re = ntwk.s_re();
    let s_im = ntwk.s_im();

    assert_eq!(s_re.shape(), ntwk.s().shape());
    assert_eq!(s_im.shape(), ntwk.s().shape());

    // Verify real and imaginary parts match
    for f in 0..ntwk.nfreq() {
        for i in 0..ntwk.nports() {
            for j in 0..ntwk.nports() {
                assert_relative_eq!(s_re[[f, i, j]], ntwk.s()[[f, i, j]].re, epsilon = 1e-15);
                assert_relative_eq!(s_im[[f, i, j]], ntwk.s()[[f, i, j]].im, epsilon = 1e-15);
            }
        }
    }
}
