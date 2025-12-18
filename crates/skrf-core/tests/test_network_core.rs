//! Core Network tests
//!
//! Tests for Network construction, copy, and equality operations.
//! Translated from Python scikit-rf test_network.py

use approx::assert_relative_eq;
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::network::Network;

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

/// Helper to create a simple 2-port test network
fn create_test_network() -> Network {
    let freq = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Linear);
    let mut s = Array3::<Complex64>::zeros((10, 2, 2));
    for f in 0..10 {
        s[[f, 0, 0]] = Complex64::new(0.1 * f as f64, 0.05 * f as f64);
        s[[f, 0, 1]] = Complex64::new(0.8, 0.1);
        s[[f, 1, 0]] = Complex64::new(0.8, 0.1);
        s[[f, 1, 1]] = Complex64::new(0.1, -0.05 * f as f64);
    }
    let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0)
}

// ============================================================================
// Network Copy Tests (Python line 84-95)
// ============================================================================

#[test]
fn test_network_copy() {
    let n = create_test_network();
    let n2 = n.clone();

    // Verify frequency matches
    assert_eq!(n.nfreq(), n2.nfreq());
    assert_eq!(n.nports(), n2.nports());

    // Verify S-parameters match
    for f in 0..n.nfreq() {
        for i in 0..n.nports() {
            for j in 0..n.nports() {
                assert_relative_eq!(n.s()[[f, i, j]].re, n2.s()[[f, i, j]].re, epsilon = 1e-15);
                assert_relative_eq!(n.s()[[f, i, j]].im, n2.s()[[f, i, j]].im, epsilon = 1e-15);
            }
        }
    }

    // Verify z0 matches
    for p in 0..n.nports() {
        assert_relative_eq!(n.z0()[p].re, n2.z0()[p].re, epsilon = 1e-15);
    }
}

// ============================================================================
// Constructor Tests (Python line 292-400)
// ============================================================================

#[test]
fn test_constructor_empty() {
    // Verify that a default/empty network can be created
    let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
    let s = Array3::<Complex64>::zeros((1, 1, 1));
    let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
    let ntwk = Network::new(freq, s, z0);

    assert_eq!(ntwk.nports(), 1);
    assert_eq!(ntwk.nfreq(), 1);
}

#[test]
fn test_constructor_from_values() {
    // Create network from f, s, z0 arrays
    let freq = Frequency::new(1.0, 2.0, 2, FrequencyUnit::GHz, SweepType::Linear);
    let mut s = Array3::<Complex64>::zeros((2, 2, 2));
    s[[0, 0, 0]] = Complex64::new(1.0, 2.0);
    s[[1, 0, 0]] = Complex64::new(3.0, 4.0);
    let z0 = Array1::from_vec(vec![Complex64::new(50.0, 0.0), Complex64::new(75.0, 0.0)]);
    let ntwk = Network::new(freq, s, z0);

    assert_eq!(ntwk.nports(), 2);
    assert_eq!(ntwk.nfreq(), 2);
    assert_relative_eq!(ntwk.s()[[0, 0, 0]].re, 1.0, epsilon = 1e-15);
    assert_relative_eq!(ntwk.s()[[0, 0, 0]].im, 2.0, epsilon = 1e-15);
    assert_relative_eq!(ntwk.z0()[1].re, 75.0, epsilon = 1e-15);
}

#[test]
fn test_constructor_from_touchstone() {
    // Load from .s2p file
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    assert_eq!(ntwk.nports(), 2);
    assert!(ntwk.nfreq() > 0);
    assert_relative_eq!(ntwk.z0()[0].re, 50.0, epsilon = 1e-10);
}

#[test]
fn test_constructor_from_touchstone_1port() {
    // Load 1-port short from .s1p file
    let path = format!("{}/short.s1p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load short.s1p");

    assert_eq!(ntwk.nports(), 1);
    assert!(ntwk.nfreq() > 0);

    // A short should have S11 ≈ -1 (magnitude close to 1)
    for f in 0..ntwk.nfreq() {
        let s11 = ntwk.s()[[f, 0, 0]];
        let mag = s11.norm();
        assert!(
            mag > 0.9 && mag < 1.1,
            "Short S11 magnitude should be ≈ 1, got {}",
            mag
        );
    }
}

#[test]
fn test_constructor_from_touchstone_4port() {
    // Load 4-port network
    let path = format!("{}/cst_example_4ports.s4p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load cst_example_4ports.s4p");

    assert_eq!(ntwk.nports(), 4);
    assert!(ntwk.nfreq() > 0);
}

// ============================================================================
// Cascade Tests (Python line 539-545)
// ============================================================================

#[test]
fn test_cascade() {
    // Test that ntwk1 ** ntwk2 == ntwk3 (verified in Python with ADS)
    let path1 = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let path2 = format!("{}/ntwk2.s2p", TEST_DATA_DIR);
    let path3 = format!("{}/ntwk3.s2p", TEST_DATA_DIR);

    let ntwk1 = Network::from_touchstone(&path1).expect("Failed to load ntwk1.s2p");
    let ntwk2 = Network::from_touchstone(&path2).expect("Failed to load ntwk2.s2p");
    let ntwk3 = Network::from_touchstone(&path3).expect("Failed to load ntwk3.s2p");

    let result = ntwk1.cascade(&ntwk2).expect("Cascade failed");

    // Compare result with ntwk3
    assert_eq!(result.nfreq(), ntwk3.nfreq());
    for f in 0..result.nfreq() {
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    result.s()[[f, i, j]].re,
                    ntwk3.s()[[f, i, j]].re,
                    epsilon = 1e-4
                );
                assert_relative_eq!(
                    result.s()[[f, i, j]].im,
                    ntwk3.s()[[f, i, j]].im,
                    epsilon = 1e-4
                );
            }
        }
    }
}

// ============================================================================
// Flip and Renumber Tests (Python line 1023-1046)
// ============================================================================

#[test]
fn test_flip() {
    let ntwk = create_test_network();
    let flipped = ntwk.flipped().expect("Flip failed");

    // Verify port swap: S11 <-> S22, S12 <-> S21
    for f in 0..ntwk.nfreq() {
        assert_relative_eq!(
            ntwk.s()[[f, 0, 0]].re,
            flipped.s()[[f, 1, 1]].re,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            ntwk.s()[[f, 1, 1]].re,
            flipped.s()[[f, 0, 0]].re,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            ntwk.s()[[f, 0, 1]].re,
            flipped.s()[[f, 1, 0]].re,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            ntwk.s()[[f, 1, 0]].re,
            flipped.s()[[f, 0, 1]].re,
            epsilon = 1e-15
        );
    }
}

#[test]
fn test_renumber() {
    let ntwk = create_test_network();
    let renumbered = ntwk.renumbered(&[0, 1], &[1, 0]).expect("Renumber failed");

    // Renumber [0,1] -> [1,0] should swap ports like flip
    for f in 0..ntwk.nfreq() {
        assert_relative_eq!(
            ntwk.s()[[f, 0, 0]].re,
            renumbered.s()[[f, 1, 1]].re,
            epsilon = 1e-15
        );
    }
}

// ============================================================================
// Subnetwork Tests (Python line 2070-2108)
// ============================================================================

#[test]
fn test_subnetwork() {
    // Load 4-port network and extract 2-port subnetwork
    let path = format!("{}/cst_example_4ports.s4p", TEST_DATA_DIR);
    let ntwk4 = Network::from_touchstone(&path).expect("Failed to load cst_example_4ports.s4p");

    // Extract ports 0 and 1
    let sub = ntwk4
        .subnetwork(&[0, 1])
        .expect("Subnetwork extraction failed");

    assert_eq!(sub.nports(), 2);
    assert_eq!(sub.nfreq(), ntwk4.nfreq());

    // Verify S-parameters match the original submatrix
    for f in 0..sub.nfreq() {
        assert_relative_eq!(
            sub.s()[[f, 0, 0]].re,
            ntwk4.s()[[f, 0, 0]].re,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            sub.s()[[f, 0, 1]].re,
            ntwk4.s()[[f, 0, 1]].re,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            sub.s()[[f, 1, 0]].re,
            ntwk4.s()[[f, 1, 0]].re,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            sub.s()[[f, 1, 1]].re,
            ntwk4.s()[[f, 1, 1]].re,
            epsilon = 1e-15
        );
    }
}

// ============================================================================
// Equality Tests (Python line 2140-2165)
// ============================================================================

#[test]
fn test_equality() {
    let n1 = create_test_network();
    let n2 = n1.clone();

    // Same networks should be equal
    assert_eq!(n1.nfreq(), n2.nfreq());
    assert_eq!(n1.nports(), n2.nports());

    // Verify all S-parameters match
    for f in 0..n1.nfreq() {
        for i in 0..n1.nports() {
            for j in 0..n1.nports() {
                let diff = (n1.s()[[f, i, j]] - n2.s()[[f, i, j]]).norm();
                assert!(diff < 1e-10, "S-parameters should match");
            }
        }
    }
}

// ============================================================================
// De-embedding Tests (Python line 1048-1061)
// ============================================================================

#[test]
fn test_de_embed_by_inv() {
    // Test: ntwk1.inv ** ntwk3 == ntwk2
    let path1 = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let path2 = format!("{}/ntwk2.s2p", TEST_DATA_DIR);
    let path3 = format!("{}/ntwk3.s2p", TEST_DATA_DIR);

    let ntwk1 = Network::from_touchstone(&path1).expect("Failed to load ntwk1.s2p");
    let ntwk2 = Network::from_touchstone(&path2).expect("Failed to load ntwk2.s2p");
    let ntwk3 = Network::from_touchstone(&path3).expect("Failed to load ntwk3.s2p");

    let ntwk1_inv = ntwk1.inv().expect("Inverse failed");
    let result = ntwk1_inv.cascade(&ntwk3).expect("Cascade failed");

    // Compare result with ntwk2
    for f in 0..result.nfreq() {
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    result.s()[[f, i, j]].re,
                    ntwk2.s()[[f, i, j]].re,
                    epsilon = 1e-4
                );
                assert_relative_eq!(
                    result.s()[[f, i, j]].im,
                    ntwk2.s()[[f, i, j]].im,
                    epsilon = 1e-4
                );
            }
        }
    }
}

#[test]
fn test_deembed() {
    // Test the deembed method directly
    let path1 = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let path3 = format!("{}/ntwk3.s2p", TEST_DATA_DIR);

    let ntwk1 = Network::from_touchstone(&path1).expect("Failed to load ntwk1.s2p");
    let ntwk3 = Network::from_touchstone(&path3).expect("Failed to load ntwk3.s2p");

    let result = ntwk3.deembed(&ntwk1).expect("Deembed failed");

    // Result should be a valid 2-port network
    assert_eq!(result.nports(), 2);
    assert_eq!(result.nfreq(), ntwk3.nfreq());
}
