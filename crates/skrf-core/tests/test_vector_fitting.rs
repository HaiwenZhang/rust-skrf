//! Vector Fitting Tests
//!
//! Tests for the Vector Fitting algorithm.
//! Translated from Python scikit-rf test_vectorfitting.py
//!
//! Test coverage:
//! - Basic fitting with different pole configurations
//! - Logarithmic pole spacing  
//! - Fitting without constant/proportional terms
//! - Model order calculation
//! - RMS error verification with Python-matching thresholds

use ndarray::{Array1, Array3};
use num_complex::Complex64;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use skrf_core::network::Network;
use skrf_core::vector_fitting::{InitPoleSpacing, VectorFitting};
use std::f64::consts::PI;

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

// ============================================================================
// Helper Functions
// ============================================================================

/// Load the ring_slot test network from file
fn load_ring_slot() -> Option<Network> {
    let path = format!("{}/ring_slot.s2p", TEST_DATA_DIR);
    Network::from_touchstone(&path).ok()
}

/// Create a simple 1-port test network for basic functionality tests
fn create_simple_1port() -> Network {
    let freq = Frequency::new(1.0e9, 10.0e9, 21, FrequencyUnit::Hz, SweepType::Linear);
    let nfreq = 21;
    let mut s = Array3::<Complex64>::zeros((nfreq, 1, 1));

    // Simple lowpass-like response
    for (i, f) in freq.f().iter().enumerate() {
        let omega = 2.0 * PI * f;
        let pole = -2.0 * PI * 5e9;
        s[[i, 0, 0]] = Complex64::new(pole, 0.0) / Complex64::new(pole, omega);
    }

    let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
    Network::new(freq, s, z0)
}

// ============================================================================
// Ring Slot Tests - Matching Python Thresholds
// Python: test_vectorfitting.py lines 15-46
// ============================================================================

/// Test vector fitting with proportional term on ring_slot data
/// Python: test_ringslot_with_proportional - threshold 0.02
#[test]
fn test_ringslot_with_proportional() {
    let nw = match load_ring_slot() {
        Some(n) => n,
        None => {
            eprintln!("Warning: ring_slot.s2p not found, skipping test");
            return;
        }
    };
    
    let mut vf = VectorFitting::new();
    
    // Python: n_poles_real=2, n_poles_cmplx=0, fit_proportional=True, fit_constant=True
    let result = vf.vector_fit(
        &nw,
        2,                          // n_poles_real
        0,                          // n_poles_cmplx
        InitPoleSpacing::Linear,
        true,                       // fit_constant
        true,                       // fit_proportional
    );

    assert!(result.is_ok(), "Vector fit should succeed");
    
    // Python threshold: 0.02
    if let Some(rms) = vf.get_rms_error(&nw, 0, 0) {
        assert!(rms < 0.02, 
            "RMS error should be less than 0.02 (Python threshold), got {}", rms);
    }
}

/// Test vector fitting with logarithmic pole spacing
/// Python: test_ringslot_default_log - threshold 0.01
#[test]
fn test_ringslot_default_log() {
    let nw = match load_ring_slot() {
        Some(n) => n,
        None => {
            eprintln!("Warning: ring_slot.s2p not found, skipping test");
            return;
        }
    };
    
    let mut vf = VectorFitting::new();

    // Python: n_poles_real=4, n_poles_cmplx=0, init_pole_spacing='log'
    let result = vf.vector_fit(
        &nw,
        4,                          // n_poles_real
        0,                          // n_poles_cmplx
        InitPoleSpacing::Logarithmic,
        true,                       // fit_constant
        false,                      // fit_proportional
    );

    assert!(result.is_ok(), "Vector fit with log spacing should succeed");
    
    // Python threshold: 0.01
    if let Some(rms) = vf.get_rms_error(&nw, 0, 0) {
        assert!(rms < 0.01, 
            "RMS error should be less than 0.01 (Python threshold), got {}", rms);
    }
}

/// Test vector fitting without proportional or constant terms
/// Python: test_ringslot_without_prop_const - threshold 0.01
#[test]
fn test_ringslot_without_prop_const() {
    let nw = match load_ring_slot() {
        Some(n) => n,
        None => {
            eprintln!("Warning: ring_slot.s2p not found, skipping test");
            return;
        }
    };
    
    let mut vf = VectorFitting::new();

    // Python: n_poles_real=4, n_poles_cmplx=0, fit_proportional=False, fit_constant=False
    let result = vf.vector_fit(
        &nw,
        4,                          // n_poles_real
        0,                          // n_poles_cmplx
        InitPoleSpacing::Linear,
        false,                      // fit_constant
        false,                      // fit_proportional
    );

    assert!(result.is_ok(), "Vector fit without const/prop should succeed");
    
    // Python threshold: 0.01
    if let Some(rms) = vf.get_rms_error(&nw, 0, 0) {
        assert!(rms < 0.01, 
            "RMS error should be less than 0.01 (Python threshold), got {}", rms);
    }
}

// ============================================================================
// Model Order Tests (Python line 162-178)
// ============================================================================

/// Test that model order is calculated correctly
#[test]
fn test_get_model_order() {
    let nw = create_simple_1port();
    let mut vf = VectorFitting::new();

    // Fit with known pole count: 2 real + 2 complex = 2 + 2*2 = 6
    let result = vf.vector_fit(
        &nw,
        2,                          // n_poles_real
        2,                          // n_poles_cmplx
        InitPoleSpacing::Linear,
        true,
        false,
    );

    assert!(result.is_ok());
    
    if let Some(order) = vf.get_model_order() {
        // Model order = n_real + 2 * n_complex = 2 + 2*2 = 6
        assert_eq!(order, 6, "Model order should be 6 for 2 real + 2 complex poles");
    }
}

/// Test model order with only real poles
#[test]
fn test_get_model_order_real_only() {
    let nw = create_simple_1port();
    let mut vf = VectorFitting::new();

    let result = vf.vector_fit(
        &nw,
        3,                          // n_poles_real
        0,                          // n_poles_cmplx
        InitPoleSpacing::Linear,
        true,
        false,
    );

    assert!(result.is_ok());
    
    if let Some(order) = vf.get_model_order() {
        assert_eq!(order, 3, "Model order should be 3 for 3 real poles");
    }
}

/// Test model order with only complex poles
#[test]
fn test_get_model_order_complex_only() {
    let nw = create_simple_1port();
    let mut vf = VectorFitting::new();

    let result = vf.vector_fit(
        &nw,
        0,                          // n_poles_real
        3,                          // n_poles_cmplx
        InitPoleSpacing::Linear,
        true,
        false,
    );

    assert!(result.is_ok());
    
    if let Some(order) = vf.get_model_order() {
        // Model order = 2 * n_complex = 2*3 = 6
        assert_eq!(order, 6, "Model order should be 6 for 3 complex poles");
    }
}

// ============================================================================
// Model Response Tests
// ============================================================================

/// Test that model response can be evaluated
#[test]
fn test_get_model_response() {
    let nw = create_simple_1port();
    let mut vf = VectorFitting::new();

    let result = vf.vector_fit(
        &nw,
        2,
        1,
        InitPoleSpacing::Linear,
        true,
        false,
    );

    assert!(result.is_ok());

    // Evaluate model at original frequencies
    let freqs: Vec<f64> = nw.f().to_vec();
    let response = vf.get_model_response(0, 0, &freqs);

    assert!(response.is_some(), "Model response should be computable");
    
    if let Some(resp) = response {
        assert_eq!(resp.len(), freqs.len(), "Response should have same length as freqs");
        
        // All values should be finite
        for val in resp.iter() {
            assert!(val.re.is_finite() && val.im.is_finite(), 
                    "Model response should be finite");
        }
    }
}

// ============================================================================
// RMS Error Tests
// ============================================================================

/// Test RMS error calculation
#[test]
fn test_get_rms_error() {
    let nw = create_simple_1port();
    let mut vf = VectorFitting::new();

    let result = vf.vector_fit(
        &nw,
        3,
        1,
        InitPoleSpacing::Linear,
        true,
        false,
    );

    assert!(result.is_ok());

    let rms = vf.get_rms_error(&nw, 0, 0);
    assert!(rms.is_some(), "RMS error should be computable");
    
    if let Some(err) = rms {
        // RMS error should be positive and finite
        assert!(err >= 0.0, "RMS error should be non-negative");
        assert!(err.is_finite(), "RMS error should be finite");
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test VectorFitting before fitting is done
#[test]
fn test_before_fit() {
    let vf = VectorFitting::new();
    
    // Model order should be None before fitting
    assert!(vf.get_model_order().is_none(), "Model order should be None before fit");
}

/// Test fitting with file-loaded network
#[test]
fn test_fit_from_file() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let nw_result = Network::from_touchstone(&path);
    
    if let Ok(nw) = nw_result {
        let mut vf = VectorFitting::new();
        
        let result = vf.vector_fit(
            &nw,
            2,
            2,
            InitPoleSpacing::Linear,
            true,
            false,
        );

        assert!(result.is_ok(), "Fitting file-loaded network should succeed");
        
        if let Some(order) = vf.get_model_order() {
            assert!(order > 0, "Model order should be positive");
        }
    }
}
