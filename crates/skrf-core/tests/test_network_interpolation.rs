//! Network Interpolation Tests
//!
//! Tests for frequency interpolation and resampling.
//! Translated from Python scikit-rf test_network.py (lines 1540-1590)

use approx::assert_relative_eq;
use skrf_core::network::Network;

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

// ============================================================================
// Basic Interpolation Tests (Python line 1540-1562)
// ============================================================================

#[test]
fn test_network_resample() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let original_nfreq = ntwk.nfreq();

    // Resample to half the points
    let resampled = ntwk.resample(original_nfreq / 2);

    assert_eq!(resampled.nfreq(), original_nfreq / 2);
    assert_eq!(resampled.nports(), ntwk.nports());

    // Verify frequency range is similar
    let orig_f = ntwk.f();
    let resamp_f = resampled.f();
    assert_relative_eq!(resamp_f[0], orig_f[0], epsilon = 1e9); // Within 1 GHz
}

#[test]
fn test_network_resample_more_points() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let original_nfreq = ntwk.nfreq();

    // Resample to double the points
    let resampled = ntwk.resample(original_nfreq * 2);

    assert_eq!(resampled.nfreq(), original_nfreq * 2);
    assert_eq!(resampled.nports(), ntwk.nports());
}

// ============================================================================
// Frequency Slicing Tests (Python line 1582-1590)
// ============================================================================

#[test]
fn test_network_crop_frequency() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let orig_f = ntwk.f();
    let f_start = orig_f[0];
    let f_end = orig_f[orig_f.len() - 1];
    let f_mid = (f_start + f_end) / 2.0;

    // Crop to lower half of frequency range
    let cropped = ntwk.cropped(f_start, f_mid);

    // Cropped network should have fewer frequency points
    assert!(
        cropped.nfreq() < ntwk.nfreq(),
        "Cropped should have fewer points"
    );
    assert!(
        cropped.nfreq() > 0,
        "Cropped should have at least one point"
    );
    assert_eq!(
        cropped.nports(),
        ntwk.nports(),
        "Cropped should have same ports"
    );
}

// ============================================================================
// Interpolation Quality Tests
// ============================================================================

#[test]
fn test_interpolation_preserves_endpoints() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    // Create a resampled network with different number of points
    let resampled = ntwk.resample(ntwk.nfreq() + 10);

    // First and last frequencies should be approximately the same
    let orig_f = ntwk.f();
    let resamp_f = resampled.f();

    assert_relative_eq!(resamp_f[0], orig_f[0], epsilon = 1e9);
    assert_relative_eq!(
        resamp_f[resamp_f.len() - 1],
        orig_f[orig_f.len() - 1],
        epsilon = 1e9
    );
}

#[test]
fn test_interpolation_preserves_z0() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    let resampled = ntwk.resample(ntwk.nfreq() * 2);

    // Z0 should be preserved
    for p in 0..ntwk.nports() {
        assert_relative_eq!(resampled.z0()[p].re, ntwk.z0()[p].re, epsilon = 1e-10);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_interpolation_single_point() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to load ntwk1.s2p");

    // Resample to single point
    let single = ntwk.resample(1);

    assert_eq!(single.nfreq(), 1);
    assert_eq!(single.nports(), ntwk.nports());
}
