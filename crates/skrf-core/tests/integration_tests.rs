//! Integration tests for Touchstone file parsing
//!
//! These tests read real Touchstone files and verify correct parsing.

use approx::assert_relative_eq;
use skrf_core::network::Network;
use skrf_core::touchstone::Touchstone;

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

#[test]
fn test_read_simple_touchstone() {
    let path = format!("{}/simple_touchstone.s2p", TEST_DATA_DIR);
    let ts = Touchstone::from_file(&path).expect("Failed to load simple_touchstone.s2p");

    // Verify basic properties
    assert_eq!(ts.nports, 2);
    assert_eq!(ts.nfreq(), 2);
    assert_eq!(ts.z0[0], 50.0); // Note: file has (50+50j) but we only parse real part

    // Verify frequency data (1.0 GHz and 1.1 GHz)
    let f = ts.frequency.f();
    assert_relative_eq!(f[0], 1.0e9, epsilon = 1e3);
    assert_relative_eq!(f[1], 1.1e9, epsilon = 1e3);

    // Verify S-parameter data at first frequency point
    // Line: 1.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0
    // This is: S11(1+2j), S21(3+4j), S12(5+6j), S22(7+8j)
    let s = &ts.s[0];
    assert_relative_eq!(s[0][0].re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(s[0][0].im, 2.0, epsilon = 1e-10);
    assert_relative_eq!(s[1][0].re, 3.0, epsilon = 1e-10); // S21
    assert_relative_eq!(s[1][0].im, 4.0, epsilon = 1e-10);
    assert_relative_eq!(s[0][1].re, 5.0, epsilon = 1e-10); // S12
    assert_relative_eq!(s[0][1].im, 6.0, epsilon = 1e-10);
    assert_relative_eq!(s[1][1].re, 7.0, epsilon = 1e-10); // S22
    assert_relative_eq!(s[1][1].im, 8.0, epsilon = 1e-10);
}

#[test]
fn test_read_ntwk1_touchstone() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ts = Touchstone::from_file(&path).expect("Failed to load ntwk1.s2p");

    assert_eq!(ts.nports, 2);
    assert!(ts.nfreq() > 0);
    assert_eq!(ts.z0[0], 50.0);
}

#[test]
fn test_read_short_s1p() {
    let path = format!("{}/short.s1p", TEST_DATA_DIR);
    let ts = Touchstone::from_file(&path).expect("Failed to load short.s1p");

    assert_eq!(ts.nports, 1);
    assert!(ts.nfreq() > 0);

    // A short should have S11 ≈ -1 (reflection coefficient)
    // Check that values are reasonable (magnitude close to 1)
    for s_matrix in &ts.s {
        let s11 = s_matrix[0][0];
        let mag = s11.norm();
        assert!(
            mag > 0.9 && mag < 1.1,
            "Short S11 magnitude should be ≈ 1, got {}",
            mag
        );
    }
}

#[test]
fn test_network_from_touchstone() {
    let path = format!("{}/simple_touchstone.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to create Network");

    assert_eq!(ntwk.nports(), 2);
    assert_eq!(ntwk.nfreq(), 2);
    assert_eq!(ntwk.z0()[0].re, 50.0);

    // Verify S-parameter access
    let s = ntwk.s();
    assert_relative_eq!(s[[0, 0, 0]].re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(s[[0, 0, 0]].im, 2.0, epsilon = 1e-10);
}

#[test]
fn test_network_s_db() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to create Network");

    let s_db = ntwk.s_db();
    let s_mag = ntwk.s_mag();
    let s_deg = ntwk.s_deg();

    // Verify dimensions match
    assert_eq!(s_db.shape(), ntwk.s().shape());
    assert_eq!(s_mag.shape(), ntwk.s().shape());
    assert_eq!(s_deg.shape(), ntwk.s().shape());

    // Verify dB calculation: 20*log10(mag)
    for i in 0..ntwk.nfreq() {
        let mag = s_mag[[i, 0, 0]];
        let db = s_db[[i, 0, 0]];
        if mag > 0.0 {
            assert_relative_eq!(db, 20.0 * mag.log10(), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_network_z_parameters() {
    let path = format!("{}/simple_touchstone.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to create Network");

    // Get Z-parameters
    let z = ntwk.z();

    // Z-parameters should have same shape as S-parameters
    assert_eq!(z.shape(), ntwk.s().shape());
}

#[test]
fn test_network_y_parameters() {
    let path = format!("{}/simple_touchstone.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to create Network");

    // Get Y-parameters
    let y = ntwk.y();

    // Y-parameters should have same shape as S-parameters
    assert_eq!(y.shape(), ntwk.s().shape());
}
