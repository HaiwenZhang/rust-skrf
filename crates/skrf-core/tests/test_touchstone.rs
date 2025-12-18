//! Touchstone I/O Tests
//!
//! Migrated from skrf/skrf/io/tests/test_touchstone.py

use approx::assert_relative_eq;
use num_complex::Complex64;
use skrf_core::network::Network;
use skrf_core::touchstone::Touchstone;

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

/// Test reading simple_touchstone.s2p and compare with known true values
/// Migrated from: test_read_data
#[allow(clippy::needless_range_loop)]
#[test]
fn test_read_data() {
    let path = format!("{}/simple_touchstone.s2p", TEST_DATA_DIR);
    let ts = Touchstone::from_file(&path).expect("Failed to load file");

    // Expected values from Python test
    let f_true = [1.0e9, 1.1e9];
    let s_true = [
        // First frequency point
        [
            [Complex64::new(1.0, 2.0), Complex64::new(5.0, 6.0)],
            [Complex64::new(3.0, 4.0), Complex64::new(7.0, 8.0)],
        ],
        // Second frequency point
        [
            [Complex64::new(9.0, 10.0), Complex64::new(13.0, 14.0)],
            [Complex64::new(11.0, 12.0), Complex64::new(15.0, 16.0)],
        ],
    ];

    // Verify frequency
    assert_eq!(ts.nfreq(), 2);
    let f = ts.frequency.f();
    assert_relative_eq!(f[0], f_true[0], epsilon = 1e3);
    assert_relative_eq!(f[1], f_true[1], epsilon = 1e3);

    // Verify S-parameters
    for freq_idx in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let expected = s_true[freq_idx][i][j];
                let actual = ts.s[freq_idx][i][j];
                assert_relative_eq!(actual.re, expected.re, epsilon = 1e-10);
                assert_relative_eq!(actual.im, expected.im, epsilon = 1e-10);
            }
        }
    }

    // Verify z0
    assert_eq!(ts.z0[0], 50.0);
}

/// Test reading short.s1p - should have S11 ≈ -1
#[test]
fn test_read_short() {
    let path = format!("{}/short.s1p", TEST_DATA_DIR);
    let ts = Touchstone::from_file(&path).expect("Failed to load file");

    assert_eq!(ts.nports, 1);
    assert!(ts.nfreq() > 0);

    // Short circuit should have |S11| ≈ 1 and phase ≈ 180°
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

/// Test reading open.s1p - should have S11 ≈ +1
#[test]
fn test_read_open() {
    let path = format!("{}/open.s1p", TEST_DATA_DIR);
    let ts = Touchstone::from_file(&path).expect("Failed to load file");

    assert_eq!(ts.nports, 1);
    assert!(ts.nfreq() > 0);

    // Open circuit should have |S11| ≈ 1 and phase ≈ 0°
    for s_matrix in &ts.s {
        let s11 = s_matrix[0][0];
        let mag = s11.norm();
        assert!(
            mag > 0.9 && mag < 1.1,
            "Open S11 magnitude should be ≈ 1, got {}",
            mag
        );
    }
}

/// Test reading match.s1p - should have S11 ≈ 0
#[test]
fn test_read_match() {
    let path = format!("{}/match.s1p", TEST_DATA_DIR);
    let ts = Touchstone::from_file(&path).expect("Failed to load file");

    assert_eq!(ts.nports, 1);
    assert!(ts.nfreq() > 0);

    // Matched load should have |S11| ≈ 0
    for s_matrix in &ts.s {
        let s11 = s_matrix[0][0];
        let mag = s11.norm();
        assert!(mag < 0.1, "Match S11 magnitude should be ≈ 0, got {}", mag);
    }
}

/// Test reading 3-port file with comments
#[test]
fn test_read_comments() {
    let path = format!("{}/comments.s3p", TEST_DATA_DIR);
    let ts = Touchstone::from_file(&path).expect("Failed to load file");

    assert_eq!(ts.nports, 3);
    assert!(!ts.comments.is_empty(), "Should have comments");
}

/// Test reading ntwk1.s2p through ntwk3.s2p
#[test]
fn test_read_ntwk_files() {
    for i in 1..=3 {
        let path = format!("{}/ntwk{}.s2p", TEST_DATA_DIR, i);
        let ts =
            Touchstone::from_file(&path).unwrap_or_else(|_| panic!("Failed to load ntwk{}.s2p", i));

        assert_eq!(ts.nports, 2);
        assert!(ts.nfreq() > 0);
        assert_eq!(ts.z0[0], 50.0);
    }
}

/// Test Network creation from Touchstone files
#[test]
fn test_network_from_all_test_files() {
    let files = [
        "simple_touchstone.s2p",
        "ntwk1.s2p",
        "short.s1p",
        "open.s1p",
        "match.s1p",
    ];

    for file in &files {
        let path = format!("{}/{}", TEST_DATA_DIR, file);
        let ntwk = Network::from_touchstone(&path)
            .unwrap_or_else(|_| panic!("Failed to create Network from {}", file));

        assert!(ntwk.nfreq() > 0, "{} should have frequency points", file);
        assert!(ntwk.nports() > 0, "{} should have ports", file);
    }
}

/// Test S-parameter conversions on real data
#[test]
fn test_network_parameter_conversions() {
    let path = format!("{}/ntwk1.s2p", TEST_DATA_DIR);
    let ntwk = Network::from_touchstone(&path).expect("Failed to create Network");

    // Get various representations
    let s_db = ntwk.s_db();
    let s_mag = ntwk.s_mag();
    let s_deg = ntwk.s_deg();
    let z = ntwk.z();
    let y = ntwk.y();

    // Verify all have correct dimensions
    let expected_shape = [ntwk.nfreq(), ntwk.nports(), ntwk.nports()];
    assert_eq!(s_db.shape(), expected_shape);
    assert_eq!(s_mag.shape(), expected_shape);
    assert_eq!(s_deg.shape(), expected_shape);
    assert_eq!(z.shape(), expected_shape);
    assert_eq!(y.shape(), expected_shape);

    // Verify dB is 20*log10(mag)
    for i in 0..ntwk.nfreq() {
        for m in 0..ntwk.nports() {
            for n in 0..ntwk.nports() {
                let mag = s_mag[[i, m, n]];
                let db = s_db[[i, m, n]];
                if mag > 1e-15 {
                    assert_relative_eq!(db, 20.0 * mag.log10(), epsilon = 1e-10);
                }
            }
        }
    }
}
