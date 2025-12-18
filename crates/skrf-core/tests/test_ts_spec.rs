//! Touchstone Specification Tests
//!
//! Migrated from skrf/skrf/io/tests/test_ts_spec.py
//!
//! Structure matches Python tests exactly where possible.
//!
//! Note:
//! - V2 Noise data not fully implemented (ignored assertions)
//! - V2 Per-port Reference impedance not fully implemented (ignored assertions validation)
//! - Write functionality tests placeholders included

use approx::assert_relative_eq;
use num_complex::Complex64;
use skrf_core::network::Network;
use skrf_core::touchstone::parser::{ParameterType, SParamFormat};

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/ts");

#[test]
fn test_ex_1() {
    let path = format!("{}/ex_1.ts", TEST_DATA_DIR);
    let _ts = Network::from_touchstone(&path).expect("Failed to load ex_1.ts");
}

#[test]
fn test_ex_2() {
    let path = format!("{}/ex_2.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_2.ts");

    // ref = rf.Network(
    //     f=np.arange(1,6),
    //     z=(np.arange(5) + 11) * np.exp(1j*np.arange(10,60,10) * np.pi / 180),
    //     f_unit="mhz"
    // )
    // Note: Rust Network stores S-parameters. Converted from Z in file?
    // ex_2.ts contains Z-parameters.
    // Network::from_touchstone parses file. If file says 'Z', S params in Network are converted.
    // We should check Z parameters of the loaded network to match the reference construction.

    let f_true = [1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6]; // 1-5 MHz

    // Check Frequency
    assert_eq!(ts.nfreq(), 5);
    let f = ts.frequency.f();
    for i in 0..5 {
        assert_relative_eq!(f[i], f_true[i], epsilon = 1e-3);
    }

    // Check Z parameters
    let z = ts.z(); // [nfreq, 1, 1] for 1-port? ex_2.ts is 1-port?
                    // ex_2.ts: [Number of Ports] 1

    for i in 0..5 {
        let mag = (i as f64) + 11.0;
        let deg = (i as f64 + 1.0) * 10.0;
        let expected_z = Complex64::from_polar(mag, deg.to_radians());

        let actual_z = z[[i, 0, 0]];
        // Allow some tolerance for text parsing
        assert_relative_eq!(actual_z.re, expected_z.re, epsilon = 1e-3);
        assert_relative_eq!(actual_z.im, expected_z.im, epsilon = 1e-3);
    }
}

#[test]
fn test_ex_2_write() {
    let path = format!("{}/ex_2.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_2.ts");

    let contents = ts
        .to_touchstone_contents(SParamFormat::MA, ParameterType::Z)
        .unwrap();
    let ts_read_back = Network::from_touchstone_content(&contents, 1).unwrap();

    assert_eq!(ts.nfreq(), ts_read_back.nfreq());
    for (v1, v2) in ts.s().iter().zip(ts_read_back.s().iter()) {
        assert_relative_eq!(v1.re, v2.re, epsilon = 1e-3);
        assert_relative_eq!(v1.im, v2.im, epsilon = 1e-3);
    }
}

#[test]
fn test_ex_3() {
    let path = format!("{}/ex_3.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_3.ts");

    // ref: f=[1,2] GHz, S matrix
    let f_true = [1.0e9, 2.0e9];

    assert_eq!(ts.nfreq(), 2);
    let f = ts.frequency.f();
    for i in 0..2 {
        assert_relative_eq!(f[i], f_true[i], epsilon = 1e-3);
    }

    // Data check:
    // f=1: 111, 121, 112, 122 (Row major? Python says [[111, 112], [121, 122]])
    // Wait, Python ref: s=[[[111, 112], [121, 122]]...] meaning S11=111, S12=112, S21=121, S22=122
    // File content ex_3.ts says: 1 111 0 121 0 112 0 122 0
    // Format: Version 2.0... [Two-Port Data Order] 21_12 means 11 21 12 22
    // So 111=S11, 121=S21, 112=S12, 122=S22
    // Which matches [[111, 112], [121, 122]]

    let _s = ts.s();
    // Helper to check S at freq index 0
    let check_s = |ts: &Network, f_idx, r, c, val: f64| {
        let s_val = ts.s[[f_idx, r, c]];
        assert_relative_eq!(s_val.re, val, epsilon = 1e-3);
        assert_relative_eq!(s_val.im, 0.0, epsilon = 1e-3);
    };

    // Freq 0 (1 GHz)
    check_s(&ts, 0, 0, 0, 111.0); // S11
    check_s(&ts, 0, 0, 1, 112.0); // S12
    check_s(&ts, 0, 1, 0, 121.0); // S21
    check_s(&ts, 0, 1, 1, 122.0); // S22

    // Freq 1 (2 GHz)
    check_s(&ts, 1, 0, 0, 211.0);
    check_s(&ts, 1, 0, 1, 212.0);
    check_s(&ts, 1, 1, 0, 221.0);
    check_s(&ts, 1, 1, 1, 222.0);
}

#[test]
fn test_ex_4() {
    let path = format!("{}/ex_4.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_4.ts");

    // Only 1 freq point
    assert_eq!(ts.nfreq(), 1);

    // Verify S parameters (partial check)
    // s=[[[11, 12, 13, 14], ...]]
    let s = ts.s();
    assert_relative_eq!(s[[0, 0, 0]].re, 11.0, epsilon = 1e-3);
    assert_relative_eq!(s[[0, 3, 3]].re, 44.0, epsilon = 1e-3);

    // Verify Z0 - Python: z0=[50, 75, 0.01, 0.01]
    assert_relative_eq!(ts.z0[0].re, 50.0);
    assert_relative_eq!(ts.z0[1].re, 75.0);
    assert_relative_eq!(ts.z0[2].re, 0.01);
    assert_relative_eq!(ts.z0[3].re, 0.01);
}

#[test]
fn test_ts_example_5_6() {
    // Tests ex_5.ts and ex_6.ts
    // Both should contain the same data but different formatting/mixed mode
    // Reference constructed manually in Python

    for filename in ["ex_5.ts", "ex_6.ts"] {
        let path = format!("{}/{}", TEST_DATA_DIR, filename);
        let ts = Network::from_touchstone(&path)
            .unwrap_or_else(|_| panic!("Failed to load {}", filename));

        // 4 ports, 2 freq points (5e9, 6e9)
        assert_eq!(ts.nports(), 4);
        assert_eq!(ts.nfreq(), 2);

        // Check S11 magnitude at 5GHz = 0.6
        // Check S11 phase at 5GHz = 161.24
        let s = ts.s();
        let s11 = s[[0, 0, 0]];
        assert_relative_eq!(s11.norm(), 0.6, epsilon = 1e-3);
        assert_relative_eq!(s11.arg().to_degrees(), 161.24, epsilon = 1e-2);
    }
}

#[test]
fn test_ts_example_7() {
    let path = format!("{}/ex_7.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_7.ts");

    // f = 100 to 500 MHz step 100 (5 points)
    assert_eq!(ts.nfreq(), 5);

    // Check first Z value: 74.25 * exp(j * -4 deg)
    let z = ts.z();
    let z00 = z[[0, 0, 0]];

    let expected = Complex64::from_polar(74.25, (-4.0_f64).to_radians());
    assert_relative_eq!(z00.re, expected.re, epsilon = 1e-3);
    assert_relative_eq!(z00.im, expected.im, epsilon = 1e-3);
}

#[test]
fn test_example_8() {
    let path = format!("{}/ex_8.s1p", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_8.s1p");

    assert_eq!(ts.nports(), 1);
    assert_eq!(ts.nfreq(), 1);

    // S11 = 0.894 ∠ -12.136
    let s = ts.s();
    let s11 = s[[0, 0, 0]];
    assert_relative_eq!(s11.norm(), 0.894, epsilon = 1e-3);
    assert_relative_eq!(s11.arg().to_degrees(), -12.136, epsilon = 1e-2);
}

#[test]
fn test_ts_example_9_10() {
    let n1 = Network::from_touchstone(&format!("{}/ex_9.s1p", TEST_DATA_DIR)).unwrap();
    let n2 = Network::from_touchstone(&format!("{}/ex_10.ts", TEST_DATA_DIR)).unwrap();

    // assert np.allclose(ex_9.z, ex_10.z)
    let z1 = n1.z();
    let z2 = n2.z();

    assert_eq!(z1.shape(), z2.shape());
    for (val1, val2) in z1.iter().zip(z2.iter()) {
        assert_relative_eq!(val1.re, val2.re, epsilon = 1e-3);
        assert_relative_eq!(val1.im, val2.im, epsilon = 1e-3);
    }
}

#[test]
fn test_ex_9_write() {
    let path = format!("{}/ex_9.s1p", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).unwrap();
    let contents = ts
        .to_touchstone_contents(SParamFormat::MA, ParameterType::Z)
        .unwrap();
    let ts_read_back = Network::from_touchstone_content(&contents, 1).unwrap();
    assert_eq!(ts.nfreq(), ts_read_back.nfreq());
}

#[test]
fn test_ex_10_write() {
    let path = format!("{}/ex_10.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).unwrap();
    let contents = ts
        .to_touchstone_contents(SParamFormat::MA, ParameterType::Z)
        .unwrap();
    let ts_read_back = Network::from_touchstone_content(&contents, 1).unwrap();
    assert_eq!(ts.nfreq(), ts_read_back.nfreq());
}

#[test]
fn test_ts_example_11_12() {
    let n1 = Network::from_touchstone(&format!("{}/ex_11.s2p", TEST_DATA_DIR)).unwrap();
    let n2 = Network::from_touchstone(&format!("{}/ex_12.ts", TEST_DATA_DIR)).unwrap();

    // assert ex_11 == ex_12 (S-parameters equal)
    let s1 = n1.s();
    let s2 = n2.s();

    assert_eq!(s1.shape(), s2.shape());
    for (val1, val2) in s1.iter().zip(s2.iter()) {
        assert_relative_eq!(val1.re, val2.re, epsilon = 1e-3);
        assert_relative_eq!(val1.im, val2.im, epsilon = 1e-3);
    }
}

#[test]
fn test_ts_example_12_load() {
    let _n_s = Network::from_touchstone(&format!("{}/ex_12.ts", TEST_DATA_DIR)).unwrap();
    let _n_g = Network::from_touchstone(&format!("{}/ex_12_g.ts", TEST_DATA_DIR)).unwrap();
}

#[test]
fn test_ex_12_g_write() {
    let path = format!("{}/ex_12_g.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).unwrap();
    let contents = ts
        .to_touchstone_contents(SParamFormat::MA, ParameterType::G)
        .unwrap();
    let ts_read_back = Network::from_touchstone_content(&contents, 2).unwrap();
    assert_eq!(ts.nfreq(), ts_read_back.nfreq());
}

#[test]
fn test_ex_12_h_write() {
    let path = format!("{}/ex_12.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).unwrap();
    let contents = ts
        .to_touchstone_contents(SParamFormat::MA, ParameterType::H)
        .unwrap();
    let ts_read_back = Network::from_touchstone_content(&contents, 2).unwrap();
    assert_eq!(ts.nfreq(), ts_read_back.nfreq());
}

#[test]
fn test_ts_example_13() {
    let path = format!("{}/ex_13.s2p", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_13.s2p");

    assert_eq!(ts.nfreq(), 3);

    // Check S-parameters at freq index 0 (1 GHz)
    // Python ref: 3.926e-01-0.1211j
    let s00 = ts.s[[0, 0, 0]];
    assert_relative_eq!(s00.re, 0.3926, epsilon = 1e-4);
    assert_relative_eq!(s00.im, -0.1211, epsilon = 1e-4);
}

#[test]
fn test_ts_example_14() {
    let path = format!("{}/ex_14.s4p", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_14.s4p");
    assert_eq!(ts.nports(), 4);
    assert_eq!(ts.nfreq(), 3);

    // Check S11 magnitude at 5GHz = 0.6
    let s00 = ts.s[[0, 0, 0]];
    assert_relative_eq!(s00.norm(), 0.6, epsilon = 1e-3);
}

#[test]
fn test_ts_example_17() {
    let path = format!("{}/ex_17.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_17.ts");

    assert!(ts.noisy, "ex_17.ts should contain noise data");
    assert_eq!(ts.nports(), 2);
    assert_eq!(ts.nfreq(), 2);

    // S11 magnitude at 2GHz = 0.95
    let s00 = ts.s[[0, 0, 0]];
    assert_relative_eq!(s00.norm(), 0.95, epsilon = 1e-3);
}

#[test]
fn test_ts_example_16() {
    let path = format!("{}/ex_16.ts", TEST_DATA_DIR);
    let ts = Network::from_touchstone(&path).expect("Failed to load ex_16.ts");

    assert_eq!(
        ts.mixed_mode_order,
        vec!["D2,3", "D6,5", "C2,3", "C6,5", "S4", "S1"]
    );

    assert_eq!(ts.nports(), 6);
    // Check raw Z0 from file (single-ended)
    assert_relative_eq!(ts.z0[0].re, 50.0);
    assert_relative_eq!(ts.z0[1].re, 75.0);
    assert_relative_eq!(ts.z0[2].re, 75.0);
    assert_relative_eq!(ts.z0[3].re, 50.0);
    assert_relative_eq!(ts.z0[4].re, 0.01);
    assert_relative_eq!(ts.z0[5].re, 0.01);
}
