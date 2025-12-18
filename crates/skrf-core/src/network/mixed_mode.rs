//! Mixed-mode S-parameter conversions
//!
//! Provides conversions between single-ended and differential/common-mode
//! S-parameters for multi-port networks.
//!
//! Mixed-mode S-parameters describe the behavior of differential signaling
//! systems in terms of differential and common mode.

use ndarray::{Array2, Array3};
use num_complex::Complex64;

use super::core::Network;
use crate::constants::NEAR_ZERO;

/// Mixed-mode S-parameter matrix for a 4-port network
///
/// Organized as:
/// ```text
/// | Sdd11 Sdd12 | Sdc11 Sdc12 |
/// | Sdd21 Sdd22 | Sdc21 Sdc22 |
/// |-------------|-------------|
/// | Scd11 Scd12 | Scc11 Scc12 |
/// | Scd21 Scd22 | Scc21 Scc22 |
/// ```
#[derive(Debug)]
pub struct MixedModeParams {
    /// Differential-to-differential S-parameters [nfreq, 2, 2]
    pub sdd: Array3<Complex64>,
    /// Common-to-common S-parameters [nfreq, 2, 2]
    pub scc: Array3<Complex64>,
    /// Differential-to-common (mode conversion) [nfreq, 2, 2]
    pub sdc: Array3<Complex64>,
    /// Common-to-differential (mode conversion) [nfreq, 2, 2]
    pub scd: Array3<Complex64>,
}

impl Network {
    /// Convert single-ended S-parameters to generalized mixed-mode
    ///
    /// `port_pairs` is a list of (positive, negative) port indices.
    /// Any ports not in `port_pairs` are treated as single-ended and included
    /// after the mixed-mode ports.
    ///
    /// The resulting ports will be ordered as:
    /// [Differential modes, Common modes, Remaining single-ended modes]
    pub fn se2gmm(&self, port_pairs: &[(usize, usize)]) -> Array3<Complex64> {
        let nfreq = self.nfreq();
        let nports = self.nports();
        let n_pairs = port_pairs.len();

        // Find remaining single-ended ports
        let mut paired_ports = std::collections::HashSet::new();
        for &(p, n) in port_pairs {
            paired_ports.insert(p);
            paired_ports.insert(n);
        }

        let mut single_ports = Vec::new();
        for i in 0..nports {
            if !paired_ports.contains(&i) {
                single_ports.push(i);
            }
        }

        let out_ports = 2 * n_pairs + single_ports.len();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        // Build transformation matrix M [out_ports, nports]
        let mut m = Array2::<Complex64>::zeros((out_ports, nports));

        // Differential modes (D1, D2, ...)
        for i in 0..n_pairs {
            let (p, n) = port_pairs[i];
            m[[i, p]] = Complex64::new(inv_sqrt2, 0.0);
            m[[i, n]] = Complex64::new(-inv_sqrt2, 0.0);
        }

        // Common modes (C1, C2, ...)
        for i in 0..n_pairs {
            let (p, n) = port_pairs[i];
            m[[n_pairs + i, p]] = Complex64::new(inv_sqrt2, 0.0);
            m[[n_pairs + i, n]] = Complex64::new(inv_sqrt2, 0.0);
        }

        // Single-ended modes
        for i in 0..single_ports.len() {
            let p = single_ports[i];
            m[[2 * n_pairs + i, p]] = Complex64::new(1.0, 0.0);
        }

        let mut s_mm = Array3::<Complex64>::zeros((nfreq, out_ports, out_ports));
        let m_t = m.t().to_owned();

        for f in 0..nfreq {
            let s_f = self.s.slice(ndarray::s![f, .., ..]);
            // S_mm = M * S * M^T
            let term = m.dot(&s_f);
            let s_mm_f = term.dot(&m_t);
            s_mm.slice_mut(ndarray::s![f, .., ..]).assign(&s_mm_f);
        }

        s_mm
    }

    /// Convert 4-port single-ended S-parameters to mixed-mode
    ///
    /// Port mapping (standard 4-port differential):
    /// - Port 0, 1: Differential pair 1 (positive, negative)
    /// - Port 2, 3: Differential pair 2 (positive, negative)
    pub fn to_mixed_mode(&self) -> Option<MixedModeParams> {
        if self.nports() != 4 {
            return None;
        }

        let s_mm = self.se2gmm(&[(0, 1), (2, 3)]);

        // Extract sub-matrices
        let sdd = s_mm.slice(ndarray::s![.., 0..2, 0..2]).to_owned();
        let sdc = s_mm.slice(ndarray::s![.., 0..2, 2..4]).to_owned();
        let scd = s_mm.slice(ndarray::s![.., 2..4, 0..2]).to_owned();
        let scc = s_mm.slice(ndarray::s![.., 2..4, 2..4]).to_owned();

        Some(MixedModeParams { sdd, scc, sdc, scd })
    }

    /// Get differential insertion loss (Sdd21) in dB
    pub fn sdd21_db(&self) -> Option<Array2<f64>> {
        let mm = self.to_mixed_mode()?;
        let nfreq = mm.sdd.shape()[0];
        let mut result = Array2::<f64>::zeros((nfreq, 1));

        for f in 0..nfreq {
            result[[f, 0]] = 20.0 * mm.sdd[[f, 1, 0]].norm().log10();
        }

        Some(result)
    }

    /// Get differential return loss (Sdd11) in dB
    pub fn sdd11_db(&self) -> Option<Array2<f64>> {
        let mm = self.to_mixed_mode()?;
        let nfreq = mm.sdd.shape()[0];
        let mut result = Array2::<f64>::zeros((nfreq, 1));

        for f in 0..nfreq {
            result[[f, 0]] = 20.0 * mm.sdd[[f, 0, 0]].norm().log10();
        }

        Some(result)
    }

    /// Get common-mode rejection ratio in dB
    pub fn cmrr_db(&self) -> Option<Array2<f64>> {
        let mm = self.to_mixed_mode()?;
        let nfreq = mm.sdd.shape()[0];
        let mut result = Array2::<f64>::zeros((nfreq, 1));

        for f in 0..nfreq {
            let sdd21 = mm.sdd[[f, 1, 0]].norm();
            let scd21 = mm.scd[[f, 1, 0]].norm();

            if scd21 > NEAR_ZERO {
                result[[f, 0]] = 20.0 * (sdd21 / scd21).log10();
            } else {
                result[[f, 0]] = f64::INFINITY;
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_mixed_mode_conversion() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        // Create an ideal differential thru (ports 0-1 to 2-3)
        let mut s = Array3::<Complex64>::zeros((1, 4, 4));

        // S21 and S31 (thru paths): assume symmetric differential
        s[[0, 2, 0]] = Complex64::new(0.707, 0.0); // Port 0 -> Port 2
        s[[0, 3, 1]] = Complex64::new(0.707, 0.0); // Port 1 -> Port 3
        s[[0, 0, 2]] = Complex64::new(0.707, 0.0); // Port 2 -> Port 0
        s[[0, 1, 3]] = Complex64::new(0.707, 0.0); // Port 3 -> Port 1

        let z0 = Array1::from_elem(4, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);

        let mm = ntwk.to_mixed_mode();
        assert!(mm.is_some());

        let mm = mm.unwrap();
        assert_eq!(mm.sdd.shape(), &[1, 2, 2]);
        assert_eq!(mm.scc.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_generalized_mixed_mode() {
        // Create 4-port network
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
        let mut s = Array3::<Complex64>::zeros((1, 4, 4));

        // Simple case: Thru between 0-2 and 1-3
        s[[0, 0, 2]] = Complex64::new(1.0, 0.0);
        s[[0, 2, 0]] = Complex64::new(1.0, 0.0);
        s[[0, 1, 3]] = Complex64::new(1.0, 0.0);
        s[[0, 3, 1]] = Complex64::new(1.0, 0.0);

        let z0 = Array1::<Complex64>::from_elem(4, Complex64::new(50.0, 0.0));
        let net = Network::new(freq, s, z0);

        // Standard 4-port mapping: (0,1) and (2,3)
        let s_mm = net.se2gmm(&[(0, 1), (2, 3)]);

        // Sdd21 should be 1.0 (D2 -> D1 thru)
        // Index mapping: d1=0, d2=1, c1=2, c2=3
        // Sdd21 should be 1.0 (D2 -> D1 thru)
        // Index mapping: d1=0, d2=1, c1=2, c2=3
        assert!((s_mm[[0, 0, 1]].norm() - 1.0).abs() < 1e-12);
        assert!((s_mm[[0, 1, 0]].norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_6port_mixed_mode() {
        // Create 6-port network (3 differential pairs)
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
        let mut s = Array3::<Complex64>::zeros((1, 6, 6));

        // Differential thru from D1(0,1) to D2(2,3)
        // D1+ = (0-1)/sqrt(2), D2+ = (2-3)/sqrt(2)
        // S(0,2)=0.5, S(1,3)=0.5, S(0,3)=-0.5, S(1,2)=-0.5
        s[[0, 0, 2]] = Complex64::new(0.5, 0.0);
        s[[0, 1, 3]] = Complex64::new(0.5, 0.0);
        s[[0, 0, 3]] = Complex64::new(-0.5, 0.0);
        s[[0, 1, 2]] = Complex64::new(-0.5, 0.0);

        // Symmetric
        s[[0, 2, 0]] = s[[0, 0, 2]];
        s[[0, 3, 1]] = s[[0, 1, 3]];
        s[[0, 3, 0]] = s[[0, 0, 3]];
        s[[0, 2, 1]] = s[[0, 1, 2]];

        let z0 = Array1::<Complex64>::from_elem(6, Complex64::new(50.0, 0.0));
        let net = Network::new(freq, s, z0);

        let pairs = &[(0, 1), (2, 3), (4, 5)];
        let s_mm = net.se2gmm(pairs);

        // n_pairs = 3
        // Indices: D1=0, D2=1, D3=2, C1=3, C2=4, C3=5
        // Sdd21 (index [0,1]) should be 1.0
        assert_relative_eq!(s_mm[[0, 0, 1]].norm(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_8port_mixed_mode_custom_mapping() {
        // Create 8-port network
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);
        let mut s = Array3::<Complex64>::zeros((1, 8, 8));

        // Mixed Mode Thru: D1(0,7) -> D2(3,4)
        // S(0,3)=0.5, S(7,4)=0.5, S(0,4)=-0.5, S(7,3)=-0.5
        s[[0, 0, 3]] = Complex64::new(0.5, 0.0);
        s[[0, 7, 4]] = Complex64::new(0.5, 0.0);
        s[[0, 0, 4]] = Complex64::new(-0.5, 0.0);
        s[[0, 7, 3]] = Complex64::new(-0.5, 0.0);

        // Symmetric
        s[[0, 3, 0]] = s[[0, 0, 3]];
        s[[0, 4, 7]] = s[[0, 7, 4]];
        s[[0, 4, 0]] = s[[0, 0, 4]];
        s[[0, 3, 7]] = s[[0, 7, 3]];

        let z0 = Array1::<Complex64>::from_elem(8, Complex64::new(50.0, 0.0));
        let net = Network::new(freq, s, z0);

        let pairs = &[(0, 7), (3, 4), (1, 2), (5, 6)];
        let s_mm = net.se2gmm(pairs);

        // n_pairs = 4
        // Indices: D1=0, D2=1, D3=2, D4=3, C1=4, C2=5, C3=6, C4=7
        assert_relative_eq!(s_mm[[0, 0, 1]].norm(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(s_mm[[0, 1, 0]].norm(), 1.0, epsilon = 1e-12);
    }
}
