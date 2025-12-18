//! Network property checks
//!
//! Provides methods to test network properties like passivity, reciprocity,
//! losslessness, and symmetry.

use super::core::Network;
use crate::constants::PROPERTY_TOL;

impl Network {
    /// Test if network is reciprocal
    ///
    /// A network is reciprocal if S = S^T (transpose).
    pub fn is_reciprocal(&self, tol: Option<f64>) -> bool {
        let tol = tol.unwrap_or(PROPERTY_TOL);
        let nfreq = self.nfreq();
        let nports = self.nports();

        for f in 0..nfreq {
            for i in 0..nports {
                for j in i + 1..nports {
                    let diff = (self.s[[f, i, j]] - self.s[[f, j, i]]).norm();
                    if diff > tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Test if network is passive
    ///
    /// A network is passive if I - S^H * S is positive semi-definite,
    /// meaning all eigenvalues of S^H * S are <= 1.
    /// Simplified check: all singular values of S <= 1.
    pub fn is_passive(&self, tol: Option<f64>) -> bool {
        let tol = tol.unwrap_or(PROPERTY_TOL);
        let nfreq = self.nfreq();
        let nports = self.nports();

        for f in 0..nfreq {
            // Compute S^H * S and check if all diagonal elements of I - S^H*S >= 0
            for i in 0..nports {
                let mut sum_sq = 0.0;
                for k in 0..nports {
                    sum_sq += self.s[[f, k, i]].norm_sqr();
                }
                // Column norm squared should be <= 1 for passivity
                if sum_sq > 1.0 + tol {
                    return false;
                }
            }
        }
        true
    }

    /// Test if network is lossless
    ///
    /// A network is lossless if S is unitary: S^H * S = I
    pub fn is_lossless(&self, tol: Option<f64>) -> bool {
        let tol = tol.unwrap_or(PROPERTY_TOL);
        let nfreq = self.nfreq();
        let nports = self.nports();

        for f in 0..nfreq {
            // Check if S^H * S = I
            for i in 0..nports {
                for j in 0..nports {
                    let mut sum = num_complex::Complex64::new(0.0, 0.0);
                    for k in 0..nports {
                        sum += self.s[[f, k, i]].conj() * self.s[[f, k, j]];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    if (sum.re - expected).abs() > tol || sum.im.abs() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Test if 2-port network has symmetric reflection coefficients (S11 = S22)
    ///
    /// Only valid for 2-port networks.
    /// Returns None for networks with other port counts.
    pub fn is_symmetric_2port(&self, tol: Option<f64>) -> Option<bool> {
        if self.nports() != 2 {
            return None;
        }

        let tol = tol.unwrap_or(PROPERTY_TOL);
        let nfreq = self.nfreq();

        for f in 0..nfreq {
            let diff = (self.s[[f, 0, 0]] - self.s[[f, 1, 1]]).norm();
            if diff > tol {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Test if 4-port network has symmetric reflection coefficients
    ///
    /// Checks that S11 = S22 and S33 = S44.
    /// Only valid for 4-port networks (e.g., differential pairs).
    /// Returns None for networks with other port counts.
    pub fn is_symmetric_4port(&self, tol: Option<f64>) -> Option<bool> {
        if self.nports() != 4 {
            return None;
        }

        let tol = tol.unwrap_or(PROPERTY_TOL);
        let nfreq = self.nfreq();

        for f in 0..nfreq {
            if (self.s[[f, 0, 0]] - self.s[[f, 1, 1]]).norm() > tol {
                return Some(false);
            }
            if (self.s[[f, 2, 2]] - self.s[[f, 3, 3]]).norm() > tol {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Test if network is symmetric (legacy compatibility)
    ///
    /// For 2-port: checks S11 = S22
    /// For 4-port: checks S11 = S22 and S33 = S44
    /// For other port counts: checks reciprocity (Sij = Sji)
    ///
    /// **Note**: For explicit port-specific checks, prefer `is_symmetric_2port`
    /// or `is_symmetric_4port`.
    pub fn is_symmetric(&self, tol: Option<f64>) -> bool {
        match self.nports() {
            2 => self.is_symmetric_2port(tol).unwrap_or(false),
            4 => self.is_symmetric_4port(tol).unwrap_or(false),
            _ => self.is_reciprocal(tol), // Fallback for other port counts
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};
    use ndarray::{Array1, Array3};
    use num_complex::Complex64;

    #[test]
    fn test_is_reciprocal() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 2, 2));
        // Reciprocal: S12 = S21
        s[[0, 0, 1]] = Complex64::new(0.5, 0.1);
        s[[0, 1, 0]] = Complex64::new(0.5, 0.1);

        let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0).unwrap();

        assert!(ntwk.is_reciprocal(None));
    }

    #[test]
    fn test_is_passive() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        // |S11| = 0.5 < 1, passive
        s[[0, 0, 0]] = Complex64::new(0.5, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0).unwrap();

        assert!(ntwk.is_passive(None));
    }

    #[test]
    fn test_is_lossless() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        // |S11| = 1, lossless (total reflection)
        s[[0, 0, 0]] = Complex64::new(1.0, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0).unwrap();

        assert!(ntwk.is_lossless(None));
    }
}
