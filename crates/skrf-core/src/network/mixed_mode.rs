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
    /// Convert 4-port single-ended S-parameters to mixed-mode
    ///
    /// Port mapping (standard 4-port differential):
    /// - Port 0, 1: Differential pair 1 (positive, negative)
    /// - Port 2, 3: Differential pair 2 (positive, negative)
    ///
    /// The transformation matrix M is:
    /// ```text
    /// M = 1/√2 * | 1 -1  0  0 |  (d1)
    ///            | 0  0  1 -1 |  (d2)
    ///            | 1  1  0  0 |  (c1)
    ///            | 0  0  1  1 |  (c2)
    /// ```
    ///
    /// S_mm = M * S * M^T
    pub fn to_mixed_mode(&self) -> Option<MixedModeParams> {
        if self.nports() != 4 {
            return None;
        }

        let nfreq = self.nfreq();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        // Transformation matrix M
        let m: [[f64; 4]; 4] = [
            [inv_sqrt2, -inv_sqrt2, 0.0, 0.0], // d1
            [0.0, 0.0, inv_sqrt2, -inv_sqrt2], // d2
            [inv_sqrt2, inv_sqrt2, 0.0, 0.0],  // c1
            [0.0, 0.0, inv_sqrt2, inv_sqrt2],  // c2
        ];

        let mut sdd = Array3::<Complex64>::zeros((nfreq, 2, 2));
        let mut scc = Array3::<Complex64>::zeros((nfreq, 2, 2));
        let mut sdc = Array3::<Complex64>::zeros((nfreq, 2, 2));
        let mut scd = Array3::<Complex64>::zeros((nfreq, 2, 2));

        for f in 0..nfreq {
            // Compute S_mm = M * S * M^T
            let mut s_mm = [[Complex64::new(0.0, 0.0); 4]; 4];

            for i in 0..4 {
                for j in 0..4 {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for k in 0..4 {
                        for l in 0..4 {
                            sum += Complex64::new(m[i][k] * m[j][l], 0.0) * self.s[[f, k, l]];
                        }
                    }
                    s_mm[i][j] = sum;
                }
            }

            // Extract sub-matrices
            // Sdd: rows 0-1, cols 0-1
            sdd[[f, 0, 0]] = s_mm[0][0];
            sdd[[f, 0, 1]] = s_mm[0][1];
            sdd[[f, 1, 0]] = s_mm[1][0];
            sdd[[f, 1, 1]] = s_mm[1][1];

            // Sdc: rows 0-1, cols 2-3
            sdc[[f, 0, 0]] = s_mm[0][2];
            sdc[[f, 0, 1]] = s_mm[0][3];
            sdc[[f, 1, 0]] = s_mm[1][2];
            sdc[[f, 1, 1]] = s_mm[1][3];

            // Scd: rows 2-3, cols 0-1
            scd[[f, 0, 0]] = s_mm[2][0];
            scd[[f, 0, 1]] = s_mm[2][1];
            scd[[f, 1, 0]] = s_mm[3][0];
            scd[[f, 1, 1]] = s_mm[3][1];

            // Scc: rows 2-3, cols 2-3
            scc[[f, 0, 0]] = s_mm[2][2];
            scc[[f, 0, 1]] = s_mm[2][3];
            scc[[f, 1, 0]] = s_mm[3][2];
            scc[[f, 1, 1]] = s_mm[3][3];
        }

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
    ///
    /// CMRR = |Sdd21| / |Scd21|
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
}
