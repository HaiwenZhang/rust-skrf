//! Active network parameters
//!
//! Provides active S, Z, Y parameters and VSWR for phased array applications.
//!
//! Active parameters represent the effective reflection/impedance when
//! multiple ports are excited simultaneously with specific amplitudes.

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::core::Network;
use crate::constants::NEAR_ZERO;

impl Network {
    /// Calculate active S-parameters for a given excitation
    ///
    /// The active S-parameter at port m is:
    /// active(s)_m = Σ s_mi * (a_i / a_m)
    ///
    /// # Arguments
    /// * `a` - Complex excitation amplitudes for each port
    ///
    /// # Returns
    /// Active S-parameters array of shape [nfreq, nports]
    pub fn s_active(&self, a: &Array1<Complex64>) -> Option<Array2<Complex64>> {
        let nports = self.nports();
        if a.len() != nports {
            return None;
        }

        let nfreq = self.nfreq();
        let mut s_act = Array2::<Complex64>::zeros((nfreq, nports));

        for f in 0..nfreq {
            for m in 0..nports {
                // Skip if excitation at port m is zero
                if a[m].norm() < NEAR_ZERO {
                    continue;
                }

                let inv_am = Complex64::new(1.0, 0.0) / a[m];
                let mut sum = Complex64::new(0.0, 0.0);

                for i in 0..nports {
                    sum += self.s[[f, m, i]] * a[i] * inv_am;
                }

                s_act[[f, m]] = sum;
            }
        }

        Some(s_act)
    }

    /// Calculate active Z-parameters for a given excitation
    ///
    /// active(z)_m = z0_m * (1 + active(s)_m) / (1 - active(s)_m)
    ///
    /// # Arguments
    /// * `a` - Complex excitation amplitudes for each port
    pub fn z_active(&self, a: &Array1<Complex64>) -> Option<Array2<Complex64>> {
        let s_act = self.s_active(a)?;
        let nfreq = self.nfreq();
        let nports = self.nports();
        let mut z_act = Array2::<Complex64>::zeros((nfreq, nports));

        for f in 0..nfreq {
            for m in 0..nports {
                let s = s_act[[f, m]];
                let one = Complex64::new(1.0, 0.0);
                let denom = one - s;

                if denom.norm() > NEAR_ZERO {
                    z_act[[f, m]] = self.z0[m] * (one + s) / denom;
                } else {
                    // Open circuit approximation
                    z_act[[f, m]] = Complex64::new(f64::INFINITY, 0.0);
                }
            }
        }

        Some(z_act)
    }

    /// Calculate active Y-parameters for a given excitation
    ///
    /// active(y)_m = y0_m * (1 - active(s)_m) / (1 + active(s)_m)
    ///
    /// # Arguments
    /// * `a` - Complex excitation amplitudes for each port
    pub fn y_active(&self, a: &Array1<Complex64>) -> Option<Array2<Complex64>> {
        let s_act = self.s_active(a)?;
        let nfreq = self.nfreq();
        let nports = self.nports();
        let mut y_act = Array2::<Complex64>::zeros((nfreq, nports));

        for f in 0..nfreq {
            for m in 0..nports {
                let s = s_act[[f, m]];
                let one = Complex64::new(1.0, 0.0);
                let y0 = one / self.z0[m]; // Admittance
                let denom = one + s;

                if denom.norm() > NEAR_ZERO {
                    y_act[[f, m]] = y0 * (one - s) / denom;
                } else {
                    // Short circuit approximation
                    y_act[[f, m]] = Complex64::new(f64::INFINITY, 0.0);
                }
            }
        }

        Some(y_act)
    }

    /// Calculate active VSWR for a given excitation
    ///
    /// active(vswr)_m = (1 + |active(s)_m|) / (1 - |active(s)_m|)
    ///
    /// # Arguments
    /// * `a` - Complex excitation amplitudes for each port
    pub fn vswr_active(&self, a: &Array1<Complex64>) -> Option<Array2<f64>> {
        let s_act = self.s_active(a)?;
        let nfreq = self.nfreq();
        let nports = self.nports();
        let mut vswr_act = Array2::<f64>::zeros((nfreq, nports));

        for f in 0..nfreq {
            for m in 0..nports {
                let mag = s_act[[f, m]].norm();
                if mag >= 1.0 {
                    vswr_act[[f, m]] = f64::INFINITY;
                } else {
                    vswr_act[[f, m]] = (1.0 + mag) / (1.0 - mag);
                }
            }
        }

        Some(vswr_act)
    }

    /// Calculate stability circle for a 2-port network
    ///
    /// Returns the center and radius of the stability circle for the given port.
    /// - target_port = 0: Source stability circle
    /// - target_port = 1: Load stability circle
    ///
    /// # Returns
    /// (centers, radii) where centers is `[nfreq]` complex values and radii is `[nfreq]` real values
    pub fn stability_circle(&self, target_port: usize) -> Option<(Array1<Complex64>, Array1<f64>)> {
        if self.nports() != 2 || target_port > 1 {
            return None;
        }

        let nfreq = self.nfreq();
        let mut centers = Array1::<Complex64>::zeros(nfreq);
        let mut radii = Array1::<f64>::zeros(nfreq);

        for f in 0..nfreq {
            let s11 = self.s[[f, 0, 0]];
            let s12 = self.s[[f, 0, 1]];
            let s21 = self.s[[f, 1, 0]];
            let s22 = self.s[[f, 1, 1]];

            // Determinant: D = S11*S22 - S12*S21
            let delta = s11 * s22 - s12 * s21;
            let s12_s21 = s12 * s21;

            if target_port == 1 {
                // Load stability circle
                // C_L = (S22 - D*S11*)* / (|S22|² - |D|²)
                // R_L = |S12*S21 / (|S22|² - |D|²)|
                let denom = s22.norm_sqr() - delta.norm_sqr();
                if denom.abs() > NEAR_ZERO {
                    centers[f] = ((s22 - delta * s11.conj()).conj()) / Complex64::new(denom, 0.0);
                    radii[f] = (s12_s21 / Complex64::new(denom, 0.0)).norm();
                }
            } else {
                // Source stability circle
                // C_S = (S11 - D*S22*)* / (|S11|² - |D|²)
                // R_S = |S12*S21 / (|S11|² - |D|²)|
                let denom = s11.norm_sqr() - delta.norm_sqr();
                if denom.abs() > NEAR_ZERO {
                    centers[f] = ((s11 - delta * s22.conj()).conj()) / Complex64::new(denom, 0.0);
                    radii[f] = (s12_s21 / Complex64::new(denom, 0.0)).norm();
                }
            }
        }

        Some((centers, radii))
    }

    /// Calculate constant gain circle for a 2-port network
    ///
    /// Returns circles on Smith chart for constant gain.
    /// - target_port = 0: Source gain circle (for input matching)
    /// - target_port = 1: Load gain circle (for output matching)
    ///
    /// # Arguments
    /// * `target_port` - 0 for source, 1 for load
    /// * `gain_db` - Desired gain in dB
    ///
    /// # Returns
    /// (centers, radii) for each frequency point
    pub fn gain_circle(
        &self,
        target_port: usize,
        gain_db: f64,
    ) -> Option<(Array1<Complex64>, Array1<f64>)> {
        if self.nports() != 2 || target_port > 1 {
            return None;
        }

        let nfreq = self.nfreq();
        let mut centers = Array1::<Complex64>::zeros(nfreq);
        let mut radii = Array1::<f64>::zeros(nfreq);
        let g = 10.0_f64.powf(gain_db / 10.0);

        for f in 0..nfreq {
            let s11 = self.s[[f, 0, 0]];
            let s22 = self.s[[f, 1, 1]];
            let s12 = self.s[[f, 0, 1]];
            let s21 = self.s[[f, 1, 0]];
            let delta = s11 * s22 - s12 * s21;

            if let Some((center, radius)) =
                compute_gain_circle_params(s11, s22, s21, delta, g, target_port)
            {
                centers[f] = center;
                radii[f] = radius;
            }
        }

        Some((centers, radii))
    }
}

/// Helper function to compute gain circle center and radius
///
/// Encapsulates the gain circle calculation for both source and load cases.
/// Returns None if the calculation is singular.
fn compute_gain_circle_params(
    s11: Complex64,
    s22: Complex64,
    s21: Complex64,
    delta: Complex64,
    g: f64,
    target_port: usize,
) -> Option<(Complex64, f64)> {
    let s21_mag_sq = s21.norm_sqr();
    if s21_mag_sq <= NEAR_ZERO {
        return None;
    }

    // Select parameters based on target port
    let (s_ref, s_other, c) = if target_port == 1 {
        // Load circle: reference is S11, other is S22
        let c = s22 - delta * s11.conj();
        (s11, s22, c)
    } else {
        // Source circle: reference is S22, other is S11
        let c = s11 - delta * s22.conj();
        (s22, s11, c)
    };

    let s_ref_mag_sq = s_ref.norm_sqr();
    let s_other_mag_sq = s_other.norm_sqr();
    let c_mag_sq = c.norm_sqr();

    // Normalized gain
    let gs = g * (1.0 - s_ref_mag_sq) / s21_mag_sq;

    // Denominator for circle calculation
    let denom = 1.0 + gs * (s_other_mag_sq - delta.norm_sqr());
    if denom.abs() <= NEAR_ZERO {
        return None;
    }

    // Circle center
    let center = gs * c.conj() / Complex64::new(denom, 0.0);

    // Circle radius
    let r_sq = 1.0 - (2.0 * gs * (1.0 - s_other_mag_sq + c_mag_sq)) / denom
        + gs * gs * c_mag_sq / (denom * denom);
    let radius = if r_sq > 0.0 { r_sq.sqrt() } else { 0.0 };

    Some((center, radius))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_s_active_single_port() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        // 2-port with S21 = 0.8, S11 = 0.1, S22 = 0.1, S12 = 0.8
        let mut s = Array3::<Complex64>::zeros((1, 2, 2));
        s[[0, 0, 0]] = Complex64::new(0.1, 0.0);
        s[[0, 0, 1]] = Complex64::new(0.8, 0.0);
        s[[0, 1, 0]] = Complex64::new(0.8, 0.0);
        s[[0, 1, 1]] = Complex64::new(0.1, 0.0);

        let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0).unwrap();

        // Excite only port 0
        let a = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        let s_act = ntwk.s_active(&a).unwrap();

        // Active S at port 0 should be S00 (only port 0 is excited with a0/a0 = 1)
        assert_relative_eq!(s_act[[0, 0]].re, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_vswr_active() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        s[[0, 0, 0]] = Complex64::new(0.5, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0).unwrap();

        let a = Array1::from_vec(vec![Complex64::new(1.0, 0.0)]);
        let vswr = ntwk.vswr_active(&a).unwrap();

        // VSWR = (1 + 0.5) / (1 - 0.5) = 3.0
        assert_relative_eq!(vswr[[0, 0]], 3.0, epsilon = 1e-10);
    }
}
