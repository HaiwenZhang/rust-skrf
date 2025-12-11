//! Derived network properties
//!
//! Provides computed properties like S-parameter magnitudes, dB values, phases,
//! VSWR, passivity, reciprocity, and group delay.

use ndarray::Array3;

use super::core::Network;
use crate::constants::NEAR_ZERO;

impl Network {
    /// Get S-parameter magnitude in dB
    #[inline]
    pub fn s_db(&self) -> Array3<f64> {
        self.s.mapv(|c| 20.0 * c.norm().log10())
    }

    /// Get S-parameter magnitude (linear)
    #[inline]
    pub fn s_mag(&self) -> Array3<f64> {
        self.s.mapv(|c| c.norm())
    }

    /// Get S-parameter phase in degrees
    #[inline]
    pub fn s_deg(&self) -> Array3<f64> {
        self.s.mapv(|c| c.arg() * 180.0 / std::f64::consts::PI)
    }

    /// Get S-parameter phase in radians
    #[inline]
    pub fn s_rad(&self) -> Array3<f64> {
        self.s.mapv(|c| c.arg())
    }

    /// Get VSWR (Voltage Standing Wave Ratio)
    ///
    /// VSWR = (1 + |S|) / (1 - |S|)
    /// Returns an array of shape [nfreq, nports, nports].
    /// Typically only diagonal elements (S11, S22, etc.) are meaningful.
    pub fn vswr(&self) -> Array3<f64> {
        self.s.mapv(|c| {
            let mag = c.norm();
            if mag >= 1.0 {
                f64::INFINITY
            } else {
                (1.0 + mag) / (1.0 - mag)
            }
        })
    }

    /// Passivity metric for a multi-port network
    ///
    /// Returns (I - S^H * S), where S^H is the conjugate transpose.
    /// For a passive network, this matrix should be positive semi-definite.
    /// The diagonal elements represent the power absorbed at each port.
    pub fn passivity(&self) -> Array3<f64> {
        let nfreq = self.nfreq();
        let nports = self.nports();
        let mut result = Array3::<f64>::zeros((nfreq, nports, nports));

        for f in 0..nfreq {
            let s_f = self.s.slice(ndarray::s![f, .., ..]);
            // S^H = conj(transpose(S))
            let s_h = s_f.mapv(|c| c.conj()).reversed_axes();
            // S^H * S
            let shs = s_h.dot(&s_f);

            // I - S^H * S (extract real part)
            for i in 0..nports {
                for j in 0..nports {
                    let identity_ij = if i == j { 1.0 } else { 0.0 };
                    result[[f, i, j]] = identity_ij - shs[[i, j]].re;
                }
            }
        }
        result
    }

    /// Reciprocity metric: S - S^T
    ///
    /// For a reciprocal network, this should be zero.
    /// Returns the magnitude of (S_ij - S_ji) for each element.
    pub fn reciprocity(&self) -> Array3<f64> {
        let nfreq = self.nfreq();
        let nports = self.nports();
        let mut result = Array3::<f64>::zeros((nfreq, nports, nports));

        for f in 0..nfreq {
            for i in 0..nports {
                for j in 0..nports {
                    let diff = self.s[[f, i, j]] - self.s[[f, j, i]];
                    result[[f, i, j]] = diff.norm();
                }
            }
        }
        result
    }

    /// Group delay in seconds
    ///
    /// Defined as: -d(phase)/d(omega) = -d(arg(S))/d(2*pi*f)
    /// Returns array of shape [nfreq-1, nports, nports] due to differentiation.
    pub fn group_delay(&self) -> Option<Array3<f64>> {
        let nfreq = self.nfreq();
        if nfreq < 2 {
            return None;
        }

        let nports = self.nports();
        let f = self.frequency.f();
        let mut gd = Array3::<f64>::zeros((nfreq - 1, nports, nports));

        for fi in 0..nfreq - 1 {
            let df = f[fi + 1] - f[fi];
            if df.abs() < NEAR_ZERO {
                continue;
            }
            let domega = 2.0 * std::f64::consts::PI * df;

            for i in 0..nports {
                for j in 0..nports {
                    let phase1 = self.s[[fi, i, j]].arg();
                    let phase2 = self.s[[fi + 1, i, j]].arg();

                    // Unwrap phase difference
                    let mut dphase = phase2 - phase1;
                    while dphase > std::f64::consts::PI {
                        dphase -= 2.0 * std::f64::consts::PI;
                    }
                    while dphase < -std::f64::consts::PI {
                        dphase += 2.0 * std::f64::consts::PI;
                    }

                    gd[[fi, i, j]] = -dphase / domega;
                }
            }
        }

        Some(gd)
    }

    /// Stability factor K (Rollett stability factor)
    ///
    /// For a 2-port network, K > 1 and |Δ| < 1 indicates unconditional stability.
    /// K = (1 - |S11|² - |S22|² + |Δ|²) / (2|S12||S21|)
    /// where Δ = S11*S22 - S12*S21
    ///
    /// Returns array of shape [nfreq] containing K values.
    pub fn stability(&self) -> Option<ndarray::Array1<f64>> {
        if self.nports() != 2 {
            return None;
        }

        let nfreq = self.nfreq();
        let mut k = ndarray::Array1::<f64>::zeros(nfreq);

        for f in 0..nfreq {
            let s11 = self.s[[f, 0, 0]];
            let s12 = self.s[[f, 0, 1]];
            let s21 = self.s[[f, 1, 0]];
            let s22 = self.s[[f, 1, 1]];

            let delta = s11 * s22 - s12 * s21;
            let s12_s21_mag = (s12.norm() * s21.norm()).max(NEAR_ZERO);

            let numerator = 1.0 - s11.norm_sqr() - s22.norm_sqr() + delta.norm_sqr();
            k[f] = numerator / (2.0 * s12_s21_mag);
        }

        Some(k)
    }

    /// Maximum Stable Gain (MSG)
    ///
    /// MSG = |S21| / |S12|
    /// Only meaningful when K < 1 (potentially unstable).
    pub fn max_stable_gain(&self) -> Option<ndarray::Array1<f64>> {
        if self.nports() != 2 {
            return None;
        }

        let nfreq = self.nfreq();
        let mut msg = ndarray::Array1::<f64>::zeros(nfreq);

        for f in 0..nfreq {
            let s12 = self.s[[f, 0, 1]];
            let s21 = self.s[[f, 1, 0]];

            if s12.norm() > NEAR_ZERO {
                msg[f] = s21.norm() / s12.norm();
            } else {
                msg[f] = f64::INFINITY;
            }
        }

        Some(msg)
    }

    /// Maximum Available Gain (MAG)
    ///
    /// MAG = (|S21|/|S12|) * (K - sqrt(K² - 1))
    /// Only valid when K > 1 (unconditionally stable).
    pub fn max_gain(&self) -> Option<ndarray::Array1<f64>> {
        let k = self.stability()?;
        let msg = self.max_stable_gain()?;
        let nfreq = self.nfreq();

        let mut mag = ndarray::Array1::<f64>::zeros(nfreq);

        for f in 0..nfreq {
            if k[f] > 1.0 {
                mag[f] = msg[f] * (k[f] - (k[f] * k[f] - 1.0).sqrt());
            } else {
                mag[f] = msg[f]; // Use MSG when K <= 1
            }
        }

        Some(mag)
    }

    /// S-parameter phase in degrees with unwrapping
    ///
    /// Removes 360° jumps from phase data.
    pub fn s_deg_unwrap(&self) -> Array3<f64> {
        let nfreq = self.nfreq();
        let nports = self.nports();
        let mut result = Array3::<f64>::zeros((nfreq, nports, nports));

        for i in 0..nports {
            for j in 0..nports {
                let mut prev_phase = 0.0;
                let mut offset = 0.0;

                for f in 0..nfreq {
                    let phase = self.s[[f, i, j]].arg() * 180.0 / std::f64::consts::PI;

                    if f > 0 {
                        let diff = phase - prev_phase;
                        if diff > 180.0 {
                            offset -= 360.0;
                        } else if diff < -180.0 {
                            offset += 360.0;
                        }
                    }

                    prev_phase = phase;
                    result[[f, i, j]] = phase + offset;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};
    use approx::assert_relative_eq;
    use ndarray::Array1;
    use num_complex::Complex64;

    #[test]
    fn test_s_db() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        // |S11| = 0.1 -> -20 dB
        s[[0, 0, 0]] = Complex64::new(0.1, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);
        let s_db = ntwk.s_db();

        assert_relative_eq!(s_db[[0, 0, 0]], -20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vswr() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        // |S11| = 0.5 -> VSWR = 3.0
        s[[0, 0, 0]] = Complex64::new(0.5, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);
        let vswr = ntwk.vswr();

        assert_relative_eq!(vswr[[0, 0, 0]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reciprocity() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 2, 2));
        // Reciprocal network: S12 = S21
        s[[0, 0, 1]] = Complex64::new(0.5, 0.1);
        s[[0, 1, 0]] = Complex64::new(0.5, 0.1);

        let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);
        let recip = ntwk.reciprocity();

        assert_relative_eq!(recip[[0, 0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(recip[[0, 1, 0]], 0.0, epsilon = 1e-10);
    }
}
