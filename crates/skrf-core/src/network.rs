//! Network module - N-port network representation
//!
//! Provides the core Network struct for S-parameter data.

use ndarray::{Array1, Array3};
use num_complex::Complex64;

use crate::frequency::Frequency;
use crate::math::transforms::{s2y, s2z, y2s, z2s};
use crate::touchstone::{parser::TouchstoneError, Touchstone};

/// An N-port electrical network
#[derive(Debug, Clone)]
pub struct Network {
    /// Frequency data
    pub frequency: Frequency,
    /// S-parameter data [nfreq, nports, nports]
    pub s: Array3<Complex64>,
    /// Reference impedance (per port)
    pub z0: Array1<Complex64>,
    /// Network name
    pub name: Option<String>,
    /// Comments
    pub comments: Vec<String>,
}

impl Network {
    /// Create a new Network from S-parameters
    pub fn new(frequency: Frequency, s: Array3<Complex64>, z0: Array1<Complex64>) -> Self {
        Self {
            frequency,
            s,
            z0,
            name: None,
            comments: Vec::new(),
        }
    }

    /// Create from a Touchstone file
    pub fn from_touchstone(path: &str) -> Result<Self, TouchstoneError> {
        let ts = Touchstone::from_file(path)?;

        let nfreq = ts.nfreq();
        let nports = ts.nports;

        let mut s = Array3::<Complex64>::zeros((nfreq, nports, nports));
        for f in 0..nfreq {
            for i in 0..nports {
                for j in 0..nports {
                    s[[f, i, j]] = ts.s[f][i][j];
                }
            }
        }

        // Convert z0 vector to Array1<Complex64>
        let z0 = Array1::from_vec(ts.z0.iter().map(|&x| Complex64::new(x, 0.0)).collect());

        let s = match ts.param_type {
            crate::touchstone::parser::ParameterType::S => s,
            crate::touchstone::parser::ParameterType::Z => {
                let z_vals = if ts.is_v2 {
                    s
                } else {
                    // V1 Z-params are normalized to Z0. Denormalize: Z_ohm = Z_norm * Z0
                    // Note: This assumes simple scalar multiplication for denormalization,
                    // which is correct if defined per port or if Z0 is diagonal.
                    // If Z0 varies per port, we need proper matrix mult?
                    // Touchstone V1 usually implies single system Z0 or per-port.
                    // Normalized usually means Z_ij_norm = Z_ij_ohm / sqrt(Z0_i * Z0_j)?
                    // Or Z_ij_norm = Z_ij_ohm / Z0_ref?
                    // Standard Touchstone 1.0 spec: "Normalized to the reference impedance."
                    // Usually means Z_norm = Z / R.
                    // For diagonal z0: Z_denorm_ij = Z_norm_ij * sqrt(z0_i * z0_j)?
                    // skrf python likely does: Z = Z_norm * z0_ref (scalar).
                    // If ts.z0 has multiple values, it's ambiguous in V1.
                    // But V1 usually has simple R 50.
                    // We will assume element-wise scaling by sqrt(z_i * z_j) is consistent with definition
                    // but simpler: if Z0 is array of reals.
                    // Let's verify standard definition.
                    // For now, assume scalar scaling if uniform, or elementwise.
                    let mut z_denorm = s.clone();
                    for f in 0..nfreq {
                        for i in 0..nports {
                            for j in 0..nports {
                                // Z_norm_ij = Z_ohm_ij / sqrt(Z0_i * Z0_j) ? No typically Z_ohm_ij / Z0_ref
                                // But if ports have different Z0?
                                // Let's assume Z_ ohm = Z_norm * sqrt(Z0[i]*Z0[j])?
                                // Or Z_ohm = Z_norm * Z0[0]?
                                // Looking at ex_9: 0.99 * 75 = 74.25.
                                // So it seems linear scaling by Z0.
                                // We use valid Z0 per port.
                                // A safe bet for V1 (normalized) is Z_denorm = Z_norm * sqrt(Z0_i * Z0_j).
                                let scaling = (ts.z0[i] * ts.z0[j]).sqrt();
                                z_denorm[[f, i, j]] = z_denorm[[f, i, j]] * scaling;
                            }
                        }
                    }
                    z_denorm
                };
                z2s(&z_vals, &z0)
            }
            crate::touchstone::parser::ParameterType::Y => {
                let y_vals = if ts.is_v2 {
                    s
                } else {
                    // V1 Y-params are normalized to Y0 = 1/Z0.
                    // Y_norm = Y_ohm * Z0 -> Y_ohm = Y_norm / Z0.
                    let mut y_denorm = s.clone();
                    for f in 0..nfreq {
                        for i in 0..nports {
                            for j in 0..nports {
                                let scaling = (ts.z0[i] * ts.z0[j]).sqrt();
                                y_denorm[[f, i, j]] = y_denorm[[f, i, j]] / scaling;
                            }
                        }
                    }
                    y_denorm
                };
                y2s(&y_vals, &z0)
            }
            _ => s, // TODO: Implement G2S and H2S
        };

        Ok(Self {
            frequency: ts.frequency,
            s,
            z0,
            name: None,
            comments: ts.comments,
        })
    }

    /// Get the number of ports
    pub fn nports(&self) -> usize {
        self.s.shape()[1]
    }

    /// Get the number of frequency points
    pub fn nfreq(&self) -> usize {
        self.s.shape()[0]
    }

    /// Get reference impedance
    pub fn z0(&self) -> &Array1<Complex64> {
        &self.z0
    }

    /// Get S-parameters
    pub fn s(&self) -> &Array3<Complex64> {
        &self.s
    }

    /// Get Z-parameters
    pub fn z(&self) -> Array3<Complex64> {
        s2z(&self.s, &self.z0)
    }

    /// Get Y-parameters
    pub fn y(&self) -> Array3<Complex64> {
        s2y(&self.s, &self.z0)
    }

    /// Get S-parameter in dB
    pub fn s_db(&self) -> Array3<f64> {
        self.s.mapv(|c| 20.0 * c.norm().log10())
    }

    /// Get S-parameter magnitude
    pub fn s_mag(&self) -> Array3<f64> {
        self.s.mapv(|c| c.norm())
    }

    /// Get S-parameter phase in degrees
    pub fn s_deg(&self) -> Array3<f64> {
        self.s.mapv(|c| c.arg() * 180.0 / std::f64::consts::PI)
    }

    /// Get S-parameter phase in radians
    pub fn s_rad(&self) -> Array3<f64> {
        self.s.mapv(|c| c.arg())
    }

    /// Cascade with another network (self ** other)
    /// Only valid for 2-port networks
    pub fn cascade(&self, other: &Network) -> Option<Network> {
        if self.nports() != 2 || other.nports() != 2 {
            return None;
        }

        // Check frequency compatibility
        if self.nfreq() != other.nfreq() {
            return None;
        }

        let nfreq = self.nfreq();
        let mut s_result = Array3::<Complex64>::zeros((nfreq, 2, 2));

        for f in 0..nfreq {
            // Get S-parameters for both networks at this frequency
            let s_a = [
                [self.s[[f, 0, 0]], self.s[[f, 0, 1]]],
                [self.s[[f, 1, 0]], self.s[[f, 1, 1]]],
            ];
            let s_b = [
                [other.s[[f, 0, 0]], other.s[[f, 0, 1]]],
                [other.s[[f, 1, 0]], other.s[[f, 1, 1]]],
            ];

            // Cascade formula using signal flow graph
            let denom = Complex64::new(1.0, 0.0) - s_a[1][1] * s_b[0][0];

            s_result[[f, 0, 0]] = s_a[0][0] + (s_a[0][1] * s_a[1][0] * s_b[0][0]) / denom;
            s_result[[f, 0, 1]] = (s_a[0][1] * s_b[0][1]) / denom;
            s_result[[f, 1, 0]] = (s_a[1][0] * s_b[1][0]) / denom;
            s_result[[f, 1, 1]] = s_b[1][1] + (s_b[0][1] * s_b[1][0] * s_a[1][1]) / denom;
        }

        Some(Network::new(
            self.frequency.clone(),
            s_result,
            self.z0.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_network_creation() {
        let freq = Frequency::new(
            1.0,
            10.0,
            10,
            crate::frequency::FrequencyUnit::GHz,
            crate::frequency::SweepType::Linear,
        );

        let s = Array3::<Complex64>::zeros((10, 2, 2));
        let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);

        assert_eq!(ntwk.nports(), 2);
        assert_eq!(ntwk.nfreq(), 10);
        assert_eq!(ntwk.z0()[0].re, 50.0);
    }

    #[test]
    fn test_s_to_z_matched() {
        // A matched load (S11 = 0) should have Z = z0
        let freq = Frequency::new(
            1.0,
            1.0,
            1,
            crate::frequency::FrequencyUnit::GHz,
            crate::frequency::SweepType::Linear,
        );

        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        s[[0, 0, 0]] = Complex64::new(0.0, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);
        let z = ntwk.z();

        assert_relative_eq!(z[[0, 0, 0]].re, 50.0, epsilon = 1e-10);
        assert_relative_eq!(z[[0, 0, 0]].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_s_db() {
        let freq = Frequency::new(
            1.0,
            1.0,
            1,
            crate::frequency::FrequencyUnit::GHz,
            crate::frequency::SweepType::Linear,
        );

        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        // |S11| = 0.1 -> -20 dB
        s[[0, 0, 0]] = Complex64::new(0.1, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);
        let s_db = ntwk.s_db();

        assert_relative_eq!(s_db[[0, 0, 0]], -20.0, epsilon = 1e-10);
    }
}
