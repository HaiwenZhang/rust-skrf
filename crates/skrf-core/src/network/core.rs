//! Core Network struct and constructors
//!
//! Contains the fundamental Network data structure and factory methods.

use ndarray::{Array1, Array3};
use num_complex::Complex64;

use crate::frequency::Frequency;
use crate::math::transforms::{y2s, z2s};
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
                    // V1 Z-params are normalized. Denormalize: Z_ohm = Z_norm * sqrt(Z0_i * Z0_j)
                    let mut z_denorm = s.clone();
                    for f in 0..nfreq {
                        for i in 0..nports {
                            for j in 0..nports {
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
                    // V1 Y-params are normalized to Y0 = 1/Z0
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

    /// Create from Touchstone content string
    ///
    /// This is useful for WASM environments where file system access is not available.
    ///
    /// # Arguments
    /// * `content` - Touchstone file content as string
    /// * `nports` - Number of ports (typically derived from file extension, e.g., .s2p = 2 ports)
    ///
    /// # Example
    /// ```ignore
    /// let content = std::fs::read_to_string("test.s2p")?;
    /// let ntwk = Network::from_touchstone_content(&content, 2)?;
    /// ```
    pub fn from_touchstone_content(content: &str, nports: usize) -> Result<Self, TouchstoneError> {
        let ts = Touchstone::from_str(content, nports)?;

        let nfreq = ts.nfreq();
        let n = ts.nports;

        let mut s = Array3::<Complex64>::zeros((nfreq, n, n));
        for f in 0..nfreq {
            for i in 0..n {
                for j in 0..n {
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
                    let mut z_denorm = s.clone();
                    for f in 0..nfreq {
                        for i in 0..n {
                            for j in 0..n {
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
                    let mut y_denorm = s.clone();
                    for f in 0..nfreq {
                        for i in 0..n {
                            for j in 0..n {
                                let scaling = (ts.z0[i] * ts.z0[j]).sqrt();
                                y_denorm[[f, i, j]] = y_denorm[[f, i, j]] / scaling;
                            }
                        }
                    }
                    y_denorm
                };
                y2s(&y_vals, &z0)
            }
            _ => s,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(ntwk.z0[0].re, 50.0);
    }
}
