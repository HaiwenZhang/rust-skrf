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
    /// Mixed-mode order (for TS 2.0)
    pub mixed_mode_order: Vec<String>,
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
            mixed_mode_order: Vec::new(),
        }
    }

    /// Create from a Touchstone file
    pub fn from_touchstone(path: &str) -> Result<Self, TouchstoneError> {
        let ts = Touchstone::from_file(path)?;
        Ok(Self::from_touchstone_data(ts))
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
        Ok(Self::from_touchstone_data(ts))
    }

    /// Internal: Convert Touchstone data to Network
    ///
    /// Shared logic for both file and string-based construction.
    fn from_touchstone_data(ts: Touchstone) -> Self {
        let nfreq = ts.nfreq();
        let nports = ts.nports;

        // Convert Vec<Vec<Vec<Complex64>>> to Array3<Complex64> using from_shape_fn
        let s = Array3::from_shape_fn((nfreq, nports, nports), |(f, i, j)| ts.s[f][i][j]);

        // Convert z0 vector to Array1<Complex64>
        let z0 = Array1::from_vec(ts.z0.iter().map(|&x| Complex64::new(x, 0.0)).collect());

        // Apply parameter type conversion and V1 denormalization
        let s = Self::convert_params_to_s(s, &ts, &z0);

        Self {
            frequency: ts.frequency,
            s,
            z0,
            name: None,
            comments: ts.comments,
            mixed_mode_order: ts.mixed_mode_order,
        }
    }

    /// Internal: Convert parameters to S-parameters with V1 denormalization
    fn convert_params_to_s(
        params: Array3<Complex64>,
        ts: &Touchstone,
        z0: &Array1<Complex64>,
    ) -> Array3<Complex64> {
        use crate::touchstone::parser::ParameterType;

        match ts.param_type {
            ParameterType::S => params,
            ParameterType::Z => {
                let z_vals = Self::denormalize_v1_params(&params, ts, true);
                z2s(&z_vals, z0)
            }
            ParameterType::Y => {
                let y_vals = Self::denormalize_v1_params(&params, ts, false);
                y2s(&y_vals, z0)
            }
            _ => params, // G and H: pass through (TODO: implement g2s, h2s)
        }
    }

    /// Internal: Denormalize V1 parameters (Z or Y)
    ///
    /// For Z-params: multiply by sqrt(Z0_i * Z0_j)
    /// For Y-params: divide by sqrt(Z0_i * Z0_j)
    fn denormalize_v1_params(
        params: &Array3<Complex64>,
        ts: &Touchstone,
        is_z_params: bool,
    ) -> Array3<Complex64> {
        if ts.is_v2 {
            return params.clone(); // V2 params are not normalized
        }

        let nports = params.shape()[1];

        // Pre-compute scaling matrix: sqrt(z0[i] * z0[j]) for all i,j
        let scaling_2d = ndarray::Array2::from_shape_fn((nports, nports), |(i, j)| {
            Complex64::new((ts.z0[i] * ts.z0[j]).sqrt(), 0.0)
        });

        // Apply scaling via broadcasting (2D scaling applies to each frequency slice)
        if is_z_params {
            params * &scaling_2d
        } else {
            params / &scaling_2d
        }
    }

    /// Get the number of ports
    #[inline]
    pub fn nports(&self) -> usize {
        self.s.shape()[1]
    }

    /// Get the number of frequency points
    #[inline]
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
