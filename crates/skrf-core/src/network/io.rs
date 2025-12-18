//! Network I/O functions
//!
//! Provides methods for writing Network to various file formats.

use std::path::Path;

use super::core::Network;
use crate::math::transforms::{s2g, s2h, s2y, s2z};
use crate::touchstone::parser::{ParameterType, SParamFormat, Touchstone, TouchstoneError};
use anyhow::Result;

impl Network {
    /// Write the network to a Touchstone file
    ///
    /// # Arguments
    /// * `path` - Path to output file (should have .sNp extension)
    /// * `format` - Output format (RI, MA, or DB). Default is RI.
    ///
    /// # Example
    /// ```ignore
    /// let ntwk = Network::from_touchstone("input.s2p")?;
    /// ntwk.write_touchstone("output.s2p", SParamFormat::RI)?;
    /// ```
    pub fn write_touchstone<P: AsRef<Path>>(
        &self,
        path: P,
        format: SParamFormat,
        param_type: ParameterType,
    ) -> Result<(), TouchstoneError> {
        let ts = self.to_touchstone(format, param_type);
        ts.write(path)
    }

    pub fn to_touchstone_contents(
        &self,
        format: SParamFormat,
        param_type: ParameterType,
    ) -> Result<String, TouchstoneError> {
        let ts = self.to_touchstone(format, param_type);
        Ok(ts.to_string())
    }

    /// Convert Network to Touchstone format
    pub fn to_touchstone(&self, format: SParamFormat, param_type: ParameterType) -> Touchstone {
        let nports = self.nports();
        let nfreq = self.nfreq();

        // Convert S to target parameter type if needed
        let data = match param_type {
            ParameterType::S => self.s.clone(),
            ParameterType::Z => {
                let physical_z = s2z(&self.s, &self.z0);
                self.normalize_v1_params(&physical_z, true)
            }
            ParameterType::Y => {
                let physical_y = s2y(&self.s, &self.z0);
                self.normalize_v1_params(&physical_y, false)
            }
            ParameterType::G => s2g(&self.s, &self.z0).unwrap_or_else(|| self.s.clone()),
            ParameterType::H => s2h(&self.s, &self.z0).unwrap_or_else(|| self.s.clone()),
        };

        // Convert array to Vec<Vec<Vec<Complex64>>>
        let mut data_vec = Vec::with_capacity(nfreq);
        for f_idx in 0..nfreq {
            let data_at_freq = data.index_axis(ndarray::Axis(0), f_idx);
            let matrix: Vec<Vec<num_complex::Complex64>> = data_at_freq
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();
            data_vec.push(matrix);
        }

        // Convert z0 to Vec<f64> (use real part only)
        let z0: Vec<f64> = (0..nports).map(|i| self.z0[i].re).collect();

        Touchstone {
            nports,
            frequency: self.frequency.clone(),
            s: data_vec,
            z0,
            comments: self.comments.clone(),
            format,
            param_type,
            is_v2: false,
            mixed_mode_order: Vec::new(),
            noisy: false,
        }
    }

    fn normalize_v1_params(
        &self,
        params: &ndarray::Array3<num_complex::Complex64>,
        is_z_params: bool,
    ) -> ndarray::Array3<num_complex::Complex64> {
        let nports = self.nports();
        let nfreq = self.nfreq();
        let mut result = params.clone();

        for f in 0..nfreq {
            for i in 0..nports {
                for j in 0..nports {
                    let z0_i = self.z0[i].re;
                    let z0_j = self.z0[j].re;
                    let scale = (z0_i * z0_j).sqrt();
                    if is_z_params {
                        result[[f, i, j]] /= scale;
                    } else {
                        result[[f, i, j]] *= scale;
                    }
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
    use ndarray::{Array1, Array3};
    use num_complex::Complex64;

    #[test]
    fn test_to_touchstone() {
        let freq = Frequency::new(1.0, 2.0, 2, FrequencyUnit::GHz, SweepType::Linear);
        let mut s = Array3::<Complex64>::zeros((2, 2, 2));

        s[[0, 0, 0]] = Complex64::new(0.1, 0.0);
        s[[0, 0, 1]] = Complex64::new(0.9, 0.0);
        s[[0, 1, 0]] = Complex64::new(0.9, 0.0);
        s[[0, 1, 1]] = Complex64::new(0.1, 0.0);

        let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0).unwrap();

        let ts = ntwk.to_touchstone(SParamFormat::RI, ParameterType::S);

        assert_eq!(ts.nports, 2);
        assert_eq!(ts.nfreq(), 2);
        assert_eq!(ts.z0[0], 50.0);
    }
}
