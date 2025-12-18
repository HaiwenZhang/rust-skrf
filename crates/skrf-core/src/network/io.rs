//! Network I/O functions
//!
//! Provides methods for writing Network to various file formats.

use std::path::Path;

use super::core::Network;
use crate::touchstone::parser::{SParamFormat, Touchstone, TouchstoneError};

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
    ) -> Result<(), TouchstoneError> {
        let ts = self.to_touchstone(format);
        ts.write(path)
    }

    /// Convert Network to Touchstone format
    pub fn to_touchstone(&self, format: SParamFormat) -> Touchstone {
        let nports = self.nports();
        let nfreq = self.nfreq();

        // Convert s-parameter array to Vec<Vec<Vec<Complex64>>>
        let mut s_data = Vec::with_capacity(nfreq);
        for f_idx in 0..nfreq {
            let s_at_freq = self.s.index_axis(ndarray::Axis(0), f_idx);
            let s_matrix: Vec<Vec<num_complex::Complex64>> = s_at_freq
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();
            s_data.push(s_matrix);
        }

        // Convert z0 to Vec<f64> (use real part only)
        let z0: Vec<f64> = (0..nports).map(|i| self.z0[i].re).collect();

        Touchstone {
            nports,
            frequency: self.frequency.clone(),
            s: s_data,
            z0,
            comments: self.comments.clone(),
            format,
            param_type: crate::touchstone::parser::ParameterType::S,
            is_v2: false,
            mixed_mode_order: Vec::new(),
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
    fn test_to_touchstone() {
        let freq = Frequency::new(1.0, 2.0, 2, FrequencyUnit::GHz, SweepType::Linear);
        let mut s = Array3::<Complex64>::zeros((2, 2, 2));

        s[[0, 0, 0]] = Complex64::new(0.1, 0.0);
        s[[0, 0, 1]] = Complex64::new(0.9, 0.0);
        s[[0, 1, 0]] = Complex64::new(0.9, 0.0);
        s[[0, 1, 1]] = Complex64::new(0.1, 0.0);

        let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0).unwrap();

        let ts = ntwk.to_touchstone(SParamFormat::RI);

        assert_eq!(ts.nports, 2);
        assert_eq!(ts.nfreq(), 2);
        assert_eq!(ts.z0[0], 50.0);
    }
}
