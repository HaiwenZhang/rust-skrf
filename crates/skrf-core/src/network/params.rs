//! Network parameter properties (S, Z, Y, T)
//!
//! Provides access to various network parameter representations.

use ndarray::{Array1, Array3};
use num_complex::Complex64;

use super::core::Network;
use crate::frequency::Frequency;
use crate::math::transforms::{s2t, s2y, s2z};

impl Network {
    /// Get reference impedance
    pub fn z0(&self) -> &Array1<Complex64> {
        &self.z0
    }

    /// Get S-parameters
    pub fn s(&self) -> &Array3<Complex64> {
        &self.s
    }

    /// Get frequency object
    pub fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    /// Get frequency vector in Hz
    pub fn f(&self) -> &[f64] {
        self.frequency.f()
    }

    /// Get Z-parameters (impedance)
    pub fn z(&self) -> Array3<Complex64> {
        s2z(&self.s, &self.z0)
    }

    /// Get Y-parameters (admittance)
    pub fn y(&self) -> Array3<Complex64> {
        s2y(&self.s, &self.z0)
    }

    /// Get T-parameters (scattering transfer)
    ///
    /// Only valid for 2-port networks. Returns None for non-2-port networks.
    pub fn t(&self) -> Option<Array3<Complex64>> {
        s2t(&self.s)
    }

    /// Get S-parameter real part
    pub fn s_re(&self) -> Array3<f64> {
        self.s.mapv(|c| c.re)
    }

    /// Get S-parameter imaginary part
    pub fn s_im(&self) -> Array3<f64> {
        self.s.mapv(|c| c.im)
    }

    /// Get ABCD parameters (chain/cascade parameters)
    ///
    /// Only valid for 2-port networks. Returns None for non-2-port networks.
    /// ABCD matrix organization: [[A, B], [C, D]]
    pub fn a(&self) -> Option<Array3<Complex64>> {
        crate::math::transforms::s2a(&self.s, &self.z0)
    }

    /// Alias for a() - get ABCD parameters
    pub fn abcd(&self) -> Option<Array3<Complex64>> {
        self.a()
    }

    /// Get H-parameters (hybrid parameters)
    ///
    /// Only valid for 2-port networks. Returns None for non-2-port networks.
    /// H-matrix: [[h11 (impedance), h12 (voltage)], [h21 (current), h22 (admittance)]]
    pub fn h(&self) -> Option<Array3<Complex64>> {
        crate::math::transforms::s2h(&self.s, &self.z0)
    }

    /// Get G-parameters (inverse hybrid parameters)
    ///
    /// Only valid for 2-port networks. Returns None for non-2-port networks.
    /// G-matrix: [[g11 (admittance), g12 (current)], [g21 (voltage), g22 (impedance)]]
    pub fn g(&self) -> Option<Array3<Complex64>> {
        crate::math::transforms::s2g(&self.s, &self.z0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};
    use approx::assert_relative_eq;

    #[test]
    fn test_s_to_z_matched() {
        // A matched load (S11 = 0) should have Z = z0
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        s[[0, 0, 0]] = Complex64::new(0.0, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0).unwrap();
        let z = ntwk.z();

        assert_relative_eq!(z[[0, 0, 0]].re, 50.0, epsilon = 1e-10);
        assert_relative_eq!(z[[0, 0, 0]].im, 0.0, epsilon = 1e-10);
    }
}
