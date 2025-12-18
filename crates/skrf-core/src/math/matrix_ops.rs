//! Matrix operation helpers
//!
//! Provides utility functions for common matrix operations to reduce
//! boilerplate nested loops in transform functions.

use ndarray::{Array2, Array3};
use num_complex::Complex64;

use crate::constants::NEAR_ZERO;

/// Create a diagonal matrix from an Array1 of values
#[inline]
pub fn diag_matrix(values: &[Complex64]) -> Array2<Complex64> {
    let n = values.len();
    let mut m = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        m[[i, i]] = values[i];
    }
    m
}

/// Create a diagonal matrix with sqrt of reference impedances
#[inline]
pub fn sqrt_z0_matrix(z0: &ndarray::Array1<Complex64>) -> Array2<Complex64> {
    let sqrt_vals: Vec<Complex64> = z0.iter().map(|z| z.sqrt()).collect();
    diag_matrix(&sqrt_vals)
}

/// Create a diagonal matrix with inverse sqrt of reference impedances
#[inline]
pub fn inv_sqrt_z0_matrix(z0: &ndarray::Array1<Complex64>) -> Array2<Complex64> {
    let inv_sqrt_vals: Vec<Complex64> = z0
        .iter()
        .map(|z| Complex64::new(1.0, 0.0) / z.sqrt())
        .collect();
    diag_matrix(&inv_sqrt_vals)
}

/// Create a diagonal matrix from reference impedances
#[inline]
pub fn z0_diag_matrix(z0: &ndarray::Array1<Complex64>) -> Array2<Complex64> {
    diag_matrix(z0.as_slice().unwrap())
}

/// Apply a per-frequency matrix transformation: `result[f] = left * s[f] * right`
///
/// This is a common pattern in S-parameter conversions where we sandwich
/// each frequency slice between two matrices.
pub fn transform_per_freq(
    s: &Array3<Complex64>,
    left: &Array2<Complex64>,
    right: &Array2<Complex64>,
) -> Array3<Complex64> {
    let nfreq = s.shape()[0];
    let nports = s.shape()[1];
    let mut result = Array3::<Complex64>::zeros((nfreq, nports, nports));

    for f in 0..nfreq {
        let s_f = s.slice(ndarray::s![f, .., ..]);
        let transformed = left.dot(&s_f).dot(right);
        result
            .slice_mut(ndarray::s![f, .., ..])
            .assign(&transformed);
    }

    result
}

/// Invert a 2x2 complex matrix
///
/// Returns None if matrix is singular (determinant near zero).
#[inline]
pub fn invert_2x2(m: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    if m.shape() != [2, 2] {
        return None;
    }

    let a = m[[0, 0]];
    let b = m[[0, 1]];
    let c = m[[1, 0]];
    let d = m[[1, 1]];
    let det = a * d - b * c;

    if det.norm() < NEAR_ZERO {
        return None;
    }

    let inv_det = Complex64::new(1.0, 0.0) / det;
    let mut result = Array2::<Complex64>::zeros((2, 2));
    result[[0, 0]] = d * inv_det;
    result[[0, 1]] = -b * inv_det;
    result[[1, 0]] = -c * inv_det;
    result[[1, 1]] = a * inv_det;
    Some(result)
}

/// Extract S-parameters at a single frequency into a 2D array
#[inline]
pub fn extract_freq_slice(s: &Array3<Complex64>, f: usize) -> Array2<Complex64> {
    s.slice(ndarray::s![f, .., ..]).to_owned()
}

/// Identity matrix of given size
#[inline]
pub fn identity(n: usize) -> Array2<Complex64> {
    Array2::<Complex64>::eye(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_diag_matrix() {
        let vals = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let m = diag_matrix(&vals);
        assert_eq!(m[[0, 0]], Complex64::new(1.0, 0.0));
        assert_eq!(m[[1, 1]], Complex64::new(2.0, 0.0));
        assert_eq!(m[[2, 2]], Complex64::new(3.0, 0.0));
        assert_eq!(m[[0, 1]], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_sqrt_z0_matrix() {
        let z0 = Array1::from_vec(vec![Complex64::new(100.0, 0.0), Complex64::new(25.0, 0.0)]);
        let m = sqrt_z0_matrix(&z0);
        assert!((m[[0, 0]].re - 10.0).abs() < 1e-10);
        assert!((m[[1, 1]].re - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_2x2() {
        let mut m = Array2::<Complex64>::zeros((2, 2));
        m[[0, 0]] = Complex64::new(1.0, 0.0);
        m[[0, 1]] = Complex64::new(2.0, 0.0);
        m[[1, 0]] = Complex64::new(3.0, 0.0);
        m[[1, 1]] = Complex64::new(4.0, 0.0);

        let inv = invert_2x2(&m).unwrap();
        let identity_approx = m.dot(&inv);

        assert!((identity_approx[[0, 0]].re - 1.0).abs() < 1e-10);
        assert!((identity_approx[[1, 1]].re - 1.0).abs() < 1e-10);
        assert!(identity_approx[[0, 1]].norm() < 1e-10);
        assert!(identity_approx[[1, 0]].norm() < 1e-10);
    }
}
