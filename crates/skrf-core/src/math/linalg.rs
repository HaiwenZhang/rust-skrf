//! Linear algebra operations
//!
//! This module provides a unified interface for matrix operations.
//! Currently uses nalgebra as the backend, but the API is designed
//! to allow swapping to faer in the future without changing callers.
//!
//! The key benefit: all ndarray<->linalg conversions are contained here,
//! eliminating scattered conversion code throughout the codebase.

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Result of least squares solve
pub struct LstsqResult {
    pub solution: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub condition: f64,
}

// ============================================================================
// Conversion helpers (internal)
// ============================================================================

/// Convert ndarray Array2<Complex64> to nalgebra DMatrix<Complex<f64>>
#[inline]
fn to_na_complex(a: &Array2<Complex64>) -> DMatrix<nalgebra::Complex<f64>> {
    let (m, n) = a.dim();
    DMatrix::from_fn(m, n, |i, j| {
        nalgebra::Complex::new(a[[i, j]].re, a[[i, j]].im)
    })
}

/// Convert nalgebra DMatrix<Complex<f64>> to ndarray Array2<Complex64>
#[inline]
fn from_na_complex(m: &DMatrix<nalgebra::Complex<f64>>) -> Array2<Complex64> {
    let rows = m.nrows();
    let cols = m.ncols();
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        Complex64::new(m[(i, j)].re, m[(i, j)].im)
    })
}

/// Convert ndarray Array2<f64> to nalgebra DMatrix<f64>
#[inline]
fn to_na_real(a: &Array2<f64>) -> DMatrix<f64> {
    let (m, n) = a.dim();
    DMatrix::from_fn(m, n, |i, j| a[[i, j]])
}

/// Convert nalgebra DMatrix<f64> to ndarray Array2<f64>
#[inline]
fn from_na_real(m: &DMatrix<f64>) -> Array2<f64> {
    let rows = m.nrows();
    let cols = m.ncols();
    Array2::from_shape_fn((rows, cols), |(i, j)| m[(i, j)])
}

// ============================================================================
// Matrix inversion
// ============================================================================

/// Invert a complex matrix
///
/// Returns None if matrix is singular or non-square.
pub fn inv_complex(a: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    let (m, n) = a.dim();
    if m != n || m == 0 {
        return None;
    }

    let mat = to_na_complex(a);
    mat.try_inverse().map(|inv| from_na_complex(&inv))
}

/// Invert a real matrix
///
/// Returns None if matrix is singular or non-square.
pub fn inv_real(a: &Array2<f64>) -> Option<Array2<f64>> {
    let (m, n) = a.dim();
    if m != n || m == 0 {
        return None;
    }

    let mat = to_na_real(a);
    mat.try_inverse().map(|inv| from_na_real(&inv))
}

// ============================================================================
// Eigenvalue decomposition
// ============================================================================

/// Compute complex eigenvalues of a real matrix
///
/// Returns error if matrix is not square.
pub fn eigenvalues(a: &Array2<f64>) -> Result<Vec<Complex64>, &'static str> {
    let (m, n) = a.dim();
    if m != n {
        return Err("Matrix must be square");
    }
    if m == 0 {
        return Ok(Vec::new());
    }

    let mat = to_na_real(a);
    let eigs = mat.complex_eigenvalues();

    Ok(eigs.iter().map(|e| Complex64::new(e.re, e.im)).collect())
}

// ============================================================================
// Singular Value Decomposition
// ============================================================================

/// Compute singular values of a complex matrix
pub fn singular_values(a: &Array2<Complex64>) -> Vec<f64> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Vec::new();
    }

    let mat = to_na_complex(a);
    let svd = mat.svd(false, false);
    svd.singular_values.iter().cloned().collect()
}

/// Full SVD of complex matrix: (U, S, Vh)
///
/// Returns (U, singular_values, Vh) where:
/// - U is m x m
/// - Vh is n x n
pub fn svd_complex(
    a: &Array2<Complex64>,
) -> Result<(Array2<Complex64>, Vec<f64>, Array2<Complex64>), &'static str> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err("Empty matrix");
    }

    let mat = to_na_complex(a);
    let svd = mat.svd(true, true);

    let u = svd.u.ok_or("SVD failed: no U matrix")?;
    let vh = svd.v_t.ok_or("SVD failed: no Vh matrix")?;
    let sigma: Vec<f64> = svd.singular_values.iter().cloned().collect();

    // Convert back to ndarray
    let u_arr = from_na_complex(&u);
    let vh_arr = from_na_complex(&vh);

    Ok((u_arr, sigma, vh_arr))
}

/// Full SVD of real matrix: (U, S, Vh)
#[allow(dead_code)]
pub fn svd_real(a: &Array2<f64>) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), &'static str> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err("Empty matrix");
    }

    let mat = to_na_real(a);
    let svd = mat.svd(true, true);

    let u = svd.u.ok_or("SVD failed: no U matrix")?;
    let vh = svd.v_t.ok_or("SVD failed: no Vh matrix")?;
    let sigma: Vec<f64> = svd.singular_values.iter().cloned().collect();

    let u_arr = from_na_real(&u);
    let vh_arr = from_na_real(&vh);

    Ok((u_arr, sigma, vh_arr))
}

// ============================================================================
// QR Decomposition
// ============================================================================

/// QR decomposition, returns R matrix only (upper triangular)
pub fn qr_r(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Array2::zeros((0, n));
    }

    let mat = to_na_real(a);
    let qr = mat.qr();
    let r = qr.r();

    let k = m.min(n);
    let mut result = Array2::<f64>::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            result[[i, j]] = r[(i, j)];
        }
    }

    result
}

// ============================================================================
// Least Squares
// ============================================================================

/// Solve least squares problem Ax = b using SVD
///
/// Returns solution vector, singular values, and condition number.
pub fn lstsq(a: &Array2<f64>, b: &Array1<f64>) -> Result<LstsqResult, &'static str> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err("Empty matrix");
    }
    if b.len() != m {
        return Err("Dimension mismatch");
    }

    let a_na = to_na_real(a);
    let b_na = DVector::from_fn(m, |i, _| b[i]);

    // SVD-based least squares solve
    let svd = a_na.svd(true, true);

    let solution = svd.solve(&b_na, 1e-14).map_err(|_| "SVD solve failed")?;

    let x: Vec<f64> = solution.iter().cloned().collect();

    // Get singular values for condition number
    let singular_values: Vec<f64> = svd.singular_values.iter().cloned().collect();
    let condition = if !singular_values.is_empty() && singular_values.last().unwrap().abs() > 1e-15
    {
        singular_values[0] / singular_values.last().unwrap()
    } else {
        f64::INFINITY
    };

    Ok(LstsqResult {
        solution: x,
        singular_values,
        condition,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_inv_real_identity() {
        let eye = Array2::<f64>::eye(3);
        let inv = inv_real(&eye).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(inv[[i, j]], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(inv[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_inv_complex() {
        let mut a = Array2::<Complex64>::zeros((2, 2));
        a[[0, 0]] = Complex64::new(1.0, 0.0);
        a[[0, 1]] = Complex64::new(2.0, 0.0);
        a[[1, 0]] = Complex64::new(3.0, 0.0);
        a[[1, 1]] = Complex64::new(4.0, 0.0);

        let inv = inv_complex(&a).unwrap();

        // A * A^(-1) should be identity
        let product = a.dot(&inv);
        assert_relative_eq!(product[[0, 0]].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(product[[1, 1]].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(product[[0, 1]].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(product[[1, 0]].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_singular_values() {
        let mut a = Array2::<Complex64>::zeros((2, 2));
        a[[0, 0]] = Complex64::new(3.0, 0.0);
        a[[1, 1]] = Complex64::new(4.0, 0.0);

        let sv = singular_values(&a);
        assert_eq!(sv.len(), 2);
        // Diagonal matrix: singular values are absolute values of diagonal
        assert_relative_eq!(sv[0], 4.0, epsilon = 1e-10);
        assert_relative_eq!(sv[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_eigenvalues() {
        // 2x2 matrix with known eigenvalues
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 0.0, 3.0]).unwrap();
        let eigs = eigenvalues(&a).unwrap();
        assert_eq!(eigs.len(), 2);
        // Eigenvalues of upper triangular matrix are diagonal elements
        // Should be 1 and 3
        let mut reals: Vec<f64> = eigs.iter().map(|e| e.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_relative_eq!(reals[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(reals[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qr_r() {
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let r = qr_r(&a);
        assert_eq!(r.dim(), (2, 2));
        // R should be upper triangular
        assert_relative_eq!(r[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lstsq_overdetermined() {
        // Solve [1,1; 1,2; 1,3] * x = [1; 2; 3]
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = lstsq(&a, &b).unwrap();
        assert_eq!(result.solution.len(), 2);
        // Least squares solution should exist
        assert!(result.condition < 100.0);
    }
}
