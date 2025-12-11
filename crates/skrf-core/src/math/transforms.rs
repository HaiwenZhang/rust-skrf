//! S-parameter transformation functions
//!
//! Provides conversions between S, Z, Y, T, and ABCD parameters.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

use crate::constants::NEAR_ZERO;
use crate::math::matrix_ops::{inv_sqrt_z0_matrix, sqrt_z0_matrix, z0_diag_matrix};

/// Convert S-parameters to Z-parameters
///
/// Formula: Z = F * (I + S) * inv(I - S) * F
/// where F is diagonal matrix of sqrt(z0)
pub fn s2z(s: &Array3<Complex64>, z0: &Array1<Complex64>) -> Array3<Complex64> {
    let nfreq = s.shape()[0];
    let nports = s.shape()[1];

    // Safety check for z0 dimension
    assert_eq!(nports, z0.len(), "z0 length must match number of ports");

    let mut z = Array3::<Complex64>::zeros((nfreq, nports, nports));
    let identity = Array2::<Complex64>::eye(nports);
    let f_mat = sqrt_z0_matrix(z0);

    for f in 0..nfreq {
        let s_f = s.slice(ndarray::s![f, .., ..]);
        let i_plus_s = &identity + &s_f;
        let i_minus_s = &identity - &s_f;

        // Z = F * (I + S) * inv(I - S) * F
        if let Some(inv_i_minus_s) = invert_matrix(&i_minus_s.to_owned()) {
            let term = i_plus_s.dot(&inv_i_minus_s);
            let z_f = f_mat.dot(&term).dot(&f_mat);
            z.slice_mut(ndarray::s![f, .., ..]).assign(&z_f);
        }
    }

    z
}

/// Convert Z-parameters to S-parameters
///
/// Formula: S = F^-1 * (Z - Z0_diag) * inv(Z + Z0_diag) * F
/// where F is diagonal matrix of sqrt(z0)
pub fn z2s(z: &Array3<Complex64>, z0: &Array1<Complex64>) -> Array3<Complex64> {
    let nfreq = z.shape()[0];
    let nports = z.shape()[1];
    assert_eq!(nports, z0.len());

    let mut s = Array3::<Complex64>::zeros((nfreq, nports, nports));

    let f_mat = sqrt_z0_matrix(z0);
    let inv_f_mat = inv_sqrt_z0_matrix(z0);
    let z0_diag = z0_diag_matrix(z0);

    for f in 0..nfreq {
        let z_f = z.slice(ndarray::s![f, .., ..]);

        let z_minus_z0 = &z_f - &z0_diag;
        let z_plus_z0 = &z_f + &z0_diag;

        if let Some(inv_term) = invert_matrix(&z_plus_z0.to_owned()) {
            let term = z_minus_z0.dot(&inv_term);
            // S = F^-1 * Term * F
            let s_f = inv_f_mat.dot(&term).dot(&f_mat);
            s.slice_mut(ndarray::s![f, .., ..]).assign(&s_f);
        }
    }

    s
}

/// Convert S-parameters to Y-parameters
///
/// Formula: Y = G * (I - S) * inv(I + S) * G
/// where G is diagonal 1/sqrt(z0) ? No, Y0 = 1/Z0.
/// Y = Y0_sqrt * (I - S) * (I + S)^-1 * Y0_sqrt
/// where Y0_sqrt = 1/sqrt(z0).
pub fn s2y(s: &Array3<Complex64>, z0: &Array1<Complex64>) -> Array3<Complex64> {
    let nfreq = s.shape()[0];
    let nports = s.shape()[1];
    assert_eq!(nports, z0.len());

    let mut y = Array3::<Complex64>::zeros((nfreq, nports, nports));
    let identity = Array2::<Complex64>::eye(nports);

    // G matrix (1/sqrt(z0))
    let g_mat = inv_sqrt_z0_matrix(z0);

    for f in 0..nfreq {
        let s_f = s.slice(ndarray::s![f, .., ..]);
        let i_minus_s = &identity - &s_f;
        let i_plus_s = &identity + &s_f;

        if let Some(inv_i_plus_s) = invert_matrix(&i_plus_s.to_owned()) {
            let term = i_minus_s.dot(&inv_i_plus_s);
            // Y = G * Term * G
            let y_f = g_mat.dot(&term).dot(&g_mat);
            y.slice_mut(ndarray::s![f, .., ..]).assign(&y_f);
        }
    }

    y
}

/// Convert Y-parameters to S-parameters
///
/// Formula: S = (I - Z0*Y) * inv(I + Z0*Y)  -- Correct only for equal Z0?
/// Generalized: S = (I - G_sqrt_inv * Y * G_sqrt_inv) ...
/// Actually skrf uses: S = (Y0 - Y)(Y0 + Y)^-1 is wrong?
/// Using Y2Z then Z2S is easier/safer given we implemented Z2S generalized.
/// Or explicitly: S = F (Yref - Y) (Yref + Y)^-1 F^-1 ?
/// Let's reuse z2s(y2z(y)) concept or implement directly.
/// skrf impl: y -> s directly ?
/// def y2s(y, z0=50): return z2s(y2z(y), z0)
/// We'll use the inverse relationship: S = F (I - Y Z0) (I + Y Z0)^-1 F^-1 ? No.
/// Let's stick to Skrf logic:
/// S = (I - Y') (I + Y')^-1 where Y' = sqrt(Z0) Y sqrt(Z0) (normalized Y)
pub fn y2s(y: &Array3<Complex64>, z0: &Array1<Complex64>) -> Array3<Complex64> {
    let nfreq = y.shape()[0];
    let nports = y.shape()[1];
    assert_eq!(nports, z0.len());

    let mut s = Array3::<Complex64>::zeros((nfreq, nports, nports));
    let identity = Array2::<Complex64>::eye(nports);

    // F matrix (sqrt(z0))
    let f_mat = sqrt_z0_matrix(z0);

    for f in 0..nfreq {
        let y_f = y.slice(ndarray::s![f, .., ..]);

        // Normalized Y' = F * Y * F
        let y_prime = f_mat.dot(&y_f).dot(&f_mat);

        let i_minus_y = &identity - &y_prime;
        let i_plus_y = &identity + &y_prime;

        if let Some(inv_term) = invert_matrix(&i_plus_y.to_owned()) {
            let s_f = i_minus_y.dot(&inv_term);
            s.slice_mut(ndarray::s![f, .., ..]).assign(&s_f);
        }
    }

    s
}

/// Convert S-parameters to T-parameters (scattering transfer parameters)
///
/// Only valid for 2-port networks.
/// T = [[T11, T12], [T21, T22]] where:
/// T11 = -(S11*S22 - S12*S21) / S21
/// T12 = S11 / S21
/// T21 = -S22 / S21
/// T22 = 1 / S21
pub fn s2t(s: &Array3<Complex64>) -> Option<Array3<Complex64>> {
    let nfreq = s.shape()[0];
    let nports = s.shape()[1];

    if nports != 2 {
        return None; // T-params only defined for 2-port networks
    }

    let mut t = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        let s11 = s[[f, 0, 0]];
        let s12 = s[[f, 0, 1]];
        let s21 = s[[f, 1, 0]];
        let s22 = s[[f, 1, 1]];

        if s21.norm() < NEAR_ZERO {
            return None; // S21 must be non-zero
        }

        let inv_s21 = Complex64::new(1.0, 0.0) / s21;
        let det_s = s11 * s22 - s12 * s21;

        t[[f, 0, 0]] = -det_s * inv_s21;
        t[[f, 0, 1]] = s11 * inv_s21;
        t[[f, 1, 0]] = -s22 * inv_s21;
        t[[f, 1, 1]] = inv_s21;
    }

    Some(t)
}

/// Convert T-parameters to S-parameters
///
/// Only valid for 2-port networks.
/// S = [[S11, S12], [S21, S22]] where:
/// S11 = T12 / T22
/// S12 = (T11*T22 - T12*T21) / T22
/// S21 = 1 / T22
/// S22 = -T21 / T22
pub fn t2s(t: &Array3<Complex64>) -> Option<Array3<Complex64>> {
    let nfreq = t.shape()[0];
    let nports = t.shape()[1];

    if nports != 2 {
        return None;
    }

    let mut s = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        let t11 = t[[f, 0, 0]];
        let t12 = t[[f, 0, 1]];
        let t21 = t[[f, 1, 0]];
        let t22 = t[[f, 1, 1]];

        if t22.norm() < NEAR_ZERO {
            return None; // T22 must be non-zero
        }

        let inv_t22 = Complex64::new(1.0, 0.0) / t22;
        let det_t = t11 * t22 - t12 * t21;

        s[[f, 0, 0]] = t12 * inv_t22;
        s[[f, 0, 1]] = det_t * inv_t22;
        s[[f, 1, 0]] = inv_t22;
        s[[f, 1, 1]] = -t21 * inv_t22;
    }

    Some(s)
}

/// Convert S-parameters to ABCD parameters (chain/cascade parameters)
///
/// Only valid for 2-port networks.
/// ABCD matrix: [[A, B], [C, D]]
///
/// Formulas (for Z0 = z0_common):
/// A = ((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21)
/// B = Z0 * ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21)
/// C = (1/Z0) * ((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21)
/// D = ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)
pub fn s2a(s: &Array3<Complex64>, z0: &Array1<Complex64>) -> Option<Array3<Complex64>> {
    let nfreq = s.shape()[0];
    let nports = s.shape()[1];

    if nports != 2 || z0.len() != 2 {
        return None;
    }

    let mut a = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        let s11 = s[[f, 0, 0]];
        let s12 = s[[f, 0, 1]];
        let s21 = s[[f, 1, 0]];
        let s22 = s[[f, 1, 1]];

        if s21.norm() < NEAR_ZERO {
            return None; // S21 must be non-zero for ABCD
        }

        // Use geometric mean of port impedances for ABCD
        let z01 = z0[0];
        let z02 = z0[1];
        let sqrt_z01 = z01.sqrt();
        let sqrt_z02 = z02.sqrt();

        let one = Complex64::new(1.0, 0.0);
        let two = Complex64::new(2.0, 0.0);
        let inv_2s21 = one / (two * s21);

        // A = ((1+S11)(1-S22) + S12*S21) / (2*S21) * sqrt(z02/z01)
        let a_val = ((one + s11) * (one - s22) + s12 * s21) * inv_2s21 * (sqrt_z02 / sqrt_z01);

        // B = Z0 * ((1+S11)(1+S22) - S12*S21) / (2*S21) where Z0 = sqrt(z01*z02)
        let z0_common = (z01 * z02).sqrt();
        let b_val = z0_common * ((one + s11) * (one + s22) - s12 * s21) * inv_2s21;

        // C = (1/Z0) * ((1-S11)(1-S22) - S12*S21) / (2*S21)
        let c_val = ((one - s11) * (one - s22) - s12 * s21) * inv_2s21 / z0_common;

        // D = ((1-S11)(1+S22) + S12*S21) / (2*S21) * sqrt(z01/z02)
        let d_val = ((one - s11) * (one + s22) + s12 * s21) * inv_2s21 * (sqrt_z01 / sqrt_z02);

        a[[f, 0, 0]] = a_val;
        a[[f, 0, 1]] = b_val;
        a[[f, 1, 0]] = c_val;
        a[[f, 1, 1]] = d_val;
    }

    Some(a)
}

/// Convert ABCD parameters to S-parameters
///
/// Only valid for 2-port networks.
/// Formulas (for Z0 = z0_common):
/// S11 = (A + B/Z0 - C*Z0 - D) / (A + B/Z0 + C*Z0 + D)
/// S12 = 2*(A*D - B*C) / (A + B/Z0 + C*Z0 + D)
/// S21 = 2 / (A + B/Z0 + C*Z0 + D)
/// S22 = (-A + B/Z0 - C*Z0 + D) / (A + B/Z0 + C*Z0 + D)
pub fn a2s(abcd: &Array3<Complex64>, z0: &Array1<Complex64>) -> Option<Array3<Complex64>> {
    let nfreq = abcd.shape()[0];
    let nports = abcd.shape()[1];

    if nports != 2 || z0.len() != 2 {
        return None;
    }

    let mut s = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        let a = abcd[[f, 0, 0]];
        let b = abcd[[f, 0, 1]];
        let c = abcd[[f, 1, 0]];
        let d = abcd[[f, 1, 1]];

        let z01 = z0[0];
        let z02 = z0[1];
        let sqrt_z01 = z01.sqrt();
        let sqrt_z02 = z02.sqrt();

        // Adjust for port impedances
        let a_adj = a * sqrt_z01 / sqrt_z02;
        let d_adj = d * sqrt_z02 / sqrt_z01;
        let z0_common = (z01 * z02).sqrt();
        let b_adj = b / z0_common;
        let c_adj = c * z0_common;

        let two = Complex64::new(2.0, 0.0);
        let denom = a_adj + b_adj + c_adj + d_adj;

        if denom.norm() < NEAR_ZERO {
            return None;
        }

        s[[f, 0, 0]] = (a_adj + b_adj - c_adj - d_adj) / denom;
        s[[f, 0, 1]] = two * (a_adj * d_adj - b_adj * c_adj) / denom;
        s[[f, 1, 0]] = two / denom;
        s[[f, 1, 1]] = (-a_adj + b_adj - c_adj + d_adj) / denom;
    }

    Some(s)
}

/// Convert S-parameters to H-parameters (hybrid parameters)
///
/// Only valid for 2-port networks.
/// H-matrix: [[h11, h12], [h21, h22]]
/// where h11 is impedance, h12 is voltage ratio, h21 is current ratio, h22 is admittance
pub fn s2h(s: &Array3<Complex64>, z0: &Array1<Complex64>) -> Option<Array3<Complex64>> {
    let nfreq = s.shape()[0];
    let nports = s.shape()[1];

    if nports != 2 || z0.len() != 2 {
        return None;
    }

    let mut h = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        let s11 = s[[f, 0, 0]];
        let s12 = s[[f, 0, 1]];
        let s21 = s[[f, 1, 0]];
        let s22 = s[[f, 1, 1]];

        let z01 = z0[0];
        let z02 = z0[1];
        let one = Complex64::new(1.0, 0.0);
        let two = Complex64::new(2.0, 0.0);

        // Denominator: (1-S11)(1+S22) + S12*S21
        let denom = (one - s11) * (one + s22) + s12 * s21;
        if denom.norm() < NEAR_ZERO {
            return None;
        }

        // h11 = z01 * ((1+S11)(1+S22) - S12*S21) / denom
        h[[f, 0, 0]] = z01 * ((one + s11) * (one + s22) - s12 * s21) / denom;

        // h12 = 2*S12 / denom
        h[[f, 0, 1]] = two * s12 / denom;

        // h21 = -2*S21 / denom
        h[[f, 1, 0]] = -two * s21 / denom;

        // h22 = (1/z02) * ((1-S11)(1-S22) - S12*S21) / denom
        h[[f, 1, 1]] = ((one - s11) * (one - s22) - s12 * s21) / (z02 * denom);
    }

    Some(h)
}

/// Convert H-parameters to S-parameters
pub fn h2s(h: &Array3<Complex64>, z0: &Array1<Complex64>) -> Option<Array3<Complex64>> {
    let nfreq = h.shape()[0];
    let nports = h.shape()[1];

    if nports != 2 || z0.len() != 2 {
        return None;
    }

    let mut s = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        let h11 = h[[f, 0, 0]];
        let h12 = h[[f, 0, 1]];
        let h21 = h[[f, 1, 0]];
        let h22 = h[[f, 1, 1]];

        let z01 = z0[0];
        let z02 = z0[1];
        let one = Complex64::new(1.0, 0.0);
        let two = Complex64::new(2.0, 0.0);

        // Denominator: (h11 + z01)(1 + h22*z02) - h12*h21*z02
        let denom = (h11 + z01) * (one + h22 * z02) - h12 * h21 * z02;
        if denom.norm() < NEAR_ZERO {
            return None;
        }

        // S11 = ((h11 - z01)(1 + h22*z02) - h12*h21*z02) / denom
        s[[f, 0, 0]] = ((h11 - z01) * (one + h22 * z02) - h12 * h21 * z02) / denom;

        // S12 = 2*h12*z02 / denom
        s[[f, 0, 1]] = two * h12 * z02 / denom;

        // S21 = -2*h21 / denom
        s[[f, 1, 0]] = -two * h21 / denom;

        // S22 = ((h11 + z01)(h22*z02 - 1) + h12*h21*z02) / denom
        s[[f, 1, 1]] = ((h11 + z01) * (h22 * z02 - one) + h12 * h21 * z02) / denom;
    }

    Some(s)
}

/// Convert S-parameters to G-parameters (inverse hybrid parameters)
///
/// Only valid for 2-port networks.
/// G-matrix: [[g11, g12], [g21, g22]]
/// where g11 is admittance, g12 is current ratio, g21 is voltage ratio, g22 is impedance
pub fn s2g(s: &Array3<Complex64>, z0: &Array1<Complex64>) -> Option<Array3<Complex64>> {
    let nfreq = s.shape()[0];
    let nports = s.shape()[1];

    if nports != 2 || z0.len() != 2 {
        return None;
    }

    let mut g = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        let s11 = s[[f, 0, 0]];
        let s12 = s[[f, 0, 1]];
        let s21 = s[[f, 1, 0]];
        let s22 = s[[f, 1, 1]];

        let z01 = z0[0];
        let z02 = z0[1];
        let one = Complex64::new(1.0, 0.0);
        let two = Complex64::new(2.0, 0.0);

        // Denominator: (1+S11)(1-S22) + S12*S21
        let denom = (one + s11) * (one - s22) + s12 * s21;
        if denom.norm() < NEAR_ZERO {
            return None;
        }

        // g11 = (1/z01) * ((1-S11)(1-S22) - S12*S21) / denom
        g[[f, 0, 0]] = ((one - s11) * (one - s22) - s12 * s21) / (z01 * denom);

        // g12 = -2*S12 / denom
        g[[f, 0, 1]] = -two * s12 / denom;

        // g21 = 2*S21 / denom
        g[[f, 1, 0]] = two * s21 / denom;

        // g22 = z02 * ((1+S11)(1+S22) - S12*S21) / denom
        g[[f, 1, 1]] = z02 * ((one + s11) * (one + s22) - s12 * s21) / denom;
    }

    Some(g)
}

/// Convert G-parameters to S-parameters
pub fn g2s(g: &Array3<Complex64>, z0: &Array1<Complex64>) -> Option<Array3<Complex64>> {
    let nfreq = g.shape()[0];
    let nports = g.shape()[1];

    if nports != 2 || z0.len() != 2 {
        return None;
    }

    let mut s = Array3::<Complex64>::zeros((nfreq, 2, 2));

    for f in 0..nfreq {
        let g11 = g[[f, 0, 0]];
        let g12 = g[[f, 0, 1]];
        let g21 = g[[f, 1, 0]];
        let g22 = g[[f, 1, 1]];

        let z01 = z0[0];
        let z02 = z0[1];
        let one = Complex64::new(1.0, 0.0);
        let two = Complex64::new(2.0, 0.0);

        // Denominator: (1 + g11*z01)(g22 + z02) - g12*g21*z01
        let denom = (one + g11 * z01) * (g22 + z02) - g12 * g21 * z01;
        if denom.norm() < NEAR_ZERO {
            return None;
        }

        // S11 = ((g11*z01 - 1)(g22 + z02) + g12*g21*z01) / denom
        s[[f, 0, 0]] = ((g11 * z01 - one) * (g22 + z02) + g12 * g21 * z01) / denom;

        // S12 = -2*g12*z01 / denom
        s[[f, 0, 1]] = -two * g12 * z01 / denom;

        // S21 = 2*g21 / denom
        s[[f, 1, 0]] = two * g21 / denom;

        // S22 = ((1 + g11*z01)(g22 - z02) - g12*g21*z01) / denom
        s[[f, 1, 1]] = ((one + g11 * z01) * (g22 - z02) - g12 * g21 * z01) / denom;
    }

    Some(s)
}

/// Matrix inversion using nalgebra's optimized LU decomposition
///
/// Uses nalgebra's try_inverse() which provides numerically stable
/// LU-based matrix inversion with partial pivoting.
fn invert_matrix(m: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    use nalgebra::{Complex, DMatrix};

    let n = m.shape()[0];
    if n != m.shape()[1] {
        return None;
    }

    if n == 0 {
        return Some(Array2::<Complex64>::zeros((0, 0)));
    }

    // Convert ndarray to nalgebra DMatrix
    let na_mat = DMatrix::from_fn(n, n, |i, j| Complex::new(m[[i, j]].re, m[[i, j]].im));

    // Use nalgebra's optimized LU-based inversion
    let inv = na_mat.try_inverse()?;

    // Convert back to ndarray
    let mut result = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = Complex64::new(inv[(i, j)].re, inv[(i, j)].im);
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_s2z_z2s_roundtrip() {
        // Create a simple 1-port S-parameter (S11 = 0.5)
        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        s[[0, 0, 0]] = Complex64::new(0.5, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let z = s2z(&s, &z0);
        let s_back = z2s(&z, &z0);

        assert_relative_eq!(s_back[[0, 0, 0]].re, 0.5, epsilon = 1e-10);
        assert_relative_eq!(s_back[[0, 0, 0]].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_s2y_y2s_roundtrip() {
        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        s[[0, 0, 0]] = Complex64::new(0.3, 0.1);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let y = s2y(&s, &z0);
        let s_back = y2s(&y, &z0);

        assert_relative_eq!(s_back[[0, 0, 0]].re, 0.3, epsilon = 1e-10);
        assert_relative_eq!(s_back[[0, 0, 0]].im, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_matched_load() {
        // A matched load has S11 = 0, which corresponds to Z = z0
        let mut s = Array3::<Complex64>::zeros((1, 1, 1));
        s[[0, 0, 0]] = Complex64::new(0.0, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let z = s2z(&s, &z0);

        assert_relative_eq!(z[[0, 0, 0]].re, 50.0, epsilon = 1e-10);
        assert_relative_eq!(z[[0, 0, 0]].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_open_circuit() {
        // An open circuit has S11 = 1, which corresponds to Z = infinity
        // We test the inverse: Z = very large -> S11 ≈ 1
        let mut z = Array3::<Complex64>::zeros((1, 1, 1));
        z[[0, 0, 0]] = Complex64::new(1e10, 0.0);

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let s = z2s(&z, &z0);

        assert_relative_eq!(s[[0, 0, 0]].re, 1.0, epsilon = 1e-5);
    }
}
