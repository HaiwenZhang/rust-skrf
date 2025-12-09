//! S-parameter transformation functions
//!
//! Provides conversions between S, Z, Y, T, and ABCD parameters.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;

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

    // Create F matrix (diagonal sqrt(z0))
    // We create it per frequency if needed (not here) or once
    let sqrt_z0 = z0.mapv(|x| x.sqrt());
    let mut f_mat = Array2::<Complex64>::zeros((nports, nports));
    for i in 0..nports {
        f_mat[[i, i]] = sqrt_z0[i];
    }

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
/// Formula: S = inv(F) * (Z - Zref) * inv(Z + Zref) * F ?
/// Actually generalized: S = F^-1 (Z - G) (Z + G)^-1 F
/// where G is diagonal z0.
/// Simplified: S = inv(Z/z0 + I) * (Z/z0 - I) ??? No.
/// Skrf uses: S = F^-1 (Z - Z0_diag) (Z + Z0_diag)^-1 F
pub fn z2s(z: &Array3<Complex64>, z0: &Array1<Complex64>) -> Array3<Complex64> {
    let nfreq = z.shape()[0];
    let nports = z.shape()[1];
    assert_eq!(nports, z0.len());

    let mut s = Array3::<Complex64>::zeros((nfreq, nports, nports));
    let _identity = Array2::<Complex64>::eye(nports);

    // F matrix
    let sqrt_z0 = z0.mapv(|x| x.sqrt());
    let mut f_mat = Array2::<Complex64>::zeros((nports, nports));
    let mut inv_f_mat = Array2::<Complex64>::zeros((nports, nports));
    let mut z0_diag = Array2::<Complex64>::zeros((nports, nports));

    for i in 0..nports {
        f_mat[[i, i]] = sqrt_z0[i];
        inv_f_mat[[i, i]] = Complex64::new(1.0, 0.0) / sqrt_z0[i];
        z0_diag[[i, i]] = z0[i];
    }

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
    let mut g_mat = Array2::<Complex64>::zeros((nports, nports));
    for i in 0..nports {
        g_mat[[i, i]] = Complex64::new(1.0, 0.0) / z0[i].sqrt();
    }

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
    let mut f_mat = Array2::<Complex64>::zeros((nports, nports));
    let mut inv_f_mat = Array2::<Complex64>::zeros((nports, nports));
    for i in 0..nports {
        let v = z0[i].sqrt();
        f_mat[[i, i]] = v;
        inv_f_mat[[i, i]] = Complex64::new(1.0, 0.0) / v;
    }

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

/// Simple 2x2 matrix inversion for complex matrices
fn invert_matrix(m: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    let n = m.shape()[0];
    if n != m.shape()[1] {
        return None;
    }

    if n == 1 {
        let det = m[[0, 0]];
        if det.norm() < 1e-15 {
            return None;
        }
        return Some(Array2::from_elem((1, 1), Complex64::new(1.0, 0.0) / det));
    }

    if n == 2 {
        let a = m[[0, 0]];
        let b = m[[0, 1]];
        let c = m[[1, 0]];
        let d = m[[1, 1]];
        let det = a * d - b * c;

        if det.norm() < 1e-15 {
            return None;
        }

        let inv_det = Complex64::new(1.0, 0.0) / det;
        let mut result = Array2::<Complex64>::zeros((2, 2));
        result[[0, 0]] = d * inv_det;
        result[[0, 1]] = -b * inv_det;
        result[[1, 0]] = -c * inv_det;
        result[[1, 1]] = a * inv_det;
        return Some(result);
    }

    // For larger matrices (up to 4 used in tests), we need implementation.
    // Implementing Gauss-Jordan elimination for small N.
    if n <= 10 {
        return invert_matrix_gauss(m);
    }

    // Logic for large N missing
    None
}

/// Gauss-Jordan elimination for matrix inversion
fn invert_matrix_gauss(m: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    let n = m.shape()[0];
    let mut a = m.clone();
    let mut inv = Array2::<Complex64>::eye(n);

    for i in 0..n {
        // Find pivot
        let mut pivot_idx = i;
        let mut pivot_val = a[[i, i]].norm();
        for k in i + 1..n {
            if a[[k, i]].norm() > pivot_val {
                pivot_idx = k;
                pivot_val = a[[k, i]].norm();
            }
        }

        if pivot_val < 1e-15 {
            return None; // Singular
        }

        // Swap rows
        if pivot_idx != i {
            for j in 0..n {
                let tmp = a[[i, j]];
                a[[i, j]] = a[[pivot_idx, j]];
                a[[pivot_idx, j]] = tmp;
                let tmp = inv[[i, j]];
                inv[[i, j]] = inv[[pivot_idx, j]];
                inv[[pivot_idx, j]] = tmp;
            }
        }

        // Scale row
        let scale = a[[i, i]];
        for j in 0..n {
            a[[i, j]] = a[[i, j]] / scale;
            inv[[i, j]] = inv[[i, j]] / scale;
        }

        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = a[[k, i]];
                for j in 0..n {
                    let val_a = a[[i, j]];
                    a[[k, j]] = a[[k, j]] - factor * val_a;

                    let val_inv = inv[[i, j]];
                    inv[[k, j]] = inv[[k, j]] - factor * val_inv;
                }
            }
        }
    }

    Some(inv)
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
