//! Core algorithms for Vector Fitting
//!
//! Implements pole initialization, pole relocation (QR + eigenvalue),
//! and residue fitting (least squares).

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Initial pole spacing type
#[derive(Debug, Clone, Copy)]
pub enum InitPoleSpacing {
    Linear,
    Logarithmic,
}

/// Result of pole relocation iteration
#[derive(Debug)]
pub struct PoleRelocationResult {
    pub poles: Array1<Complex64>,
    pub d_res: f64,
    pub condition: f64,
    pub rank_deficiency: i32,
    pub singular_vals: Vec<f64>,
}

/// Initialize starting poles across the frequency range
///
/// # Arguments
/// * `freqs` - Normalized frequencies
/// * `n_poles_real` - Number of real poles
/// * `n_poles_cmplx` - Number of complex conjugate poles
/// * `spacing` - Linear or logarithmic spacing
pub fn init_poles(
    freqs: &[f64],
    n_poles_real: usize,
    n_poles_cmplx: usize,
    spacing: InitPoleSpacing,
) -> Array1<Complex64> {
    let n_poles = n_poles_real + n_poles_cmplx;
    let mut poles = Array1::<Complex64>::zeros(n_poles);

    if freqs.is_empty() || n_poles == 0 {
        return poles;
    }

    let f_min = freqs.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_max = freqs.iter().cloned().fold(0.0, f64::max);

    // Poles cannot be at f=0
    let f_min = if f_min == 0.0 {
        if freqs.len() > 1 {
            freqs[1] / 1000.0
        } else {
            1e-6
        }
    } else {
        f_min
    };

    // Generate pole frequencies
    let real_freqs: Vec<f64> = match spacing {
        InitPoleSpacing::Linear => linspace(f_min, f_max, n_poles_real),
        InitPoleSpacing::Logarithmic => logspace(f_min, f_max, n_poles_real),
    };

    let cmplx_freqs: Vec<f64> = match spacing {
        InitPoleSpacing::Linear => linspace(f_min, f_max, n_poles_cmplx),
        InitPoleSpacing::Logarithmic => logspace(f_min, f_max, n_poles_cmplx),
    };

    // Add real poles (negative real part for stability)
    for (i, f) in real_freqs.iter().enumerate() {
        let omega = 2.0 * PI * f;
        poles[i] = Complex64::new(-omega, 0.0);
    }

    // Add complex-conjugate poles (store only positive imaginary part)
    for (i, f) in cmplx_freqs.iter().enumerate() {
        let omega = 2.0 * PI * f;
        poles[n_poles_real + i] = Complex64::new(-0.01 * omega, omega);
    }

    poles
}

/// Calculate model order from poles
///
/// Order = N_real + 2 * N_complex
pub fn get_model_order(poles: &Array1<Complex64>) -> usize {
    let mut order = 0;
    for pole in poles.iter() {
        if pole.im == 0.0 {
            order += 1; // Real pole
        } else {
            order += 2; // Complex conjugate pair
        }
    }
    order
}

/// Pole relocation algorithm
///
/// Uses QR decomposition and eigenvalue extraction to relocate poles.
pub fn pole_relocation(
    poles: &Array1<Complex64>,
    freqs: &[f64],
    freq_responses: &Array2<Complex64>,
    weights: &[f64],
    fit_constant: bool,
    fit_proportional: bool,
) -> Result<PoleRelocationResult, String> {
    let n_responses = freq_responses.nrows();
    let n_freqs = freq_responses.ncols();
    let n_poles = poles.len();

    if n_freqs == 0 || n_poles == 0 {
        return Err("Empty input".to_string());
    }

    // Build s = j * omega
    let omega: Vec<f64> = freqs.iter().map(|f| 2.0 * PI * f).collect();
    let s: Vec<Complex64> = omega.iter().map(|w| Complex64::new(0.0, *w)).collect();

    // Separate real and complex poles
    let idx_real: Vec<usize> = poles
        .iter()
        .enumerate()
        .filter(|(_, p)| p.im == 0.0)
        .map(|(i, _)| i)
        .collect();
    let idx_cmplx: Vec<usize> = poles
        .iter()
        .enumerate()
        .filter(|(_, p)| p.im != 0.0)
        .map(|(i, _)| i)
        .collect();

    let n_real = idx_real.len();
    let n_cmplx = idx_cmplx.len();

    // Model order determines number of columns for residue terms
    let model_order = get_model_order(poles);

    // Number of columns: model_order for residues + optional d + optional e + model_order + 1 for d_res
    let _n_cols_unused =
        model_order + (if fit_constant { 1 } else { 0 }) + (if fit_proportional { 1 } else { 0 });
    let n_cols_used = model_order + 1; // c_res terms + d_res

    // Build coefficient matrix indices
    let mut idx_res_real: Vec<usize> = Vec::new();
    let mut idx_res_cmplx_re: Vec<usize> = Vec::new();
    let mut idx_res_cmplx_im: Vec<usize> = Vec::new();

    let mut col = 0;
    for pole in poles.iter() {
        if pole.im == 0.0 {
            idx_res_real.push(col);
            col += 1;
        } else {
            idx_res_cmplx_re.push(col);
            idx_res_cmplx_im.push(col + 1);
            col += 2;
        }
    }

    // Build coefficient matrices for real and complex poles
    // coeff_real[freq, pole_idx] = 1 / (s - pole)
    let mut coeff_real = Array2::<Complex64>::zeros((n_freqs, n_real));
    for (i, &pole_idx) in idx_real.iter().enumerate() {
        let pole = poles[pole_idx];
        for (f, &s_f) in s.iter().enumerate() {
            coeff_real[[f, i]] = Complex64::new(1.0, 0.0) / (s_f - pole);
        }
    }

    // coeff_complex_re and coeff_complex_im for complex conjugate pairs
    let mut coeff_cmplx_re = Array2::<Complex64>::zeros((n_freqs, n_cmplx));
    let mut coeff_cmplx_im = Array2::<Complex64>::zeros((n_freqs, n_cmplx));
    for (i, &pole_idx) in idx_cmplx.iter().enumerate() {
        let pole = poles[pole_idx];
        for (f, &s_f) in s.iter().enumerate() {
            let term1 = Complex64::new(1.0, 0.0) / (s_f - pole);
            let term2 = Complex64::new(1.0, 0.0) / (s_f - pole.conj());
            coeff_cmplx_re[[f, i]] = term1 + term2;
            coeff_cmplx_im[[f, i]] = Complex64::i() * (term1 - term2);
        }
    }

    // Build the compressed coefficient matrix using QR decomposition
    // For simplicity in this first implementation, we'll use a direct least-squares approach
    // This is less efficient than the full QR-based fast algorithm but correct

    // Build A_fast matrix by stacking weighted response equations
    let total_rows = n_responses * n_freqs + 1; // +1 for extra equation
    let mut a_real = Array2::<f64>::zeros((total_rows, n_cols_used));
    let mut b_real = Array1::<f64>::zeros(total_rows);

    let weight_extra = {
        let norm: f64 = freq_responses
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        norm / (n_responses * n_freqs) as f64
    };

    // Fill in the coefficient matrix
    for resp_idx in 0..n_responses {
        let w = weights[resp_idx].sqrt();
        for freq_idx in 0..n_freqs {
            let row = resp_idx * n_freqs + freq_idx;
            let h = freq_responses[[resp_idx, freq_idx]];

            // Real pole terms: -h / (s - p)
            for (i, _) in idx_real.iter().enumerate() {
                let coeff = coeff_real[[freq_idx, i]];
                let col = idx_res_real[i];
                a_real[[row, col]] = (-h * coeff).re * w;
            }

            // Complex pole terms
            for (i, _) in idx_cmplx.iter().enumerate() {
                let coeff_re = coeff_cmplx_re[[freq_idx, i]];
                let coeff_im = coeff_cmplx_im[[freq_idx, i]];
                let col_re = idx_res_cmplx_re[i];
                let col_im = idx_res_cmplx_im[i];
                a_real[[row, col_re]] = (-h * coeff_re).re * w;
                a_real[[row, col_im]] = (-h * coeff_im).re * w;
            }

            // d_res term: -h
            a_real[[row, n_cols_used - 1]] = -h.re * w;
        }
    }

    // Extra equation to avoid trivial solution
    let last_row = total_rows - 1;
    for (i, _) in idx_real.iter().enumerate() {
        let col = idx_res_real[i];
        let sum: f64 = (0..n_freqs).map(|f| coeff_real[[f, i]].re).sum();
        a_real[[last_row, col]] = sum * weight_extra.sqrt();
    }
    for (i, _) in idx_cmplx.iter().enumerate() {
        let col_re = idx_res_cmplx_re[i];
        let col_im = idx_res_cmplx_im[i];
        let sum_re: f64 = (0..n_freqs).map(|f| coeff_cmplx_re[[f, i]].re).sum();
        let sum_im: f64 = (0..n_freqs).map(|f| coeff_cmplx_im[[f, i]].re).sum();
        a_real[[last_row, col_re]] = sum_re * weight_extra.sqrt();
        a_real[[last_row, col_im]] = sum_im * weight_extra.sqrt();
    }
    a_real[[last_row, n_cols_used - 1]] = n_freqs as f64 * weight_extra.sqrt();
    b_real[last_row] = (n_responses * n_freqs) as f64 * weight_extra.sqrt();

    // Solve least squares using nalgebra
    let (x, singular_vals, condition) = solve_least_squares(&a_real, &b_real)?;

    // Extract c_res and d_res
    let d_res = x[n_cols_used - 1];

    // Check d_res validity
    let d_res = if d_res.abs() < 1e-8 {
        1e-8 * d_res.signum()
    } else {
        d_res
    };

    // Build test matrix H for eigenvalue extraction
    let h_size = model_order;
    let mut h_matrix = Array2::<f64>::zeros((h_size, h_size));

    // Place poles on diagonal and subtract c_res / d_res
    for (i, &pole_idx) in idx_real.iter().enumerate() {
        let col = idx_res_real[i];
        h_matrix[[col, col]] = poles[pole_idx].re;
        // Subtract c_res / d_res from entire row
        let c_res_i = x[col];
        for j in 0..h_size {
            h_matrix[[col, j]] -= c_res_i / d_res;
        }
    }

    for (i, &pole_idx) in idx_cmplx.iter().enumerate() {
        let col_re = idx_res_cmplx_re[i];
        let col_im = idx_res_cmplx_im[i];
        let pole = poles[pole_idx];

        h_matrix[[col_re, col_re]] = pole.re;
        h_matrix[[col_re, col_im]] = pole.im;
        h_matrix[[col_im, col_re]] = -pole.im;
        h_matrix[[col_im, col_im]] = pole.re;

        // Subtract 2 * c_res / d_res from real part row
        let c_res_re = x[col_re];
        for j in 0..h_size {
            h_matrix[[col_re, j]] -= 2.0 * c_res_re / d_res;
        }
    }

    // Extract eigenvalues
    let new_poles = eigenvalues_to_poles(&h_matrix)?;

    Ok(PoleRelocationResult {
        poles: new_poles,
        d_res,
        condition,
        rank_deficiency: 0,
        singular_vals,
    })
}

/// Fit residues using least squares
pub fn fit_residues(
    poles: &Array1<Complex64>,
    freqs: &[f64],
    freq_responses: &Array2<Complex64>,
    fit_constant: bool,
    fit_proportional: bool,
) -> Result<(Array2<Complex64>, Array1<f64>, Array1<f64>), String> {
    let n_responses = freq_responses.nrows();
    let n_freqs = freq_responses.ncols();
    let n_poles = poles.len();

    if n_freqs == 0 || n_poles == 0 {
        return Err("Empty input".to_string());
    }

    let omega: Vec<f64> = freqs.iter().map(|f| 2.0 * PI * f).collect();
    let s: Vec<Complex64> = omega.iter().map(|w| Complex64::new(0.0, *w)).collect();

    // Separate indices
    let idx_real: Vec<usize> = poles
        .iter()
        .enumerate()
        .filter(|(_, p)| p.im == 0.0)
        .map(|(i, _)| i)
        .collect();
    let idx_cmplx: Vec<usize> = poles
        .iter()
        .enumerate()
        .filter(|(_, p)| p.im != 0.0)
        .map(|(i, _)| i)
        .collect();

    let model_order = get_model_order(poles);
    let n_cols =
        model_order + (if fit_constant { 1 } else { 0 }) + (if fit_proportional { 1 } else { 0 });

    // Build column indices
    let mut idx_res_real: Vec<usize> = Vec::new();
    let mut idx_res_cmplx_re: Vec<usize> = Vec::new();
    let mut idx_res_cmplx_im: Vec<usize> = Vec::new();

    let mut col = 0;
    for pole in poles.iter() {
        if pole.im == 0.0 {
            idx_res_real.push(col);
            col += 1;
        } else {
            idx_res_cmplx_re.push(col);
            idx_res_cmplx_im.push(col + 1);
            col += 2;
        }
    }

    let idx_constant = if fit_constant { Some(col) } else { None };
    if fit_constant {
        col += 1;
    }
    let idx_proportional = if fit_proportional { Some(col) } else { None };

    // Build coefficient matrix A [n_freqs, n_cols]
    let mut a_matrix = Array2::<Complex64>::zeros((n_freqs, n_cols));

    for f_idx in 0..n_freqs {
        let s_f = s[f_idx];

        // Real pole coefficients
        for (i, &pole_idx) in idx_real.iter().enumerate() {
            let col = idx_res_real[i];
            let pole = poles[pole_idx];
            a_matrix[[f_idx, col]] = Complex64::new(1.0, 0.0) / (s_f - pole);
        }

        // Complex pole coefficients
        for (i, &pole_idx) in idx_cmplx.iter().enumerate() {
            let pole = poles[pole_idx];
            let term1 = Complex64::new(1.0, 0.0) / (s_f - pole);
            let term2 = Complex64::new(1.0, 0.0) / (s_f - pole.conj());
            a_matrix[[f_idx, idx_res_cmplx_re[i]]] = term1 + term2;
            a_matrix[[f_idx, idx_res_cmplx_im[i]]] = Complex64::i() * (term1 - term2);
        }

        // Constant and proportional
        if let Some(idx) = idx_constant {
            a_matrix[[f_idx, idx]] = Complex64::new(1.0, 0.0);
        }
        if let Some(idx) = idx_proportional {
            a_matrix[[f_idx, idx]] = s_f;
        }
    }

    // Solve least squares for each response
    let mut residues = Array2::<Complex64>::zeros((n_responses, n_poles));
    let mut constant_coeff = Array1::<f64>::zeros(n_responses);
    let mut proportional_coeff = Array1::<f64>::zeros(n_responses);

    for resp_idx in 0..n_responses {
        let b: Array1<Complex64> = freq_responses.row(resp_idx).to_owned();

        // Stack real and imaginary parts
        let a_ri = stack_real_imag_matrix(&a_matrix);
        let b_ri = stack_real_imag_vector(&b);

        let (x, _, _) = solve_least_squares(&a_ri, &b_ri)?;

        // Extract residues
        for (i, &pole_idx) in idx_real.iter().enumerate() {
            let col = idx_res_real[i];
            residues[[resp_idx, pole_idx]] = Complex64::new(x[col], 0.0);
        }
        for (i, &pole_idx) in idx_cmplx.iter().enumerate() {
            let col_re = idx_res_cmplx_re[i];
            let col_im = idx_res_cmplx_im[i];
            residues[[resp_idx, pole_idx]] = Complex64::new(x[col_re], x[col_im]);
        }

        if let Some(idx) = idx_constant {
            constant_coeff[resp_idx] = x[idx];
        }
        if let Some(idx) = idx_proportional {
            proportional_coeff[resp_idx] = x[idx];
        }
    }

    Ok((residues, constant_coeff, proportional_coeff))
}

// Helper functions

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![(start + end) / 2.0];
    }
    (0..n)
        .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
        .collect()
}

fn logspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 || start <= 0.0 || end <= 0.0 {
        return linspace(start, end, n);
    }
    let log_start = start.ln();
    let log_end = end.ln();
    linspace(log_start, log_end, n)
        .iter()
        .map(|x| x.exp())
        .collect()
}

fn stack_real_imag_matrix(a: &Array2<Complex64>) -> Array2<f64> {
    let (rows, cols) = a.dim();
    let mut result = Array2::<f64>::zeros((2 * rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            result[[i, j]] = a[[i, j]].re;
            result[[rows + i, j]] = a[[i, j]].im;
        }
    }
    result
}

fn stack_real_imag_vector(v: &Array1<Complex64>) -> Array1<f64> {
    let n = v.len();
    let mut result = Array1::<f64>::zeros(2 * n);
    for i in 0..n {
        result[i] = v[i].re;
        result[n + i] = v[i].im;
    }
    result
}

fn solve_least_squares(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    use nalgebra::{DMatrix, DVector};

    let (m, n) = a.dim();

    // Convert to nalgebra matrices
    let a_na = DMatrix::from_fn(m, n, |i, j| a[[i, j]]);
    let b_na = DVector::from_fn(m, |i, _| b[i]);

    // SVD-based least squares solve
    let svd = a_na.clone().svd(true, true);

    let solution = svd
        .solve(&b_na, 1e-14)
        .map_err(|_| "SVD solve failed".to_string())?;

    let x: Vec<f64> = solution.iter().cloned().collect();

    // Get singular values for condition number
    let singular_vals: Vec<f64> = svd.singular_values.iter().cloned().collect();
    let condition = if !singular_vals.is_empty() && singular_vals.last().unwrap().abs() > 1e-15 {
        singular_vals[0] / singular_vals.last().unwrap()
    } else {
        f64::INFINITY
    };

    Ok((x, singular_vals, condition))
}

fn eigenvalues_to_poles(h: &Array2<f64>) -> Result<Array1<Complex64>, String> {
    use nalgebra::DMatrix;

    let n = h.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let h_na = DMatrix::from_fn(n, n, |i, j| h[[i, j]]);

    // Compute eigenvalues
    let eigen = h_na.complex_eigenvalues();

    // Filter: keep only poles with positive (or zero) imaginary part
    // and flip unstable poles
    let mut poles: Vec<Complex64> = Vec::new();
    for ev in eigen.iter() {
        let pole = Complex64::new(ev.re, ev.im);
        if pole.im >= 0.0 {
            // Flip unstable poles (make real part negative)
            let stable_pole = Complex64::new(-pole.re.abs(), pole.im);
            poles.push(stable_pole);
        }
    }

    Ok(Array1::from_vec(poles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_init_poles_linear() {
        let freqs = vec![1e9, 2e9, 3e9, 4e9, 5e9];
        let poles = init_poles(&freqs, 2, 2, InitPoleSpacing::Linear);

        assert_eq!(poles.len(), 4);
        // First 2 are real (imaginary part = 0)
        assert_eq!(poles[0].im, 0.0);
        assert_eq!(poles[1].im, 0.0);
        // Last 2 are complex (imaginary part != 0)
        assert!(poles[2].im > 0.0);
        assert!(poles[3].im > 0.0);
    }

    #[test]
    fn test_get_model_order() {
        let poles = Array1::from_vec(vec![
            Complex64::new(-1.0, 0.0), // real
            Complex64::new(-2.0, 0.0), // real
            Complex64::new(-0.1, 1.0), // complex
        ]);
        assert_eq!(get_model_order(&poles), 4); // 2 + 2 = 4
    }

    #[test]
    fn test_linspace() {
        let result = linspace(0.0, 10.0, 5);
        assert_eq!(result.len(), 5);
        assert_relative_eq!(result[0], 0.0);
        assert_relative_eq!(result[4], 10.0);
        assert_relative_eq!(result[2], 5.0);
    }
}
