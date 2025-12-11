//! Core algorithms for Vector Fitting
//!
//! Implements pole initialization, pole relocation (QR + eigenvalue),
//! and residue fitting (least squares).

use ndarray::{Array1, Array2, Array3};
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
    use super::constants::MIN_FREQUENCY_FRACTION;
    let f_min = if f_min == 0.0 {
        if freqs.len() > 1 {
            freqs[1] / 1000.0
        } else {
            MIN_FREQUENCY_FRACTION
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
    use super::constants::COMPLEX_POLE_DAMPING_RATIO;
    for (i, f) in cmplx_freqs.iter().enumerate() {
        let omega = 2.0 * PI * f;
        poles[n_poles_real + i] = Complex64::new(-COMPLEX_POLE_DAMPING_RATIO * omega, omega);
    }

    poles
}

/// Calculate model order from poles
///
/// Order = N_real + 2 * N_complex
pub fn get_model_order(poles: &Array1<Complex64>) -> usize {
    super::poles::model_order_from_poles(poles)
}

// ============================================================================
// Helper structures for pole_relocation decomposition
// ============================================================================

/// Indices for categorized poles (real vs complex)
struct PoleIndices {
    /// Indices of real poles in the original array
    real: Vec<usize>,
    /// Indices of complex poles in the original array
    complex: Vec<usize>,
    /// Column indices for real residues
    res_real: Vec<usize>,
    /// Column indices for complex residue real parts
    res_cmplx_re: Vec<usize>,
    /// Column indices for complex residue imaginary parts
    res_cmplx_im: Vec<usize>,
}

impl PoleIndices {
    /// Compute pole indices from pole array
    fn from_poles(poles: &Array1<Complex64>) -> Self {
        let real: Vec<usize> = poles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.im == 0.0)
            .map(|(i, _)| i)
            .collect();

        let complex: Vec<usize> = poles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.im != 0.0)
            .map(|(i, _)| i)
            .collect();

        let n_real = real.len();
        let n_cmplx = complex.len();

        let res_real: Vec<usize> = (0..n_real).collect();
        let res_cmplx_re: Vec<usize> = (0..n_cmplx).map(|i| n_real + 2 * i).collect();
        let res_cmplx_im: Vec<usize> = (0..n_cmplx).map(|i| n_real + 2 * i + 1).collect();

        Self {
            real,
            complex,
            res_real,
            res_cmplx_re,
            res_cmplx_im,
        }
    }

    #[inline]
    fn n_real(&self) -> usize {
        self.real.len()
    }

    #[inline]
    fn n_complex(&self) -> usize {
        self.complex.len()
    }
}

/// Build H matrix for eigenvalue extraction in pole relocation
///
/// The H matrix is constructed such that its eigenvalues are the new poles.
fn build_h_matrix(
    poles: &Array1<Complex64>,
    indices: &PoleIndices,
    c_res: &[f64],
    d_res: f64,
    model_order: usize,
) -> Array2<f64> {
    let mut h = Array2::<f64>::zeros((model_order, model_order));

    // Real poles contribution
    for (i, &pole_idx) in indices.real.iter().enumerate() {
        let col = indices.res_real[i];
        h[[col, col]] = poles[pole_idx].re;
        // Subtract c_res / d_res from entire row
        let c_i = c_res[col];
        for j in 0..model_order {
            h[[col, j]] -= c_i / d_res;
        }
    }

    // Complex poles contribution
    for (i, &pole_idx) in indices.complex.iter().enumerate() {
        let col_re = indices.res_cmplx_re[i];
        let col_im = indices.res_cmplx_im[i];
        let pole = poles[pole_idx];

        h[[col_re, col_re]] = pole.re;
        h[[col_re, col_im]] = pole.im;
        h[[col_im, col_re]] = -pole.im;
        h[[col_im, col_im]] = pole.re;

        // Subtract 2 * c_res_re / d_res from real part row
        let c_re = c_res[col_re];
        for j in 0..model_order {
            h[[col_re, j]] -= 2.0 * c_re / d_res;
        }
    }

    h
}

/// Pole relocation algorithm following Python scikit-rf implementation
///
/// Uses QR decomposition for fast solving and eigenvalue extraction to relocate poles.
/// This implementation matches the Python version which stacks real and imaginary parts.
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
    let n_samples = n_responses * n_freqs;

    if n_freqs == 0 || n_poles == 0 {
        return Err("Empty input".to_string());
    }

    // Build s = j * omega
    let omega: Vec<f64> = freqs.iter().map(|f| 2.0 * PI * f).collect();
    let s: Vec<Complex64> = omega.iter().map(|w| Complex64::new(0.0, *w)).collect();

    // Calculate weight for extra equation
    let weight_extra = {
        let mut sum = 0.0;
        for resp_idx in 0..n_responses {
            for freq_idx in 0..n_freqs {
                let hw = weights[resp_idx] * freq_responses[[resp_idx, freq_idx]].norm();
                sum += hw * hw;
            }
        }
        (sum.sqrt() / n_samples as f64).sqrt()
    };

    // Sqrt of weights for rows
    let weights_sqrt: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();

    // Use PoleIndices struct for cleaner indexing
    let indices = PoleIndices::from_poles(poles);
    let n_real = indices.n_real();
    let n_cmplx = indices.n_complex();
    let model_order = get_model_order(poles);

    // Column indices for the coefficient matrix
    // Layout: [c terms for unused part] [c_res and d_res for used part]
    let mut n_cols_unused = model_order;
    let idx_constant: Option<usize> = if fit_constant {
        let idx = n_cols_unused;
        n_cols_unused += 1;
        Some(idx)
    } else {
        None
    };
    let idx_proportional: Option<usize> = if fit_proportional {
        let idx = n_cols_unused;
        n_cols_unused += 1;
        Some(idx)
    } else {
        None
    };
    let n_cols_used = model_order + 1; // c_res terms + d_res

    // Calculate coefficient matrices for poles
    // coeff_real[freq, pole_idx] = 1 / (s - pole)
    let mut coeff_real = Array2::<Complex64>::zeros((n_freqs, n_real));
    for (i, &pole_idx) in indices.real.iter().enumerate() {
        let pole = poles[pole_idx];
        for (f, &s_f) in s.iter().enumerate() {
            coeff_real[[f, i]] = Complex64::new(1.0, 0.0) / (s_f - pole);
        }
    }

    // Complex pole coefficients
    let mut coeff_cmplx_re = Array2::<Complex64>::zeros((n_freqs, n_cmplx));
    let mut coeff_cmplx_im = Array2::<Complex64>::zeros((n_freqs, n_cmplx));
    for (i, &pole_idx) in indices.complex.iter().enumerate() {
        let pole = poles[pole_idx];
        for (f, &s_f) in s.iter().enumerate() {
            let term1 = Complex64::new(1.0, 0.0) / (s_f - pole);
            let term2 = Complex64::new(1.0, 0.0) / (s_f - pole.conj());
            coeff_cmplx_re[[f, i]] = term1 + term2;
            coeff_cmplx_im[[f, i]] = Complex64::i() * (term1 - term2);
        }
    }

    // Build full complex coefficient matrix A[n_responses, n_freqs, n_cols_unused + n_cols_used]
    let n_cols_total = n_cols_unused + n_cols_used;
    let mut a_matrix = Array3::<Complex64>::zeros((n_responses, n_freqs, n_cols_total));

    for resp_idx in 0..n_responses {
        for freq_idx in 0..n_freqs {
            let h = freq_responses[[resp_idx, freq_idx]];

            // Part 1: First sum of rational functions (unused part - for c)
            for (i, _) in indices.real.iter().enumerate() {
                a_matrix[[resp_idx, freq_idx, indices.res_real[i]]] = coeff_real[[freq_idx, i]];
            }
            for (i, _) in indices.complex.iter().enumerate() {
                a_matrix[[resp_idx, freq_idx, indices.res_cmplx_re[i]]] =
                    coeff_cmplx_re[[freq_idx, i]];
                a_matrix[[resp_idx, freq_idx, indices.res_cmplx_im[i]]] =
                    coeff_cmplx_im[[freq_idx, i]];
            }

            // Part 2: Constant and proportional terms
            if let Some(idx) = idx_constant {
                a_matrix[[resp_idx, freq_idx, idx]] = Complex64::new(1.0, 0.0);
            }
            if let Some(idx) = idx_proportional {
                a_matrix[[resp_idx, freq_idx, idx]] = s[freq_idx];
            }

            // Part 3: Second sum multiplied by -h (used part - for c_res)
            for (i, _) in indices.real.iter().enumerate() {
                a_matrix[[resp_idx, freq_idx, n_cols_unused + indices.res_real[i]]] =
                    -h * coeff_real[[freq_idx, i]];
            }
            for (i, _) in indices.complex.iter().enumerate() {
                a_matrix[[resp_idx, freq_idx, n_cols_unused + indices.res_cmplx_re[i]]] =
                    -h * coeff_cmplx_re[[freq_idx, i]];
                a_matrix[[resp_idx, freq_idx, n_cols_unused + indices.res_cmplx_im[i]]] =
                    -h * coeff_cmplx_im[[freq_idx, i]];
            }

            // Part 4: d_res term (last column)
            a_matrix[[resp_idx, freq_idx, n_cols_total - 1]] = -h;
        }
    }

    // Stack real and imaginary parts horizontally: A_ri = [A.real, A.imag]
    // Then apply QR decomposition to compress
    let dim_m = 2 * n_freqs;
    let dim_n = n_cols_total;
    let dim_k = dim_m.min(dim_n);

    // Perform QR on each response and extract R
    let mut r_matrices = Array3::<f64>::zeros((n_responses, dim_k, dim_n));

    for resp_idx in 0..n_responses {
        // Build A_ri for this response: stack real and imaginary parts
        let mut a_ri = Array2::<f64>::zeros((dim_m, dim_n));
        for freq_idx in 0..n_freqs {
            for col in 0..n_cols_total {
                let c = a_matrix[[resp_idx, freq_idx, col]];
                a_ri[[freq_idx, col]] = c.re;
                a_ri[[n_freqs + freq_idx, col]] = c.im;
            }
        }

        // QR decomposition
        let r = qr_r(&a_ri)?;
        for i in 0..dim_k {
            for j in 0..dim_n {
                r_matrices[[resp_idx, i, j]] = r[[i, j]];
            }
        }
    }

    // Extract R22 (the part for c_res and d_res)
    let (n_rows_r12, n_rows_r22) = if dim_k == dim_m {
        (n_freqs, n_freqs)
    } else {
        (n_cols_unused, n_cols_used)
    };

    // Build A_fast by stacking weighted R22 matrices
    let dim0 = n_responses * n_rows_r22 + 1;
    let mut a_fast = Array2::<f64>::zeros((dim0, n_cols_used));

    for resp_idx in 0..n_responses {
        let w = weights_sqrt[resp_idx];
        for row in 0..n_rows_r22 {
            for col in 0..n_cols_used {
                a_fast[[resp_idx * n_rows_r22 + row, col]] =
                    w * r_matrices[[resp_idx, n_rows_r12 + row, n_cols_unused + col]];
            }
        }
    }

    // Extra equation to avoid trivial solution (last row)
    let last_row = dim0 - 1;
    for (i, _) in indices.real.iter().enumerate() {
        let sum: f64 = (0..n_freqs).map(|f| coeff_real[[f, i]].re).sum();
        a_fast[[last_row, indices.res_real[i]]] = sum;
    }
    for (i, _) in indices.complex.iter().enumerate() {
        let sum_re: f64 = (0..n_freqs).map(|f| coeff_cmplx_re[[f, i]].re).sum();
        let sum_im: f64 = (0..n_freqs).map(|f| coeff_cmplx_im[[f, i]].re).sum();
        a_fast[[last_row, indices.res_cmplx_re[i]]] = sum_re;
        a_fast[[last_row, indices.res_cmplx_im[i]]] = sum_im;
    }
    a_fast[[last_row, n_cols_used - 1]] = n_freqs as f64;

    // Apply weighting to last row
    for col in 0..n_cols_used {
        a_fast[[last_row, col]] *= weight_extra;
    }

    // Column scaling for numerical stability
    use super::constants::NORM_TOLERANCE;
    let mut scaling = Array1::<f64>::zeros(n_cols_used);
    for col in 0..n_cols_used {
        let norm: f64 = (0..dim0)
            .map(|row| a_fast[[row, col]].powi(2))
            .sum::<f64>()
            .sqrt();
        scaling[col] = if norm > NORM_TOLERANCE {
            1.0 / norm
        } else {
            1.0
        };
    }
    for row in 0..dim0 {
        for col in 0..n_cols_used {
            a_fast[[row, col]] *= scaling[col];
        }
    }

    // Right-hand side vector
    let mut b = Array1::<f64>::zeros(dim0);
    b[last_row] = weight_extra * n_samples as f64;

    // Solve least squares
    let (mut x, singular_vals, condition) = solve_least_squares(&a_fast, &b)?;

    // Undo scaling
    for col in 0..n_cols_used {
        x[col] *= scaling[col];
    }

    let c_res: Vec<f64> = x[..n_cols_used - 1].to_vec();
    let mut d_res = x[n_cols_used - 1];

    // Check if d_res is too small
    use super::constants::RESIDUE_TOLERANCE;
    if d_res.abs() < RESIDUE_TOLERANCE {
        d_res = RESIDUE_TOLERANCE * d_res.signum();
    }

    // Build test matrix H for eigenvalue extraction (use helper function)
    let h_matrix = build_h_matrix(poles, &indices, &c_res, d_res, model_order);

    // Extract eigenvalues to get new poles
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
    let n_real = idx_real.len();
    let n_cmplx = idx_cmplx.len();
    let idx_res_real: Vec<usize> = (0..n_real).collect();
    let idx_res_cmplx_re: Vec<usize> = (0..n_cmplx).map(|i| n_real + 2 * i).collect();
    let idx_res_cmplx_im: Vec<usize> = (0..n_cmplx).map(|i| n_real + 2 * i + 1).collect();

    let idx_constant = if fit_constant {
        Some(model_order)
    } else {
        None
    };
    let idx_proportional = if fit_proportional {
        Some(model_order + if fit_constant { 1 } else { 0 })
    } else {
        None
    };

    // Build coefficient matrix A [n_freqs, n_cols]
    let mut a_matrix = Array2::<Complex64>::zeros((n_freqs, n_cols));

    for f_idx in 0..n_freqs {
        let s_f = s[f_idx];

        // Real pole coefficients
        for (i, &pole_idx) in idx_real.iter().enumerate() {
            let pole = poles[pole_idx];
            a_matrix[[f_idx, idx_res_real[i]]] = Complex64::new(1.0, 0.0) / (s_f - pole);
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
            residues[[resp_idx, pole_idx]] = Complex64::new(x[idx_res_real[i]], 0.0);
        }
        for (i, &pole_idx) in idx_cmplx.iter().enumerate() {
            residues[[resp_idx, pole_idx]] =
                Complex64::new(x[idx_res_cmplx_re[i]], x[idx_res_cmplx_im[i]]);
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

// ============================================================================
// Linear algebra operations - delegated to linalg module
// ============================================================================

use crate::math::linalg;

/// QR decomposition returning only R matrix (upper triangular)
fn qr_r(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    Ok(linalg::qr_r(a))
}

fn solve_least_squares(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let result = linalg::lstsq(a, b).map_err(|e| e.to_string())?;
    Ok((result.solution, result.singular_values, result.condition))
}

fn eigenvalues_to_poles(h: &Array2<f64>) -> Result<Array1<Complex64>, String> {
    let n = h.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Compute eigenvalues
    let eigen = linalg::eigenvalues(h).map_err(|e| e.to_string())?;

    // Filter: keep only poles with positive (or zero) imaginary part
    // and flip unstable poles
    let mut poles: Vec<Complex64> = Vec::new();
    for pole in eigen.iter() {
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
