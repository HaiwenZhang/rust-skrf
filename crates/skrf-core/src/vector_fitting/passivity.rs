//! Passivity testing and enforcement for Vector Fitting models
//!
//! Implements passivity analysis using the half-size test matrix method and
//! passivity enforcement using singular value perturbation.
//!
//! References:
//! - B. Gustavsen and A. Semlyen, "Fast Passivity Assessment for S-Parameter Rational
//!   Models Via a Half-Size Test Matrix," IEEE TMTT, vol. 56, no. 12, 2008
//! - T. Dhaene et al., "Efficient Algorithm for Passivity Enforcement of S-Parameter-
//!   Based Macromodels," IEEE TMTT, vol. 57, no. 2, 2009

use ndarray::{s, Array1, Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// State-space representation matrices (A, B, C, D, E)
pub struct StateSpaceMatrices {
    /// System matrix A [model_order * nports, model_order * nports]
    pub a: Array2<f64>,
    /// Input matrix B [model_order * nports, nports]
    pub b: Array2<f64>,
    /// Output matrix C [nports, model_order * nports]
    pub c: Array2<f64>,
    /// Feedthrough matrix D [nports, nports]
    pub d: Array2<f64>,
    /// Proportional matrix E [nports, nports]
    pub e: Array2<f64>,
}

/// Result of passivity test
#[derive(Debug, Clone)]
pub struct PassivityTestResult {
    /// Frequency bands where passivity is violated [[f_start, f_stop], ...]
    pub violation_bands: Vec<[f64; 2]>,
    /// Maximum singular value observed
    pub max_singular_value: f64,
}

impl PassivityTestResult {
    /// Check if model is passive (no violation bands)
    pub fn is_passive(&self) -> bool {
        self.violation_bands.is_empty()
    }
}

/// Build state-space matrices A, B, C, D, E from poles, residues, and coefficients
///
/// The state-space representation is:
///   x' = A * x + B * u
///   y  = C * x + D * u + s * E * u
///
/// where u is the input (incident power waves) and y is the output (reflected power waves).
pub fn build_state_space_matrices(
    poles: &Array1<Complex64>,
    residues: &Array2<Complex64>,
    constant_coeff: &Array1<f64>,
    proportional_coeff: &Array1<f64>,
    nports: usize,
) -> StateSpaceMatrices {
    // Use PoleSet to compute model order
    use super::poles::PoleSet;
    let pole_set = PoleSet::from_array(poles);
    let model_order = pole_set.model_order();
    let n_matrix = nports * model_order;

    // Initialize matrices
    let mut a = Array2::<f64>::eye(n_matrix);
    let mut b = Array2::<f64>::zeros((n_matrix, nports));

    // Build A and B matrices
    let mut i_a = 0;
    for j in 0..nports {
        for pole in poles.iter() {
            if pole.im == 0.0 {
                // Real pole
                a[[i_a, i_a]] = pole.re;
                b[[i_a, j]] = 1.0;
                i_a += 1;
            } else {
                // Complex-conjugate pole pair
                a[[i_a, i_a]] = pole.re;
                a[[i_a, i_a + 1]] = pole.im;
                a[[i_a + 1, i_a]] = -pole.im;
                a[[i_a + 1, i_a + 1]] = pole.re;
                b[[i_a, j]] = 2.0;
                i_a += 2;
            }
        }
    }

    // Build C matrix (residues)
    let mut c = Array2::<f64>::zeros((nports, n_matrix));
    for i in 0..nports {
        for j in 0..nports {
            let i_response = i * nports + j;
            let mut j_residues = 0;

            for (k, pole) in poles.iter().enumerate() {
                let residue = residues[[i_response, k]];
                let offset = j * model_order;

                if pole.im == 0.0 {
                    c[[i, offset + j_residues]] = residue.re;
                    j_residues += 1;
                } else {
                    c[[i, offset + j_residues]] = residue.re;
                    c[[i, offset + j_residues + 1]] = residue.im;
                    j_residues += 2;
                }
            }
        }
    }

    // Build D matrix (constants)
    let mut d = Array2::<f64>::zeros((nports, nports));
    for i in 0..nports {
        for j in 0..nports {
            let i_response = i * nports + j;
            d[[i, j]] = constant_coeff[i_response];
        }
    }

    // Build E matrix (proportional coefficients)
    let mut e = Array2::<f64>::zeros((nports, nports));
    for i in 0..nports {
        for j in 0..nports {
            let i_response = i * nports + j;
            e[[i, j]] = proportional_coeff[i_response];
        }
    }

    StateSpaceMatrices { a, b, c, d, e }
}

/// Calculate S-matrices from state-space representation at given frequencies
///
/// S(s) = D + s * E + C * (s * I - A)^(-1) * B
pub fn get_s_from_state_space(freqs: &[f64], ss: &StateSpaceMatrices) -> Array3<Complex64> {
    let n_freqs = freqs.len();
    let nports = ss.d.nrows();
    let dim_a = ss.a.nrows();

    let mut s_matrices = Array3::<Complex64>::zeros((n_freqs, nports, nports));

    for (f_idx, &freq) in freqs.iter().enumerate() {
        let s = Complex64::new(0.0, 2.0 * PI * freq);

        // Compute (s * I - A)^(-1)
        let mut s_minus_a = Array2::<Complex64>::zeros((dim_a, dim_a));
        for i in 0..dim_a {
            for j in 0..dim_a {
                if i == j {
                    s_minus_a[[i, j]] = s - Complex64::new(ss.a[[i, j]], 0.0);
                } else {
                    s_minus_a[[i, j]] = Complex64::new(-ss.a[[i, j]], 0.0);
                }
            }
        }

        // Matrix inversion using nalgebra
        let inv_s_minus_a = match invert_complex_matrix(&s_minus_a) {
            Some(inv) => inv,
            None => continue, // Skip singular case
        };

        // Convert B and C to complex
        let b_complex = ss.b.mapv(|x| Complex64::new(x, 0.0));
        let c_complex = ss.c.mapv(|x| Complex64::new(x, 0.0));

        // S = D + s * E + C * inv(s*I - A) * B
        let temp = inv_s_minus_a.dot(&b_complex);
        let s_matrix = c_complex.dot(&temp);

        for i in 0..nports {
            for j in 0..nports {
                let d_ij = Complex64::new(ss.d[[i, j]], 0.0);
                let e_ij = Complex64::new(ss.e[[i, j]], 0.0);
                s_matrices[[f_idx, i, j]] = s_matrix[[i, j]] + d_ij + s * e_ij;
            }
        }
    }

    s_matrices
}

/// Perform passivity test using half-size test matrix method
///
/// Returns frequency bands where passivity violations occur.
pub fn passivity_test(
    poles: &Array1<Complex64>,
    residues: &Array2<Complex64>,
    constant_coeff: &Array1<f64>,
    proportional_coeff: &Array1<f64>,
    nports: usize,
) -> Result<PassivityTestResult, String> {
    // Check for proportional coefficients (not supported)
    if proportional_coeff.iter().any(|&e| e != 0.0) {
        return Err(
            "Passivity testing with nonzero proportional coefficients is not supported".to_string(),
        );
    }

    // Get state-space matrices
    let ss =
        build_state_space_matrices(poles, residues, constant_coeff, proportional_coeff, nports);

    // Build half-size test matrix P
    // P = (A - B * inv(D - I) * C) * (A - B * inv(D + I) * C)
    let identity = Array2::<f64>::eye(nports);

    // D - I and D + I
    let d_minus_i = &ss.d - &identity;
    let d_plus_i = &ss.d + &identity;

    // Invert D - I and D + I
    let inv_d_minus_i = match invert_real_matrix(&d_minus_i) {
        Some(inv) => inv,
        None => return Err("Cannot invert D - I matrix".to_string()),
    };
    let inv_d_plus_i = match invert_real_matrix(&d_plus_i) {
        Some(inv) => inv,
        None => return Err("Cannot invert D + I matrix".to_string()),
    };

    // B * inv(D-I) * C and B * inv(D+I) * C
    let prod_neg = ss.b.dot(&inv_d_minus_i).dot(&ss.c);
    let prod_pos = ss.b.dot(&inv_d_plus_i).dot(&ss.c);

    // P = (A - prod_neg) * (A - prod_pos)
    let a_minus_neg = &ss.a - &prod_neg;
    let a_minus_pos = &ss.a - &prod_pos;
    let p = a_minus_neg.dot(&a_minus_pos);

    // Extract eigenvalues of P
    let p_eigs = eigenvalues_real_matrix(&p)?;

    // Find purely imaginary square roots of eigenvalues
    use super::constants::EIGENVALUE_TOLERANCE;
    let mut freqs_violation: Vec<f64> = Vec::new();
    for eig in p_eigs.iter() {
        let sqrt_eig = eig.sqrt();
        if sqrt_eig.re.abs() < EIGENVALUE_TOLERANCE {
            let freq = sqrt_eig.im.abs() / (2.0 * PI);
            freqs_violation.push(freq);
        }
    }

    // Include DC if not already present
    if !freqs_violation.contains(&0.0) {
        freqs_violation.push(0.0);
    }

    // Sort frequencies
    freqs_violation.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Identify frequency bands of passivity violations
    let mut violation_bands: Vec<[f64; 2]> = Vec::new();
    let mut max_singular_value = 0.0_f64;

    for (i, &freq) in freqs_violation.iter().enumerate() {
        let (f_start, f_stop, f_center) = if i == freqs_violation.len() - 1 {
            // Last band extends to infinity
            (freq, f64::INFINITY, 1.1 * freq.max(1.0))
        } else {
            let f_next = freqs_violation[i + 1];
            (freq, f_next, 0.5 * (freq + f_next))
        };

        // Calculate S-matrix at center frequency
        let s_matrices = get_s_from_state_space(&[f_center], &ss);

        // Extract S-matrix for this frequency
        let s_center = s_matrices.slice(s![0, .., ..]);

        // Compute singular values using simplified SVD
        let sigma = singular_values_complex(&s_center.to_owned())?;

        let sigma_max = sigma.iter().cloned().fold(0.0_f64, f64::max);
        max_singular_value = max_singular_value.max(sigma_max);

        if sigma_max > 1.0 {
            violation_bands.push([f_start, f_stop]);
        }
    }

    Ok(PassivityTestResult {
        violation_bands,
        max_singular_value,
    })
}

// ============================================================================
// Linear algebra operations - delegated to linalg module
// ============================================================================

use crate::math::linalg;

fn invert_complex_matrix(a: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    linalg::inv_complex(a)
}

fn invert_real_matrix(a: &Array2<f64>) -> Option<Array2<f64>> {
    linalg::inv_real(a)
}

fn eigenvalues_real_matrix(a: &Array2<f64>) -> Result<Vec<Complex64>, String> {
    linalg::eigenvalues(a).map_err(|e| e.to_string())
}

fn singular_values_complex(a: &Array2<Complex64>) -> Result<Vec<f64>, String> {
    Ok(linalg::singular_values(a))
}

/// Perform full SVD on a complex matrix, returning U, S, Vh
#[allow(clippy::type_complexity)]
fn svd_complex_full(
    a: &Array2<Complex64>,
) -> Result<(Array2<Complex64>, Vec<f64>, Array2<Complex64>), String> {
    linalg::svd_complex(a).map_err(|e| e.to_string())
}

/// Perform full SVD on a real matrix
#[allow(clippy::type_complexity)]
#[allow(dead_code)]
fn svd_real_full(a: &Array2<f64>) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), String> {
    linalg::svd_real(a).map_err(|e| e.to_string())
}

/// Result of passivity enforcement
#[derive(Debug, Clone)]
pub struct PassivityEnforceResult {
    /// Updated residues after enforcement
    pub residues: Array2<Complex64>,
    /// History of max singular values during iterations
    pub history_max_sigma: Vec<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether enforcement was successful
    pub success: bool,
}

// ============================================================================
// Helper functions for passivity_enforce decomposition
// ============================================================================

/// Precompute (s*I - A)^-1 * B for all evaluation frequencies
///
/// Returns a 3D array [n_samples, dim_a, nports] of frequency coefficients.
fn precompute_frequency_coefficients(
    ss: &StateSpaceMatrices,
    freqs: &[f64],
    nports: usize,
) -> Array3<Complex64> {
    let dim_a = ss.a.nrows();
    let n_samples = freqs.len();
    let mut coeffs = Array3::<Complex64>::zeros((n_samples, dim_a, nports));

    for (f_idx, &freq) in freqs.iter().enumerate() {
        let s = Complex64::new(0.0, 2.0 * PI * freq);

        let mut s_minus_a = Array2::<Complex64>::zeros((dim_a, dim_a));
        for i in 0..dim_a {
            for j in 0..dim_a {
                if i == j {
                    s_minus_a[[i, j]] = s - Complex64::new(ss.a[[i, j]], 0.0);
                } else {
                    s_minus_a[[i, j]] = Complex64::new(-ss.a[[i, j]], 0.0);
                }
            }
        }

        if let Some(inv) = invert_complex_matrix(&s_minus_a) {
            let b_complex = ss.b.mapv(|x| Complex64::new(x, 0.0));
            let coeff = inv.dot(&b_complex);
            for i in 0..dim_a {
                for j in 0..nports {
                    coeffs[[f_idx, i, j]] = coeff[[i, j]];
                }
            }
        }
    }

    coeffs
}

/// Convert C matrix back to residues format
///
/// Transforms the perturbed C matrix into the standard residue format.
fn c_matrix_to_residues(
    c_t: &Array2<f64>,
    poles: &Array1<Complex64>,
    original_residues: &Array2<Complex64>,
    nports: usize,
    model_order: usize,
) -> Array2<Complex64> {
    let mut new_residues = original_residues.clone();

    for i in 0..nports {
        for j in 0..nports {
            let i_response = i * nports + j;
            let mut k = j * model_order; // C_t column index

            for (z, pole) in poles.iter().enumerate() {
                if pole.im == 0.0 {
                    new_residues[[i_response, z]] = Complex64::new(c_t[[i, k]], 0.0);
                    k += 1;
                } else {
                    new_residues[[i_response, z]] = Complex64::new(c_t[[i, k]], c_t[[i, k + 1]]);
                    k += 2;
                }
            }
        }
    }

    new_residues
}

/// Enforce passivity of the vector fitted model
///
/// Uses iterative singular value perturbation to enforce passivity.
///
/// # Arguments
/// * `poles` - Fitted poles
/// * `residues` - Fitted residues [n_responses, n_poles]
/// * `constant_coeff` - Constant coefficients
/// * `proportional_coeff` - Proportional coefficients
/// * `nports` - Number of ports
/// * `f_max` - Maximum frequency of interest (Hz)
/// * `n_samples` - Number of frequency samples for evaluation
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// `PassivityEnforceResult` with updated residues
#[allow(clippy::too_many_arguments)]
pub fn passivity_enforce(
    poles: &Array1<Complex64>,
    residues: &Array2<Complex64>,
    constant_coeff: &Array1<f64>,
    proportional_coeff: &Array1<f64>,
    nports: usize,
    max_iterations: usize,
) -> Result<PassivityEnforceResult, String> {
    // Check for proportional coefficients (not supported)
    if proportional_coeff.iter().any(|&e| e != 0.0) {
        return Err(
            "Passivity enforcement with nonzero proportional coefficients is not supported"
                .to_string(),
        );
    }

    // Check if already passive
    let test_result = passivity_test(poles, residues, constant_coeff, proportional_coeff, nports)?;
    if test_result.is_passive() {
        return Ok(PassivityEnforceResult {
            residues: residues.clone(),
            history_max_sigma: vec![test_result.max_singular_value],
            iterations: 0,
            success: true,
        });
    }

    // Get state-space matrices
    let ss =
        build_state_space_matrices(poles, residues, constant_coeff, proportional_coeff, nports);

    // Mutable C matrix for perturbation
    let mut c_t = ss.c.clone();

    // Count model order
    use super::poles::PoleSet;
    let pole_set = PoleSet::from_array(poles);
    let model_order = pole_set.model_order();

    use super::constants::PASSIVITY_DAMPING_FACTOR;
    let mut sigma_max;
    let mut t = 0;
    let mut history_max_sigma: Vec<f64> = Vec::new();
    let mut current_damping = PASSIVITY_DAMPING_FACTOR;
    let mut best_c = c_t.clone();
    let mut min_sigma = test_result.max_singular_value;

    // Iterative passivity enforcement
    while t < max_iterations {
        // 1. Get current residues and test passivity
        let current_residues = c_matrix_to_residues(&c_t, poles, residues, nports, model_order);
        let test_result = match passivity_test(
            poles,
            &current_residues,
            constant_coeff,
            proportional_coeff,
            nports,
        ) {
            Ok(res) => res,
            Err(_) => break,
        };

        sigma_max = test_result.max_singular_value;
        history_max_sigma.push(sigma_max);

        if sigma_max < min_sigma {
            min_sigma = sigma_max;
            best_c = c_t.clone();
        } else if sigma_max > min_sigma * 1.1 {
            // Divergence detected, revert and reduce damping
            c_t = best_c.clone();
            current_damping *= 0.5;
            if current_damping < 1e-4 {
                break;
            }
        }

        if test_result.is_passive() || sigma_max <= 1.0 {
            break;
        }

        // 2. Build thorough evaluation frequencies targeting violation bands + DC Guard
        let mut sample_freqs = Vec::new();
        // Commercial preference: Always guard DC to prevent offset drift
        sample_freqs.push(0.0);

        for band in &test_result.violation_bands {
            sample_freqs.push(band[0]);
            if band[1].is_infinite() {
                sample_freqs.push(band[0] * 1.5);
                sample_freqs.push(band[0] * 3.0);
                sample_freqs.push(band[0] * 10.0);
            } else {
                sample_freqs.push(band[1]);
                sample_freqs.push(0.5 * (band[0] + band[1]));
                if band[1] > 2.0 * band[0] {
                    sample_freqs.push(0.75 * band[0] + 0.25 * band[1]);
                    sample_freqs.push(0.25 * band[0] + 0.75 * band[1]);
                }
            }
        }
        sample_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sample_freqs.dedup();

        let n_fit_samples = sample_freqs.len();

        // 3. Precompute coefficients and S-matrices with weighting
        let fit_coeffs = precompute_frequency_coefficients(&ss, &sample_freqs, nports);
        let c_complex = c_t.mapv(|x| Complex64::new(x, 0.0));
        let d_complex = ss.d.mapv(|x| Complex64::new(x, 0.0));

        let mut s_viol = Array3::<Complex64>::zeros((n_fit_samples, nports, nports));
        let mut sample_weights = vec![1.0; n_fit_samples];

        for f_idx in 0..n_fit_samples {
            let coeff_slice = fit_coeffs.slice(s![f_idx, .., ..]);
            let s_f = c_complex.dot(&coeff_slice.to_owned()) + &d_complex;

            if let Ok((u, sigma, vh)) = svd_complex_full(&s_f.to_owned()) {
                use super::constants::PASSIVITY_DELTA_THRESHOLD;
                let delta = PASSIVITY_DELTA_THRESHOLD;

                let this_max_sig = sigma.iter().cloned().fold(0.0, f64::max);

                // Weighting logic: Priority to high violations
                // Points with no violation get low weight (just keep them where they are)
                // DC Guard gets extremely high weight
                if sample_freqs[f_idx] == 0.0 {
                    sample_weights[f_idx] = 1000.0;
                } else if this_max_sig > 1.0 {
                    sample_weights[f_idx] = (this_max_sig - 1.0).sqrt() * 10.0 + 1.0;
                } else {
                    sample_weights[f_idx] = 0.1;
                }

                let sigma_viol: Vec<f64> = sigma
                    .iter()
                    .map(|&s| if s > delta { s - delta } else { 0.0 })
                    .collect();

                for i in 0..nports {
                    for j in 0..nports {
                        let mut val = Complex64::new(0.0, 0.0);
                        for k in 0..nports.min(sigma_viol.len()) {
                            val += u[[i, k]] * sigma_viol[k] * vh[[k, j]];
                        }
                        s_viol[[f_idx, i, j]] = val;
                    }
                }
            }
        }

        // 4. Fit violation using Weighted Least Squares per response (i, j)
        for i in 0..nports {
            for j in 0..nports {
                let mut lapack_a = Vec::with_capacity(n_fit_samples * 2 * model_order);
                let mut lapack_b = Vec::with_capacity(n_fit_samples * 2);
                let mut count = 0;

                for f_idx in 0..n_fit_samples {
                    let viol = s_viol[[f_idx, i, j]];
                    let w = sample_weights[f_idx];

                    // Add samples if they are violations OR for DC guard
                    if viol.norm() > 1e-15 || sample_freqs[f_idx] == 0.0 {
                        count += 1;
                        let offset = j * model_order;

                        for k in 0..model_order {
                            let coeff = fit_coeffs[[f_idx, offset + k, j]];
                            lapack_a.push(coeff.re * w);
                        }
                        lapack_b.push(viol.re * w);

                        for k in 0..model_order {
                            let coeff = fit_coeffs[[f_idx, offset + k, j]];
                            lapack_a.push(coeff.im * w);
                        }
                        lapack_b.push(viol.im * w);
                    }
                }

                if count > 0 {
                    let a_mat = Array2::from_shape_vec((count * 2, model_order), lapack_a).unwrap();
                    let b_vec = Array1::from_vec(lapack_b);

                    if let Ok(result) = linalg::lstsq(&a_mat, &b_vec) {
                        for (k, &perturbation) in result.solution.iter().enumerate() {
                            let idx = j * model_order + k;
                            if idx < c_t.ncols() {
                                c_t[[i, idx]] -= perturbation * current_damping;
                            }
                        }
                    }
                }
            }
        }

        t += 1;
    }

    // Convert C_t back to residues format (use helper function)
    let new_residues = c_matrix_to_residues(&c_t, poles, residues, nports, model_order);

    // Final passivity check
    let final_test = passivity_test(
        poles,
        &new_residues,
        constant_coeff,
        proportional_coeff,
        nports,
    )?;
    let success = final_test.is_passive();

    Ok(PassivityEnforceResult {
        residues: new_residues,
        history_max_sigma,
        iterations: t,
        success,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_state_space_matrices() {
        let poles = Array1::from_vec(vec![
            Complex64::new(-1e9, 0.0),   // real
            Complex64::new(-0.5e9, 1e9), // complex
        ]);
        let residues = Array2::from_shape_vec(
            (1, 2),
            vec![Complex64::new(0.5, 0.0), Complex64::new(0.2, 0.1)],
        )
        .unwrap();
        let constant_coeff = Array1::from_vec(vec![0.1]);
        let proportional_coeff = Array1::from_vec(vec![0.0]);

        let ss =
            build_state_space_matrices(&poles, &residues, &constant_coeff, &proportional_coeff, 1);

        assert_eq!(ss.a.dim(), (3, 3)); // 1 real + 2 for complex = 3
        assert_eq!(ss.b.dim(), (3, 1));
        assert_eq!(ss.c.dim(), (1, 3));
        assert_eq!(ss.d.dim(), (1, 1));
        assert_eq!(ss.e.dim(), (1, 1));
    }

    #[test]
    fn test_passivity_enforce_simple() {
        // Create a non-passive system: S(s) = 0.5 + 1.0e9 / (s + 1.0e9)
        // At s=0, S(0) = 1.5 -> Non-passive
        let poles = Array1::from_vec(vec![Complex64::new(-1.0e9, 0.0)]);
        let residues = Array2::from_shape_vec((1, 1), vec![Complex64::new(1.0e9, 0.0)]).unwrap();
        let constant_coeff = Array1::from_vec(vec![0.5]);
        let proportional_coeff = Array1::from_vec(vec![0.0]);

        // Test initial passivity
        let test_init =
            passivity_test(&poles, &residues, &constant_coeff, &proportional_coeff, 1).unwrap();
        assert!(!test_init.is_passive());
        assert!(test_init.max_singular_value > 1.2);

        // Enforce passivity
        let result = passivity_enforce(
            &poles,
            &residues,
            &constant_coeff,
            &proportional_coeff,
            1,
            20, // max_iterations
        )
        .unwrap();

        // Check if max singular value decreased
        let test_final = passivity_test(
            &poles,
            &result.residues,
            &constant_coeff,
            &proportional_coeff,
            1,
        )
        .unwrap();

        // It should be closer to passive
        assert!(test_final.max_singular_value < test_init.max_singular_value);
        if result.success {
            assert!(test_final.max_singular_value <= 1.001);
        }
    }
}
