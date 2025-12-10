//! Model response evaluation
//!
//! Functions for evaluating the pole-residue model at given frequencies.

use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Evaluate model response at given frequencies
///
/// Computes: H(s) = d + e*s + sum_k[ r_k / (s - p_k) + conj(r_k) / (s - conj(p_k)) ]
/// where complex poles contribute their conjugate pair.
///
/// # Arguments
/// * `poles` - Array of poles (only positive imaginary for conjugate pairs)
/// * `residues` - Array of residues corresponding to poles
/// * `d` - Constant coefficient
/// * `e` - Proportional coefficient
/// * `freqs` - Frequencies in Hz at which to evaluate
///
/// # Returns
/// Complex frequency response at each frequency
pub fn evaluate_response(
    poles: &Array1<Complex64>,
    residues: &Array1<Complex64>,
    d: f64,
    e: f64,
    freqs: &[f64],
) -> Array1<Complex64> {
    let n_freqs = freqs.len();
    let mut response = Array1::<Complex64>::zeros(n_freqs);

    for (f_idx, &freq) in freqs.iter().enumerate() {
        let s = Complex64::new(0.0, 2.0 * PI * freq);

        // Start with constant and proportional terms
        let mut h = Complex64::new(d, 0.0) + Complex64::new(e, 0.0) * s;

        // Add pole-residue contributions
        for (p_idx, &pole) in poles.iter().enumerate() {
            let residue = residues[p_idx];

            if pole.im == 0.0 {
                // Real pole: single term
                h += residue / (s - pole);
            } else {
                // Complex conjugate pole pair
                h += residue / (s - pole) + residue.conj() / (s - pole.conj());
            }
        }

        response[f_idx] = h;
    }

    response
}

/// Calculate RMS error between model and target responses
#[allow(dead_code)]
pub fn rms_error(model_response: &Array1<Complex64>, target_response: &Array1<Complex64>) -> f64 {
    if model_response.len() != target_response.len() || model_response.is_empty() {
        return f64::NAN;
    }

    let n = model_response.len() as f64;
    let error_sum: f64 = model_response
        .iter()
        .zip(target_response.iter())
        .map(|(m, t)| (m - t).norm_sqr())
        .sum();

    (error_sum / n).sqrt()
}

/// Calculate maximum absolute error between model and target responses
#[allow(dead_code)]
pub fn max_error(model_response: &Array1<Complex64>, target_response: &Array1<Complex64>) -> f64 {
    if model_response.len() != target_response.len() || model_response.is_empty() {
        return f64::NAN;
    }

    model_response
        .iter()
        .zip(target_response.iter())
        .map(|(m, t)| (m - t).norm())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_evaluate_response_constant() {
        // Model with only constant term
        let poles = Array1::<Complex64>::zeros(0);
        let residues = Array1::<Complex64>::zeros(0);
        let d = 0.5;
        let e = 0.0;
        let freqs = vec![1e9, 2e9, 3e9];

        let response = evaluate_response(&poles, &residues, d, e, &freqs);

        for h in response.iter() {
            assert_relative_eq!(h.re, 0.5, epsilon = 1e-10);
            assert_relative_eq!(h.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_evaluate_response_single_real_pole() {
        // Single real pole: H(s) = r / (s - p) + d
        let pole = Complex64::new(-1e9, 0.0);
        let residue = Complex64::new(1e9, 0.0);
        let poles = Array1::from_vec(vec![pole]);
        let residues = Array1::from_vec(vec![residue]);
        let d = 0.0;
        let e = 0.0;
        let freqs = vec![0.0];

        let response = evaluate_response(&poles, &residues, d, e, &freqs);

        // At s=0: H(0) = 1e9 / (0 - (-1e9)) = 1e9 / 1e9 = 1
        assert_relative_eq!(response[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(response[0].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rms_error() {
        let model = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);
        let target = Array1::from_vec(vec![
            Complex64::new(1.1, 0.0),
            Complex64::new(2.1, 0.0),
            Complex64::new(3.1, 0.0),
        ]);

        let error = rms_error(&model, &target);
        assert_relative_eq!(error, 0.1, epsilon = 1e-10);
    }
}
