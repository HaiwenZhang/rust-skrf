//! Time-domain analysis functions
//!
//! Provides impulse response and step response calculations via FFT.
//!
//! Note: For accurate time-domain results, frequency data should:
//! - Start from 0 Hz (DC)
//! - Have uniform frequency spacing

use ndarray::{Array1, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

use super::core::Network;

/// Window function types
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// No window (rectangular)
    None,
    /// Hamming window
    Hamming,
    /// Hanning window
    Hanning,
    /// Blackman window
    Blackman,
}

impl Network {
    /// Calculate time-domain impulse response
    ///
    /// Uses inverse FFT to transform S-parameters to time domain.
    /// Returns (time_vector, impulse_response) where impulse_response
    /// has shape [ntime, nports, nports].
    ///
    /// # Arguments
    /// * `window` - Window function to apply
    /// * `pad` - Number of zero-padding points
    pub fn impulse_response(
        &self,
        window: WindowType,
        pad: usize,
    ) -> Option<(Array1<f64>, Array3<Complex64>)> {
        let nfreq = self.nfreq();
        if nfreq < 2 {
            return None;
        }

        let nports = self.nports();
        let f = self.frequency.f();

        // Calculate frequency step
        let df = f[1] - f[0];
        if df <= 0.0 {
            return None;
        }

        // Total points including padding
        let n_total = nfreq + pad;
        // Time step from frequency range
        let dt = 1.0 / (df * n_total as f64);

        // Apply window function
        let windowed_s = apply_window(&self.s, window);

        // Perform IFFT for each S-parameter
        let mut ir = Array3::<Complex64>::zeros((n_total, nports, nports));

        for i in 0..nports {
            for j in 0..nports {
                // Extract frequency-domain data
                let mut freq_data = vec![Complex64::new(0.0, 0.0); n_total];
                for k in 0..nfreq {
                    freq_data[k] = windowed_s[[k, i, j]];
                }

                // Simple IFFT implementation
                let time_data = ifft(&freq_data);

                // FFT shift to center zero-time
                let shift = n_total / 2;
                for k in 0..n_total {
                    let src_idx = (k + shift) % n_total;
                    ir[[k, i, j]] = time_data[src_idx];
                }
            }
        }

        // Create time vector centered at 0
        let mut t = Array1::<f64>::zeros(n_total);
        let t_offset = (n_total as f64 / 2.0) * dt;
        for k in 0..n_total {
            t[k] = k as f64 * dt - t_offset;
        }

        Some((t, ir))
    }

    /// Calculate time-domain step response
    ///
    /// Step response is the cumulative integral of the impulse response.
    /// Returns (time_vector, step_response).
    pub fn step_response(
        &self,
        window: WindowType,
        pad: usize,
    ) -> Option<(Array1<f64>, Array3<f64>)> {
        let (t, ir) = self.impulse_response(window, pad)?;

        let n_total = t.len();
        let nports = self.nports();
        let dt = if n_total > 1 { t[1] - t[0] } else { 1.0 };

        // Cumulative trapezoidal integration
        let mut sr = Array3::<f64>::zeros((n_total, nports, nports));

        for i in 0..nports {
            for j in 0..nports {
                let mut cumsum = 0.0;
                for k in 0..n_total {
                    if k > 0 {
                        // Trapezoidal rule
                        let y0 = ir[[k - 1, i, j]].re;
                        let y1 = ir[[k, i, j]].re;
                        cumsum += (y0 + y1) * dt / 2.0;
                    }
                    sr[[k, i, j]] = cumsum;
                }
            }
        }

        Some((t, sr))
    }
}

/// Apply window function to S-parameters using vectorized broadcasting
fn apply_window(s: &Array3<Complex64>, window: WindowType) -> Array3<Complex64> {
    let nfreq = s.shape()[0];
    let nports = s.shape()[1];

    let w = match window {
        WindowType::None => vec![1.0; nfreq],
        WindowType::Hamming => cosine_window(nfreq, &[0.54, 0.46]),
        WindowType::Hanning => cosine_window(nfreq, &[0.5, 0.5]),
        WindowType::Blackman => cosine_window(nfreq, &[0.42, 0.5, 0.08]),
    };

    // Vectorized: broadcast 1D window to 3D shape and multiply
    let w_array = Array1::from_vec(w);
    let w_3d = w_array
        .into_shape_with_order((nfreq, 1, 1))
        .unwrap()
        .broadcast((nfreq, nports, nports))
        .unwrap()
        .mapv(|x| Complex64::new(x, 0.0));

    s * &w_3d
}

/// Generalized cosine window: w[n] = sum((-1)^k * coeffs[k] * cos(k * 2π * n / (N-1)))
///
/// This unifies Hamming, Hanning, and Blackman windows with different coefficients.
fn cosine_window(n: usize, coeffs: &[f64]) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| {
            let x = 2.0 * PI * i as f64 / (n - 1) as f64;
            coeffs
                .iter()
                .enumerate()
                .map(|(k, &c)| {
                    let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                    sign * c * (k as f64 * x).cos()
                })
                .sum()
        })
        .collect()
}

/// IFFT using rustfft library (O(n log n) implementation)
///
/// Note: rustfft 6.x uses num_complex::Complex directly, so no type conversion needed.
fn ifft(data: &[Complex64]) -> Vec<Complex64> {
    use rustfft::FftPlanner;

    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // rustfft 6.x uses the same Complex64 type as num_complex - no conversion needed
    let mut buffer: Vec<Complex64> = data.to_vec();

    // Create IFFT planner and perform IFFT
    let mut planner = FftPlanner::new();
    let ifft_plan = planner.plan_fft_inverse(n);
    ifft_plan.process(&mut buffer);

    // Normalize in-place (more efficient than creating new vec)
    let scale = 1.0 / n as f64;
    for c in buffer.iter_mut() {
        *c *= scale;
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};

    #[test]
    fn test_impulse_response() {
        // Create a simple through with linear phase
        let freq = Frequency::new(0.0, 10.0, 11, FrequencyUnit::GHz, SweepType::Linear);
        let nfreq = 11;
        let mut s = Array3::<Complex64>::zeros((nfreq, 1, 1));

        for f in 0..nfreq {
            // Unity magnitude, linear phase
            let phase = -2.0 * PI * f as f64 * 0.1; // 0.1 ns delay
            s[[f, 0, 0]] = Complex64::from_polar(1.0, phase);
        }

        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);

        let result = ntwk.impulse_response(WindowType::Hamming, 0);
        assert!(result.is_some());

        let (t, ir) = result.unwrap();
        assert_eq!(t.len(), 11);
        assert_eq!(ir.shape(), &[11, 1, 1]);
    }
}
