//! Noise parameters for 2-port networks
//!
//! Provides noise figure calculations and noise circles for
//! low-noise amplifier design.

use num_complex::Complex64;

use super::core::Network;
use crate::constants::NEAR_ZERO;

/// Noise parameters for a 2-port network
#[derive(Debug, Clone)]
pub struct NoiseParams {
    /// Frequency points (Hz)
    pub f_noise: Vec<f64>,
    /// Minimum noise figure (linear, not dB)
    pub nfmin: Vec<f64>,
    /// Optimum source reflection coefficient
    pub gamma_opt: Vec<Complex64>,
    /// Equivalent noise resistance (normalized to Z0)
    pub rn: Vec<f64>,
}

impl Network {
    /// Set noise parameters for the network
    ///
    /// # Arguments
    /// * `f_noise` - Noise frequency points (Hz)
    /// * `nfmin_db` - Minimum noise figure in dB
    /// * `gamma_opt` - Optimum source reflection coefficient
    /// * `rn` - Equivalent noise resistance (Ohms)
    pub fn create_noise_params(
        f_noise: Vec<f64>,
        nfmin_db: Vec<f64>,
        gamma_opt: Vec<Complex64>,
        rn: Vec<f64>,
    ) -> NoiseParams {
        let nfmin: Vec<f64> = nfmin_db
            .iter()
            .map(|&db| 10.0_f64.powf(db / 10.0))
            .collect();
        NoiseParams {
            f_noise,
            nfmin,
            gamma_opt,
            rn,
        }
    }

    /// Calculate noise figure for given source reflection coefficient
    ///
    /// NF = NFmin + 4*Rn*|Γs - Γopt|² / ((1-|Γs|²)*|1+Γopt|²)
    ///
    /// # Arguments
    /// * `params` - Noise parameters
    /// * `gamma_s` - Source reflection coefficient
    ///
    /// # Returns
    /// Noise figure in dB for each frequency point
    pub fn noise_figure(params: &NoiseParams, gamma_s: Complex64) -> Vec<f64> {
        let mut nf_db = Vec::with_capacity(params.f_noise.len());

        for i in 0..params.f_noise.len() {
            let nfmin = params.nfmin[i];
            let gamma_opt = params.gamma_opt[i];
            let rn = params.rn[i];

            let gamma_s_mag_sq = gamma_s.norm_sqr();
            let diff = gamma_s - gamma_opt;
            let one_plus_gamma_opt = Complex64::new(1.0, 0.0) + gamma_opt;

            if gamma_s_mag_sq < 1.0 && one_plus_gamma_opt.norm_sqr() > NEAR_ZERO {
                let nf = nfmin
                    + 4.0 * rn * diff.norm_sqr()
                        / ((1.0 - gamma_s_mag_sq) * one_plus_gamma_opt.norm_sqr());
                nf_db.push(10.0 * nf.log10());
            } else {
                nf_db.push(f64::INFINITY);
            }
        }

        nf_db
    }

    /// Calculate noise figure circle (constant NF contour)
    ///
    /// Returns center and radius for plotting on Smith chart.
    ///
    /// # Arguments
    /// * `params` - Noise parameters at single frequency point
    /// * `nf_db` - Desired noise figure in dB
    /// * `idx` - Frequency index
    ///
    /// # Returns
    /// (center, radius) for the noise circle
    pub fn noise_circle(params: &NoiseParams, nf_db: f64, idx: usize) -> Option<(Complex64, f64)> {
        if idx >= params.f_noise.len() {
            return None;
        }

        let nfmin = params.nfmin[idx];
        let gamma_opt = params.gamma_opt[idx];
        let rn = params.rn[idx];

        let nf = 10.0_f64.powf(nf_db / 10.0);
        let n_factor = (nf - nfmin) / (4.0 * rn);

        if n_factor <= 0.0 {
            return None; // NF must be greater than NFmin
        }

        let one_plus_gamma_opt = Complex64::new(1.0, 0.0) + gamma_opt;
        let gamma_opt_mag_sq = gamma_opt.norm_sqr();
        let n = n_factor * one_plus_gamma_opt.norm_sqr();

        // Center: C_NF = Γopt / (1 + N)
        let center = gamma_opt / Complex64::new(1.0 + n, 0.0);

        // Radius: R_NF = sqrt(N² + N*(1 - |Γopt|²)) / (1 + N)
        let r_sq = (n * n + n * (1.0 - gamma_opt_mag_sq)) / ((1.0 + n) * (1.0 + n));
        let radius = if r_sq > 0.0 { r_sq.sqrt() } else { 0.0 };

        Some((center, radius))
    }
}

/// Calculate optimum source impedance from gamma_opt
#[allow(dead_code)]
pub fn gamma_to_z(gamma: Complex64, z0: f64) -> Complex64 {
    let one = Complex64::new(1.0, 0.0);
    Complex64::new(z0, 0.0) * (one + gamma) / (one - gamma)
}

/// Calculate optimum source admittance from gamma_opt
#[allow(dead_code)]
pub fn gamma_to_y(gamma: Complex64, z0: f64) -> Complex64 {
    let one = Complex64::new(1.0, 0.0);
    (one - gamma) / (Complex64::new(z0, 0.0) * (one + gamma))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_figure_at_optimum() {
        // At optimum source, NF should equal NFmin
        let params = NoiseParams {
            f_noise: vec![1e9],
            nfmin: vec![10.0_f64.powf(0.5 / 10.0)], // 0.5 dB
            gamma_opt: vec![Complex64::new(0.5, 0.3)],
            rn: vec![0.1],
        };

        let gamma_s = params.gamma_opt[0]; // Source at optimum
        let nf_db = Network::noise_figure(&params, gamma_s);

        // Should be very close to NFmin = 0.5 dB
        assert!((nf_db[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_noise_circle() {
        let params = NoiseParams {
            f_noise: vec![1e9],
            nfmin: vec![10.0_f64.powf(0.5 / 10.0)],
            gamma_opt: vec![Complex64::new(0.3, 0.2)],
            rn: vec![0.2],
        };

        let result = Network::noise_circle(&params, 1.0, 0);
        assert!(result.is_some());

        let (center, radius) = result.unwrap();
        assert!(center.norm() < 1.0); // Center should be inside Smith chart
        assert!(radius > 0.0);
    }
}
