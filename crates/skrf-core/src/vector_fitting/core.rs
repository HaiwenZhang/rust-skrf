//! Core VectorFitting struct and main fitting routine

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::algorithms::{self, InitPoleSpacing, PoleRelocationResult};
use super::model;
use crate::network::Network;

/// Vector Fitting result for storing fitted model parameters
#[derive(Debug, Clone)]
pub struct VectorFitting {
    /// Fitted poles (complex, only positive imaginary for conjugate pairs)
    pub poles: Option<Array1<Complex64>>,

    /// Fitted residues [n_responses, n_poles]
    pub residues: Option<Array2<Complex64>>,

    /// Proportional coefficients (usually 0 for S-params)
    pub proportional_coeff: Option<Array1<f64>>,

    /// Constant coefficients
    pub constant_coeff: Option<Array1<f64>>,

    /// Maximum iterations for pole relocation
    pub max_iterations: usize,

    /// Convergence tolerance
    pub max_tol: f64,

    /// Wall-clock time of last fit (in seconds)
    pub wall_clock_time: f64,
}

impl Default for VectorFitting {
    fn default() -> Self {
        Self {
            poles: None,
            residues: None,
            proportional_coeff: None,
            constant_coeff: None,
            max_iterations: 100,
            max_tol: 1e-6,
            wall_clock_time: 0.0,
        }
    }
}

impl VectorFitting {
    /// Create a new VectorFitting instance with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Perform vector fitting on a network
    ///
    /// # Arguments
    /// * `network` - The network to fit
    /// * `n_poles_real` - Number of initial real poles
    /// * `n_poles_cmplx` - Number of initial complex conjugate poles
    /// * `init_pole_spacing` - Type of initial pole spacing (Linear or Log)
    /// * `fit_constant` - Include a constant term in the fit
    /// * `fit_proportional` - Include a proportional term in the fit
    ///
    /// # Returns
    /// `Ok(())` on success, error message on failure
    pub fn vector_fit(
        &mut self,
        network: &Network,
        n_poles_real: usize,
        n_poles_cmplx: usize,
        init_pole_spacing: InitPoleSpacing,
        fit_constant: bool,
        fit_proportional: bool,
    ) -> Result<(), String> {
        use std::time::Instant;
        let timer_start = Instant::now();

        let nfreq = network.nfreq();
        let nports = network.nports();

        if nfreq < 2 {
            return Err("Network must have at least 2 frequency points".to_string());
        }

        // Get frequencies in Hz
        let freqs: Vec<f64> = network.frequency.f().to_vec();

        // Normalize frequencies for numerical stability
        let norm: f64 = freqs.iter().sum::<f64>() / freqs.len() as f64;
        let freqs_norm: Vec<f64> = freqs.iter().map(|f| f / norm).collect();

        // Initialize poles
        let mut poles =
            algorithms::init_poles(&freqs_norm, n_poles_real, n_poles_cmplx, init_pole_spacing);

        // Stack frequency responses as a single vector
        // Stacking order (row-major): s11, s12, ..., s21, s22, ...
        let n_responses = nports * nports;
        let mut freq_responses = Array2::<Complex64>::zeros((n_responses, nfreq));

        for i in 0..nports {
            for j in 0..nports {
                let idx = i * nports + j;
                for f in 0..nfreq {
                    freq_responses[[idx, f]] = network.s[[f, i, j]];
                }
            }
        }

        // Calculate response weights (based on norm)
        let weights_responses: Vec<f64> = (0..n_responses)
            .map(|idx| {
                let row = freq_responses.row(idx);
                row.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
            })
            .collect();

        // Iterative pole relocation
        let mut converged = false;
        let mut max_singular = 1.0;

        for iteration in 0..self.max_iterations {
            let result = algorithms::pole_relocation(
                &poles,
                &freqs_norm,
                &freq_responses,
                &weights_responses,
                fit_constant,
                fit_proportional,
            );

            match result {
                Ok(PoleRelocationResult {
                    poles: new_poles,
                    d_res: _,
                    singular_vals,
                    ..
                }) => {
                    poles = new_poles;

                    // Check convergence
                    let new_max_singular = singular_vals.iter().cloned().fold(0.0, f64::max);
                    let delta_max = (1.0 - new_max_singular / max_singular).abs();
                    max_singular = new_max_singular;

                    if delta_max < self.max_tol {
                        if converged {
                            // Really converged
                            break;
                        } else {
                            // Might be converged, do one more iteration
                            converged = true;
                        }
                    } else {
                        converged = false;
                    }
                }
                Err(e) => {
                    return Err(format!(
                        "Pole relocation failed at iteration {}: {}",
                        iteration, e
                    ))
                }
            }
        }

        // Fit residues with final poles
        let residue_result = algorithms::fit_residues(
            &poles,
            &freqs_norm,
            &freq_responses,
            fit_constant,
            fit_proportional,
        );

        match residue_result {
            Ok((residues, constant_coeff, proportional_coeff)) => {
                // Un-normalize and store results
                self.poles = Some(poles.mapv(|p| p * norm));
                self.residues = Some(residues.mapv(|r| r * norm));
                self.constant_coeff = Some(constant_coeff);
                self.proportional_coeff = Some(proportional_coeff.mapv(|e| e / norm));
            }
            Err(e) => return Err(format!("Residue fitting failed: {}", e)),
        }

        self.wall_clock_time = timer_start.elapsed().as_secs_f64();
        Ok(())
    }

    /// Get the model response at specified frequencies
    ///
    /// # Arguments
    /// * `i` - Row index of the response
    /// * `j` - Column index of the response
    /// * `freqs` - Frequencies at which to evaluate (Hz)
    ///
    /// # Returns
    /// Complex frequency response array
    pub fn get_model_response(
        &self,
        i: usize,
        j: usize,
        freqs: &[f64],
    ) -> Option<Array1<Complex64>> {
        let poles = self.poles.as_ref()?;
        let residues = self.residues.as_ref()?;
        let constant_coeff = self.constant_coeff.as_ref()?;
        let proportional_coeff = self.proportional_coeff.as_ref()?;

        let n_ports = (constant_coeff.len() as f64).sqrt() as usize;
        let idx = i * n_ports + j;

        if idx >= residues.nrows() {
            return None;
        }

        let response_residues = residues.row(idx);
        let d = constant_coeff[idx];
        let e = proportional_coeff[idx];

        Some(model::evaluate_response(
            poles,
            &response_residues.to_owned(),
            d,
            e,
            freqs,
        ))
    }

    /// Calculate the model order
    ///
    /// Model order = N_real + 2 * N_complex
    pub fn get_model_order(&self) -> Option<usize> {
        self.poles.as_ref().map(algorithms::get_model_order)
    }

    /// Get the RMS error between the fitted model and original network
    pub fn get_rms_error(&self, network: &Network, i: usize, j: usize) -> Option<f64> {
        let freqs: Vec<f64> = network.frequency.f().to_vec();
        let model_response = self.get_model_response(i, j, &freqs)?;

        let nfreq = network.nfreq();
        let mut error_sum = 0.0;
        for f in 0..nfreq {
            let orig = network.s[[f, i, j]];
            let model = model_response[f];
            error_sum += (orig - model).norm_sqr();
        }

        Some((error_sum / nfreq as f64).sqrt())
    }

    /// Write SPICE subcircuit netlist to a file
    ///
    /// Creates an equivalent N-port subcircuit based on its vector fitted scattering (S)
    /// parameter responses in SPICE simulator netlist syntax (compatible with LTspice,
    /// ngspice, Xyce, ...).
    ///
    /// # Arguments
    /// * `path` - Path to the output file
    /// * `network` - The network that was fitted (for port info and z0)
    /// * `model_name` - Name of the subcircuit (default: "s_equivalent")
    /// * `create_reference_pins` - If true, create separate reference pins for each port
    ///
    /// # Returns
    /// `Ok(())` on success, error message on failure
    pub fn write_spice_subcircuit_s(
        &self,
        path: &std::path::Path,
        network: &Network,
        model_name: Option<&str>,
        create_reference_pins: bool,
    ) -> Result<(), String> {
        let poles = self.poles.as_ref().ok_or("Model not fitted yet")?;
        let residues = self.residues.as_ref().ok_or("Model not fitted yet")?;
        let constant_coeff = self.constant_coeff.as_ref().ok_or("Model not fitted yet")?;
        let proportional_coeff = self
            .proportional_coeff
            .as_ref()
            .ok_or("Model not fitted yet")?;

        let name = model_name.unwrap_or("s_equivalent");

        super::spice::write_spice_subcircuit_s_to_file(
            path,
            poles,
            residues,
            constant_coeff,
            proportional_coeff,
            network.z0(),
            network.nports(),
            name,
            create_reference_pins,
        )
        .map_err(|e| format!("Failed to write SPICE file: {}", e))
    }

    /// Generate SPICE subcircuit netlist as a string
    ///
    /// # Arguments
    /// * `network` - The network that was fitted (for port info and z0)
    /// * `model_name` - Name of the subcircuit (default: "s_equivalent")
    /// * `create_reference_pins` - If true, create separate reference pins for each port
    ///
    /// # Returns
    /// SPICE netlist as a String
    pub fn generate_spice_subcircuit_s(
        &self,
        network: &Network,
        model_name: Option<&str>,
        create_reference_pins: bool,
    ) -> Result<String, String> {
        let poles = self.poles.as_ref().ok_or("Model not fitted yet")?;
        let residues = self.residues.as_ref().ok_or("Model not fitted yet")?;
        let constant_coeff = self.constant_coeff.as_ref().ok_or("Model not fitted yet")?;
        let proportional_coeff = self
            .proportional_coeff
            .as_ref()
            .ok_or("Model not fitted yet")?;

        let name = model_name.unwrap_or("s_equivalent");

        Ok(super::spice::generate_spice_subcircuit_s(
            poles,
            residues,
            constant_coeff,
            proportional_coeff,
            network.z0(),
            network.nports(),
            name,
            create_reference_pins,
        ))
    }

    /// Perform passivity test on the fitted model
    ///
    /// Evaluates the passivity of the vector fitted model using the half-size test matrix method.
    /// Returns frequency bands where passivity violations occur.
    ///
    /// # Arguments
    /// * `nports` - Number of ports in the original network
    ///
    /// # Returns
    /// `PassivityTestResult` containing violation bands and max singular value
    pub fn passivity_test(
        &self,
        nports: usize,
    ) -> Result<super::passivity::PassivityTestResult, String> {
        let poles = self.poles.as_ref().ok_or("Model not fitted yet")?;
        let residues = self.residues.as_ref().ok_or("Model not fitted yet")?;
        let constant_coeff = self.constant_coeff.as_ref().ok_or("Model not fitted yet")?;
        let proportional_coeff = self
            .proportional_coeff
            .as_ref()
            .ok_or("Model not fitted yet")?;

        super::passivity::passivity_test(
            poles,
            residues,
            constant_coeff,
            proportional_coeff,
            nports,
        )
    }

    /// Check if the fitted model is passive
    ///
    /// # Arguments
    /// * `nports` - Number of ports in the original network
    ///
    /// # Returns
    /// `true` if model is passive, `false` otherwise
    pub fn is_passive(&self, nports: usize) -> Result<bool, String> {
        let result = self.passivity_test(nports)?;
        Ok(result.is_passive())
    }

    /// Get the state-space representation matrices (A, B, C, D, E)
    ///
    /// # Arguments
    /// * `nports` - Number of ports in the original network
    pub fn get_state_space_matrices(
        &self,
        nports: usize,
    ) -> Result<super::passivity::StateSpaceMatrices, String> {
        let poles = self.poles.as_ref().ok_or("Model not fitted yet")?;
        let residues = self.residues.as_ref().ok_or("Model not fitted yet")?;
        let constant_coeff = self.constant_coeff.as_ref().ok_or("Model not fitted yet")?;
        let proportional_coeff = self
            .proportional_coeff
            .as_ref()
            .ok_or("Model not fitted yet")?;

        Ok(super::passivity::build_state_space_matrices(
            poles,
            residues,
            constant_coeff,
            proportional_coeff,
            nports,
        ))
    }

    /// Enforce passivity of the fitted model
    ///
    /// Uses iterative singular value perturbation to enforce passivity.
    ///
    /// # Arguments
    /// * `nports` - Number of ports in the original network
    /// * `f_max` - Maximum frequency of interest (Hz)
    /// * `n_samples` - Number of frequency samples for evaluation (default: 200)
    ///
    /// # Returns
    /// `Ok(())` on success, updating internal residues
    pub fn passivity_enforce(
        &mut self,
        nports: usize,
    ) -> Result<super::passivity::PassivityEnforceResult, String> {
        let poles = self.poles.as_ref().ok_or("Model not fitted yet")?;
        let residues = self.residues.as_ref().ok_or("Model not fitted yet")?;
        let constant_coeff = self.constant_coeff.as_ref().ok_or("Model not fitted yet")?;
        let proportional_coeff = self
            .proportional_coeff
            .as_ref()
            .ok_or("Model not fitted yet")?;
        let result = super::passivity::passivity_enforce(
            poles,
            residues,
            constant_coeff,
            proportional_coeff,
            nports,
            self.max_iterations,
        )?;

        // Update residues with enforced values
        self.residues = Some(result.residues.clone());

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_fitting_new() {
        let vf = VectorFitting::new();
        assert!(vf.poles.is_none());
        assert!(vf.residues.is_none());
        assert_eq!(vf.max_iterations, 100);
        assert_eq!(vf.max_tol, 1e-6);
    }

    #[test]
    fn test_get_model_order() {
        let mut vf = VectorFitting::new();

        // 2 real poles + 2 complex poles = 2 + 2*2 = 6
        let poles = Array1::from_vec(vec![
            Complex64::new(-1.0, 0.0), // real
            Complex64::new(-2.0, 0.0), // real
            Complex64::new(-0.1, 1.0), // complex
            Complex64::new(-0.2, 2.0), // complex
        ]);
        vf.poles = Some(poles);

        assert_eq!(vf.get_model_order(), Some(6));
    }
}
