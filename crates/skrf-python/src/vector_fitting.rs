//! Python bindings for VectorFitting class

use num_complex::Complex64;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use skrf_core::vector_fitting::{InitPoleSpacing, VectorFitting};

use crate::network::PyNetwork;

/// Python wrapper for VectorFitting
#[pyclass(name = "VectorFitting")]
#[derive(Default)]
pub struct PyVectorFitting {
    inner: VectorFitting,
}

#[pymethods]
impl PyVectorFitting {
    /// Create a new VectorFitting instance
    #[new]
    pub fn new() -> Self {
        Self {
            inner: VectorFitting::new(),
        }
    }

    /// Perform vector fitting on a network
    ///
    /// Args:
    ///     network: The network to fit
    ///     n_poles_real: Number of initial real poles (default: 2)
    ///     n_poles_cmplx: Number of initial complex conjugate poles (default: 2)
    ///     init_pole_spacing: Type of initial pole spacing ('linear' or 'log', default: 'linear')
    ///     fit_constant: Include a constant term in the fit (default: True)
    ///     fit_proportional: Include a proportional term in the fit (default: False)
    #[pyo3(signature = (network, n_poles_real=2, n_poles_cmplx=2, init_pole_spacing="linear", fit_constant=true, fit_proportional=false))]
    pub fn vector_fit(
        &mut self,
        network: &PyNetwork,
        n_poles_real: usize,
        n_poles_cmplx: usize,
        init_pole_spacing: &str,
        fit_constant: bool,
        fit_proportional: bool,
    ) -> PyResult<()> {
        let spacing = match init_pole_spacing.to_lowercase().as_str() {
            "linear" | "lin" => InitPoleSpacing::Linear,
            "log" | "logarithmic" => InitPoleSpacing::Logarithmic,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid pole spacing: {}. Use 'linear' or 'log'",
                    init_pole_spacing
                )))
            }
        };

        self.inner
            .vector_fit(
                network.inner(),
                n_poles_real,
                n_poles_cmplx,
                spacing,
                fit_constant,
                fit_proportional,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Get the RMS error between the fitted model and original network
    ///
    /// Args:
    ///     network: The network that was fitted
    ///     i: Row index of the response (0-based)
    ///     j: Column index of the response (0-based)
    ///
    /// Returns:
    ///     RMS error value
    pub fn get_rms_error(&self, network: &PyNetwork, i: usize, j: usize) -> PyResult<f64> {
        self.inner
            .get_rms_error(network.inner(), i, j)
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("RMS error calculation failed")
            })
    }

    /// Calculate the model order
    ///
    /// Model order = N_real + 2 * N_complex
    ///
    /// Returns:
    ///     Model order as integer
    pub fn get_model_order(&self) -> PyResult<usize> {
        self.inner
            .get_model_order()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted yet"))
    }

    /// Get the model response at specified frequencies
    ///
    /// Args:
    ///     i: Row index of the response (0-based)
    ///     j: Column index of the response (0-based)
    ///     freqs: Frequencies at which to evaluate (Hz)
    ///
    /// Returns:
    ///     Complex frequency response as numpy array
    pub fn get_model_response<'py>(
        &self,
        py: Python<'py>,
        i: usize,
        j: usize,
        freqs: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        self.inner
            .get_model_response(i, j, &freqs)
            .map(|arr| arr.to_pyarray(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Model response calculation failed")
            })
    }

    /// Get fitted poles as numpy array
    #[getter]
    pub fn poles<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        self.inner
            .poles
            .as_ref()
            .map(|arr| arr.to_pyarray(py))
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted yet"))
    }

    /// Get fitted residues as numpy array [n_responses, n_poles]
    #[getter]
    pub fn residues<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<Complex64>>> {
        self.inner
            .residues
            .as_ref()
            .map(|arr| arr.to_pyarray(py))
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted yet"))
    }

    /// Get constant coefficients as numpy array
    #[getter]
    pub fn constant_coeff<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.inner
            .constant_coeff
            .as_ref()
            .map(|arr| arr.to_pyarray(py))
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted yet"))
    }

    /// Get proportional coefficients as numpy array
    #[getter]
    pub fn proportional_coeff<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.inner
            .proportional_coeff
            .as_ref()
            .map(|arr| arr.to_pyarray(py))
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted yet"))
    }

    /// Maximum iterations for pole relocation
    #[getter]
    pub fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    #[setter]
    pub fn set_max_iterations(&mut self, value: usize) {
        self.inner.max_iterations = value;
    }

    /// Convergence tolerance
    #[getter]
    pub fn max_tol(&self) -> f64 {
        self.inner.max_tol
    }

    #[setter]
    pub fn set_max_tol(&mut self, value: f64) {
        self.inner.max_tol = value;
    }

    /// Wall-clock time of last fit (in seconds)
    #[getter]
    pub fn wall_clock_time(&self) -> f64 {
        self.inner.wall_clock_time
    }

    fn __repr__(&self) -> String {
        match self.inner.get_model_order() {
            Some(order) => format!("VectorFitting(order={})", order),
            None => "VectorFitting(not fitted)".to_string(),
        }
    }

    /// Write SPICE subcircuit netlist to a file
    ///
    /// Creates an equivalent N-port subcircuit based on its vector fitted S-parameter
    /// responses in SPICE simulator netlist syntax (compatible with LTspice, ngspice, Xyce).
    ///
    /// Args:
    ///     file: Path to the output file (usually .sp extension)
    ///     network: The network that was fitted
    ///     fitted_model_name: Name of the subcircuit (default: "s_equivalent")
    ///     create_reference_pins: If True, create separate reference pins for each port
    #[pyo3(signature = (file, network, fitted_model_name="s_equivalent", create_reference_pins=false))]
    pub fn write_spice_subcircuit_s(
        &self,
        file: &str,
        network: &PyNetwork,
        fitted_model_name: &str,
        create_reference_pins: bool,
    ) -> PyResult<()> {
        let path = std::path::Path::new(file);
        self.inner
            .write_spice_subcircuit_s(
                path,
                network.inner(),
                Some(fitted_model_name),
                create_reference_pins,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Generate SPICE subcircuit netlist as a string
    ///
    /// Args:
    ///     network: The network that was fitted
    ///     fitted_model_name: Name of the subcircuit (default: "s_equivalent")
    ///     create_reference_pins: If True, create separate reference pins for each port
    ///
    /// Returns:
    ///     SPICE netlist as a string
    #[pyo3(signature = (network, fitted_model_name="s_equivalent", create_reference_pins=false))]
    pub fn generate_spice_subcircuit_s(
        &self,
        network: &PyNetwork,
        fitted_model_name: &str,
        create_reference_pins: bool,
    ) -> PyResult<String> {
        self.inner
            .generate_spice_subcircuit_s(
                network.inner(),
                Some(fitted_model_name),
                create_reference_pins,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Perform passivity test on the fitted model
    ///
    /// Evaluates the passivity of the vector fitted model using the half-size test matrix method.
    /// Returns frequency bands where passivity violations occur.
    ///
    /// Args:
    ///     network: The network that was fitted
    ///
    /// Returns:
    ///     List of [f_start, f_stop] frequency bands with passivity violations
    pub fn passivity_test(&self, network: &PyNetwork) -> PyResult<Vec<[f64; 2]>> {
        let result = self
            .inner
            .passivity_test(network.inner().nports())
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(result.violation_bands)
    }

    /// Check if the fitted model is passive
    ///
    /// Args:
    ///     network: The network that was fitted
    ///
    /// Returns:
    ///     True if model is passive, False otherwise
    pub fn is_passive(&self, network: &PyNetwork) -> PyResult<bool> {
        self.inner
            .is_passive(network.inner().nports())
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Enforce passivity of the fitted model
    ///
    /// Uses iterative singular value perturbation with Weighted Least Squares
    /// to enforce passivity. This modifies the internal residues to make the model passive.
    ///
    /// Args:
    ///     network: The network that was fitted
    ///
    /// Returns:
    ///     Tuple of (success: bool, iterations: int, history_max_sigma: list)
    pub fn passivity_enforce(&mut self, network: &PyNetwork) -> PyResult<(bool, usize, Vec<f64>)> {
        let result = self
            .inner
            .passivity_enforce(network.inner().nports())
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok((result.success, result.iterations, result.history_max_sigma))
    }
}
