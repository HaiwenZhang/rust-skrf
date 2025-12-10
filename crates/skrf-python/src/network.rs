//! Python bindings for Network class

use num_complex::Complex64;
use numpy::{PyArray1, PyArray3, ToPyArray};
use pyo3::prelude::*;
use skrf_core::network::Network;

use crate::frequency::PyFrequency;

/// Python wrapper for Network
#[pyclass(name = "Network")]
pub struct PyNetwork {
    inner: Network,
}

#[pymethods]
impl PyNetwork {
    /// Load a Network from a Touchstone file
    ///
    /// Args:
    ///     path: Path to the Touchstone file (.s1p, .s2p, etc.)
    ///
    /// Returns:
    ///     Network object
    #[staticmethod]
    pub fn from_touchstone(path: &str) -> PyResult<Self> {
        Network::from_touchstone(path)
            .map(|n| Self { inner: n })
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to load Touchstone: {}", e))
            })
    }

    /// Get S-parameters as complex numpy array [nfreq, nports, nports]
    #[getter]
    pub fn s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<Complex64>> {
        self.inner.s.to_pyarray(py)
    }

    /// Get S-parameters magnitude in dB as numpy array [nfreq, nports, nports]
    #[getter]
    pub fn s_db<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.s_db().to_pyarray(py)
    }

    /// Get S-parameters magnitude (linear) as numpy array [nfreq, nports, nports]
    #[getter]
    pub fn s_mag<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.s_mag().to_pyarray(py)
    }

    /// Get S-parameters phase in degrees as numpy array [nfreq, nports, nports]
    #[getter]
    pub fn s_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.s_deg().to_pyarray(py)
    }

    /// Get S-parameters phase in radians as numpy array [nfreq, nports, nports]
    #[getter]
    pub fn s_rad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.s_rad().to_pyarray(py)
    }

    /// Get S-parameters real part as numpy array [nfreq, nports, nports]
    #[getter]
    pub fn s_re<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.s_re().to_pyarray(py)
    }

    /// Get S-parameters imaginary part as numpy array [nfreq, nports, nports]
    #[getter]
    pub fn s_im<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.s_im().to_pyarray(py)
    }

    /// Get Z-parameters as complex numpy array [nfreq, nports, nports]
    #[getter]
    pub fn z<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<Complex64>> {
        self.inner.z().to_pyarray(py)
    }

    /// Get Y-parameters as complex numpy array [nfreq, nports, nports]
    #[getter]
    pub fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<Complex64>> {
        self.inner.y().to_pyarray(py)
    }

    /// Get reference impedance as numpy array [nports]
    #[getter]
    pub fn z0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.inner.z0().to_pyarray(py)
    }

    /// Get frequency array in Hz as numpy array
    #[getter]
    pub fn f<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.f().to_pyarray(py)
    }

    /// Get the Frequency object
    #[getter]
    pub fn frequency(&self) -> PyFrequency {
        PyFrequency::from_frequency(self.inner.frequency.clone())
    }

    /// Number of ports
    #[getter]
    pub fn nports(&self) -> usize {
        self.inner.nports()
    }

    /// Number of frequency points
    #[getter]
    pub fn nfreq(&self) -> usize {
        self.inner.nfreq()
    }

    /// Network name
    #[getter]
    pub fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// Get VSWR as numpy array [nfreq, nports]
    #[getter]
    pub fn vswr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.inner.vswr().to_pyarray(py)
    }

    /// Check if network is reciprocal
    ///
    /// Args:
    ///     tol: Tolerance for reciprocity check (default: 1e-6)
    #[pyo3(signature = (tol=None))]
    pub fn is_reciprocal(&self, tol: Option<f64>) -> bool {
        self.inner.is_reciprocal(tol)
    }

    /// Check if network is passive
    ///
    /// Args:
    ///     tol: Tolerance for passivity check (default: 1e-6)
    #[pyo3(signature = (tol=None))]
    pub fn is_passive(&self, tol: Option<f64>) -> bool {
        self.inner.is_passive(tol)
    }

    /// Check if network is lossless
    ///
    /// Args:
    ///     tol: Tolerance for lossless check (default: 1e-6)
    #[pyo3(signature = (tol=None))]
    pub fn is_lossless(&self, tol: Option<f64>) -> bool {
        self.inner.is_lossless(tol)
    }

    /// Check if network is symmetric
    ///
    /// Args:
    ///     tol: Tolerance for symmetry check (default: 1e-6)
    #[pyo3(signature = (tol=None))]
    pub fn is_symmetric(&self, tol: Option<f64>) -> bool {
        self.inner.is_symmetric(tol)
    }

    fn __repr__(&self) -> String {
        let name = self.name().unwrap_or_else(|| "unnamed".to_string());
        format!(
            "Network('{}', {} ports, {} freqs, {:.3} GHz - {:.3} GHz)",
            name,
            self.nports(),
            self.nfreq(),
            self.inner.frequency.start() / 1e9,
            self.inner.frequency.stop() / 1e9
        )
    }
}

impl PyNetwork {
    /// Get the inner Network reference (for internal use)
    pub fn inner(&self) -> &Network {
        &self.inner
    }

    /// Create from existing Network
    pub fn from_network(network: Network) -> Self {
        Self { inner: network }
    }
}
