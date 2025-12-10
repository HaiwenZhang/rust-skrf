//! Python bindings for Frequency class

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};

/// Python wrapper for Frequency
#[pyclass(name = "Frequency")]
#[derive(Clone)]
pub struct PyFrequency {
    inner: Frequency,
}

#[pymethods]
impl PyFrequency {
    /// Create a new Frequency object
    ///
    /// Args:
    ///     start: Start frequency in Hz
    ///     stop: Stop frequency in Hz  
    ///     npoints: Number of frequency points
    ///     unit: Frequency unit ('Hz', 'kHz', 'MHz', 'GHz', 'THz')
    ///     sweep_type: Sweep type ('linear' or 'log')
    #[new]
    #[pyo3(signature = (start, stop, npoints, unit="Hz", sweep_type="linear"))]
    pub fn new(
        start: f64,
        stop: f64,
        npoints: usize,
        unit: &str,
        sweep_type: &str,
    ) -> PyResult<Self> {
        let freq_unit = match unit.to_lowercase().as_str() {
            "hz" => FrequencyUnit::Hz,
            "khz" => FrequencyUnit::KHz,
            "mhz" => FrequencyUnit::MHz,
            "ghz" => FrequencyUnit::GHz,
            "thz" => FrequencyUnit::THz,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid frequency unit: {}. Use 'Hz', 'kHz', 'MHz', 'GHz', or 'THz'",
                    unit
                )))
            }
        };

        let sweep = match sweep_type.to_lowercase().as_str() {
            "linear" | "lin" => SweepType::Linear,
            "log" | "logarithmic" => SweepType::Log,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid sweep type: {}. Use 'linear' or 'log'",
                    sweep_type
                )))
            }
        };

        Ok(Self {
            inner: Frequency::new(start, stop, npoints, freq_unit, sweep),
        })
    }

    /// Get frequency array in Hz as numpy array
    #[getter]
    pub fn f<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.f().to_pyarray(py)
    }

    /// Get scaled frequency array (in the specified unit) as numpy array
    #[getter]
    pub fn f_scaled<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.f_scaled().to_pyarray(py)
    }

    /// Start frequency in Hz
    #[getter]
    pub fn start(&self) -> f64 {
        self.inner.start()
    }

    /// Stop frequency in Hz
    #[getter]
    pub fn stop(&self) -> f64 {
        self.inner.stop()
    }

    /// Number of frequency points
    #[getter]
    pub fn npoints(&self) -> usize {
        self.inner.npoints()
    }

    /// Frequency unit as string
    #[getter]
    pub fn unit(&self) -> &str {
        match self.inner.unit() {
            FrequencyUnit::Hz => "Hz",
            FrequencyUnit::KHz => "kHz",
            FrequencyUnit::MHz => "MHz",
            FrequencyUnit::GHz => "GHz",
            FrequencyUnit::THz => "THz",
        }
    }

    /// Sweep type as string
    #[getter]
    pub fn sweep_type(&self) -> &str {
        match self.inner.sweep_type() {
            SweepType::Linear => "linear",
            SweepType::Log => "log",
        }
    }

    /// Center frequency in Hz
    #[getter]
    pub fn center(&self) -> f64 {
        self.inner.center()
    }

    /// Frequency span in Hz
    #[getter]
    pub fn span(&self) -> f64 {
        self.inner.span()
    }

    fn __repr__(&self) -> String {
        format!(
            "Frequency({} {} - {} {}, {} points, {})",
            self.inner.start() / self.inner.multiplier(),
            self.unit(),
            self.inner.stop() / self.inner.multiplier(),
            self.unit(),
            self.npoints(),
            self.sweep_type()
        )
    }

    fn __len__(&self) -> usize {
        self.npoints()
    }
}

impl PyFrequency {
    /// Get the inner Frequency reference (for internal use)
    pub fn inner(&self) -> &Frequency {
        &self.inner
    }

    /// Create from existing Frequency
    pub fn from_frequency(freq: Frequency) -> Self {
        Self { inner: freq }
    }
}
