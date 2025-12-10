//! Python bindings for skrf-core
//!
//! This module exposes skrf-core functionality to Python via PyO3.
//!
//! ## Classes
//!
//! - `Frequency` - Frequency band representation
//! - `Network` - N-port RF network with S/Z/Y parameters
//! - `VectorFitting` - Vector Fitting algorithm for rational approximation

use pyo3::prelude::*;

mod frequency;
mod network;
mod vector_fitting;

pub use frequency::PyFrequency;
pub use network::PyNetwork;
pub use vector_fitting::PyVectorFitting;

/// skrf_python - Python bindings for scikit-rf in Rust
///
/// This module provides high-performance RF/microwave network analysis
/// capabilities implemented in Rust.
///
/// Example:
///     >>> import skrf_python as skrf
///     >>> nw = skrf.Network.from_touchstone("device.s2p")
///     >>> print(nw.nports, nw.nfreq)
///     >>> s11_db = nw.s_db[:, 0, 0]
#[pymodule]
fn skrf_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes with clean names (no Py prefix in Python)
    m.add_class::<PyFrequency>()?;
    m.add_class::<PyNetwork>()?;
    m.add_class::<PyVectorFitting>()?;

    // Add module version
    m.add("__version__", "0.1.0")?;

    Ok(())
}
