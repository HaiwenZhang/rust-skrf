//! Python bindings for skrf-core
//!
//! This module exposes skrf-core functionality to Python via PyO3.

use pyo3::prelude::*;

#[pyfunction]
fn hello() -> String {
    "Hello from skrf-python!".to_string()
}

#[pymodule]
fn skrf_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
