//! Vector Fitting algorithm for rational approximation of frequency responses
//!
//! This module provides a Rust implementation of the Vector Fitting algorithm
//! for approximating frequency domain responses with rational functions in
//! pole-residue form.
//!
//! # References
//!
//! - B. Gustavsen, A. Semlyen, "Rational Approximation of Frequency Domain Responses
//!   by Vector Fitting", IEEE Trans. Power Delivery, vol. 14, no. 3, 1999
//! - B. Gustavsen, "Improving the Pole Relocating Properties of Vector Fitting",
//!   IEEE Trans. Power Delivery, vol. 21, no. 3, 2006

mod algorithms;
mod core;
mod model;
pub mod passivity;
pub mod spice;

pub use self::core::VectorFitting;
pub use algorithms::{InitPoleSpacing, PoleRelocationResult};
pub use passivity::{PassivityEnforceResult, PassivityTestResult};
