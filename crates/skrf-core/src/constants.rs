//! Numerical constants for RF calculations
//!
//! Provides standardized tolerance values and other numerical constants
//! used throughout the library.

/// Tolerance for detecting near-zero values in division and singularity checks.
/// Used to prevent division by zero and detect ill-conditioned matrices.
pub const NEAR_ZERO: f64 = 1e-15;

/// Default tolerance for property checks (passivity, reciprocity, etc).
/// This is the tolerance used when None is passed to property check functions.
pub const PROPERTY_TOL: f64 = 1e-12;
