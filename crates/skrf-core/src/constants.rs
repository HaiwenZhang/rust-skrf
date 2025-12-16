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

/// Tolerance for SVD solve in least squares problems.
/// Used in Vector Fitting and other numerical algorithms.
pub const SVD_TOLERANCE: f64 = 1e-14;

/// Tolerance for pole/residue calculations in Vector Fitting.
/// Used to detect near-zero residues and avoid trivial solutions.
pub const POLE_RESIDUE_TOL: f64 = 1e-8;

/// Tolerance for detecting DC (zero frequency).
/// Used in extrapolation and time-domain analysis.
pub const DC_FREQ_TOL: f64 = 1e-10;

/// Tolerance for column scaling in numerical algorithms.
/// Used to avoid division by zero during matrix normalization.
pub const COLUMN_SCALE_TOL: f64 = 1e-15;
