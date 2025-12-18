//! Constants for Vector Fitting algorithms
//!
//! Centralizes magic numbers to improve code clarity and maintainability.

// ============================================================================
// Numerical tolerances
// ============================================================================

/// Default tolerance for convergence checking
pub const DEFAULT_TOLERANCE: f64 = 1e-6;

/// Tolerance for residue values (minimum d_res threshold)
pub const RESIDUE_TOLERANCE: f64 = 1e-8;

/// Tolerance for considering a pole as real (imaginary part threshold)
pub const REAL_POLE_TOLERANCE: f64 = 1e-12;

/// Tolerance for eigenvalue square root comparison
pub const EIGENVALUE_TOLERANCE: f64 = 1e-6;

/// Tolerance for detecting zero norm in column scaling
pub const NORM_TOLERANCE: f64 = 1e-15;

/// Tolerance for violation detection in passivity enforcement
pub const VIOLATION_TOLERANCE: f64 = 1e-12;

// ============================================================================
// Passivity enforcement parameters
// ============================================================================

/// Delta threshold for singular value perturbation in passivity enforcement
/// (maximum allowed singular value before perturbation)
pub const PASSIVITY_DELTA_THRESHOLD: f64 = 0.999;

/// Damping factor for C matrix perturbation updates
pub const PASSIVITY_DAMPING_FACTOR: f64 = 0.1;

/// Frequency margin for evaluation (evaluation extends beyond f_max by this factor)
pub const PASSIVITY_FREQ_MARGIN: f64 = 1.2;

// ============================================================================
// Pole initialization parameters
// ============================================================================

/// Damping ratio for complex pole initialization
/// (real part = -DAMPING_RATIO * omega)
pub const COMPLEX_POLE_DAMPING_RATIO: f64 = 0.01;

/// Minimum frequency fraction when f_min = 0
pub const MIN_FREQUENCY_FRACTION: f64 = 1e-6;
