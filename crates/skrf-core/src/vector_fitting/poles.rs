//! Pole set abstraction for Vector Fitting
//!
//! Provides a categorized view of poles (real vs complex) to eliminate
//! repeated type checking throughout the codebase.

use ndarray::Array1;
use num_complex::Complex64;

/// Tolerance for determining if a pole is real (imaginary part ~ 0)
const REAL_POLE_TOLERANCE: f64 = 1e-12;

/// A categorized set of poles
///
/// Separates poles into real and complex conjugate pairs for efficient
/// processing. Complex poles are stored with positive imaginary parts only
/// (the conjugate is implicit).
#[derive(Debug, Clone)]
pub struct PoleSet {
    /// Real poles (stored as real values)
    real_poles: Vec<f64>,
    /// Complex poles (stored with positive imaginary part)
    complex_poles: Vec<Complex64>,
}

impl PoleSet {
    /// Create a PoleSet from an array of complex poles
    ///
    /// Poles with |imag| < tolerance are treated as real.
    /// Complex poles are expected to have positive imaginary parts.
    pub fn from_array(poles: &Array1<Complex64>) -> Self {
        let mut real_poles = Vec::new();
        let mut complex_poles = Vec::new();

        for pole in poles.iter() {
            if pole.im.abs() < REAL_POLE_TOLERANCE {
                real_poles.push(pole.re);
            } else {
                // Store only positive imaginary (conjugate is implicit)
                if pole.im > 0.0 {
                    complex_poles.push(*pole);
                }
                // Skip negative imaginary parts (they're conjugates)
            }
        }

        Self {
            real_poles,
            complex_poles,
        }
    }

    /// Create a PoleSet from separate real and complex pole vectors
    pub fn from_parts(real_poles: Vec<f64>, complex_poles: Vec<Complex64>) -> Self {
        Self {
            real_poles,
            complex_poles,
        }
    }

    /// Number of real poles
    #[inline]
    pub fn n_real(&self) -> usize {
        self.real_poles.len()
    }

    /// Number of complex pole pairs
    #[inline]
    pub fn n_complex(&self) -> usize {
        self.complex_poles.len()
    }

    /// Total number of poles (real + complex)
    #[inline]
    pub fn len(&self) -> usize {
        self.real_poles.len() + self.complex_poles.len()
    }

    /// Check if pole set is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.real_poles.is_empty() && self.complex_poles.is_empty()
    }

    /// Model order = n_real + 2 * n_complex
    ///
    /// This is the dimension of the state-space representation.
    #[inline]
    pub fn model_order(&self) -> usize {
        self.real_poles.len() + 2 * self.complex_poles.len()
    }

    /// Iterate over real poles
    #[inline]
    pub fn iter_real(&self) -> impl Iterator<Item = f64> + '_ {
        self.real_poles.iter().copied()
    }

    /// Iterate over complex poles (positive imaginary only)
    #[inline]
    pub fn iter_complex(&self) -> impl Iterator<Item = Complex64> + '_ {
        self.complex_poles.iter().copied()
    }

    /// Iterate over all poles as Complex64
    ///
    /// Real poles are converted to Complex64 with zero imaginary part.
    pub fn iter_all(&self) -> impl Iterator<Item = Complex64> + '_ {
        self.real_poles
            .iter()
            .map(|&r| Complex64::new(r, 0.0))
            .chain(self.complex_poles.iter().copied())
    }

    /// Convert back to `Array1<Complex64>`
    ///
    /// Returns poles in the original format (real + complex with positive imag).
    pub fn to_array(&self) -> Array1<Complex64> {
        let mut poles = Vec::with_capacity(self.len());
        for &r in &self.real_poles {
            poles.push(Complex64::new(r, 0.0));
        }
        for &c in &self.complex_poles {
            poles.push(c);
        }
        Array1::from_vec(poles)
    }

    /// Get real poles as slice
    #[inline]
    pub fn real_poles(&self) -> &[f64] {
        &self.real_poles
    }

    /// Get complex poles as slice
    #[inline]
    pub fn complex_poles(&self) -> &[Complex64] {
        &self.complex_poles
    }
}

/// Compute model order from a pole array directly
///
/// Convenience function for cases where a full PoleSet is not needed.
#[inline]
pub fn model_order_from_poles(poles: &Array1<Complex64>) -> usize {
    let mut n_real = 0;
    let mut n_complex = 0;

    for pole in poles.iter() {
        if pole.im.abs() < REAL_POLE_TOLERANCE {
            n_real += 1;
        } else if pole.im > 0.0 {
            n_complex += 1;
        }
    }

    n_real + 2 * n_complex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pole_set_from_array() {
        let poles = Array1::from_vec(vec![
            Complex64::new(-1e9, 0.0),   // real
            Complex64::new(-2e9, 0.0),   // real
            Complex64::new(-0.5e9, 1e9), // complex
            Complex64::new(-0.3e9, 2e9), // complex
        ]);

        let ps = PoleSet::from_array(&poles);

        assert_eq!(ps.n_real(), 2);
        assert_eq!(ps.n_complex(), 2);
        assert_eq!(ps.len(), 4);
        assert_eq!(ps.model_order(), 6); // 2 + 2*2 = 6
    }

    #[test]
    fn test_pole_set_real_only() {
        let poles = Array1::from_vec(vec![
            Complex64::new(-1.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-3.0, 0.0),
        ]);

        let ps = PoleSet::from_array(&poles);

        assert_eq!(ps.n_real(), 3);
        assert_eq!(ps.n_complex(), 0);
        assert_eq!(ps.model_order(), 3);
    }

    #[test]
    fn test_pole_set_complex_only() {
        let poles = Array1::from_vec(vec![Complex64::new(-0.1, 1.0), Complex64::new(-0.2, 2.0)]);

        let ps = PoleSet::from_array(&poles);

        assert_eq!(ps.n_real(), 0);
        assert_eq!(ps.n_complex(), 2);
        assert_eq!(ps.model_order(), 4); // 2*2 = 4
    }

    #[test]
    fn test_pole_set_to_array() {
        let poles = Array1::from_vec(vec![Complex64::new(-1.0, 0.0), Complex64::new(-0.5, 1.0)]);

        let ps = PoleSet::from_array(&poles);
        let recovered = ps.to_array();

        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].re, -1.0);
        assert_eq!(recovered[0].im, 0.0);
        assert_eq!(recovered[1].re, -0.5);
        assert_eq!(recovered[1].im, 1.0);
    }

    #[test]
    fn test_model_order_from_poles() {
        let poles = Array1::from_vec(vec![
            Complex64::new(-1.0, 0.0), // real: +1
            Complex64::new(-0.5, 1.0), // complex: +2
            Complex64::new(-0.3, 2.0), // complex: +2
        ]);

        assert_eq!(model_order_from_poles(&poles), 5); // 1 + 2 + 2 = 5
    }
}
