//! Unit conversion functions
//!
//! Provides conversions between different representations of complex numbers
//! (magnitude, dB, phase, real/imaginary).

use num_complex::Complex64;
use std::f64::consts::PI;

/// Convert complex number to magnitude
pub fn complex_2_magnitude(z: Complex64) -> f64 {
    z.norm()
}

/// Convert complex number to dB (20*log10(|z|))
pub fn complex_2_db(z: Complex64) -> f64 {
    20.0 * z.norm().log10()
}

/// Convert complex number to dB10 (10*log10(|z|))  
pub fn complex_2_db10(z: Complex64) -> f64 {
    10.0 * z.norm().log10()
}

/// Convert complex number to phase in radians
pub fn complex_2_radian(z: Complex64) -> f64 {
    z.arg()
}

/// Convert complex number to phase in degrees
pub fn complex_2_degree(z: Complex64) -> f64 {
    z.arg() * 180.0 / PI
}

/// Convert complex number to (magnitude, arc_length)
pub fn complex_2_quadrature(z: Complex64) -> (f64, f64) {
    let mag = z.norm();
    let arc = z.arg() * mag;
    (mag, arc)
}

/// Convert complex number to (real, imaginary)
pub fn complex_2_reim(z: Complex64) -> (f64, f64) {
    (z.re, z.im)
}

/// Convert magnitude to dB (20*log10(mag))
pub fn magnitude_2_db(mag: f64) -> f64 {
    20.0 * mag.log10()
}

/// Alias for magnitude_2_db
pub fn mag_2_db(mag: f64) -> f64 {
    magnitude_2_db(mag)
}

/// Convert magnitude to dB10 (10*log10(mag))
pub fn mag_2_db10(mag: f64) -> f64 {
    10.0 * mag.log10()
}

/// Convert dB to magnitude (10^(dB/20))
pub fn db_2_magnitude(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Alias for db_2_magnitude
pub fn db_2_mag(db: f64) -> f64 {
    db_2_magnitude(db)
}

/// Convert dB10 to magnitude (10^(dB/10))
pub fn db10_2_mag(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

/// Convert (magnitude, degree) to complex
pub fn magdeg_2_reim(mag: f64, deg: f64) -> Complex64 {
    let rad = deg * PI / 180.0;
    Complex64::from_polar(mag, rad)
}

/// Convert (dB, degree) to complex
pub fn dbdeg_2_reim(db: f64, deg: f64) -> Complex64 {
    let mag = db_2_magnitude(db);
    magdeg_2_reim(mag, deg)
}

/// Convert radians to degrees
pub fn radian_2_degree(rad: f64) -> f64 {
    rad * 180.0 / PI
}

/// Convert degrees to radians
pub fn degree_2_radian(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Convert Nepers to dB
pub fn np_2_db(np: f64) -> f64 {
    np * 20.0 / 10.0_f64.ln()
}

/// Convert dB to Nepers
pub fn db_2_np(db: f64) -> f64 {
    db * 10.0_f64.ln() / 20.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_complex_2_magnitude() {
        // 5 = |3 + 4j|
        let z = Complex64::new(3.0, 4.0);
        assert_relative_eq!(complex_2_magnitude(z), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_2_db10() {
        // 10 dB = 10 * log10(|6 + 8j|) = 10 * log10(10)
        let z = Complex64::new(6.0, 8.0);
        assert_relative_eq!(complex_2_db10(z), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_2_degree() {
        // 90° = angle(0 + 1j)
        let z = Complex64::new(0.0, 1.0);
        assert_relative_eq!(complex_2_degree(z), 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_2_quadrature() {
        // (2, π) = (|2j|, angle(2j) * |2j|)
        let z = Complex64::new(0.0, 2.0);
        let (mag, arc) = complex_2_quadrature(z);
        assert_relative_eq!(mag, 2.0, epsilon = 1e-10);
        assert_relative_eq!(arc, PI, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_2_reim() {
        let z = Complex64::new(1.0, 2.0);
        assert_eq!(complex_2_reim(z), (1.0, 2.0));
    }

    #[test]
    fn test_magnitude_2_db() {
        assert_relative_eq!(magnitude_2_db(10.0), 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_db_2_magnitude() {
        assert_relative_eq!(db_2_magnitude(20.0), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magdeg_2_reim() {
        let z = magdeg_2_reim(1.0, 90.0);
        assert_relative_eq!(z.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(z.im, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dbdeg_2_reim() {
        let z = dbdeg_2_reim(20.0, 90.0);
        assert_relative_eq!(z.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(z.im, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_np_2_db() {
        // 1 Np = 20/ln(10) dB
        assert_relative_eq!(np_2_db(1.0), 20.0 / 10.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_db_2_np() {
        // 1 dB = ln(10)/20 Np
        assert_relative_eq!(db_2_np(1.0), 10.0_f64.ln() / 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_radian_2_degree() {
        assert_relative_eq!(radian_2_degree(PI), 180.0, epsilon = 1e-10);
    }
}
