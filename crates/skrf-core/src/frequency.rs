//! Frequency module - represents a frequency band
//!
//! Provides a convenient way to work with frequency vectors with units.

/// Frequency unit enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrequencyUnit {
    Hz,
    #[default]
    KHz,
    MHz,
    GHz,
    THz,
}

impl FrequencyUnit {
    /// Get the multiplier to convert to Hz
    pub fn multiplier(&self) -> f64 {
        match self {
            FrequencyUnit::Hz => 1.0,
            FrequencyUnit::KHz => 1e3,
            FrequencyUnit::MHz => 1e6,
            FrequencyUnit::GHz => 1e9,
            FrequencyUnit::THz => 1e12,
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "hz" => Some(FrequencyUnit::Hz),
            "khz" => Some(FrequencyUnit::KHz),
            "mhz" => Some(FrequencyUnit::MHz),
            "ghz" => Some(FrequencyUnit::GHz),
            "thz" => Some(FrequencyUnit::THz),
            _ => None,
        }
    }
}

/// Sweep type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SweepType {
    #[default]
    Linear,
    Log,
}

/// A frequency band representation
#[derive(Debug, Clone)]
pub struct Frequency {
    /// Frequency vector in Hz
    f: Vec<f64>,
    /// Display unit
    unit: FrequencyUnit,
    /// Sweep type (linear or log)
    sweep_type: SweepType,
}

impl Frequency {
    /// Create a new Frequency with start/stop/npoints
    ///
    /// # Arguments
    /// * `start` - Start frequency in the specified unit
    /// * `stop` - Stop frequency in the specified unit
    /// * `npoints` - Number of frequency points
    /// * `unit` - Frequency unit
    /// * `sweep_type` - Linear or logarithmic sweep
    ///
    /// # Example
    /// ```
    /// use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
    /// let freq = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Linear);
    /// ```
    pub fn new(
        start: f64,
        stop: f64,
        npoints: usize,
        unit: FrequencyUnit,
        sweep_type: SweepType,
    ) -> Self {
        let mult = unit.multiplier();
        let start_hz = start * mult;
        let stop_hz = stop * mult;

        let f = match sweep_type {
            SweepType::Linear => {
                if npoints == 1 {
                    vec![start_hz]
                } else {
                    let step = (stop_hz - start_hz) / (npoints - 1) as f64;
                    (0..npoints).map(|i| start_hz + i as f64 * step).collect()
                }
            }
            SweepType::Log => {
                if npoints == 1 {
                    vec![start_hz]
                } else {
                    let log_start = start_hz.ln();
                    let log_stop = stop_hz.ln();
                    let log_step = (log_stop - log_start) / (npoints - 1) as f64;
                    (0..npoints)
                        .map(|i| (log_start + i as f64 * log_step).exp())
                        .collect()
                }
            }
        };

        Self {
            f,
            unit,
            sweep_type,
        }
    }

    /// Create from a frequency vector
    pub fn from_f(f: Vec<f64>, unit: FrequencyUnit) -> Self {
        let mult = unit.multiplier();
        let f_hz: Vec<f64> = f.iter().map(|&x| x * mult).collect();
        Self {
            f: f_hz,
            unit,
            sweep_type: SweepType::Linear, // default, actual sweep type unknown
        }
    }

    /// Get frequency vector in Hz
    #[inline]
    pub fn f(&self) -> &[f64] {
        &self.f
    }

    /// Get frequency vector in the current unit
    pub fn f_scaled(&self) -> Vec<f64> {
        let mult = self.unit.multiplier();
        self.f.iter().map(|&x| x / mult).collect()
    }

    /// Get the number of frequency points
    #[inline]
    pub fn npoints(&self) -> usize {
        self.f.len()
    }

    /// Get the start frequency in Hz
    #[inline]
    pub fn start(&self) -> f64 {
        *self.f.first().unwrap_or(&0.0)
    }

    /// Get the stop frequency in Hz
    #[inline]
    pub fn stop(&self) -> f64 {
        *self.f.last().unwrap_or(&0.0)
    }

    /// Get the center frequency in Hz
    pub fn center(&self) -> f64 {
        (self.start() + self.stop()) / 2.0
    }

    /// Get the current unit
    #[inline]
    pub fn unit(&self) -> FrequencyUnit {
        self.unit
    }

    /// Get the sweep type
    #[inline]
    pub fn sweep_type(&self) -> SweepType {
        self.sweep_type
    }

    /// Get the frequency span in Hz
    #[inline]
    pub fn span(&self) -> f64 {
        self.stop() - self.start()
    }

    /// Get the multiplier for the current unit
    pub fn multiplier(&self) -> f64 {
        self.unit.multiplier()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_create_linear_sweep() {
        let freq = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Linear);

        // Check that frequency is in Hz internally
        assert_eq!(freq.npoints(), 10);
        assert_relative_eq!(freq.start(), 1e9, epsilon = 1.0);
        assert_relative_eq!(freq.stop(), 10e9, epsilon = 1.0);

        // Check scaled values
        let f_scaled = freq.f_scaled();
        assert_relative_eq!(f_scaled[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(f_scaled[9], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_create_log_sweep() {
        let freq = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Log);

        // Check endpoints
        assert_relative_eq!(freq.start(), 1e9, epsilon = 1.0);
        assert_relative_eq!(freq.stop(), 10e9, epsilon = 1.0);

        // Check that ratio between adjacent points is constant
        let f = freq.f();
        let ratios: Vec<f64> = f.windows(2).map(|w| w[1] / w[0]).collect();
        for i in 1..ratios.len() {
            assert_relative_eq!(ratios[i], ratios[0], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_from_f() {
        let f = vec![1.0, 5.0, 200.0];
        let freq = Frequency::from_f(f, FrequencyUnit::KHz);

        assert_eq!(freq.npoints(), 3);
        assert_relative_eq!(freq.f()[0], 1e3, epsilon = 1e-10);
        assert_relative_eq!(freq.f()[1], 5e3, epsilon = 1e-10);
        assert_relative_eq!(freq.f()[2], 200e3, epsilon = 1e-10);
    }

    #[test]
    fn test_frequency_unit_multiplier() {
        assert_eq!(FrequencyUnit::Hz.multiplier(), 1.0);
        assert_eq!(FrequencyUnit::KHz.multiplier(), 1e3);
        assert_eq!(FrequencyUnit::MHz.multiplier(), 1e6);
        assert_eq!(FrequencyUnit::GHz.multiplier(), 1e9);
        assert_eq!(FrequencyUnit::THz.multiplier(), 1e12);
    }

    #[test]
    fn test_frequency_unit_from_str() {
        assert_eq!(FrequencyUnit::from_str("ghz"), Some(FrequencyUnit::GHz));
        assert_eq!(FrequencyUnit::from_str("GHZ"), Some(FrequencyUnit::GHz));
        assert_eq!(FrequencyUnit::from_str("MHz"), Some(FrequencyUnit::MHz));
        assert_eq!(FrequencyUnit::from_str("invalid"), None);
    }
}
