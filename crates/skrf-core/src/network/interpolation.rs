//! Frequency interpolation and cropping
//!
//! Provides methods for interpolating network data to new frequency points
//! and cropping to frequency ranges.

use ndarray::Array3;
use num_complex::Complex64;

use super::core::Network;
use crate::frequency::{Frequency, FrequencyUnit, SweepType};

impl Network {
    /// Interpolate S-parameters to a new frequency vector
    ///
    /// Uses linear interpolation in the complex domain.
    /// Points outside the original frequency range are extrapolated.
    ///
    /// # Arguments
    /// * `new_freq` - New frequency points in Hz
    pub fn interpolate(&self, new_freq: &Frequency) -> Network {
        let old_f = self.frequency.f();
        let new_f = new_freq.f();
        let nports = self.nports();
        let new_nfreq = new_f.len();

        let mut s_new = Array3::<Complex64>::zeros((new_nfreq, nports, nports));

        // Linear interpolation for each S-parameter
        for i in 0..nports {
            for j in 0..nports {
                for (nfi, &nf) in new_f.iter().enumerate() {
                    s_new[[nfi, i, j]] = interpolate_complex(old_f, &self.s, i, j, nf);
                }
            }
        }

        Network::new(new_freq.clone(), s_new, self.z0.clone())
    }

    /// Interpolate to a specified number of linearly-spaced points
    pub fn interpolate_n(&self, npoints: usize) -> Network {
        let f = self.frequency.f();
        let f_start = f.first().copied().unwrap_or(0.0);
        let f_stop = f.last().copied().unwrap_or(1.0);

        // Convert to original unit
        let unit = self.frequency.unit();
        let mult = unit.multiplier();

        let new_freq = Frequency::new(
            f_start / mult,
            f_stop / mult,
            npoints,
            unit,
            SweepType::Linear,
        );

        self.interpolate(&new_freq)
    }

    /// Crop network to a frequency range (in Hz)
    ///
    /// Returns a new network containing only points within [f_start, f_stop].
    pub fn cropped(&self, f_start: f64, f_stop: f64) -> Network {
        let f = self.frequency.f();
        let nports = self.nports();

        // Find indices within range
        let indices: Vec<usize> = f
            .iter()
            .enumerate()
            .filter(|(_, &freq)| freq >= f_start && freq <= f_stop)
            .map(|(i, _)| i)
            .collect();

        if indices.is_empty() {
            // Return empty network with same structure
            return Network::new(
                Frequency::from_f(vec![], self.frequency.unit()),
                Array3::<Complex64>::zeros((0, nports, nports)),
                self.z0.clone(),
            );
        }

        let new_nfreq = indices.len();
        let mut s_new = Array3::<Complex64>::zeros((new_nfreq, nports, nports));
        let mut f_new = Vec::with_capacity(new_nfreq);

        for (new_i, &old_i) in indices.iter().enumerate() {
            f_new.push(f[old_i]);
            for i in 0..nports {
                for j in 0..nports {
                    s_new[[new_i, i, j]] = self.s[[old_i, i, j]];
                }
            }
        }

        Network::new(
            Frequency::from_f(f_new, self.frequency.unit()),
            s_new,
            self.z0.clone(),
        )
    }

    /// Crop network to a frequency range in specified units
    pub fn cropped_unit(&self, f_start: f64, f_stop: f64, unit: FrequencyUnit) -> Network {
        let mult = unit.multiplier();
        self.cropped(f_start * mult, f_stop * mult)
    }

    /// Resample network to specified number of points
    ///
    /// Alias for interpolate_n
    pub fn resample(&self, npoints: usize) -> Network {
        self.interpolate_n(npoints)
    }

    /// Extrapolate network data to DC (0 Hz)
    ///
    /// This is required before time-domain analysis for accurate results.
    /// Uses linear extrapolation from the first few frequency points.
    ///
    /// # Arguments
    /// * `dc_sparam` - Optional S-parameter value at DC. If None, extrapolates from data.
    pub fn extrapolate_to_dc(&self, dc_sparam: Option<Complex64>) -> Network {
        let f = self.frequency.f();
        let nfreq = self.nfreq();
        let nports = self.nports();

        // Check if already starts at DC
        if nfreq > 0 && f[0].abs() < 1e-10 {
            return self.clone();
        }

        // Create new frequency vector with DC point
        let mut f_new = vec![0.0];
        f_new.extend_from_slice(f);
        let new_nfreq = f_new.len();

        // Create new S-parameter array
        let mut s_new = Array3::<Complex64>::zeros((new_nfreq, nports, nports));

        // Copy existing data
        for fi in 0..nfreq {
            for i in 0..nports {
                for j in 0..nports {
                    s_new[[fi + 1, i, j]] = self.s[[fi, i, j]];
                }
            }
        }

        // Set DC values
        for i in 0..nports {
            for j in 0..nports {
                let dc_val = match dc_sparam {
                    Some(val) => val,
                    None => {
                        // Extrapolate from first two points
                        if nfreq >= 2 {
                            let s0 = self.s[[0, i, j]];
                            let s1 = self.s[[1, i, j]];
                            let df = f[1] - f[0];
                            if df.abs() > 1e-15 {
                                // Linear extrapolation back to DC
                                s0 - (s1 - s0) * Complex64::new(f[0] / df, 0.0)
                            } else {
                                s0
                            }
                        } else if nfreq == 1 {
                            self.s[[0, i, j]]
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    }
                };
                s_new[[0, i, j]] = dc_val;
            }
        }

        Network::new(
            Frequency::from_f(f_new, self.frequency.unit()),
            s_new,
            self.z0.clone(),
        )
    }
}

/// Linear interpolation for a single complex value
fn interpolate_complex(
    f_old: &[f64],
    s: &Array3<Complex64>,
    i: usize,
    j: usize,
    f_new: f64,
) -> Complex64 {
    let n = f_old.len();

    if n == 0 {
        return Complex64::new(0.0, 0.0);
    }

    if n == 1 {
        return s[[0, i, j]];
    }

    // Find bracketing indices
    if f_new <= f_old[0] {
        // Extrapolate below
        let slope = (s[[1, i, j]] - s[[0, i, j]]) / Complex64::new(f_old[1] - f_old[0], 0.0);
        return s[[0, i, j]] + slope * Complex64::new(f_new - f_old[0], 0.0);
    }

    if f_new >= f_old[n - 1] {
        // Extrapolate above
        let slope = (s[[n - 1, i, j]] - s[[n - 2, i, j]])
            / Complex64::new(f_old[n - 1] - f_old[n - 2], 0.0);
        return s[[n - 1, i, j]] + slope * Complex64::new(f_new - f_old[n - 1], 0.0);
    }

    // Binary search for interval (O(log n) instead of O(n))
    let idx = match f_old.partition_point(|&f| f < f_new) {
        0 => 0,
        i if i >= n => n - 2,
        i => i - 1,
    };

    // Linear interpolation
    let t = (f_new - f_old[idx]) / (f_old[idx + 1] - f_old[idx]);
    s[[idx, i, j]] * Complex64::new(1.0 - t, 0.0) + s[[idx + 1, i, j]] * Complex64::new(t, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_crop() {
        let freq = Frequency::new(1.0, 10.0, 10, FrequencyUnit::GHz, SweepType::Linear);
        let nports = 1;
        let mut s = Array3::<Complex64>::zeros((10, nports, nports));
        for f in 0..10 {
            s[[f, 0, 0]] = Complex64::new(f as f64 * 0.1, 0.0);
        }
        let z0 = Array1::from_elem(nports, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq, s, z0);

        // Crop to 3-7 GHz
        let cropped = ntwk.cropped_unit(3.0, 7.0, FrequencyUnit::GHz);

        // Should have fewer frequency points
        assert!(cropped.nfreq() < ntwk.nfreq());
        assert!(cropped.nfreq() > 0);
    }

    #[test]
    fn test_interpolate_identity() {
        let freq = Frequency::new(1.0, 5.0, 5, FrequencyUnit::GHz, SweepType::Linear);
        let mut s = Array3::<Complex64>::zeros((5, 1, 1));
        for f in 0..5 {
            s[[f, 0, 0]] = Complex64::new(f as f64 * 0.2, 0.0);
        }
        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));
        let ntwk = Network::new(freq.clone(), s, z0);

        // Interpolate to same frequency - should be identity
        let interp = ntwk.interpolate(&freq);

        assert_eq!(interp.nfreq(), 5);
        for f in 0..5 {
            assert_relative_eq!(
                interp.s[[f, 0, 0]].re,
                ntwk.s[[f, 0, 0]].re,
                epsilon = 1e-10
            );
        }
    }
}
