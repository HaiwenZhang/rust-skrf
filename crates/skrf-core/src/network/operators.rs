//! Network operations
//!
//! Provides network operations like cascade, deembed, flip, renumber, etc.

use ndarray::{Array1, Array3};
use num_complex::Complex64;

use super::core::Network;
use crate::constants::NEAR_ZERO;
use crate::math::transforms::{s2t, t2s};

impl Network {
    /// Cascade with another network (self ** other)
    ///
    /// Only valid for 2-port networks. Connects port 2 of self to port 1 of other.
    pub fn cascade(&self, other: &Network) -> Option<Network> {
        if self.nports() != 2 || other.nports() != 2 {
            return None;
        }

        // Check frequency compatibility
        if self.nfreq() != other.nfreq() {
            return None;
        }

        let nfreq = self.nfreq();
        let mut s_result = Array3::<Complex64>::zeros((nfreq, 2, 2));

        for f in 0..nfreq {
            // Get S-parameters for both networks at this frequency
            let s_a = [
                [self.s[[f, 0, 0]], self.s[[f, 0, 1]]],
                [self.s[[f, 1, 0]], self.s[[f, 1, 1]]],
            ];
            let s_b = [
                [other.s[[f, 0, 0]], other.s[[f, 0, 1]]],
                [other.s[[f, 1, 0]], other.s[[f, 1, 1]]],
            ];

            // Cascade formula using signal flow graph
            let denom = Complex64::new(1.0, 0.0) - s_a[1][1] * s_b[0][0];

            s_result[[f, 0, 0]] = s_a[0][0] + (s_a[0][1] * s_a[1][0] * s_b[0][0]) / denom;
            s_result[[f, 0, 1]] = (s_a[0][1] * s_b[0][1]) / denom;
            s_result[[f, 1, 0]] = (s_a[1][0] * s_b[1][0]) / denom;
            s_result[[f, 1, 1]] = s_b[1][1] + (s_b[0][1] * s_b[1][0] * s_a[1][1]) / denom;
        }

        Some(Network::new(
            self.frequency.clone(),
            s_result,
            self.z0.clone(),
        ))
    }

    /// Flip the ports of a 2-port network (swap port 1 and port 2)
    ///
    /// Returns a new Network with ports swapped.
    pub fn flipped(&self) -> Option<Network> {
        if self.nports() != 2 {
            return None;
        }

        let nfreq = self.nfreq();
        let mut s_flipped = Array3::<Complex64>::zeros((nfreq, 2, 2));

        for f in 0..nfreq {
            // Swap indices: new[i,j] = old[1-i, 1-j]
            s_flipped[[f, 0, 0]] = self.s[[f, 1, 1]];
            s_flipped[[f, 0, 1]] = self.s[[f, 1, 0]];
            s_flipped[[f, 1, 0]] = self.s[[f, 0, 1]];
            s_flipped[[f, 1, 1]] = self.s[[f, 0, 0]];
        }

        // Also flip z0
        let z0_flipped = Array1::from_vec(vec![self.z0[1], self.z0[0]]);

        Some(Network::new(self.frequency.clone(), s_flipped, z0_flipped))
    }

    /// Get inverse S-parameters for de-embedding
    ///
    /// The inverse is defined such that cascade(self, inv(self)) = identity.
    /// Uses T-parameters: T_inv = T^-1, then convert back to S.
    pub fn inv(&self) -> Option<Network> {
        if self.nports() != 2 {
            return None;
        }

        // Convert to T-parameters
        let t = s2t(&self.s)?;
        let nfreq = self.nfreq();
        let mut t_inv = Array3::<Complex64>::zeros((nfreq, 2, 2));

        for f in 0..nfreq {
            // 2x2 matrix inversion
            let t11 = t[[f, 0, 0]];
            let t12 = t[[f, 0, 1]];
            let t21 = t[[f, 1, 0]];
            let t22 = t[[f, 1, 1]];

            let det = t11 * t22 - t12 * t21;
            if det.norm() < NEAR_ZERO {
                return None;
            }

            let inv_det = Complex64::new(1.0, 0.0) / det;
            t_inv[[f, 0, 0]] = t22 * inv_det;
            t_inv[[f, 0, 1]] = -t12 * inv_det;
            t_inv[[f, 1, 0]] = -t21 * inv_det;
            t_inv[[f, 1, 1]] = t11 * inv_det;
        }

        // Convert back to S-parameters
        let s_inv = t2s(&t_inv)?;

        Some(Network::new(self.frequency.clone(), s_inv, self.z0.clone()))
    }

    /// De-embed another network from this one
    ///
    /// Effectively: self // other = inv(other) ** self ** inv(other)
    /// For single-sided de-embedding: inv(other) ** self
    pub fn deembed(&self, other: &Network) -> Option<Network> {
        let other_inv = other.inv()?;
        other_inv.cascade(self)
    }

    /// Renumber ports according to a mapping
    ///
    /// `from_ports` and `to_ports` specify the port renumbering.
    /// For example, renumber(&[0,1], &[1,0]) swaps ports 0 and 1.
    pub fn renumbered(&self, from_ports: &[usize], to_ports: &[usize]) -> Option<Network> {
        let nports = self.nports();
        if from_ports.len() != to_ports.len() || from_ports.len() != nports {
            return None;
        }

        // Validate port indices
        for &p in from_ports.iter().chain(to_ports.iter()) {
            if p >= nports {
                return None;
            }
        }

        let nfreq = self.nfreq();
        let mut s_new = Array3::<Complex64>::zeros((nfreq, nports, nports));
        let mut z0_new = Array1::<Complex64>::zeros(nports);

        // Build reverse mapping: to_ports[i] -> from_ports[i]
        let mut mapping = vec![0usize; nports];
        for i in 0..nports {
            mapping[to_ports[i]] = from_ports[i];
        }

        for f in 0..nfreq {
            for i in 0..nports {
                for j in 0..nports {
                    s_new[[f, i, j]] = self.s[[f, mapping[i], mapping[j]]];
                }
            }
        }

        for i in 0..nports {
            z0_new[i] = self.z0[mapping[i]];
        }

        Some(Network::new(self.frequency.clone(), s_new, z0_new))
    }

    /// Extract a subnetwork with specified ports
    ///
    /// Creates a new network containing only the specified ports.
    /// Port indices are 0-based.
    pub fn subnetwork(&self, ports: &[usize]) -> Option<Network> {
        let nports = self.nports();
        let new_nports = ports.len();

        if new_nports == 0 {
            return None;
        }

        // Validate port indices
        for &p in ports {
            if p >= nports {
                return None;
            }
        }

        let nfreq = self.nfreq();
        let mut s_new = Array3::<Complex64>::zeros((nfreq, new_nports, new_nports));
        let mut z0_new = Array1::<Complex64>::zeros(new_nports);

        for (new_i, &old_i) in ports.iter().enumerate() {
            z0_new[new_i] = self.z0[old_i];
            for (new_j, &old_j) in ports.iter().enumerate() {
                for f in 0..nfreq {
                    s_new[[f, new_i, new_j]] = self.s[[f, old_i, old_j]];
                }
            }
        }

        Some(Network::new(self.frequency.clone(), s_new, z0_new))
    }
}

// Operator overloads for element-wise operations
use std::ops::{Add, Div, Mul, Sub};

impl Add for &Network {
    type Output = Option<Network>;

    /// Element-wise addition of S-parameters
    fn add(self, other: &Network) -> Option<Network> {
        if self.nfreq() != other.nfreq() || self.nports() != other.nports() {
            return None;
        }

        let s_new = &self.s + &other.s;
        Some(Network::new(self.frequency.clone(), s_new, self.z0.clone()))
    }
}

impl Sub for &Network {
    type Output = Option<Network>;

    /// Element-wise subtraction of S-parameters
    fn sub(self, other: &Network) -> Option<Network> {
        if self.nfreq() != other.nfreq() || self.nports() != other.nports() {
            return None;
        }

        let s_new = &self.s - &other.s;
        Some(Network::new(self.frequency.clone(), s_new, self.z0.clone()))
    }
}

impl Mul<Complex64> for &Network {
    type Output = Network;

    /// Multiply all S-parameters by a complex scalar
    fn mul(self, scalar: Complex64) -> Network {
        let s_new = self.s.mapv(|x| x * scalar);
        Network::new(self.frequency.clone(), s_new, self.z0.clone())
    }
}

impl Mul<f64> for &Network {
    type Output = Network;

    /// Multiply all S-parameters by a real scalar
    fn mul(self, scalar: f64) -> Network {
        let s_new = self.s.mapv(|x| x * scalar);
        Network::new(self.frequency.clone(), s_new, self.z0.clone())
    }
}

impl Div<Complex64> for &Network {
    type Output = Network;

    /// Divide all S-parameters by a complex scalar
    fn div(self, scalar: Complex64) -> Network {
        let s_new = self.s.mapv(|x| x / scalar);
        Network::new(self.frequency.clone(), s_new, self.z0.clone())
    }
}

impl Div<f64> for &Network {
    type Output = Network;

    /// Divide all S-parameters by a real scalar
    fn div(self, scalar: f64) -> Network {
        let s_new = self.s.mapv(|x| x / scalar);
        Network::new(self.frequency.clone(), s_new, self.z0.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};

    #[test]
    fn test_flip() {
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        let mut s = Array3::<Complex64>::zeros((1, 2, 2));
        s[[0, 0, 0]] = Complex64::new(0.1, 0.0); // S11
        s[[0, 1, 1]] = Complex64::new(0.2, 0.0); // S22
        s[[0, 0, 1]] = Complex64::new(0.5, 0.0); // S12
        s[[0, 1, 0]] = Complex64::new(0.5, 0.0); // S21

        let z0 = Array1::from_vec(vec![Complex64::new(50.0, 0.0), Complex64::new(75.0, 0.0)]);
        let ntwk = Network::new(freq, s, z0);
        let flipped = ntwk.flipped().unwrap();

        // After flip: S11 <-> S22, S12 <-> S21
        assert!((flipped.s[[0, 0, 0]].re - 0.2).abs() < 1e-10);
        assert!((flipped.s[[0, 1, 1]].re - 0.1).abs() < 1e-10);
        // z0 should also be swapped
        assert!((flipped.z0[0].re - 75.0).abs() < 1e-10);
        assert!((flipped.z0[1].re - 50.0).abs() < 1e-10);
    }
}
