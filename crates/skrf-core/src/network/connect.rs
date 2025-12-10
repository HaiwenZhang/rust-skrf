//! Network connection functions
//!
//! Provides algorithms for connecting n-port networks together.
//!
//! Based on the "sub-network growth" algorithm described in:
//! - Filipsson, Gunnar, "A New General Computer Algorithm for S-Matrix Calculation
//!   of Interconnected Multiports", 11th European Microwave Conference, 1981.

use ndarray::Array3;
use num_complex::Complex64;

use super::core::Network;
use crate::constants::NEAR_ZERO;

/// Connect two ports of a single n-port network's s-matrix.
///
/// Connects port `k` to port `l` on matrix `A`. This results in a (n-2)-port network.
/// Uses the sub-network growth algorithm.
///
/// # Arguments
/// * `a` - S-parameter matrix of shape [nfreq, nports, nports]
/// * `k` - First port index (0-indexed)
/// * `l` - Second port index (0-indexed)
///
/// # Returns
/// New S-parameter matrix with 2 fewer ports
pub fn innerconnect_s(a: &Array3<Complex64>, k: usize, l: usize) -> Option<Array3<Complex64>> {
    let nfreq = a.shape()[0];
    let nports = a.shape()[1];

    if k >= nports || l >= nports || k == l {
        return None;
    }

    // External ports (all ports except k and l)
    let ext_ports: Vec<usize> = (0..nports).filter(|&i| i != k && i != l).collect();
    let n_ext = ext_ports.len();

    if n_ext == 0 {
        return None; // Can't connect both ports of a 2-port
    }

    let mut c = Array3::<Complex64>::zeros((nfreq, n_ext, n_ext));

    for f in 0..nfreq {
        // Sub-matrices of internal ports
        let akl = Complex64::new(1.0, 0.0) - a[[f, k, l]];
        let alk = Complex64::new(1.0, 0.0) - a[[f, l, k]];
        let akk = a[[f, k, k]];
        let all = a[[f, l, l]];

        // Determinant
        let det = akl * alk - akk * all;
        if det.norm() < NEAR_ZERO {
            return None; // Singular matrix
        }

        // Calculate resultant S-parameters for external ports
        for (i, &ei) in ext_ports.iter().enumerate() {
            for (j, &ej) in ext_ports.iter().enumerate() {
                // Base S-parameter
                let mut sij = a[[f, ei, ej]];

                // Add connection terms
                let aik = a[[f, ei, k]];
                let ail = a[[f, ei, l]];
                let akj = a[[f, k, ej]];
                let alj = a[[f, l, ej]];

                sij +=
                    (akj * ail * alk + alj * aik * akl + akj * all * aik + alj * akk * ail) / det;

                c[[f, i, j]] = sij;
            }
        }
    }

    Some(c)
}

/// Connect two n-port networks' s-matrices together.
///
/// Connects port `k` on network `A` to port `l` on network `B`.
/// The resultant network has (nports_a + nports_b - 2) ports.
///
/// # Arguments
/// * `a` - S-parameter matrix of network A [nfreq, nports_a, nports_a]
/// * `k` - Port index on A (0-indexed)
/// * `b` - S-parameter matrix of network B [nfreq, nports_b, nports_b]
/// * `l` - Port index on B (0-indexed)
///
/// # Returns
/// New S-parameter matrix
pub fn connect_s(
    a: &Array3<Complex64>,
    k: usize,
    b: &Array3<Complex64>,
    l: usize,
) -> Option<Array3<Complex64>> {
    let nfreq = a.shape()[0];
    let nports_a = a.shape()[1];
    let nports_b = b.shape()[1];

    if k >= nports_a || l >= nports_b {
        return None;
    }

    // Frequency dimension must match
    if b.shape()[0] != nfreq {
        return None;
    }

    // Create composite matrix by placing A and B diagonally
    let nc = nports_a + nports_b;
    let mut c = Array3::<Complex64>::zeros((nfreq, nc, nc));

    for f in 0..nfreq {
        // Place A in top-left
        for i in 0..nports_a {
            for j in 0..nports_a {
                c[[f, i, j]] = a[[f, i, j]];
            }
        }
        // Place B in bottom-right
        for i in 0..nports_b {
            for j in 0..nports_b {
                c[[f, nports_a + i, nports_a + j]] = b[[f, i, j]];
            }
        }
    }

    // Connect port k on A to port (nports_a + l) on composite
    innerconnect_s(&c, k, nports_a + l)
}

impl Network {
    /// Connect two ports of this network together (innerconnect)
    ///
    /// Connects port `k` to port `l`, resulting in a (nports-2)-port network.
    pub fn innerconnect(&self, k: usize, l: usize) -> Option<Network> {
        let s_new = innerconnect_s(&self.s, k, l)?;
        let nports_new = s_new.shape()[1];

        // Build new z0 by excluding ports k and l
        let mut z0_vec = Vec::with_capacity(nports_new);
        for i in 0..self.nports() {
            if i != k && i != l {
                z0_vec.push(self.z0[i]);
            }
        }
        let z0_new = ndarray::Array1::from_vec(z0_vec);

        Some(Network::new(self.frequency.clone(), s_new, z0_new))
    }

    /// Connect this network's port `k` to another network's port `l`
    ///
    /// Returns a new network with (nports_self + nports_other - 2) ports.
    pub fn connect(&self, k: usize, other: &Network, l: usize) -> Option<Network> {
        // Check frequency compatibility
        if self.nfreq() != other.nfreq() {
            return None;
        }

        let s_new = connect_s(&self.s, k, &other.s, l)?;
        let nports_new = s_new.shape()[1];

        // Build new z0 by combining both networks' z0, excluding connected ports
        let mut z0_vec = Vec::with_capacity(nports_new);
        for i in 0..self.nports() {
            if i != k {
                z0_vec.push(self.z0[i]);
            }
        }
        for i in 0..other.nports() {
            if i != l {
                z0_vec.push(other.z0[i]);
            }
        }
        let z0_new = ndarray::Array1::from_vec(z0_vec);

        Some(Network::new(self.frequency.clone(), s_new, z0_new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};
    use ndarray::Array1;

    #[test]
    fn test_connect_two_thrus() {
        // Two thru connections should result in another thru
        let freq = Frequency::new(1.0, 1.0, 1, FrequencyUnit::GHz, SweepType::Linear);

        // Thru: S12=S21=1, S11=S22=0
        let mut s_thru = Array3::<Complex64>::zeros((1, 2, 2));
        s_thru[[0, 0, 1]] = Complex64::new(1.0, 0.0);
        s_thru[[0, 1, 0]] = Complex64::new(1.0, 0.0);

        let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));
        let thru = Network::new(freq.clone(), s_thru.clone(), z0.clone());
        let thru2 = Network::new(freq, s_thru, z0);

        // Connect port 1 of thru to port 0 of thru2
        let result = thru.connect(1, &thru2, 0);
        assert!(result.is_some());

        let connected = result.unwrap();
        assert_eq!(connected.nports(), 2);

        // Should still be approximately a thru
        assert!((connected.s[[0, 0, 1]].norm() - 1.0).abs() < 0.1);
    }
}
