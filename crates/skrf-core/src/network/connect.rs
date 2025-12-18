//! Network connection functions
//!
//! Provides algorithms for connecting n-port networks together.
//!
//! Based on the "sub-network growth" algorithm described in:
//! - Filipsson, Gunnar, "A New General Computer Algorithm for S-Matrix Calculation
//!   of Interconnected Multiports", 11th European Microwave Conference, 1981.

use anyhow::{bail, Result};
use ndarray::Array3;
use num_complex::Complex64;

use super::core::Network;

/// Generic S-parameter connection algorithm.
///
/// Connects multiple port pairs within a single network.
///
/// # Arguments
/// * `a` - S-parameter matrix [nfreq, nports, nports]
/// * `connections` - List of port pairs to connect [(k, l), ...]
pub fn innerconnect_multi_s(
    a: &Array3<Complex64>,
    connections: &[(usize, usize)],
) -> Result<Array3<Complex64>> {
    let nfreq = a.shape()[0];
    let nports = a.shape()[1];

    if connections.is_empty() {
        return Ok(a.clone());
    }

    // Identify internal and external ports
    let mut internal_set = std::collections::HashSet::new();
    for &(k, l) in connections {
        if k >= nports || l >= nports {
            bail!("Port index out of range");
        }
        if k == l {
            bail!("Cannot connect port to itself");
        }
        if !internal_set.insert(k) || !internal_set.insert(l) {
            bail!("Port connected more than once");
        }
    }

    let mut ext_ports = Vec::new();
    for i in 0..nports {
        if !internal_set.contains(&i) {
            ext_ports.push(i);
        }
    }
    let n_ext = ext_ports.len();

    // Map internal ports to their indices in the Si_i matrix
    let int_ports: Vec<usize> = connections.iter().flat_map(|&(k, l)| vec![k, l]).collect();
    let n_int = int_ports.len();

    // Connection matrix M (permutation matrix for the pairs)
    // For each pair (k, l), M[k, l] = 1 and M[l, k] = 1
    let mut m = nalgebra::DMatrix::<Complex64>::zeros(n_int, n_int);
    for i in 0..connections.len() {
        m[(2 * i, 2 * i + 1)] = Complex64::new(1.0, 0.0);
        m[(2 * i + 1, 2 * i)] = Complex64::new(1.0, 0.0);
    }

    let mut result = Array3::<Complex64>::zeros((nfreq, n_ext, n_ext));

    for f in 0..nfreq {
        // Extract sub-matrices
        // S_ee
        let mut s_ee = nalgebra::DMatrix::<Complex64>::zeros(n_ext, n_ext);
        for (i, &ei) in ext_ports.iter().enumerate() {
            for (j, &ej) in ext_ports.iter().enumerate() {
                s_ee[(i, j)] = a[[f, ei, ej]];
            }
        }

        // S_ei
        let mut s_ei = nalgebra::DMatrix::<Complex64>::zeros(n_ext, n_int);
        for (i, &ei) in ext_ports.iter().enumerate() {
            for (j, &ij) in int_ports.iter().enumerate() {
                s_ei[(i, j)] = a[[f, ei, ij]];
            }
        }

        // S_ie
        let mut s_ie = nalgebra::DMatrix::<Complex64>::zeros(n_int, n_ext);
        for (i, &ii) in int_ports.iter().enumerate() {
            for (j, &ej) in ext_ports.iter().enumerate() {
                s_ie[(i, j)] = a[[f, ii, ej]];
            }
        }

        // S_ii
        let mut s_ii = nalgebra::DMatrix::<Complex64>::zeros(n_int, n_int);
        for (i, &ii) in int_ports.iter().enumerate() {
            for (j, &ij) in int_ports.iter().enumerate() {
                s_ii[(i, j)] = a[[f, ii, ij]];
            }
        }

        // S_new = S_ee + S_ei * (I - M * S_ii)^-1 * M * S_ie
        let identity = nalgebra::DMatrix::<Complex64>::identity(n_int, n_int);
        let m_inv_block = (identity - &m * s_ii)
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("Singular matrix in connection at frequency {}", f))?;

        let s_new = s_ee + s_ei * m_inv_block * &m * s_ie;

        for i in 0..n_ext {
            for j in 0..n_ext {
                result[[f, i, j]] = s_new[(i, j)];
            }
        }
    }

    Ok(result)
}

/// Connect two ports of a single n-port network's s-matrix.
pub fn innerconnect_s(a: &Array3<Complex64>, k: usize, l: usize) -> Result<Array3<Complex64>> {
    innerconnect_multi_s(a, &[(k, l)])
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
) -> Result<Array3<Complex64>> {
    let nfreq = a.shape()[0];
    let nports_a = a.shape()[1];
    let nports_b = b.shape()[1];

    if k >= nports_a {
        bail!(
            "port k={} out of range (network A has {} ports)",
            k,
            nports_a
        );
    }
    if l >= nports_b {
        bail!(
            "port l={} out of range (network B has {} ports)",
            l,
            nports_b
        );
    }

    // Frequency dimension must match
    if b.shape()[0] != nfreq {
        bail!("frequency count mismatch: {} vs {}", nfreq, b.shape()[0]);
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
    pub fn innerconnect(&self, k: usize, l: usize) -> Result<Network> {
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

        Network::new(self.frequency.clone(), s_new, z0_new)
    }

    /// Connect this network's port `k` to another network's port `l`
    ///
    /// Returns a new network with (nports_self + nports_other - 2) ports.
    pub fn connect(&self, k: usize, other: &Network, l: usize) -> Result<Network> {
        // Check frequency compatibility
        if self.nfreq() != other.nfreq() {
            bail!(
                "frequency count mismatch: {} vs {}",
                self.nfreq(),
                other.nfreq()
            );
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

        Network::new(self.frequency.clone(), s_new, z0_new)
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
        let thru = Network::new(freq.clone(), s_thru.clone(), z0.clone()).unwrap();
        let thru2 = Network::new(freq, s_thru, z0).unwrap();

        // Connect port 1 of thru to port 0 of thru2
        let result = thru.connect(1, &thru2, 0);
        assert!(result.is_ok());

        let connected = result.expect("Connect failed");
        assert_eq!(connected.nports(), 2);

        // Should still be approximately a thru
        assert!((connected.s[[0, 0, 1]].norm() - 1.0).abs() < 0.1);
    }
}
