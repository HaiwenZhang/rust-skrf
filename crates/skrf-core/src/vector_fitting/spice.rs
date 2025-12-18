//! SPICE export functionality for Vector Fitting models

use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Generate SPICE subcircuit netlist for a vector fitted S-parameter model
///
/// Creates an equivalent N-port subcircuit based on its vector fitted scattering (S)
/// parameter responses in SPICE simulator netlist syntax (compatible with LTspice,
/// ngspice, Xyce, ...).
///
/// The circuit synthesis is based on a direct implementation of the state-space
/// representation of the vector fitted model.
///
/// # Arguments
/// * `poles` - Fitted poles
/// * `residues` - Fitted residues [n_responses, n_poles]
/// * `constant_coeff` - Constant coefficients (d terms)
/// * `proportional_coeff` - Proportional coefficients (e terms)
/// * `z0` - Reference impedances `[nports]`
/// * `nports` - Number of ports
/// * `model_name` - Name of the subcircuit
/// * `create_reference_pins` - If true, create separate reference pins for each port
///
/// # Returns
/// SPICE netlist as a String
#[allow(clippy::too_many_arguments)]
pub fn generate_spice_subcircuit_s(
    poles: &Array1<Complex64>,
    residues: &Array2<Complex64>,
    constant_coeff: &Array1<f64>,
    proportional_coeff: &Array1<f64>,
    z0: &Array1<Complex64>,
    nports: usize,
    model_name: &str,
    create_reference_pins: bool,
) -> String {
    let mut netlist = String::new();

    // Check if we need proportional term network
    let build_e = proportional_coeff.iter().any(|&e| e != 0.0);

    // Write title line
    writeln!(netlist, "* EQUIVALENT CIRCUIT FOR VECTOR FITTED S-MATRIX").unwrap();
    writeln!(netlist, "* Created using rust-skrf VectorFitting").unwrap();
    writeln!(netlist, "*").unwrap();

    // Create subcircuit pin string
    let input_nodes: String = if create_reference_pins {
        (0..nports)
            .map(|i| format!("p{} p{}_ref", i + 1, i + 1))
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        (0..nports)
            .map(|i| format!("p{}", i + 1))
            .collect::<Vec<_>>()
            .join(" ")
    };

    writeln!(netlist, ".SUBCKT {} {}", model_name, input_nodes).unwrap();

    for i in 0..nports {
        writeln!(netlist, "*").unwrap();
        writeln!(netlist, "* Port network for port {}", i + 1).unwrap();

        let node_ref_i = if create_reference_pins {
            format!("p{}_ref", i + 1)
        } else {
            "0".to_string()
        };

        // Reference impedance (real part) of port i
        let z0_i = z0[i].re;

        // Transfer gain of the controlled current sources representing incident power wave a_i
        // a_i = 1 / 2 / sqrt(Z0_i) * (V_i + Z0_i * I_i)
        let gain_vccs_a_i = 1.0 / 2.0 / z0_i.sqrt();
        let gain_cccs_a_i = z0_i.sqrt() / 2.0;

        // Transfer gain for reflected power wave b_i
        // b_i = sqrt(Z0_i) / 2 * I_b_i
        let gain_b_i = 2.0 / z0_i.sqrt();

        // Dummy voltage source for port current sensing
        writeln!(netlist, "V{} p{} s{} 0", i + 1, i + 1, i + 1).unwrap();

        // Port reference resistor Ri = Z0_i
        writeln!(netlist, "R{} s{} {} {}", i + 1, i + 1, node_ref_i, z0_i).unwrap();

        // Transfer of states and inputs from port j to input/output network of port i
        for j in 0..nports {
            let node_ref_j = if create_reference_pins {
                format!("p{}_ref", j + 1)
            } else {
                "0".to_string()
            };

            let z0_j = z0[j].re;
            let idx_s_i_j = i * nports + j;

            // VCCS and CCCS for incident wave a_j
            let gain_vccs_a_j = 1.0 / 2.0 / z0_j.sqrt();
            let gain_cccs_a_j = z0_j.sqrt() / 2.0;

            let d = constant_coeff[idx_s_i_j];
            let e = proportional_coeff[idx_s_i_j];

            if d != 0.0 {
                // Constant term d_i_j
                let g_ij = gain_b_i * d * gain_vccs_a_j;
                let f_ij = gain_b_i * d * gain_cccs_a_j;
                writeln!(
                    netlist,
                    "Gd{}_{} {} s{} p{} {} {}",
                    i + 1,
                    j + 1,
                    node_ref_i,
                    i + 1,
                    j + 1,
                    node_ref_j,
                    g_ij
                )
                .unwrap();
                writeln!(
                    netlist,
                    "Fd{}_{} {} s{} V{} {}",
                    i + 1,
                    j + 1,
                    node_ref_i,
                    i + 1,
                    j + 1,
                    f_ij
                )
                .unwrap();
            }

            if build_e && e != 0.0 {
                // Proportional term e_i_j
                let g_ij = gain_b_i * e;
                writeln!(
                    netlist,
                    "Ge{}_{} {} s{} e{} 0 {}",
                    i + 1,
                    j + 1,
                    node_ref_i,
                    i + 1,
                    j + 1,
                    g_ij
                )
                .unwrap();
            }

            // Each residue rk_i_j at port i is multiplied by its respective state signal xk_j
            for (k, &pole) in poles.iter().enumerate() {
                let residue = residues[[idx_s_i_j, k]];
                let g_re = gain_b_i * residue.re;
                let g_im = gain_b_i * residue.im;

                if pole.im == 0.0 {
                    // Real pole/residue pair
                    let xkj = format!("x{}_a{}", k + 1, j + 1);
                    writeln!(
                        netlist,
                        "Gr{}_{}_{} {} s{} {} 0 {}",
                        k + 1,
                        i + 1,
                        j + 1,
                        node_ref_i,
                        i + 1,
                        xkj,
                        g_re
                    )
                    .unwrap();
                } else {
                    // Complex-conjugate pole/residue pair
                    let xk_re_j = format!("x{}_re_a{}", k + 1, j + 1);
                    let xk_im_j = format!("x{}_im_a{}", k + 1, j + 1);
                    writeln!(
                        netlist,
                        "Gr{}_re_{}_{} {} s{} {} 0 {}",
                        k + 1,
                        i + 1,
                        j + 1,
                        node_ref_i,
                        i + 1,
                        xk_re_j,
                        g_re
                    )
                    .unwrap();
                    writeln!(
                        netlist,
                        "Gr{}_im_{}_{} {} s{} {} 0 {}",
                        k + 1,
                        i + 1,
                        j + 1,
                        node_ref_i,
                        i + 1,
                        xk_im_j,
                        g_im
                    )
                    .unwrap();
                }
            }
        }

        // Create state networks driven by this port i
        writeln!(netlist, "*").unwrap();
        writeln!(netlist, "* State networks driven by port {}", i + 1).unwrap();

        for (k, &pole) in poles.iter().enumerate() {
            let pole_re = pole.re;
            let pole_im = pole.im;

            if pole_im == 0.0 {
                // Real pole
                let xki = format!("x{}_a{}", k + 1, i + 1);
                writeln!(netlist, "Cx{}_a{} {} 0 1.0", k + 1, i + 1, xki).unwrap();
                writeln!(
                    netlist,
                    "Gx{}_a{} 0 {} p{} {} {}",
                    k + 1,
                    i + 1,
                    xki,
                    i + 1,
                    node_ref_i,
                    gain_vccs_a_i
                )
                .unwrap();
                writeln!(
                    netlist,
                    "Fx{}_a{} 0 {} V{} {}",
                    k + 1,
                    i + 1,
                    xki,
                    i + 1,
                    gain_cccs_a_i
                )
                .unwrap();
                writeln!(
                    netlist,
                    "Rp{}_a{} 0 {} {}",
                    k + 1,
                    i + 1,
                    xki,
                    -1.0 / pole_re
                )
                .unwrap();
            } else {
                // Complex pole
                let xk_re_i = format!("x{}_re_a{}", k + 1, i + 1);
                let xk_im_i = format!("x{}_im_a{}", k + 1, i + 1);

                // Real part state
                writeln!(netlist, "Cx{}_re_a{} {} 0 1.0", k + 1, i + 1, xk_re_i).unwrap();
                writeln!(
                    netlist,
                    "Gx{}_re_a{} 0 {} p{} {} {}",
                    k + 1,
                    i + 1,
                    xk_re_i,
                    i + 1,
                    node_ref_i,
                    2.0 * gain_vccs_a_i
                )
                .unwrap();
                writeln!(
                    netlist,
                    "Fx{}_re_a{} 0 {} V{} {}",
                    k + 1,
                    i + 1,
                    xk_re_i,
                    i + 1,
                    2.0 * gain_cccs_a_i
                )
                .unwrap();
                writeln!(
                    netlist,
                    "Rp{}_re_re_a{} 0 {} {}",
                    k + 1,
                    i + 1,
                    xk_re_i,
                    -1.0 / pole_re
                )
                .unwrap();
                writeln!(
                    netlist,
                    "Gp{}_re_im_a{} 0 {} {} 0 {}",
                    k + 1,
                    i + 1,
                    xk_re_i,
                    xk_im_i,
                    pole_im
                )
                .unwrap();

                // Imaginary part state
                writeln!(netlist, "Cx{}_im_a{} {} 0 1.0", k + 1, i + 1, xk_im_i).unwrap();
                writeln!(
                    netlist,
                    "Gp{}_im_re_a{} 0 {} {} 0 {}",
                    k + 1,
                    i + 1,
                    xk_im_i,
                    xk_re_i,
                    -pole_im
                )
                .unwrap();
                writeln!(
                    netlist,
                    "Rp{}_im_im_a{} 0 {} {}",
                    k + 1,
                    i + 1,
                    xk_im_i,
                    -1.0 / pole_re
                )
                .unwrap();
            }
        }

        // Create differentiation network for proportional term
        if build_e {
            writeln!(netlist, "*").unwrap();
            writeln!(
                netlist,
                "* Network with derivative of input a_{} for proportional term",
                i + 1
            )
            .unwrap();
            writeln!(netlist, "Le{} e{} 0 1.0", i + 1, i + 1).unwrap();
            writeln!(
                netlist,
                "Ge{} 0 e{} p{} {} {}",
                i + 1,
                i + 1,
                i + 1,
                node_ref_i,
                gain_vccs_a_i
            )
            .unwrap();
            writeln!(
                netlist,
                "Fe{} 0 e{} V{} {}",
                i + 1,
                i + 1,
                i + 1,
                gain_cccs_a_i
            )
            .unwrap();
        }
    }

    writeln!(netlist, ".ENDS {}", model_name).unwrap();

    netlist
}

/// Write SPICE subcircuit to a file
#[allow(clippy::too_many_arguments)]
pub fn write_spice_subcircuit_s_to_file(
    path: &Path,
    poles: &Array1<Complex64>,
    residues: &Array2<Complex64>,
    constant_coeff: &Array1<f64>,
    proportional_coeff: &Array1<f64>,
    z0: &Array1<Complex64>,
    nports: usize,
    model_name: &str,
    create_reference_pins: bool,
) -> io::Result<()> {
    let netlist = generate_spice_subcircuit_s(
        poles,
        residues,
        constant_coeff,
        proportional_coeff,
        z0,
        nports,
        model_name,
        create_reference_pins,
    );

    let mut file = File::create(path)?;
    file.write_all(netlist.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_spice_1port() {
        let poles = Array1::from_vec(vec![Complex64::new(-1e9, 0.0)]);
        let residues = Array2::from_shape_vec((1, 1), vec![Complex64::new(0.5, 0.0)]).unwrap();
        let constant_coeff = Array1::from_vec(vec![0.1]);
        let proportional_coeff = Array1::from_vec(vec![0.0]);
        let z0 = Array1::from_vec(vec![Complex64::new(50.0, 0.0)]);

        let netlist = generate_spice_subcircuit_s(
            &poles,
            &residues,
            &constant_coeff,
            &proportional_coeff,
            &z0,
            1,
            "test_model",
            false,
        );

        assert!(netlist.contains(".SUBCKT test_model"));
        assert!(netlist.contains(".ENDS test_model"));
        assert!(netlist.contains("V1 p1 s1 0"));
        assert!(netlist.contains("R1 s1 0 50"));
    }
}
