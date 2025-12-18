use crate::constants::NEAR_ZERO;
use crate::network::Network;
use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3};
use num_complex::Complex64;

/// 1-Port SOL (Short-Open-Load) Calibration
pub struct OnePortSOL {
    pub short: Network,
    pub open: Network,
    pub load: Network,
    pub short_model: Network,
    pub open_model: Network,
    pub load_model: Network,
}

impl OnePortSOL {
    pub fn new(
        short: Network,
        open: Network,
        load: Network,
        short_model: Network,
        open_model: Network,
        load_model: Network,
    ) -> Self {
        Self {
            short,
            open,
            load,
            short_model,
            open_model,
            load_model,
        }
    }

    /// Calculate error coefficients (Directivity, Source Match, Reflection Tracking)
    ///
    /// e00: Directivity
    /// e11: Source Match
    /// de: Determinant of error matrix (e00*e11 - e10*e01)
    pub fn run(&self) -> Result<(Array2<Complex64>, Array2<Complex64>, Array2<Complex64>)> {
        let nfreq = self.short.nfreq();
        let mut e00 = Array2::<Complex64>::zeros((nfreq, 1));
        let mut e11 = Array2::<Complex64>::zeros((nfreq, 1));
        let mut de = Array2::<Complex64>::zeros((nfreq, 1));

        for f in 0..nfreq {
            let sm = self.short.s[[f, 0, 0]];
            let om = self.open.s[[f, 0, 0]];
            let lm = self.load.s[[f, 0, 0]];

            let si = self.short_model.s[[f, 0, 0]];
            let oi = self.open_model.s[[f, 0, 0]];
            let li = self.load_model.s[[f, 0, 0]];

            // Solve system of 3 linear equations for e00, de, e11
            // Sm = e00 + Si*de - Sm*Si*e11
            // Om = e00 + Oi*de - Om*Oi*e11
            // Lm = e00 + Li*de - Lm*Li*e11

            // Matrix formulation: A * x = B
            // [ 1  -Si  Sm*Si ] [ e00 ]   [ Sm ]
            // [ 1  -Oi  Om*Oi ] [ de  ] = [ Om ]
            // [ 1  -Li  Lm*Li ] [ e11 ]   [ Lm ]

            let a = nalgebra::Matrix3::new(
                Complex64::new(1.0, 0.0),
                -si,
                sm * si,
                Complex64::new(1.0, 0.0),
                -oi,
                om * oi,
                Complex64::new(1.0, 0.0),
                -li,
                lm * li,
            );

            let b = nalgebra::Vector3::new(sm, om, lm);

            if let Some(x) = a.qr().solve(&b) {
                e00[[f, 0]] = x[0];
                de[[f, 0]] = x[1];
                e11[[f, 0]] = x[2];
            } else {
                return Err(anyhow!(
                    "Failed to solve calibration equations at frequency index {}",
                    f
                ));
            }
        }

        Ok((e00, e11, de))
    }

    /// Apply calibration to measured data
    pub fn apply(&self, uncalibrated: &Network) -> Result<Network> {
        let (e00, e11, de) = self.run()?;
        let nfreq = uncalibrated.nfreq();
        let mut s_cal = Array3::<Complex64>::zeros((nfreq, 1, 1));

        for f in 0..nfreq {
            let sm = uncalibrated.s[[f, 0, 0]];
            // S_actual = (Sm - e00) / (Sm*e11 - de + e00*e11)
            // Wait, Standard formula: S_actual = (Sm - e00) / (Sm*e11 - (e00*e11 - e10*e01))
            // Which is (Sm - e00) / (Sm*e11 - de)
            let den = sm * e11[[f, 0]] - de[[f, 0]];
            if den.norm() < 1e-15 {
                s_cal[[f, 0, 0]] = Complex64::new(0.0, 0.0);
            } else {
                s_cal[[f, 0, 0]] = (sm - e00[[f, 0]]) / den;
            }
        }

        Ok(Network::new(
            uncalibrated.frequency.clone(),
            s_cal,
            uncalibrated.z0.clone(),
        ))
    }
}

/// 12-Term SOLT (Short-Open-Load-Thru) Calibration for 2-port networks
pub struct Solt12Term {
    pub p1_short: Network,
    pub p1_open: Network,
    pub p1_load: Network,
    pub p2_short: Network,
    pub p2_open: Network,
    pub p2_load: Network,
    pub thru: Network,

    // Models
    pub short_model: Network,
    pub open_model: Network,
    pub load_model: Network,
    pub thru_model: Network,
}

impl Solt12Term {
    pub fn new(
        p1_short: Network,
        p1_open: Network,
        p1_load: Network,
        p2_short: Network,
        p2_open: Network,
        p2_load: Network,
        thru: Network,
        short_model: Network,
        open_model: Network,
        load_model: Network,
        thru_model: Network,
    ) -> Self {
        Self {
            p1_short,
            p1_open,
            p1_load,
            p2_short,
            p2_open,
            p2_load,
            thru,
            short_model,
            open_model,
            load_model,
            thru_model,
        }
    }

    /// Calculate all 12 error terms
    ///
    /// Returns (ForwardTerms, ReverseTerms)
    pub fn run(&self) -> Result<(ForwardErrorTerms, ReverseErrorTerms)> {
        let nfreq = self.thru.nfreq();

        // 1. Solve 3-term for Port 1 (EDF, ESF, ERF)
        let s_p1 = OnePortSOL::new(
            self.p1_short.clone(),
            self.p1_open.clone(),
            self.p1_load.clone(),
            self.short_model.clone(),
            self.open_model.clone(),
            self.load_model.clone(),
        );
        let (edf, esf, det_f) = s_p1.run()?;
        let erf = det_f.clone() - (&edf * &esf);

        // 2. Solve 3-term for Port 2 (EDR, ESR, ERR)
        let s_p2 = OnePortSOL::new(
            self.p2_short.clone(),
            self.p2_open.clone(),
            self.p2_load.clone(),
            self.short_model.clone(),
            self.open_model.clone(),
            self.load_model.clone(),
        );
        let (edr, esr, det_r) = s_p2.run()?;
        let err = det_r.clone() - (&edr * &esr);

        // 3. Solve for Load Match (ELF, ELR) and Tracking (ETF, ETR) using Thru
        let mut elf = Array2::<Complex64>::zeros((nfreq, 1));
        let mut etf = Array2::<Complex64>::zeros((nfreq, 1));
        let mut elr = Array2::<Complex64>::zeros((nfreq, 1));
        let mut etr = Array2::<Complex64>::zeros((nfreq, 1));

        for f in 0..nfreq {
            let s11m = self.thru.s[[f, 0, 0]];
            let s21m = self.thru.s[[f, 1, 0]];
            let s12m = self.thru.s[[f, 0, 1]];
            let s22m = self.thru.s[[f, 1, 1]];

            // Ideal thru: S11=0, S22=0, S21=1, S12=1 (simplified)
            let s21i = self.thru_model.s[[f, 1, 0]];

            // Forward Load Match (ELF)
            // ELF = (S11m - EDF) / (S11m*ESF - detF)
            elf[[f, 0]] = (s11m - edf[[f, 0]]) / (s11m * esf[[f, 0]] - det_f[[f, 0]]);

            // Forward Trans Tracking (ETF)
            // ETF = S21m/S21i * (1 - ESF*Si11) * (1 - ELF*Si22) ... simplifies for ideal thru
            etf[[f, 0]] = s21m * (Complex64::new(1.0, 0.0) - esf[[f, 0]] * elf[[f, 0]]) / s21i;

            // Reverse Load Match (ELR)
            elr[[f, 0]] = (s22m - edr[[f, 0]]) / (s22m * esr[[f, 0]] - det_r[[f, 0]]);

            // Reverse Trans Tracking (ETR)
            etr[[f, 0]] = s12m * (Complex64::new(1.0, 0.0) - esr[[f, 0]] * elr[[f, 0]]) / s21i;
        }

        Ok((
            ForwardErrorTerms {
                edf,
                esf,
                erf,
                elf,
                etf,
                exf: Array2::zeros((nfreq, 1)),
            },
            ReverseErrorTerms {
                edr,
                esr,
                err,
                elr,
                etr,
                exr: Array2::zeros((nfreq, 1)),
            },
        ))
    }

    /// Apply 12-term calibration
    pub fn apply(&self, uncalibrated: &Network) -> Result<Network> {
        let (fwd, rev) = self.run()?;
        let nfreq = uncalibrated.nfreq();
        let mut s_cal = Array3::<Complex64>::zeros((nfreq, 2, 2));

        for f in 0..nfreq {
            let s11m = uncalibrated.s[[f, 0, 0]];
            let s21m = uncalibrated.s[[f, 1, 0]];
            let s12m = uncalibrated.s[[f, 0, 1]];
            let s22m = uncalibrated.s[[f, 1, 1]];

            let edf = fwd.edf[[f, 0]];
            let esf = fwd.esf[[f, 0]];
            let erf = fwd.erf[[f, 0]];
            let elf = fwd.elf[[f, 0]];
            let etf = fwd.etf[[f, 0]];

            let edr = rev.edr[[f, 0]];
            let esr = rev.esr[[f, 0]];
            let err = rev.err[[f, 0]];
            let elr = rev.elr[[f, 0]];
            let etr = rev.etr[[f, 0]];

            // Standard 12-term correction
            let s11p = (s11m - edf) / erf;
            let s21p = (s21m - fwd.exf[[f, 0]]) / etf;
            let s12p = (s12m - rev.exr[[f, 0]]) / etr;
            let s22p = (s22m - edr) / err;

            let den = (Complex64::new(1.0, 0.0) + s11p * esf)
                * (Complex64::new(1.0, 0.0) + s22p * esr)
                - s21p * s12p * elf * elr;

            if den.norm() > NEAR_ZERO {
                s_cal[[f, 0, 0]] =
                    (s11p * (Complex64::new(1.0, 0.0) + s22p * esr) - s21p * s12p * elf) / den;
                s_cal[[f, 1, 0]] = s21p * (Complex64::new(1.0, 0.0) + s22p * (esr - elr)) / den;
                s_cal[[f, 0, 1]] = s12p * (Complex64::new(1.0, 0.0) + s11p * (esf - elf)) / den;
                s_cal[[f, 1, 1]] =
                    (s22p * (Complex64::new(1.0, 0.0) + s11p * esf) - s21p * s12p * elr) / den;
            }
        }

        Ok(Network::new(
            uncalibrated.frequency.clone(),
            s_cal,
            uncalibrated.z0.clone(),
        ))
    }
}

pub struct ForwardErrorTerms {
    pub edf: Array2<Complex64>,
    pub esf: Array2<Complex64>,
    pub erf: Array2<Complex64>,
    pub elf: Array2<Complex64>,
    pub etf: Array2<Complex64>,
    pub exf: Array2<Complex64>,
}

pub struct ReverseErrorTerms {
    pub edr: Array2<Complex64>,
    pub esr: Array2<Complex64>,
    pub err: Array2<Complex64>,
    pub elr: Array2<Complex64>,
    pub etr: Array2<Complex64>,
    pub exr: Array2<Complex64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::{Frequency, FrequencyUnit, SweepType};
    use ndarray::Array1;

    #[test]
    fn test_one_port_sol_calibration() -> Result<()> {
        let nfreq = 1;
        let freq = Frequency::new(1.0, 1.0, nfreq, FrequencyUnit::GHz, SweepType::Linear);
        let z0 = Array1::from_elem(1, Complex64::new(50.0, 0.0));

        // Ideal models
        let short_model = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), Complex64::new(-1.0, 0.0)),
            z0.clone(),
        );
        let open_model = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), Complex64::new(1.0, 0.0)),
            z0.clone(),
        );
        let load_model = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), Complex64::new(0.0, 0.0)),
            z0.clone(),
        );

        // Simulated measurements with error (Directivity=0.1, SourceMatch=0.05, ReflectionTracking=0.9)
        // Sm = e00 + (et * Si) / (1 - e11*Si)
        let e00 = Complex64::new(0.1, 0.0);
        let e11 = Complex64::new(0.05, 0.0);
        let et = Complex64::new(0.9, 0.0);

        let simulate = |si: Complex64| -> Complex64 {
            e00 + (et * si) / (Complex64::new(1.0, 0.0) - e11 * si)
        };

        let short_meas = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), simulate(Complex64::new(-1.0, 0.0))),
            z0.clone(),
        );
        let open_meas = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), simulate(Complex64::new(1.0, 0.0))),
            z0.clone(),
        );
        let load_meas = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), simulate(Complex64::new(0.0, 0.0))),
            z0.clone(),
        );

        let cal = OnePortSOL::new(
            short_meas,
            open_meas,
            load_meas,
            short_model,
            open_model,
            load_model,
        );

        // Calibrate a "dirty" measurement
        let si_dirty = Complex64::new(0.5, 0.2);
        let sm_dirty = simulate(si_dirty);
        let uncal = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), sm_dirty),
            z0.clone(),
        );

        let calified = cal.apply(&uncal)?;

        let s_actual = calified.s[[0, 0, 0]];
        assert!((s_actual.re - si_dirty.re).abs() < 1e-12);
        assert!((s_actual.im - si_dirty.im).abs() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_12_term_solt_calibration() -> Result<()> {
        let nfreq = 1;
        let freq = Frequency::new(1.0, 1.0, nfreq, FrequencyUnit::GHz, SweepType::Linear);
        let z0 = Array1::from_elem(2, Complex64::new(50.0, 0.0));

        let s_ideal = Array3::from_elem((nfreq, 2, 2), Complex64::new(0.0, 0.0));
        let thru_ideal = {
            let mut s = s_ideal.clone();
            s[[0, 1, 0]] = Complex64::new(1.0, 0.0);
            s[[0, 0, 1]] = Complex64::new(1.0, 0.0);
            Network::new(freq.clone(), s, z0.clone())
        };

        // Simplified: use ideal measurements to verify identity
        let short = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), Complex64::new(-1.0, 0.0)),
            Array1::from_elem(1, Complex64::new(50.0, 0.0)),
        );
        let open = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), Complex64::new(1.0, 0.0)),
            Array1::from_elem(1, Complex64::new(50.0, 0.0)),
        );
        let load = Network::new(
            freq.clone(),
            Array3::from_elem((nfreq, 1, 1), Complex64::new(0.0, 0.0)),
            Array1::from_elem(1, Complex64::new(50.0, 0.0)),
        );

        let cal = Solt12Term::new(
            short.clone(),
            open.clone(),
            load.clone(),
            short.clone(),
            open.clone(),
            load.clone(),
            thru_ideal.clone(),
            short.clone(),
            open.clone(),
            load.clone(),
            thru_ideal.clone(),
        );

        let uncal = thru_ideal.clone();
        let calified = cal.apply(&uncal)?;

        assert!((calified.s[[0, 1, 0]].re - 1.0).abs() < 1e-12);
        assert!(calified.s[[0, 0, 0]].norm() < 1e-12);

        Ok(())
    }
}
