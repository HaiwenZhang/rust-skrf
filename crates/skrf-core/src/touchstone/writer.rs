//! Touchstone file writer
//!
//! Writes S-parameter data to Touchstone format files.

use num_complex::Complex64;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::parser::{SParamFormat, Touchstone, TouchstoneError};
use crate::frequency::FrequencyUnit;

impl Touchstone {
    /// Write to a Touchstone file
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), TouchstoneError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        self.write_to(&mut writer)
    }

    /// Write to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), TouchstoneError> {
        // Write comments
        for comment in &self.comments {
            writeln!(writer, "! {}", comment)?;
        }

        // Write option line
        writeln!(
            writer,
            "# {} S {} R {}",
            match self.frequency.unit() {
                FrequencyUnit::Hz => "HZ",
                FrequencyUnit::KHz => "KHZ",
                FrequencyUnit::MHz => "MHZ",
                FrequencyUnit::GHz => "GHZ",
                FrequencyUnit::THz => "THZ",
            },
            match self.format {
                SParamFormat::RI => "RI",
                SParamFormat::MA => "MA",
                SParamFormat::DB => "DB",
            },
            if !self.z0.is_empty() {
                self.z0[0]
            } else {
                50.0
            }
        )?;

        // Write data
        let f_scaled = self.frequency.f_scaled();

        for (freq_idx, freq) in f_scaled.iter().enumerate() {
            let s_matrix = &self.s[freq_idx];

            write!(writer, "{:>15.9}", freq)?;

            // For 2-port, use standard order: S11, S21, S12, S22
            if self.nports == 2 {
                let order = [(0, 0), (1, 0), (0, 1), (1, 1)];
                for (i, j) in order {
                    let c = s_matrix[i][j];
                    let (v1, v2) = self.format_complex(c);
                    write!(writer, " {:>15.9} {:>15.9}", v1, v2)?;
                }
            } else {
                for row in s_matrix.iter() {
                    for c in row.iter() {
                        let (v1, v2) = self.format_complex(*c);
                        write!(writer, " {:>15.9} {:>15.9}", v1, v2)?;
                    }
                }
            }

            writeln!(writer)?;
        }

        Ok(())
    }

    fn format_complex(&self, c: Complex64) -> (f64, f64) {
        match self.format {
            SParamFormat::RI => (c.re, c.im),
            SParamFormat::MA => {
                let mag = c.norm();
                let deg = c.arg() * 180.0 / std::f64::consts::PI;
                (mag, deg)
            }
            SParamFormat::DB => {
                let db = 20.0 * c.norm().log10();
                let deg = c.arg() * 180.0 / std::f64::consts::PI;
                (db, deg)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests will be added when integration tests with files are set up
}
