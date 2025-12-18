//! Touchstone file parser
//!
//! Implements parsing of Touchstone v1 and v2 format files.

use num_complex::Complex64;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

use crate::frequency::{Frequency, FrequencyUnit};

/// Touchstone parsing errors
#[derive(Error, Debug)]
pub enum TouchstoneError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("Invalid option line: {0}")]
    InvalidOption(String),

    #[error("Invalid file extension: expected .sNp format")]
    InvalidExtension,
}

/// S-parameter data format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SParamFormat {
    #[default]
    RI, // Real-Imaginary
    MA, // Magnitude-Angle (degrees)
    DB, // dB-Angle (degrees)
}

impl SParamFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "RI" => Some(SParamFormat::RI),
            "MA" => Some(SParamFormat::MA),
            "DB" => Some(SParamFormat::DB),
            _ => None,
        }
    }
}

/// Touchstone file parser and data container
/// Network parameter type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParameterType {
    #[default]
    S,
    Y,
    Z,
    G,
    H,
}

impl ParameterType {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "S" => Some(ParameterType::S),
            "Y" => Some(ParameterType::Y),
            "Z" => Some(ParameterType::Z),
            "G" => Some(ParameterType::G),
            "H" => Some(ParameterType::H),
            _ => None,
        }
    }
}

/// Touchstone file parser and data container
#[derive(Debug, Clone)]
pub struct Touchstone {
    /// Number of ports
    pub nports: usize,
    /// Frequency data
    pub frequency: Frequency,
    /// Parameter data matrices: [nfreq, nports, nports]
    /// Can be S, Y, Z, G, or H parameters depending on `param_type`
    pub s: Vec<Vec<Vec<Complex64>>>,
    /// Reference impedance (per port)
    pub z0: Vec<f64>,
    /// Comments from the file
    pub comments: Vec<String>,
    /// Original format
    pub format: SParamFormat,
    /// Parameter type (S, Y, Z, G, H)
    pub param_type: ParameterType,
    /// Is this a Version 2.0 file?
    pub is_v2: bool,
    /// Mixed-mode order (if specified)
    pub mixed_mode_order: Vec<String>,
    /// Whether noise data was encountered
    pub noisy: bool,
}

impl Touchstone {
    /// Parse a Touchstone file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TouchstoneError> {
        let path = path.as_ref();

        // Determine number of ports from extension
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or(TouchstoneError::InvalidExtension)?;

        let nports = Self::parse_extension(ext)?;

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        Self::parse(reader, nports)
    }

    /// Parse extension to get number of ports
    fn parse_extension(ext: &str) -> Result<usize, TouchstoneError> {
        let ext_lower = ext.to_lowercase();
        if ext_lower.starts_with('s') && ext_lower.ends_with('p') {
            let num_str = &ext_lower[1..ext_lower.len() - 1];
            num_str
                .parse::<usize>()
                .map_err(|_| TouchstoneError::InvalidExtension)
        } else if ext_lower == "ts" {
            // Touchstone 2.0 format - need to read from file
            Ok(0) // Will be determined from file content
        } else {
            Err(TouchstoneError::InvalidExtension)
        }
    }

    /// Parse from string content
    ///
    /// This is useful for WASM environments where file system access is not available.
    ///
    /// # Arguments
    /// * `content` - Touchstone file content as string
    /// * `nports` - Number of ports (typically derived from file extension, e.g., .s2p = 2 ports)
    ///
    /// # Example
    /// ```ignore
    /// let content = "# GHz S RI R 50\n1.0 0.1 0.0 0.9 0.0 0.9 0.0 0.1 0.0";
    /// let ts = Touchstone::from_str(content, 2)?;
    /// ```
    pub fn from_str(content: &str, nports: usize) -> Result<Self, TouchstoneError> {
        let cursor = std::io::Cursor::new(content);
        Self::parse(cursor, nports)
    }

    /// Parse from a reader
    fn parse<R: BufRead>(reader: R, nports_hint: usize) -> Result<Self, TouchstoneError> {
        let mut state = ParserState::new(nports_hint);

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                continue;
            }

            // Handle comments (v1 starts with !, v2 can be anywhere but usually line based)
            if let Some(comment) = trimmed.strip_prefix('!') {
                state.comments.push(comment.trim().to_string());
                continue;
            }

            // Handle keywords
            if trimmed.starts_with('[') {
                state.parse_keyword(trimmed)?;
                continue;
            }

            // Handle option line
            if trimmed.starts_with('#') {
                if !state.option_parsed {
                    state.parse_option_line(trimmed)?;
                }
                continue;
            }

            // Handle continuation of reference impedance
            if state.expecting_reference {
                state.check_expecting_reference(trimmed);
                continue;
            }

            // Handle continuation of mixed-mode order
            if state.expecting_mixed_mode_order {
                state.check_expecting_mixed_mode_order(trimmed);
                continue;
            }

            // Parse data line
            // Check if we are in a data section (implicit for v1, explicit for v2)
            // Or if the line looks like data (starts with number/sign) even if [Network Data] missing
            let first_char = trimmed.chars().next().unwrap_or(' ');
            let looks_like_data = first_char.is_ascii_digit()
                || first_char == '-'
                || first_char == '+'
                || first_char == '.';

            if (state.is_data_section() || looks_like_data) && !state.noise_data_encountered {
                state.parse_data_line(trimmed)?;
            }
        }

        state.finalize()
    }

    /// Parse the option line (# Hz S RI R 50)
    pub fn parse_option_line(
        line: &str,
        is_v2: bool,
    ) -> Result<(FrequencyUnit, SParamFormat, f64, ParameterType), TouchstoneError> {
        let parts: Vec<&str> = line[1..].split_whitespace().collect();
        if parts.is_empty() {
            return Ok((
                FrequencyUnit::GHz,
                if is_v2 {
                    SParamFormat::MA
                } else {
                    SParamFormat::RI
                },
                50.0,
                ParameterType::S,
            ));
        }

        let mut freq_unit = FrequencyUnit::GHz;
        let mut format = SParamFormat::RI;
        let mut z0 = 50.0;
        let mut param_type = ParameterType::S;

        let mut i = 0;
        while i < parts.len() {
            let part = parts[i].to_uppercase();

            if let Some(unit) = FrequencyUnit::parse(&part) {
                freq_unit = unit;
            } else if let Some(fmt) = SParamFormat::parse(&part) {
                format = fmt;
            } else if let Some(pt) = ParameterType::parse(&part) {
                param_type = pt;
            } else if part == "R" && i + 1 < parts.len() {
                if let Ok(r) = parts[i + 1].parse::<f64>() {
                    z0 = r;
                    i += 1;
                }
            }

            i += 1;
        }

        Ok((freq_unit, format, z0, param_type))
    }

    /// Get the number of frequency points
    pub fn nfreq(&self) -> usize {
        self.s.len()
    }
}

/// Internal parser state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixFormat {
    #[default]
    Full,
    Lower,
    Upper,
}

struct ParserState {
    version: String,
    nports: usize,
    freq_unit: FrequencyUnit,
    format: SParamFormat,
    matrix_format: MatrixFormat,
    z0: Vec<f64>,
    param_type: ParameterType,
    comments: Vec<String>,
    option_parsed: bool,

    // Data accumulation
    frequencies: Vec<f64>,
    s_data: Vec<Vec<Vec<Complex64>>>,
    current_freq_data: Vec<f64>,

    // V2 specific
    is_v2: bool,
    data_section_started: bool,
    two_port_order_21_12: bool, // Default is 12_21 (S11, S21, S12, S22)
    expecting_reference: bool,
    expecting_mixed_mode_order: bool,
    noise_data_encountered: bool,
    expected_nfreq: Option<usize>,
    mixed_mode_order: Vec<String>,
}

impl ParserState {
    fn new(nports_hint: usize) -> Self {
        Self {
            version: "1.0".to_string(),
            nports: nports_hint,
            freq_unit: FrequencyUnit::GHz,
            format: SParamFormat::RI,
            matrix_format: MatrixFormat::Full,
            z0: Vec::new(),
            param_type: ParameterType::S,
            comments: Vec::new(),
            option_parsed: false,
            frequencies: Vec::new(),
            s_data: Vec::new(),
            current_freq_data: Vec::new(),
            is_v2: false,
            data_section_started: false,
            two_port_order_21_12: true, // Standard Touchstone 2-port order (x, S11, S21, S12, S22)
            expecting_reference: false,
            expecting_mixed_mode_order: false,
            noise_data_encountered: false,
            expected_nfreq: None,
            mixed_mode_order: Vec::new(),
        }
    }

    /// Determine if 2-port data uses 21_12 order (S11, S21, S12, S22)
    ///
    /// V1: Always uses 21_12 order
    /// V2: Defaults to 12_21 (S11, S12, S21, S22) unless [Two-Port Data Order] 21_12
    #[inline]
    fn use_21_12_order(&self) -> bool {
        !self.is_v2 || self.two_port_order_21_12
    }

    /// Map 2-port flat index to (row, col) based on ordering
    ///
    /// For 21_12 order: 0->S11, 1->S21, 2->S12, 3->S22
    /// For 12_21 order: 0->S11, 1->S12, 2->S21, 3->S22 (row-major)
    #[inline]
    fn map_2port_index(&self, flat_idx: usize) -> (usize, usize) {
        if self.use_21_12_order() {
            // 21_12: S11, S21, S12, S22
            match flat_idx {
                0 => (0, 0), // S11
                1 => (1, 0), // S21
                2 => (0, 1), // S12
                3 => (1, 1), // S22
                _ => (flat_idx / 2, flat_idx % 2),
            }
        } else {
            // 12_21: row-major (S11, S12, S21, S22)
            (flat_idx / 2, flat_idx % 2)
        }
    }

    fn parse_keyword(&mut self, line: &str) -> Result<(), TouchstoneError> {
        let line_lower = line.to_lowercase();
        if line_lower.starts_with("[version]") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                self.version = parts[1].to_string();
                if self.version != "1.0" {
                    self.is_v2 = true;
                }
            }
        } else if line_lower.starts_with("[number of ports]") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // [Number, of, Ports], index 3 is nports
            if parts.len() >= 4 {
                self.nports = parts[3].parse().map_err(|_| {
                    TouchstoneError::InvalidOption("Invalid port number".to_string())
                })?;
            }
        } else if line_lower.starts_with("[two-port data order]") {
            if line_lower.contains("21_12") {
                self.two_port_order_21_12 = true;
            } else if line_lower.contains("12_21") {
                self.two_port_order_21_12 = false;
            }
        } else if line_lower.starts_with("[reference]") {
            // Parse reference impedances [Reference] 50 75 ...
            // Skip first part "[reference]"
            let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
            self.z0.clear();
            for part in parts {
                if let Ok(val) = part.parse::<f64>() {
                    self.z0.push(val);
                }
            }
            if self.z0.len() < self.nports {
                self.expecting_reference = true;
            }
        } else if line_lower.starts_with("[matrix format]") {
            if line_lower.contains("lower") {
                self.matrix_format = MatrixFormat::Lower;
            } else if line_lower.contains("upper") {
                self.matrix_format = MatrixFormat::Upper;
            } else {
                self.matrix_format = MatrixFormat::Full;
            }
        } else if line_lower.starts_with("[number of frequencies]") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(n) = parts[3].parse() {
                    self.expected_nfreq = Some(n);
                    self.frequencies.reserve(n);
                    self.s_data.reserve(n);
                }
            }
        } else if line_lower.starts_with("[mixed-mode order]") {
            let parts: Vec<&str> = line.split_whitespace().skip(2).collect();
            // Keyword is [Mixed-Mode, Order], so skip 2
            for part in parts {
                self.mixed_mode_order.push(part.to_string());
            }
            if self.mixed_mode_order.len() < self.nports {
                self.expecting_mixed_mode_order = true;
            }
        } else if line_lower.starts_with("[network data]") {
            self.data_section_started = true;
        } else if line_lower.starts_with("[noise data]") {
            // Stop processing network data
            self.data_section_started = false;
            self.noise_data_encountered = true;
        } else if line_lower.starts_with("[end]") {
            // Stop parsing
            self.data_section_started = false;
            self.noise_data_encountered = true; // effectively end
        }
        Ok(())
    }

    fn check_expecting_reference(&mut self, line: &str) {
        if self.expecting_reference {
            let parts: Vec<&str> = line.split_whitespace().collect();
            for part in parts {
                if let Ok(val) = part.parse::<f64>() {
                    self.z0.push(val);
                }
            }
            if self.z0.len() >= self.nports && self.nports > 0 {
                self.expecting_reference = false;
            }
        }
    }

    fn check_expecting_mixed_mode_order(&mut self, line: &str) {
        if self.expecting_mixed_mode_order {
            let parts: Vec<&str> = line.split_whitespace().collect();
            for part in parts {
                self.mixed_mode_order.push(part.to_string());
            }
            if self.mixed_mode_order.len() >= self.nports && self.nports > 0 {
                self.expecting_mixed_mode_order = false;
            }
        }
    }

    fn parse_option_line(&mut self, line: &str) -> Result<(), TouchstoneError> {
        let (u, f, z, p) = Touchstone::parse_option_line(line, self.is_v2)?;
        self.freq_unit = u;
        self.format = f;
        self.param_type = p;
        if self.z0.is_empty() {
            // Don't override if [Reference] was set (though [Reference] usually comes after option line in V2)
            self.z0 = vec![z];
        }
        self.option_parsed = true;
        Ok(())
    }

    fn is_data_section(&self) -> bool {
        // V1: implicit data section if option parsed or no option line yet (some files omit it)
        // V2: must be after [Network Data]
        if self.is_v2 {
            self.data_section_started
        } else {
            true // For V1, we assume everything that looks like numbers is data
        }
    }

    fn parse_data_line(&mut self, line: &str) -> Result<(), TouchstoneError> {
        // Strip comments from data line if any
        let clean_line = if let Some(idx) = line.find('!') {
            &line[..idx]
        } else {
            line
        };

        let parts: Vec<&str> = clean_line.split_whitespace().collect();
        for part in parts {
            if let Ok(val) = part.parse::<f64>() {
                self.current_freq_data.push(val);
            }
        }

        self.process_buffer()
    }

    #[allow(clippy::needless_range_loop)]
    fn process_buffer(&mut self) -> Result<(), TouchstoneError> {
        let n_values = match self.matrix_format {
            MatrixFormat::Full => self.nports * self.nports,
            MatrixFormat::Lower | MatrixFormat::Upper => (self.nports * (self.nports + 1)) / 2,
        };
        let expected_values = 1 + n_values * 2;

        if self.current_freq_data.len() >= expected_values {
            // Extract one frequency point
            let freq = self.current_freq_data[0] * self.freq_unit.multiplier();

            // In Touchstone v1, a frequency decrease or equality usually indicates
            // the transition from network data to noise data.
            if !self.is_v2
                && !self.frequencies.is_empty()
                && freq <= *self.frequencies.last().unwrap()
            {
                self.noise_data_encountered = true;
                self.current_freq_data.clear();
                return Ok(());
            }

            self.frequencies.push(freq);

            let mut s_matrix = vec![vec![Complex64::new(0.0, 0.0); self.nports]; self.nports];
            let mut idx = 1;

            match self.matrix_format {
                MatrixFormat::Full => {
                    for i in 0..self.nports {
                        for j in 0..self.nports {
                            let (v1, v2) =
                                (self.current_freq_data[idx], self.current_freq_data[idx + 1]);
                            idx += 2;
                            let c = self.parse_complex_val(v1, v2);

                            // Use helper for 2-port ordering, pass through for n-port
                            let (r, c_idx) = if self.nports == 2 {
                                self.map_2port_index(i * 2 + j)
                            } else {
                                (i, j)
                            };
                            s_matrix[r][c_idx] = c;
                        }
                    }
                }
                MatrixFormat::Lower => {
                    for i in 0..self.nports {
                        for j in 0..=i {
                            let (v1, v2) =
                                (self.current_freq_data[idx], self.current_freq_data[idx + 1]);
                            idx += 2;
                            let c = self.parse_complex_val(v1, v2);
                            s_matrix[i][j] = c;
                            if i != j {
                                s_matrix[j][i] = c;
                            }
                        }
                    }
                }
                MatrixFormat::Upper => {
                    for i in 0..self.nports {
                        for j in i..self.nports {
                            let (v1, v2) =
                                (self.current_freq_data[idx], self.current_freq_data[idx + 1]);
                            idx += 2;
                            let c = self.parse_complex_val(v1, v2);
                            s_matrix[i][j] = c;
                            if i != j {
                                s_matrix[j][i] = c;
                            }
                        }
                    }
                }
            }

            self.s_data.push(s_matrix);

            // Remove consumed data, keep remainder
            self.current_freq_data.drain(0..expected_values);

            // If we have enough for another point
            if self.current_freq_data.len() >= expected_values {
                self.process_buffer()?;
            }
        }
        Ok(())
    }

    fn parse_complex_val(&self, v1: f64, v2: f64) -> Complex64 {
        match self.format {
            SParamFormat::RI => Complex64::new(v1, v2),
            SParamFormat::MA => {
                let rad = v2 * std::f64::consts::PI / 180.0;
                Complex64::from_polar(v1, rad)
            }
            SParamFormat::DB => {
                let mag = 10.0_f64.powf(v1 / 20.0);
                let rad = v2 * std::f64::consts::PI / 180.0;
                Complex64::from_polar(mag, rad)
            }
        }
    }

    fn finalize(self) -> Result<Touchstone, TouchstoneError> {
        let frequency = Frequency::from_f(
            self.frequencies
                .iter()
                .map(|&f| f / self.freq_unit.multiplier())
                .collect(),
            self.freq_unit,
        );

        // If z0 vector is not fully populated (e.g. only global z0 set), fill it
        let z0 = if self.z0.len() == self.nports {
            self.z0
        } else if !self.z0.is_empty() {
            // Use first value for all ports
            vec![self.z0[0]; self.nports]
        } else {
            // Default 50 Ohm
            vec![50.0; self.nports]
        };

        Ok(Touchstone {
            nports: self.nports,
            frequency,
            s: self.s_data,
            z0,
            comments: self.comments,
            format: self.format,
            param_type: self.param_type,
            is_v2: self.is_v2,
            mixed_mode_order: self.mixed_mode_order,
            noisy: self.noise_data_encountered,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_extension() {
        assert_eq!(Touchstone::parse_extension("s1p").unwrap(), 1);
        assert_eq!(Touchstone::parse_extension("s2p").unwrap(), 2);
        assert_eq!(Touchstone::parse_extension("S4P").unwrap(), 4);
        assert_eq!(Touchstone::parse_extension("s32p").unwrap(), 32);
    }

    #[test]
    fn test_parse_option_line() {
        let (unit, format, z0, param_type) =
            Touchstone::parse_option_line("# GHz S RI R 50", false).unwrap();
        assert_eq!(unit, FrequencyUnit::GHz);
        assert_eq!(format, SParamFormat::RI);
        assert_eq!(z0, 50.0);
        assert_eq!(param_type, ParameterType::S);

        let (unit, format, z0, param_type) =
            Touchstone::parse_option_line("# MHz S MA R 75", false).unwrap();
        assert_eq!(unit, FrequencyUnit::MHz);
        assert_eq!(format, SParamFormat::MA);
        assert_eq!(z0, 75.0);
        assert_eq!(param_type, ParameterType::S);
    }

    #[test]
    fn test_sparam_format_from_str() {
        assert_eq!(SParamFormat::parse("RI"), Some(SParamFormat::RI));
        assert_eq!(SParamFormat::parse("ma"), Some(SParamFormat::MA));
        assert_eq!(SParamFormat::parse("DB"), Some(SParamFormat::DB));
        assert_eq!(SParamFormat::parse("invalid"), None);
    }

    #[test]
    fn test_parse_v2_keywords() {
        let content = "[Version] 2.0
# MHz S RI R 50
[Number of Ports] 2
[Number of Frequencies] 1
[Reference] 50 
 75
[Mixed-Mode Order] D1,2
 C1,2
[Network Data]
1.0 0.1 0.0 0.0 0.1 0.0 0.1 0.1 0.0";
        let ts = Touchstone::from_str(content, 0).unwrap();
        assert!(ts.is_v2);
        assert_eq!(ts.nports, 2);
        assert_eq!(ts.z0, vec![50.0, 75.0]);
        assert_eq!(ts.mixed_mode_order, vec!["D1,2", "C1,2"]);
    }
}
