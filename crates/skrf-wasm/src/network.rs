//! WASM bindings for Network class

use js_sys::Float64Array;
use skrf_core::network::Network;
use wasm_bindgen::prelude::*;

/// N-port RF network for WASM
#[wasm_bindgen]
pub struct WasmNetwork {
    inner: Network,
}

/// Helper function to extract number of ports from filename extension
fn extract_nports_from_filename(filename: &str) -> Option<usize> {
    let ext = filename.rsplit('.').next()?.to_lowercase();
    if ext.starts_with('s') && ext.ends_with('p') {
        let num_str = &ext[1..ext.len() - 1];
        num_str.parse().ok()
    } else if ext == "ts" {
        // Touchstone 2.0 - will be determined from content
        Some(0)
    } else {
        None
    }
}

#[wasm_bindgen]
impl WasmNetwork {
    /// Load a Network from Touchstone file content
    ///
    /// Note: In WASM, we can't read files directly. Pass the file content as a string.
    ///
    /// @param content - The Touchstone file content as string
    /// @param filename - Original filename (e.g., "test.s2p") used to determine port count
    #[wasm_bindgen(js_name = fromTouchstoneContent)]
    pub fn from_touchstone_content(content: &str, filename: &str) -> Result<WasmNetwork, JsValue> {
        // Extract number of ports from filename extension
        let nports = extract_nports_from_filename(filename).ok_or_else(|| {
            JsValue::from_str("Invalid filename: expected .sNp extension (e.g., test.s2p)")
        })?;

        let network = Network::from_touchstone_content(content, nports)
            .map_err(|e| JsValue::from_str(&format!("Touchstone parse error: {}", e)))?;

        Ok(WasmNetwork { inner: network })
    }

    /// Number of ports
    #[wasm_bindgen(getter)]
    pub fn nports(&self) -> usize {
        self.inner.nports()
    }

    /// Number of frequency points
    #[wasm_bindgen(getter)]
    pub fn nfreq(&self) -> usize {
        self.inner.nfreq()
    }

    /// Network name
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// Get frequency array in Hz
    #[wasm_bindgen(getter)]
    pub fn f(&self) -> Float64Array {
        Float64Array::from(self.inner.f())
    }

    /// Get S-parameters magnitude in dB as flattened Float64Array
    /// Shape: [nfreq * nports * nports] in row-major order
    #[wasm_bindgen(js_name = getSDb)]
    pub fn get_s_db(&self) -> Float64Array {
        let s_db = self.inner.s_db();
        let flat: Vec<f64> = s_db.iter().cloned().collect();
        Float64Array::from(flat.as_slice())
    }

    /// Get S-parameters magnitude (linear) as flattened Float64Array
    #[wasm_bindgen(js_name = getSMag)]
    pub fn get_s_mag(&self) -> Float64Array {
        let s_mag = self.inner.s_mag();
        let flat: Vec<f64> = s_mag.iter().cloned().collect();
        Float64Array::from(flat.as_slice())
    }

    /// Get S-parameters phase in degrees as flattened Float64Array
    #[wasm_bindgen(js_name = getSDeg)]
    pub fn get_s_deg(&self) -> Float64Array {
        let s_deg = self.inner.s_deg();
        let flat: Vec<f64> = s_deg.iter().cloned().collect();
        Float64Array::from(flat.as_slice())
    }

    /// Get S-parameters real part as flattened Float64Array
    #[wasm_bindgen(js_name = getSRe)]
    pub fn get_s_re(&self) -> Float64Array {
        let s_re = self.inner.s_re();
        let flat: Vec<f64> = s_re.iter().cloned().collect();
        Float64Array::from(flat.as_slice())
    }

    /// Get S-parameters imaginary part as flattened Float64Array
    #[wasm_bindgen(js_name = getSIm)]
    pub fn get_s_im(&self) -> Float64Array {
        let s_im = self.inner.s_im();
        let flat: Vec<f64> = s_im.iter().cloned().collect();
        Float64Array::from(flat.as_slice())
    }

    /// Get a specific S-parameter in dB as Float64Array
    /// @param i - Row index (0-based)
    /// @param j - Column index (0-based)
    #[wasm_bindgen(js_name = getSDbAt)]
    pub fn get_s_db_at(&self, i: usize, j: usize) -> Float64Array {
        let s_db = self.inner.s_db();
        let nfreq = self.inner.nfreq();
        let mut result = Vec::with_capacity(nfreq);
        for f in 0..nfreq {
            result.push(s_db[[f, i, j]]);
        }
        Float64Array::from(result.as_slice())
    }

    /// Check if network is reciprocal
    #[wasm_bindgen(js_name = isReciprocal)]
    pub fn is_reciprocal(&self, tol: Option<f64>) -> bool {
        self.inner.is_reciprocal(tol)
    }

    /// Check if network is passive
    #[wasm_bindgen(js_name = isPassive)]
    pub fn is_passive(&self, tol: Option<f64>) -> bool {
        self.inner.is_passive(tol)
    }

    /// Check if network is lossless
    #[wasm_bindgen(js_name = isLossless)]
    pub fn is_lossless(&self, tol: Option<f64>) -> bool {
        self.inner.is_lossless(tol)
    }

    /// Check if network is symmetric
    #[wasm_bindgen(js_name = isSymmetric)]
    pub fn is_symmetric(&self, tol: Option<f64>) -> bool {
        self.inner.is_symmetric(tol)
    }
}

impl WasmNetwork {
    /// Get the inner Network reference
    pub fn inner(&self) -> &Network {
        &self.inner
    }

    /// Create from existing Network
    pub fn from_network(network: Network) -> Self {
        Self { inner: network }
    }
}
