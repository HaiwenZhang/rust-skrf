//! WASM bindings for Frequency class

use js_sys::Float64Array;
use skrf_core::frequency::{Frequency, FrequencyUnit, SweepType};
use wasm_bindgen::prelude::*;

/// Frequency band representation for WASM
#[wasm_bindgen]
pub struct WasmFrequency {
    inner: Frequency,
}

#[wasm_bindgen]
impl WasmFrequency {
    /// Create a new Frequency object
    ///
    /// @param start - Start frequency in the specified unit
    /// @param stop - Stop frequency in the specified unit
    /// @param npoints - Number of frequency points
    /// @param unit - Frequency unit ('Hz', 'kHz', 'MHz', 'GHz', 'THz')
    /// @param sweep_type - Sweep type ('linear' or 'log')
    #[wasm_bindgen(constructor)]
    pub fn new(
        start: f64,
        stop: f64,
        npoints: usize,
        unit: Option<String>,
        sweep_type: Option<String>,
    ) -> Result<WasmFrequency, JsValue> {
        let freq_unit = match unit.as_deref().unwrap_or("Hz").to_lowercase().as_str() {
            "hz" => FrequencyUnit::Hz,
            "khz" => FrequencyUnit::KHz,
            "mhz" => FrequencyUnit::MHz,
            "ghz" => FrequencyUnit::GHz,
            "thz" => FrequencyUnit::THz,
            u => return Err(JsValue::from_str(&format!("Invalid unit: {}", u))),
        };

        let sweep = match sweep_type
            .as_deref()
            .unwrap_or("linear")
            .to_lowercase()
            .as_str()
        {
            "linear" | "lin" => SweepType::Linear,
            "log" | "logarithmic" => SweepType::Log,
            s => return Err(JsValue::from_str(&format!("Invalid sweep type: {}", s))),
        };

        Ok(WasmFrequency {
            inner: Frequency::new(start, stop, npoints, freq_unit, sweep),
        })
    }

    /// Get frequency array in Hz as Float64Array
    #[wasm_bindgen(getter)]
    pub fn f(&self) -> Float64Array {
        let f = self.inner.f();
        Float64Array::from(f)
    }

    /// Get scaled frequency array as Float64Array
    #[wasm_bindgen(getter)]
    pub fn f_scaled(&self) -> Float64Array {
        let f_scaled = self.inner.f_scaled();
        Float64Array::from(f_scaled.as_slice())
    }

    /// Start frequency in Hz
    #[wasm_bindgen(getter)]
    pub fn start(&self) -> f64 {
        self.inner.start()
    }

    /// Stop frequency in Hz
    #[wasm_bindgen(getter)]
    pub fn stop(&self) -> f64 {
        self.inner.stop()
    }

    /// Number of frequency points
    #[wasm_bindgen(getter)]
    pub fn npoints(&self) -> usize {
        self.inner.npoints()
    }

    /// Center frequency in Hz
    #[wasm_bindgen(getter)]
    pub fn center(&self) -> f64 {
        self.inner.center()
    }

    /// Frequency span in Hz
    #[wasm_bindgen(getter)]
    pub fn span(&self) -> f64 {
        self.inner.span()
    }

    /// Frequency unit as string
    #[wasm_bindgen(getter)]
    pub fn unit(&self) -> String {
        match self.inner.unit() {
            FrequencyUnit::Hz => "Hz".to_string(),
            FrequencyUnit::KHz => "kHz".to_string(),
            FrequencyUnit::MHz => "MHz".to_string(),
            FrequencyUnit::GHz => "GHz".to_string(),
            FrequencyUnit::THz => "THz".to_string(),
        }
    }
}

impl WasmFrequency {
    /// Get the inner Frequency reference
    pub fn inner(&self) -> &Frequency {
        &self.inner
    }

    /// Create from existing Frequency
    pub fn from_frequency(freq: Frequency) -> Self {
        Self { inner: freq }
    }
}
