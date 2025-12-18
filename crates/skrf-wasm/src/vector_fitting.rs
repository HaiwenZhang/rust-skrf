//! WASM bindings for VectorFitting class

use js_sys::{Array, Float64Array};
use skrf_core::vector_fitting::{InitPoleSpacing, VectorFitting};
use wasm_bindgen::prelude::*;

use crate::network::WasmNetwork;

/// Vector Fitting for rational approximation - WASM binding
#[wasm_bindgen]
pub struct WasmVectorFitting {
    inner: VectorFitting,
}

#[wasm_bindgen]
impl WasmVectorFitting {
    /// Create a new VectorFitting instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmVectorFitting {
        WasmVectorFitting {
            inner: VectorFitting::new(),
        }
    }

    /// Perform vector fitting on a network
    ///
    /// @param network - The network to fit
    /// @param n_poles_real - Number of initial real poles (default: 2)
    /// @param n_poles_cmplx - Number of initial complex conjugate poles (default: 2)
    /// @param init_pole_spacing - Type of initial pole spacing ('linear' or 'log')
    /// @param fit_constant - Include constant term (default: true)
    /// @param fit_proportional - Include proportional term (default: false)
    #[wasm_bindgen(js_name = vectorFit)]
    pub fn vector_fit(
        &mut self,
        network: &WasmNetwork,
        n_poles_real: Option<usize>,
        n_poles_cmplx: Option<usize>,
        init_pole_spacing: Option<String>,
        fit_constant: Option<bool>,
        fit_proportional: Option<bool>,
    ) -> Result<(), JsValue> {
        let spacing = match init_pole_spacing
            .as_deref()
            .unwrap_or("linear")
            .to_lowercase()
            .as_str()
        {
            "linear" | "lin" => InitPoleSpacing::Linear,
            "log" | "logarithmic" => InitPoleSpacing::Logarithmic,
            s => return Err(JsValue::from_str(&format!("Invalid pole spacing: {}", s))),
        };

        self.inner
            .vector_fit(
                network.inner(),
                n_poles_real.unwrap_or(2),
                n_poles_cmplx.unwrap_or(2),
                spacing,
                fit_constant.unwrap_or(true),
                fit_proportional.unwrap_or(false),
            )
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Get the RMS error between the fitted model and original network
    ///
    /// @param network - The network that was fitted
    /// @param i - Row index (0-based)
    /// @param j - Column index (0-based)
    #[wasm_bindgen(js_name = getRmsError)]
    pub fn get_rms_error(&self, network: &WasmNetwork, i: usize, j: usize) -> Result<f64, JsValue> {
        self.inner
            .get_rms_error(network.inner(), i, j)
            .ok_or_else(|| JsValue::from_str("RMS error calculation failed"))
    }

    /// Get the model order
    #[wasm_bindgen(js_name = getModelOrder)]
    pub fn get_model_order(&self) -> Result<usize, JsValue> {
        self.inner
            .get_model_order()
            .ok_or_else(|| JsValue::from_str("Model not fitted yet"))
    }

    /// Get the model response at specified frequencies
    ///
    /// @param i - Row index (0-based)
    /// @param j - Column index (0-based)
    /// @param freqs - Frequencies in Hz (Float64Array)
    /// Returns: Object with {real: Float64Array, imag: Float64Array}
    #[wasm_bindgen(js_name = getModelResponse)]
    pub fn get_model_response(
        &self,
        i: usize,
        j: usize,
        freqs: Float64Array,
    ) -> Result<JsValue, JsValue> {
        let freqs_vec: Vec<f64> = freqs.to_vec();

        let response = self
            .inner
            .get_model_response(i, j, &freqs_vec)
            .ok_or_else(|| JsValue::from_str("Model response calculation failed"))?;

        // Return as object with real and imag arrays
        let real: Vec<f64> = response.iter().map(|c| c.re).collect();
        let imag: Vec<f64> = response.iter().map(|c| c.im).collect();

        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"real".into(), &Float64Array::from(real.as_slice()))?;
        js_sys::Reflect::set(&obj, &"imag".into(), &Float64Array::from(imag.as_slice()))?;

        Ok(obj.into())
    }

    /// Check if the fitted model is passive
    #[wasm_bindgen(js_name = isPassive)]
    pub fn is_passive(&self, network: &WasmNetwork) -> Result<bool, JsValue> {
        self.inner
            .is_passive(network.inner().nports())
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Perform passivity test
    /// Returns array of [f_start, f_stop] violation bands
    #[wasm_bindgen(js_name = passivityTest)]
    pub fn passivity_test(&self, network: &WasmNetwork) -> Result<Array, JsValue> {
        let result = self
            .inner
            .passivity_test(network.inner().nports())
            .map_err(|e| JsValue::from_str(&e))?;

        let arr = Array::new();
        for band in result.violation_bands {
            let band_arr = Array::new();
            band_arr.push(&JsValue::from_f64(band[0]));
            band_arr.push(&JsValue::from_f64(band[1]));
            arr.push(&band_arr);
        }

        Ok(arr)
    }

    /// Enforce passivity of the fitted model
    /// Returns: Object with {success: bool, iterations: number, historyMaxSigma: Float64Array}
    #[wasm_bindgen(js_name = passivityEnforce)]
    pub fn passivity_enforce(
        &mut self,
        network: &WasmNetwork,
        n_samples: Option<usize>,
    ) -> Result<JsValue, JsValue> {
        let f_max = network.inner().frequency.stop();
        let result = self
            .inner
            .passivity_enforce(network.inner().nports())
            .map_err(|e| JsValue::from_str(&e))?;

        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"success".into(), &JsValue::from_bool(result.success))?;
        js_sys::Reflect::set(
            &obj,
            &"iterations".into(),
            &JsValue::from_f64(result.iterations as f64),
        )?;
        js_sys::Reflect::set(
            &obj,
            &"historyMaxSigma".into(),
            &Float64Array::from(result.history_max_sigma.as_slice()),
        )?;

        Ok(obj.into())
    }

    /// Generate SPICE subcircuit netlist as a string
    #[wasm_bindgen(js_name = generateSpiceSubcircuit)]
    pub fn generate_spice_subcircuit(
        &self,
        network: &WasmNetwork,
        model_name: Option<String>,
        create_reference_pins: Option<bool>,
    ) -> Result<String, JsValue> {
        self.inner
            .generate_spice_subcircuit_s(
                network.inner(),
                model_name.as_deref(),
                create_reference_pins.unwrap_or(false),
            )
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Maximum iterations for fitting
    #[wasm_bindgen(getter, js_name = maxIterations)]
    pub fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    #[wasm_bindgen(setter, js_name = maxIterations)]
    pub fn set_max_iterations(&mut self, value: usize) {
        self.inner.max_iterations = value;
    }

    /// Convergence tolerance
    #[wasm_bindgen(getter, js_name = maxTol)]
    pub fn max_tol(&self) -> f64 {
        self.inner.max_tol
    }

    #[wasm_bindgen(setter, js_name = maxTol)]
    pub fn set_max_tol(&mut self, value: f64) {
        self.inner.max_tol = value;
    }

    /// Wall-clock time of last fit
    #[wasm_bindgen(getter, js_name = wallClockTime)]
    pub fn wall_clock_time(&self) -> f64 {
        self.inner.wall_clock_time
    }
}

impl Default for WasmVectorFitting {
    fn default() -> Self {
        Self::new()
    }
}
