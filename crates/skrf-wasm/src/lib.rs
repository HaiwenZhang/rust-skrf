//! WASM bindings for skrf-core
//!
//! This module exposes skrf-core functionality to JavaScript/WASM.

use wasm_bindgen::prelude::*;

mod frequency;
mod network;
mod vector_fitting;

pub use frequency::WasmFrequency;
pub use network::WasmNetwork;
pub use vector_fitting::WasmVectorFitting;

/// Get the library version
#[wasm_bindgen]
pub fn version() -> String {
    "0.1.0".to_string()
}
