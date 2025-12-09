//! WASM bindings for skrf-core
//!
//! This module will expose skrf-core functionality to JavaScript/WASM.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet() -> String {
    "Hello from skrf-wasm!".to_string()
}
