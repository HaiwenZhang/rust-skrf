//! skrf-core: Core RF/Microwave engineering library
//!
//! Rust implementation of scikit-rf (skrf) core functionality.
//!
//! ## Modules
//!
//! - `frequency` - Frequency band representation
//! - `math` - Mathematical functions (conversions, S-parameter transforms)
//! - `touchstone` - Touchstone file I/O
//! - `network` - N-port network representation

pub mod frequency;
pub mod math;
pub mod touchstone;
pub mod network;

pub use frequency::Frequency;
pub use network::Network;
