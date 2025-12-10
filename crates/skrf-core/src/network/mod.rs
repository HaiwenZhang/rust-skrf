//! Network module - N-port electrical network representation
//!
//! Provides the core Network struct and associated operations for
//! S-parameter and other network parameter manipulation.

mod active;
mod connect;
mod core;
mod derived;
mod interpolation;
mod io;
mod mixed_mode;
mod noise;
mod operators;
mod params;
mod properties;
mod time_domain;

pub use connect::{connect_s, innerconnect_s};
pub use core::Network;
pub use mixed_mode::MixedModeParams;
pub use noise::NoiseParams;
pub use time_domain::WindowType;
