//! Local wall-clock primitive.
//!
//! The skeleton owns its own `Timestamp` so the crate has zero dependency
//! on daruma's `shared`. the host maps to/from daruma's timestamp when
//! wiring the layer (they are the same `chrono` type, so it is a no-op).
//! Re-exported from layer-kit's shared `time` module, which every layer
//! uses so the (identical) definition lives in one place.
//!
//! Canonical UTC timestamp used across the sensemaking layer.

pub use layer_kit::time::{now, Timestamp};
