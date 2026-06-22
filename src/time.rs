//! Local wall-clock primitive.
//!
//! The skeleton owns its own `Timestamp` so the crate has zero dependency
//! on taskagent's `shared`. mcpbox maps to/from taskagent's timestamp when
//! wiring the layer (they are the same `chrono` type, so it is a no-op).

use chrono::{DateTime, Utc};

/// Canonical UTC timestamp used across the sensemaking layer.
pub type Timestamp = DateTime<Utc>;

/// Current UTC wall-clock time.
#[inline]
pub fn now() -> Timestamp {
    Utc::now()
}
