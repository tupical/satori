//! Local error type.
//!
//! Replaces `taskagent_shared::CoreError` so the skeleton is
//! dependency-free. mcpbox maps [`SensemakingError`] onto its own error
//! surface when wiring the layer.

use crate::ai::AiError;
use std::fmt;

#[derive(Debug)]
pub enum SensemakingError {
    /// AI provider failed or returned an unusable response.
    Ai(String),
    /// (De)serialization failure.
    Serde(String),
    /// Output failed validation (missing or invalid fields).
    Validation(String),
}

impl SensemakingError {
    pub fn ai(msg: impl Into<String>) -> Self {
        Self::Ai(msg.into())
    }
    pub fn serde(msg: impl Into<String>) -> Self {
        Self::Serde(msg.into())
    }
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }
}

impl fmt::Display for SensemakingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ai(m) => write!(f, "ai: {m}"),
            Self::Serde(m) => write!(f, "serde: {m}"),
            Self::Validation(m) => write!(f, "validation: {m}"),
        }
    }
}

impl std::error::Error for SensemakingError {}

impl From<AiError> for SensemakingError {
    fn from(e: AiError) -> Self {
        Self::Ai(e.to_string())
    }
}
