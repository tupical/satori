//! Sensemaking prompt catalogue.
//!
//! The prompt *rendering engine* and the shared [`SharedRegistry`] live in
//! `taskagent-ai-infra`. This module only declares the catalogue of
//! sensemaking-operation prompts — one `prompts/*.toml` per operation
//! (research) — because those prompts are operational, not infrastructure.
//!
//! All known prompts are baked into the binary via `include_str!`; the
//! first [`PromptRegistry::load`] call parses them.
//!
//! ```ignore
//! use serde::Serialize;
//! use sensemaking_oss::prompts::PromptRegistry;
//!
//! #[derive(Serialize)]
//! struct ResearchCtx<'a> { query: &'a str }
//!
//! let s = PromptRegistry::load("research", "default", &ResearchCtx { query: "why?" })?;
//! ```

use once_cell::sync::Lazy;
use serde::Serialize;
use taskagent_ai_infra::prompts::PromptRegistry as SharedRegistry;
use taskagent_shared::CoreError;

static PROMPTS: Lazy<SharedRegistry> =
    Lazy::new(|| SharedRegistry::new(&[("research", include_str!("../prompts/research.toml"))]));

/// Process-wide catalogue of sensemaking prompts. All sources are baked
/// into the binary via `include_str!`; the first `load` call parses them.
pub struct PromptRegistry;

impl PromptRegistry {
    /// Render `name` / `variant` against `params`. See
    /// [`SharedRegistry::load`] for error semantics.
    pub fn load<P: Serialize>(name: &str, variant: &str, params: &P) -> Result<String, CoreError> {
        PROMPTS.load(name, variant, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_prompt_loads() {
        for (name, _file) in PROMPTS.iter() {
            assert!(!name.is_empty());
        }
        assert!(!PROMPTS.is_empty(), "no prompts loaded");
    }
}
