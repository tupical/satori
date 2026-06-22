//! Sensemaking prompt catalogue + local rendering.
//!
//! The skeleton owns both the prompt *content* (`prompts/*.toml`) and a
//! small rendering engine over [`tinytemplate`], so it needs no shared
//! infrastructure crate. All sources are baked in via `include_str!`; the
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

use crate::error::SensemakingError;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tinytemplate::{format_unescaped, TinyTemplate};

#[derive(Debug, Deserialize)]
struct PromptVariant {
    template: String,
}

/// One parsed `prompts/*.toml` file: a set of named template variants.
/// The `[meta]` table is ignored (unknown fields are dropped by serde).
#[derive(Debug, Deserialize)]
struct PromptFile {
    #[serde(default)]
    variants: HashMap<String, PromptVariant>,
}

struct Registry(HashMap<&'static str, PromptFile>);

impl Registry {
    fn new(raw: &[(&'static str, &'static str)]) -> Self {
        let mut map = HashMap::new();
        for (name, body) in raw {
            let file: PromptFile =
                toml::from_str(body).unwrap_or_else(|e| panic!("prompt {name}: bad toml: {e}"));
            map.insert(*name, file);
        }
        Self(map)
    }

    fn load<P: Serialize>(
        &self,
        name: &str,
        variant: &str,
        params: &P,
    ) -> Result<String, SensemakingError> {
        let file = self
            .0
            .get(name)
            .ok_or_else(|| SensemakingError::ai(format!("prompt {name}: not found")))?;
        let template = file
            .variants
            .get(variant)
            .ok_or_else(|| SensemakingError::ai(format!("prompt {name}/{variant}: unknown variant")))?;

        let label = format!("{name}/{variant}");
        let mut tt = TinyTemplate::new();
        // Untrusted content carries `<`/`>`; do not HTML-escape it.
        tt.set_default_formatter(&format_unescaped);
        tt.add_template(&label, &template.template)
            .map_err(|e| SensemakingError::ai(format!("prompt {label}: bad template: {e}")))?;
        tt.render(&label, params)
            .map_err(|e| SensemakingError::ai(format!("prompt {label}: render failed: {e}")))
    }

    #[cfg(test)]
    fn iter(&self) -> impl Iterator<Item = (&'static str, &PromptFile)> {
        self.0.iter().map(|(k, v)| (*k, v))
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

static PROMPTS: Lazy<Registry> =
    Lazy::new(|| Registry::new(&[("research", include_str!("../prompts/research.toml"))]));

/// Process-wide catalogue of sensemaking prompts. All sources are baked
/// into the binary via `include_str!`; the first `load` call parses them.
pub struct PromptRegistry;

impl PromptRegistry {
    /// Render `name` / `variant` against `params`.
    pub fn load<P: Serialize>(name: &str, variant: &str, params: &P) -> Result<String, SensemakingError> {
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

    #[test]
    fn research_default_substitutes_query() {
        #[derive(Serialize)]
        struct Ctx<'a> {
            query: &'a str,
        }
        let s = PromptRegistry::load("research", "default", &Ctx { query: "why?" }).unwrap();
        assert!(s.contains("why?"), "{s}");
    }
}
