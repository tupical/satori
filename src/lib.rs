//! `sensemaking-oss` — the Sensemaking layer extracted from the TaskAgent
//! OSS core.
//!
//! This crate owns the AI **research** operation — `research` (answer a
//! free-form query, optionally grounded in the bodies of existing tasks) —
//! on top of the provider-neutral [`taskagent_ai_infra`] infrastructure
//! ([`AiProvider`] abstraction, prompt rendering engine, prompt-injection
//! hardening).
//!
//! It is a Wave-2b sibling of `planning-oss` / `intake-oss`: the operation
//! and its prompt moved out of `taskagent/crates/ai` into this separate
//! repository, while the shared infrastructure is consumed read-only via
//! the `vendor/oss` symlink (mirroring the `mcpbox.ru` vendoring pattern).
//!
//! # Contract (inherited from the core AI layer)
//! - The sensemaking layer **never** writes to storage. [`research`]
//!   returns the answer as a plain `String`; persisting it (e.g. as a
//!   `Research` comment) is the caller's responsibility, performed through
//!   the ordinary command/comment API — it is not part of this operation.
//! - All JSON is built with [`serde_json::json!`]; no string concatenation.
//! - Errors propagate as [`taskagent_shared::CoreError`].

pub mod prompts;
pub mod research;
pub mod types;

// ── Re-export the infrastructure layer ─────────────────────────────────────────
//
// Preserves the operation crate's public surface (`AiProvider`,
// `AiConfig`, `OpenAiClient`, …) so callers depend on `sensemaking_oss::*`
// without also naming `taskagent_ai_infra`.
pub use taskagent_ai_infra::{wrap_untrusted, AiConfig, AiError, AiProvider, OpenAiClient};

// The sensemaking prompt catalogue, owned by this crate.
pub use prompts::PromptRegistry;

// ── Operation re-exports ────────────────────────────────────────────────────────

pub use research::{annotate_research_output, build_research_prompt, format_task_context, research};

// ── Sensemaking type re-exports ────────────────────────────────────────────────

pub use types::{
    Confidence, LinkKind, ReconsiderTrigger, RejectedIdea, RejectedIdeaId, SensingItem,
    SensingItemId, SensingItemKind, SensingLink, SensingTarget, Source,
};
