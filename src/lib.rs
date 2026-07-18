//! `satori` — the Sensemaking layer skeleton.
//!
//! A self-contained open-core skeleton: it defines its own primitives,
//! domain output types, and a provider-neutral [`AiProvider`] seam. It has
//! **no** dependency on daruma and **no** dependency on sibling `*_oss`
//! layers. the host supplies the concrete AI provider and any daruma
//! adapters when wiring the layer into its architecture — implementations
//! live only inside the host.
//!
//! # Contract
//! - The sensemaking layer never writes to storage. [`research`] returns
//!   the answer as a plain `String`; the caller (the host) persists it.
//! - All JSON is built with [`serde_json::json!`]; no string concatenation.
//! - Errors propagate as [`SensemakingError`].

pub mod ai;
pub mod error;
pub mod index;
pub mod profiles;
pub mod prompts;
pub mod recall;
pub mod research;
pub mod semantic;
pub mod time;
pub mod types;

// ── Seam + operation re-exports ──────────────────────────────────────────────
pub use ai::{
    research_answer_tool, wrap_untrusted, AiError, AiOutput, AiProvider, AiRequest, ToolCall,
};
pub use error::SensemakingError;
pub use index::{ImpactGraph, IndexError, IndexHit, SearchIndex};
pub use profiles::{
    mine_agent_profiles, AgentProfile, PatternLifecycle, PatternSource, ProfileReport,
    ResponsibilityPattern, UserSetOverride,
};
pub use prompts::PromptRegistry;
pub use recall::{impact, lesson_recall, semantic_search, LESSON_PREFIX};
pub use research::{
    annotate_research_output, build_research_prompt, format_task_context, research, TaskContext,
};
pub use semantic::{
    cosine, hybrid_search, index_nodes, load_sidecar, rerank, save_sidecar, sidecar_path,
    suggest_links, EmbeddingProvider, FtsCandidate, LinkSuggestion, NodeInput, RankedHit,
    SemanticConfig, SemanticError, SemanticIndex, SuggestedRelation, EMBED_VERSION,
};
pub use time::Timestamp;

// ── Sensemaking type re-exports ──────────────────────────────────────────────
pub use types::{
    Actor, Confidence, LinkKind, ReconsiderTrigger, RejectedIdea, RejectedIdeaId, SensingItem,
    SensingItemId, SensingItemKind, SensingLink, SensingTarget, Source,
};
