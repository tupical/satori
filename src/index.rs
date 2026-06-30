//! Search/index seams for the Sensemaking layer.
//!
//! The knowledge operations in this crate — lesson recall, semantic search,
//! and impact analysis — are **pure** in the I/O sense: they own the query
//! shaping and result clamping, but delegate the actual lookup to a seam the
//! host implements. This mirrors the [`crate::ai::AiProvider`] pattern: no
//! daruma storage type ever leaks into the skeleton.
//!
//! Two seams are exposed because the two access patterns are genuinely
//! different:
//!
//! * [`SearchIndex`] — text / semantic lookup (FTS today, embeddings later).
//!   Backs both [`crate::recall::semantic_search`] and
//!   [`crate::recall::lesson_recall`] (lessons are just a prefixed search).
//! * [`ImpactGraph`] — downstream graph traversal. Backs
//!   [`crate::recall::impact`]; it answers "what breaks if I change this node".

use std::fmt;

// ── Errors ────────────────────────────────────────────────────────────────

/// Error raised by a [`SearchIndex`] or [`ImpactGraph`] implementation.
#[derive(Debug, Clone)]
pub struct IndexError(pub String);

impl IndexError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for IndexError {}

// ── Hit type ────────────────────────────────────────────────────────────────

/// One match returned by a search index or impact traversal.
///
/// Kept deliberately string-typed and storage-agnostic: the host maps its
/// own node / comment rows onto this struct, so no daruma id type leaks in.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct IndexHit {
    /// Opaque node / comment identifier (e.g. `task:tsk_…`, `cmt_…`).
    pub id: String,
    /// Node kind (`task`, `plan`, `document`, `comment`, …).
    pub kind: String,
    /// Short human-readable title or label.
    pub title: String,
    /// Matched body excerpt (may be empty when not applicable).
    pub snippet: String,
    /// Relevance / proximity score; higher is closer. `0.0` when unranked.
    pub score: f32,
}

impl IndexHit {
    pub fn new(
        id: impl Into<String>,
        kind: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            title: title.into(),
            snippet: String::new(),
            score: 0.0,
        }
    }

    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.snippet = snippet.into();
        self
    }

    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }
}

// ── Seams ────────────────────────────────────────────────────────────────

/// Text / semantic search seam.
///
/// Implemented in the host over daruma's storage (FTS today; an embedding
/// index later). The knowledge operations are generic over this trait, so no
/// concrete index ever leaks into the skeleton.
#[allow(async_fn_in_trait)]
pub trait SearchIndex: Send + Sync {
    /// Return up to `limit` matches for `query`, best first.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<IndexHit>, IndexError>;
}

/// Downstream-impact graph seam.
///
/// Given a node id, return the nodes that would be affected by a change to
/// it (downstream dependents through blocking / containment / ownership
/// edges). The traversal policy lives in the host; this crate only shapes
/// the request and clamps the result.
#[allow(async_fn_in_trait)]
pub trait ImpactGraph: Send + Sync {
    /// Return up to `limit` downstream dependents of `node_id`.
    async fn downstream(&self, node_id: &str, limit: usize) -> Result<Vec<IndexHit>, IndexError>;
}
