//! Semantic search over the workspace graph: a per-workspace embedding index,
//! hybrid FTS→vector rerank, and `RelatesTo` link suggestions.
//!
//! # Vector backend (decision, fixed here)
//!
//! **Pure-Rust sidecar file per workspace; cosine computed in Rust.** The
//! crate's dependency tree carries no sqlite / sqlite-vec / rusqlite, and the
//! skeleton's contract is "self-contained, zero native extensions", so vectors
//! are stored as a JSON blob (`save_sidecar` / `load_sidecar`) and similarity
//! is plain Rust math ([`cosine`]). sqlite-vec remains the migration target
//! for the cloud-sidecar task (index next to `data/workspaces/<id>/daruma.sqlite`)
//! — the seam below ([`EmbeddingProvider`], [`SemanticIndex`]) is backend-
//! neutral, so swapping storage does not touch the ranking code.
//!
//! # Contract
//! - Input arrives as plain structs ([`NodeInput`], [`FtsCandidate`]) — the
//!   host maps its daruma workspace-graph rows onto them; no daruma type
//!   leaks in (same rule as [`crate::profiles`]).
//! - Hybrid score: `α · fts_norm + (1 − α) · cosine`, α configurable
//!   ([`SemanticConfig::alpha`]). FTS candidates come from the caller's
//!   keyword index with their bm25 scores; this module only reranks them —
//!   it never invents candidates the FTS layer did not return. bm25 scores
//!   are normalized within the candidate batch (`score / max_score`).
//! - [`suggest_links`] only ever proposes [`SuggestedRelation::RelatesTo`].
//!   Semantic proximity is not an execution constraint: a `Blocks` edge is a
//!   human/plan decision, never a text-similarity outcome.
//! - Feature gating (the `SATORI_SEMANTIC_ENABLED` flag / SaaS cost gate) is
//!   enforced by the host before calling in — exactly like the AI provider
//!   gate in [`crate::ai`]. This module has no env access of its own.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::recall::{DEFAULT_LIMIT, MAX_LIMIT};
use crate::time::{now, Timestamp};

/// Sidecar format version. Bump on any layout change; [`load_sidecar`]
/// refuses older/newer files so a stale index is rebuilt, never misread.
pub const EMBED_VERSION: u32 = 1;

/// Default hybrid weight on the FTS leg (`α`); the vector leg gets `1 − α`.
pub const DEFAULT_ALPHA: f32 = 0.5;
/// Default minimum cosine for a [`LinkSuggestion`].
pub const DEFAULT_SUGGEST_THRESHOLD: f32 = 0.8;
/// Default cap on suggestions per call.
pub const DEFAULT_SUGGEST_LIMIT: usize = 20;

// ── Errors ────────────────────────────────────────────────────────────────

/// Error raised by semantic operations / storage.
#[derive(Debug)]
pub enum SemanticError {
    /// Embedding provider call failed or returned an unusable response.
    Provider(String),
    /// Sidecar I/O or (de)serialization failure.
    Storage(String),
    /// Sidecar `embedVersion` differs from [`EMBED_VERSION`] — reindex.
    Version { found: u32, expected: u32 },
    /// Sidecar was built with a different embedding model — reindex.
    ModelMismatch { stored: String, expected: String },
    /// Caller input failed validation.
    Validation(String),
}

impl SemanticError {
    pub fn provider(msg: impl Into<String>) -> Self {
        Self::Provider(msg.into())
    }
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage(msg.into())
    }
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(m) => write!(f, "provider: {m}"),
            Self::Storage(m) => write!(f, "storage: {m}"),
            Self::Version { found, expected } => {
                write!(f, "sidecar embedVersion {found}, expected {expected} — reindex required")
            }
            Self::ModelMismatch { stored, expected } => write!(
                f,
                "sidecar model '{stored}' differs from configured '{expected}' — reindex required"
            ),
            Self::Validation(m) => write!(f, "validation: {m}"),
        }
    }
}

impl std::error::Error for SemanticError {}

// ── Public input / output types ───────────────────────────────────────────

/// A workspace-graph node handed to the index: task, plan, or document
/// mirror. String-typed and storage-agnostic, like [`crate::TaskContext`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeInput {
    /// Opaque node identifier (e.g. `task:tsk_…`, `plan:pln_…`).
    pub id: String,
    /// Node kind (`task`, `plan`, `document`, …).
    pub kind: String,
    pub title: String,
    /// Longer text ground-truth for the embedding (may be empty).
    #[serde(default)]
    pub body: String,
}

impl NodeInput {
    pub fn new(
        id: impl Into<String>,
        kind: impl Into<String>,
        title: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            title: title.into(),
            body: body.into(),
        }
    }

    /// The text that gets embedded: title first, body on a new line.
    fn text(&self) -> String {
        let body = self.body.trim();
        if body.is_empty() {
            self.title.trim().to_string()
        } else {
            format!("{}\n{body}", self.title.trim())
        }
    }
}

/// One keyword-search hit supplied by the caller's FTS index, carrying its
/// bm25 score. The host owns the FTS layer; this module only reranks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FtsCandidate {
    pub id: String,
    pub kind: String,
    pub title: String,
    #[serde(default)]
    pub snippet: String,
    /// Raw bm25 (or equivalent) score; normalized within the batch.
    pub score: f32,
}

/// A reranked result: the blend plus both legs for explainability.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankedHit {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub snippet: String,
    /// Final score: `α · fts_norm + (1 − α) · vec_score`.
    pub score: f32,
    /// Batch-normalized FTS leg (`score / max_score`, `0.0` when degenerate).
    pub fts_score: f32,
    /// Cosine leg (`0.0` when the node is not in the vector index).
    pub vec_score: f32,
}

/// The only relation kind semantic similarity may propose. `Blocks` is
/// deliberately absent: blocking is an execution constraint decided by
/// humans and plans, never a text-proximity outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedRelation {
    RelatesTo,
}

/// A proposed `RelatesTo` edge between two indexed nodes, with the evidence
/// a reviewer needs to accept or dismiss it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkSuggestion {
    pub from_id: String,
    pub to_id: String,
    pub kind: SuggestedRelation,
    /// Cosine similarity of the two node embeddings.
    pub score: f32,
    /// Human-readable justification, including the threshold that fired.
    pub explanation: String,
}

// ── Provider seam ─────────────────────────────────────────────────────────

/// Any backend that can turn texts into embedding vectors.
///
/// Mirrors the [`crate::ai::AiProvider`] pattern: the skeleton owns the
/// operation, the host owns the concrete client (satori-server supplies an
/// OpenAI-compatible `/embeddings` implementation).
#[allow(async_fn_in_trait)]
pub trait EmbeddingProvider: Send + Sync {
    /// Model identifier, stamped into the sidecar so a model change forces
    /// an honest reindex instead of silently mixing vector spaces.
    fn model(&self) -> &str;
    /// Embed a batch of texts; returns one vector per input, same order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, SemanticError>;
}

// ── The index ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
struct IndexedNode {
    kind: String,
    title: String,
    vector: Vec<f32>,
}

/// In-memory per-workspace embedding index; persisted via the sidecar file.
///
/// Vectors live as plain `Vec<f32>`; similarity is computed in Rust. All
/// vectors in one index share one dimensionality (the model's).
#[derive(Clone, Debug)]
pub struct SemanticIndex {
    model: String,
    dim: usize,
    nodes: HashMap<String, IndexedNode>,
}

impl SemanticIndex {
    /// An empty index stamped for `model`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            dim: 0,
            nodes: HashMap::new(),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Vector dimensionality (`0` until the first upsert).
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    fn vector(&self, id: &str) -> Option<&[f32]> {
        self.nodes.get(id).map(|n| n.vector.as_slice())
    }

    /// Insert or replace one node's embedding. Rejects vectors whose
    /// dimensionality differs from the index's (a silent mix would poison
    /// every cosine in the workspace).
    pub fn upsert(
        &mut self,
        id: impl Into<String>,
        kind: impl Into<String>,
        title: impl Into<String>,
        vector: Vec<f32>,
    ) -> Result<(), SemanticError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(SemanticError::validation("semantic index: node id must not be empty"));
        }
        if vector.is_empty() {
            return Err(SemanticError::validation(format!(
                "semantic index: empty embedding for node '{id}'"
            )));
        }
        if self.dim == 0 {
            self.dim = vector.len();
        } else if vector.len() != self.dim {
            return Err(SemanticError::validation(format!(
                "semantic index: embedding dim {} for node '{id}' does not match index dim {}",
                vector.len(),
                self.dim
            )));
        }
        self.nodes.insert(
            id,
            IndexedNode {
                kind: kind.into(),
                title: title.into(),
                vector,
            },
        );
        Ok(())
    }

    /// Drop a node (e.g. the host deleted the underlying graph node).
    pub fn remove(&mut self, id: &str) -> bool {
        self.nodes.remove(id).is_some()
    }
}

// ── Pure math ─────────────────────────────────────────────────────────────

/// Cosine similarity in `[-1, 1]`; `0.0` for dim mismatch or a zero vector
/// (a missing/degenerate vector must look "unrelated", never crash rerank).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Clamp a caller limit into `[1, MAX_LIMIT]`, `0` meaning the default —
/// the same semantics as [`crate::recall`].
fn clamp_limit(limit: usize) -> usize {
    if limit == 0 {
        DEFAULT_LIMIT
    } else {
        limit.clamp(1, MAX_LIMIT)
    }
}

/// Hybrid rerank: blend each FTS candidate's normalized bm25 score with the
/// cosine of its indexed embedding against the query vector.
///
/// `alpha` (clamped to `[0, 1]`) weights the FTS leg. Candidates absent from
/// the vector index score `0.0` on the vector leg — they are down-ranked,
/// not dropped (FTS found them; the index may just lag).
pub fn rerank(
    index: &SemanticIndex,
    query_vec: &[f32],
    candidates: &[FtsCandidate],
    alpha: f32,
    limit: usize,
) -> Vec<RankedHit> {
    let alpha = alpha.clamp(0.0, 1.0);
    let max_fts = candidates
        .iter()
        .map(|c| c.score)
        .fold(0.0f32, f32::max);
    let mut out: Vec<RankedHit> = candidates
        .iter()
        .map(|c| {
            let fts_norm = if max_fts > 0.0 { c.score / max_fts } else { 0.0 };
            let vec_score = index
                .vector(&c.id)
                .map(|v| cosine(query_vec, v))
                .unwrap_or(0.0);
            RankedHit {
                id: c.id.clone(),
                kind: c.kind.clone(),
                title: c.title.clone(),
                snippet: c.snippet.clone(),
                score: alpha * fts_norm + (1.0 - alpha) * vec_score,
                fts_score: fts_norm,
                vec_score,
            }
        })
        .collect();
    // Deterministic order: score desc, id asc as the tie-break.
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    out.truncate(clamp_limit(limit));
    out
}

/// Propose `RelatesTo` edges between indexed node pairs whose cosine
/// similarity reaches `threshold`. Deterministic: pairs are scanned in
/// sorted-id order, results sorted by score desc with an id tie-break.
pub fn suggest_links(
    index: &SemanticIndex,
    threshold: f32,
    limit: usize,
) -> Vec<LinkSuggestion> {
    let mut ids: Vec<&String> = index.nodes.keys().collect();
    ids.sort();
    let mut out = Vec::new();
    for (i, a_id) in ids.iter().enumerate() {
        for b_id in &ids[i + 1..] {
            let score = cosine(
                &index.nodes[*a_id].vector,
                &index.nodes[*b_id].vector,
            );
            if score >= threshold {
                out.push(LinkSuggestion {
                    from_id: (*a_id).clone(),
                    to_id: (*b_id).clone(),
                    kind: SuggestedRelation::RelatesTo,
                    score,
                    explanation: format!(
                        "cosine similarity {score:.3} >= threshold {threshold:.3}; \
                         semantic proximity suggests relates_to only — never a blocking dependency"
                    ),
                });
            }
        }
    }
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.from_id.cmp(&b.from_id))
            .then_with(|| a.to_id.cmp(&b.to_id))
    });
    out.truncate(clamp_limit(limit));
    out
}

// ── Operations (provider-backed) ──────────────────────────────────────────

/// Embed a batch of graph nodes and upsert them into the workspace index.
/// Returns the number of nodes indexed. An empty batch is a no-op (`Ok(0)`).
pub async fn index_nodes<P: EmbeddingProvider>(
    provider: &P,
    index: &mut SemanticIndex,
    nodes: &[NodeInput],
) -> Result<usize, SemanticError> {
    if nodes.is_empty() {
        return Ok(0);
    }
    let texts: Vec<String> = nodes.iter().map(NodeInput::text).collect();
    let vectors = provider.embed(&texts).await?;
    if vectors.len() != nodes.len() {
        return Err(SemanticError::provider(format!(
            "embedding provider returned {} vectors for {} texts",
            vectors.len(),
            nodes.len()
        )));
    }
    for (node, vector) in nodes.iter().zip(vectors) {
        index.upsert(node.id.clone(), node.kind.clone(), node.title.clone(), vector)?;
    }
    Ok(nodes.len())
}

/// Hybrid semantic search: embed `query`, then rerank the caller's FTS
/// candidates against the workspace index. Mirrors
/// [`crate::recall::semantic_search`] in rejecting an empty query.
pub async fn hybrid_search<P: EmbeddingProvider>(
    provider: &P,
    index: &SemanticIndex,
    query: &str,
    candidates: &[FtsCandidate],
    alpha: f32,
    limit: usize,
) -> Result<Vec<RankedHit>, SemanticError> {
    let q = query.trim();
    if q.is_empty() {
        return Err(SemanticError::validation(
            "hybrid_search: query must not be empty",
        ));
    }
    let vectors = provider.embed(&[q.to_string()]).await?;
    let query_vec = vectors
        .into_iter()
        .next()
        .ok_or_else(|| SemanticError::provider("embedding provider returned no query vector"))?;
    Ok(rerank(index, &query_vec, candidates, alpha, limit))
}

// ── Sidecar persistence ───────────────────────────────────────────────────

/// Tunables for the semantic surface; the host builds this from its config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SemanticConfig {
    /// Hybrid FTS weight α ∈ [0, 1]; the vector leg gets `1 − α`.
    pub alpha: f32,
    /// Minimum cosine for a link suggestion.
    pub suggest_threshold: f32,
    /// Max suggestions per search call.
    pub suggest_limit: usize,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            suggest_threshold: DEFAULT_SUGGEST_THRESHOLD,
            suggest_limit: DEFAULT_SUGGEST_LIMIT,
        }
    }
}

/// Versioned on-disk sidecar layout (camelCase: the `embedVersion` field is
/// the cross-project convention shared with the neighbouring F5 task).
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SidecarFile {
    embed_version: u32,
    model: String,
    dim: usize,
    updated_at: Timestamp,
    nodes: Vec<SidecarNode>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SidecarNode {
    id: String,
    kind: String,
    title: String,
    vector: Vec<f32>,
}

/// Sidecar path for a workspace: `<dir>/<workspace>.embeddings.json`.
/// The workspace id is sanitized to `[A-Za-z0-9_-]` so an untrusted claim
/// can never escape `dir` (path traversal → `_`).
pub fn sidecar_path(dir: &Path, workspace: &str) -> PathBuf {
    let safe: String = workspace
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe = if safe.is_empty() { "_".into() } else { safe };
    dir.join(format!("{safe}.embeddings.json"))
}

/// Persist the index atomically (write-then-rename within the same dir).
pub fn save_sidecar(index: &SemanticIndex, path: &Path) -> Result<(), SemanticError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| SemanticError::storage(format!("create {}: {e}", parent.display())))?;
    }
    let file = SidecarFile {
        embed_version: EMBED_VERSION,
        model: index.model.clone(),
        dim: index.dim,
        updated_at: now(),
        nodes: index
            .nodes
            .iter()
            .map(|(id, n)| SidecarNode {
                id: id.clone(),
                kind: n.kind.clone(),
                title: n.title.clone(),
                vector: n.vector.clone(),
            })
            .collect(),
    };
    let bytes = serde_json::to_vec(&file)
        .map_err(|e| SemanticError::storage(format!("serialize sidecar: {e}")))?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)
        .map_err(|e| SemanticError::storage(format!("write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| SemanticError::storage(format!("rename to {}: {e}", path.display())))?;
    Ok(())
}

/// Load a sidecar, refusing stale formats and foreign vector spaces:
/// `embedVersion` must equal [`EMBED_VERSION`] and the stored model must
/// equal `expected_model`. Both mismatches mean "rebuild the index".
pub fn load_sidecar(path: &Path, expected_model: &str) -> Result<SemanticIndex, SemanticError> {
    let bytes = std::fs::read(path)
        .map_err(|e| SemanticError::storage(format!("read {}: {e}", path.display())))?;
    let file: SidecarFile = serde_json::from_slice(&bytes)
        .map_err(|e| SemanticError::storage(format!("parse {}: {e}", path.display())))?;
    if file.embed_version != EMBED_VERSION {
        return Err(SemanticError::Version {
            found: file.embed_version,
            expected: EMBED_VERSION,
        });
    }
    if file.model != expected_model {
        return Err(SemanticError::ModelMismatch {
            stored: file.model,
            expected: expected_model.to_string(),
        });
    }
    let mut index = SemanticIndex::new(expected_model);
    for node in file.nodes {
        index.upsert(node.id, node.kind, node.title, node.vector)?;
    }
    Ok(index)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fake provider: exact-text lookup, zeros as fallback.
    struct FakeEmbed {
        model: String,
        map: HashMap<String, Vec<f32>>,
        dim: usize,
    }

    impl FakeEmbed {
        fn new(dim: usize, entries: &[(&str, &[f32])]) -> Self {
            Self {
                model: "fake-embed-v1".into(),
                map: entries
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), v.to_vec()))
                    .collect(),
                dim,
            }
        }
    }

    impl EmbeddingProvider for FakeEmbed {
        fn model(&self) -> &str {
            &self.model
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, SemanticError> {
            Ok(texts
                .iter()
                .map(|t| {
                    self.map
                        .get(t)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; self.dim])
                })
                .collect())
        }
    }

    fn candidate(id: &str, score: f32) -> FtsCandidate {
        FtsCandidate {
            id: id.into(),
            kind: "task".into(),
            title: format!("title {id}"),
            snippet: String::new(),
            score,
        }
    }

    // ── cosine ──────────────────────────────────────────────────────────

    #[test]
    fn cosine_known_values() {
        let x = [1.0, 0.0];
        let y = [0.0, 1.0];
        assert_eq!(cosine(&x, &y), 0.0, "orthogonal");
        assert!((cosine(&x, &x) - 1.0).abs() < 1e-6, "identical");
        assert!((cosine(&[1.0, 1.0], &[-1.0, -1.0]) + 1.0).abs() < 1e-6, "opposite");
        // 3-4-5 triangle: cos = 3/5.
        assert!((cosine(&[3.0, 0.0], &[3.0, 4.0]) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn cosine_degenerate_inputs_are_zero_not_panic() {
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0, "dim mismatch");
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0, "zero norm");
        assert_eq!(cosine(&[], &[]), 0.0, "empty");
    }

    // ── rerank ──────────────────────────────────────────────────────────

    #[test]
    fn rerank_blends_fts_and_vector_legs() {
        let mut index = SemanticIndex::new("m");
        index.upsert("a", "task", "A", vec![1.0, 0.0]).unwrap();
        index.upsert("b", "task", "B", vec![0.0, 1.0]).unwrap();
        // "c" is an FTS hit with no embedding → vector leg 0.
        let candidates = vec![candidate("a", 10.0), candidate("b", 5.0), candidate("c", 1.0)];
        let out = rerank(&index, &[1.0, 0.0], &candidates, 0.5, 10);
        // fts norms: a=1.0, b=0.5, c=0.1; vec: a=1.0, b=0.0, c=0.0.
        assert_eq!(out[0].id, "a");
        assert!((out[0].score - 1.0).abs() < 1e-6);
        assert_eq!(out[1].id, "b");
        assert!((out[1].score - 0.25).abs() < 1e-6);
        assert_eq!(out[2].id, "c");
        assert!((out[2].score - 0.05).abs() < 1e-6);
        assert_eq!(out[2].vec_score, 0.0, "unindexed candidate keeps fts leg only");
    }

    #[test]
    fn rerank_alpha_zero_is_pure_vector_alpha_one_is_pure_fts() {
        let mut index = SemanticIndex::new("m");
        index.upsert("strong_vec", "task", "V", vec![1.0, 0.0]).unwrap();
        index.upsert("strong_fts", "task", "F", vec![0.0, 1.0]).unwrap();
        let candidates = vec![candidate("strong_vec", 1.0), candidate("strong_fts", 10.0)];
        let vec_only = rerank(&index, &[1.0, 0.0], &candidates, 0.0, 10);
        assert_eq!(vec_only[0].id, "strong_vec");
        let fts_only = rerank(&index, &[1.0, 0.0], &candidates, 1.0, 10);
        assert_eq!(fts_only[0].id, "strong_fts");
    }

    #[test]
    fn rerank_is_deterministic_and_clamps_limit() {
        let index = SemanticIndex::new("m");
        let candidates = vec![candidate("b", 1.0), candidate("a", 1.0), candidate("c", 1.0)];
        let out = rerank(&index, &[1.0], &candidates, 0.5, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "a", "tie broken by id asc");
        assert_eq!(out[1].id, "b");
    }

    #[test]
    fn rerank_zero_fts_scores_do_not_divide_by_zero() {
        let index = SemanticIndex::new("m");
        let out = rerank(&index, &[1.0], &[candidate("a", 0.0)], 0.5, 10);
        assert_eq!(out[0].fts_score, 0.0);
        assert_eq!(out[0].score, 0.0);
    }

    // ── suggest_links ───────────────────────────────────────────────────

    #[test]
    fn suggest_proposes_only_relates_to_with_explanation() {
        let mut index = SemanticIndex::new("m");
        index.upsert("x", "task", "X", vec![1.0, 0.0]).unwrap();
        index.upsert("y", "task", "Y", vec![0.95, 0.05]).unwrap();
        index.upsert("z", "task", "Z", vec![0.0, 1.0]).unwrap();
        let out = suggest_links(&index, 0.9, 10);
        assert_eq!(out.len(), 1, "only the close pair fires");
        let s = &out[0];
        assert_eq!(s.kind, SuggestedRelation::RelatesTo);
        assert_eq!(
            serde_json::to_value(s.kind).unwrap(),
            serde_json::json!("relates_to"),
            "wire form is relates_to — Blocks is never suggested"
        );
        assert_eq!(s.from_id, "x");
        assert_eq!(s.to_id, "y");
        assert!(s.explanation.contains("relates_to"));
        assert!(s.explanation.contains("never a blocking dependency"));
        assert!(s.score >= 0.9);
    }

    #[test]
    fn suggest_threshold_and_limit_are_honoured() {
        let mut index = SemanticIndex::new("m");
        index.upsert("a", "task", "A", vec![1.0, 0.0]).unwrap();
        index.upsert("b", "task", "B", vec![1.0, 0.0]).unwrap();
        assert!(suggest_links(&index, 1.1, 10).is_empty(), "above max cosine");
        assert_eq!(suggest_links(&index, 0.99, 10).len(), 1);
    }

    // ── provider-backed ops ─────────────────────────────────────────────

    #[tokio::test]
    async fn index_nodes_embeds_and_upserts() {
        let provider = FakeEmbed::new(2, &[("T one\nbody one", &[1.0, 0.0])]);
        let mut index = SemanticIndex::new(provider.model());
        let nodes = vec![
            NodeInput::new("n1", "task", "T one", "body one"),
            NodeInput::new("n2", "plan", "T two", ""),
        ];
        let n = index_nodes(&provider, &mut index, &nodes).await.unwrap();
        assert_eq!(n, 2);
        assert!(index.contains("n1"));
        assert!(index.contains("n2"));
        assert_eq!(index.dim(), 2);
        assert_eq!(index_nodes(&provider, &mut index, &[]).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn index_nodes_rejects_dim_mismatch() {
        let provider = FakeEmbed::new(2, &[("A", &[1.0, 0.0, 0.0])]);
        let mut index = SemanticIndex::new(provider.model());
        index.upsert("ok", "task", "ok", vec![1.0, 0.0]).unwrap();
        let err = index_nodes(&provider, &mut index, &[NodeInput::new("bad", "task", "A", "")])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("dim"), "got: {err}");
    }

    #[tokio::test]
    async fn hybrid_search_reranks_against_query_embedding() {
        let provider = FakeEmbed::new(
            2,
            &[
                ("rotate tokens", &[1.0, 0.0]),
                ("T one", &[1.0, 0.0]),
                ("T two", &[0.0, 1.0]),
            ],
        );
        let mut index = SemanticIndex::new(provider.model());
        index_nodes(
            &provider,
            &mut index,
            &[
                NodeInput::new("n1", "task", "T one", ""),
                NodeInput::new("n2", "task", "T two", ""),
            ],
        )
        .await
        .unwrap();
        let out = hybrid_search(
            &provider,
            &index,
            "rotate tokens",
            &[candidate("n1", 5.0), candidate("n2", 10.0)],
            0.5,
            10,
        )
        .await
        .unwrap();
        // n1 loses FTS (0.5 norm) but wins the vector leg (1.0 vs 0.0):
        // 0.5·0.5 + 0.5·1.0 = 0.75 beats 0.5·1.0 + 0.5·0.0 = 0.5.
        assert_eq!(out[0].id, "n1");
        assert!((out[0].score - 0.75).abs() < 1e-6);
        assert_eq!(out[1].id, "n2");
        assert!((out[1].score - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn hybrid_search_rejects_empty_query() {
        let provider = FakeEmbed::new(2, &[]);
        let index = SemanticIndex::new(provider.model());
        let err = hybrid_search(&provider, &index, "   ", &[], 0.5, 10)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    // ── sidecar storage ─────────────────────────────────────────────────

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "satori-semantic-test-{tag}-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sidecar_roundtrip_preserves_index() {
        let dir = temp_dir("roundtrip");
        let path = sidecar_path(&dir, "ws-1");
        let mut index = SemanticIndex::new("fake-embed-v1");
        index.upsert("a", "task", "Alpha", vec![1.0, 0.0]).unwrap();
        index.upsert("b", "plan", "Beta", vec![0.5, 0.5]).unwrap();
        save_sidecar(&index, &path).unwrap();
        // Format contract: camelCase embedVersion is on disk.
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(raw["embedVersion"], EMBED_VERSION);
        assert_eq!(raw["model"], "fake-embed-v1");
        assert_eq!(raw["dim"], 2);
        let loaded = load_sidecar(&path, "fake-embed-v1").unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.dim(), 2);
        assert_eq!(loaded.vector("a"), Some(&[1.0, 0.0][..]));
        assert_eq!(loaded.vector("b"), Some(&[0.5, 0.5][..]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sidecar_version_mismatch_forces_reindex() {
        let dir = temp_dir("version");
        let path = sidecar_path(&dir, "ws");
        let raw = serde_json::json!({
            "embedVersion": 999,
            "model": "m",
            "dim": 0,
            "updatedAt": crate::time::now(),
            "nodes": []
        });
        std::fs::write(&path, serde_json::to_vec(&raw).unwrap()).unwrap();
        let err = load_sidecar(&path, "m").unwrap_err();
        assert!(
            matches!(err, SemanticError::Version { found: 999, expected: EMBED_VERSION }),
            "got: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sidecar_model_mismatch_forces_reindex() {
        let dir = temp_dir("model");
        let path = sidecar_path(&dir, "ws");
        let mut index = SemanticIndex::new("old-model");
        index.upsert("a", "task", "A", vec![1.0]).unwrap();
        save_sidecar(&index, &path).unwrap();
        let err = load_sidecar(&path, "new-model").unwrap_err();
        assert!(matches!(err, SemanticError::ModelMismatch { .. }), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sidecar_path_sanitizes_workspace() {
        let dir = Path::new("/tmp/x");
        assert_eq!(
            sidecar_path(dir, "ws-1_ok"),
            PathBuf::from("/tmp/x/ws-1_ok.embeddings.json")
        );
        let p = sidecar_path(dir, "../../etc/passwd");
        assert_eq!(p.parent().unwrap(), dir, "traversal must not escape dir");
        assert_eq!(sidecar_path(dir, ""), PathBuf::from("/tmp/x/_.embeddings.json"));
    }
}
