//! Knowledge operations: lesson recall, semantic search, and impact.
//!
//! These three operations moved out of the daruma execution core (where they
//! lived as the `lesson_recall` MCP tool and the WorkspaceGraph `search` /
//! `impact` route handlers) into the Sensemaking layer, because they are
//! *knowledge* concerns — recalling past lessons, finding semantically related
//! material, and reasoning about behavioral impact — not strict execution.
//!
//! Each operation is **pure** in the I/O sense: it shapes the query and clamps
//! the result, delegating the actual lookup to a host-supplied seam
//! ([`SearchIndex`] / [`ImpactGraph`]). No daruma storage type leaks in.
//!
//! The *structural* WorkspaceGraph navigation (status / context / related,
//! the nodes-and-edges lineage) stays in the execution core — it is not a
//! sensemaking concern and is intentionally absent here.

use crate::index::{ImpactGraph, IndexError, IndexHit, SearchIndex};

/// Comment-body marker that distinguishes a lesson from an ordinary comment.
///
/// `lesson_recall` reproduces the core behaviour: it recalls comments whose
/// body starts with this prefix.
pub const LESSON_PREFIX: &str = "lesson:";

/// Default result cap, matching the core route's `default_limit`.
pub const DEFAULT_LIMIT: usize = 20;
/// Hard upper bound on any result set, matching the core route's clamp.
pub const MAX_LIMIT: usize = 200;

/// Clamp a caller-supplied limit into `[1, MAX_LIMIT]`, treating `0` as the
/// default. Mirrors the core route's `default_limit` semantics.
fn clamp_limit(limit: usize) -> usize {
    if limit == 0 {
        DEFAULT_LIMIT
    } else {
        limit.clamp(1, MAX_LIMIT)
    }
}

/// Recall lesson comments.
///
/// Searches the comment index for bodies starting with [`LESSON_PREFIX`];
/// an optional `query` narrows the match (the host's index decides how to
/// interpret the combined `"lesson: <query>"` term — FTS, prefix, or
/// embedding). This is the Sensemaking-layer home of the former core
/// `daruma_lesson_recall` MCP tool.
pub async fn lesson_recall<I: SearchIndex>(
    index: &I,
    query: Option<&str>,
    limit: usize,
) -> Result<Vec<IndexHit>, IndexError> {
    let term = match query.map(str::trim).filter(|s| !s.is_empty()) {
        Some(narrow) => format!("{LESSON_PREFIX} {narrow}"),
        None => LESSON_PREFIX.to_string(),
    };
    index.search(&term, clamp_limit(limit)).await
}

/// Semantic (full-text → embeddings) search over the workspace index.
///
/// The Sensemaking-layer home of the former core WorkspaceGraph `search`
/// route / `daruma_workspacegraph_search` MCP tool. Returns an error when
/// the query is empty, matching the core handler's validation.
pub async fn semantic_search<I: SearchIndex>(
    index: &I,
    query: &str,
    limit: usize,
) -> Result<Vec<IndexHit>, IndexError> {
    let q = query.trim();
    if q.is_empty() {
        return Err(IndexError::new("semantic_search: query must not be empty"));
    }
    index.search(q, clamp_limit(limit)).await
}

/// Behavioral impact analysis: downstream nodes affected by a change to
/// `node_id`.
///
/// The Sensemaking-layer home of the former core WorkspaceGraph `impact`
/// route / `daruma_workspacegraph_impact` MCP tool. Returns an error when
/// `node_id` is empty.
pub async fn impact<G: ImpactGraph>(
    graph: &G,
    node_id: &str,
    limit: usize,
) -> Result<Vec<IndexHit>, IndexError> {
    let id = node_id.trim();
    if id.is_empty() {
        return Err(IndexError::new("impact: node_id must not be empty"));
    }
    graph.downstream(id, clamp_limit(limit)).await
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Fake search index that records the last query it received and replays
    /// a fixed hit list (clamped to the requested limit).
    struct FakeIndex {
        last_query: Mutex<String>,
        hits: Vec<IndexHit>,
    }

    impl FakeIndex {
        fn new(hits: Vec<IndexHit>) -> Self {
            Self {
                last_query: Mutex::new(String::new()),
                hits,
            }
        }
        fn query(&self) -> String {
            self.last_query.lock().unwrap().clone()
        }
    }

    impl SearchIndex for FakeIndex {
        async fn search(
            &self,
            query: &str,
            limit: usize,
        ) -> Result<Vec<IndexHit>, IndexError> {
            *self.last_query.lock().unwrap() = query.to_string();
            Ok(self.hits.iter().take(limit).cloned().collect())
        }
    }

    struct FakeGraph {
        last_node: Mutex<String>,
        last_limit: Mutex<usize>,
        hits: Vec<IndexHit>,
    }

    impl FakeGraph {
        fn new(hits: Vec<IndexHit>) -> Self {
            Self {
                last_node: Mutex::new(String::new()),
                last_limit: Mutex::new(0),
                hits,
            }
        }
        fn node(&self) -> String {
            self.last_node.lock().unwrap().clone()
        }
        fn limit(&self) -> usize {
            *self.last_limit.lock().unwrap()
        }
    }

    impl ImpactGraph for FakeGraph {
        async fn downstream(
            &self,
            node_id: &str,
            limit: usize,
        ) -> Result<Vec<IndexHit>, IndexError> {
            *self.last_node.lock().unwrap() = node_id.to_string();
            *self.last_limit.lock().unwrap() = limit;
            Ok(self.hits.iter().take(limit).cloned().collect())
        }
    }

    fn hit(id: &str) -> IndexHit {
        IndexHit::new(id, "comment", "t")
    }

    #[tokio::test]
    async fn lesson_recall_without_query_uses_bare_prefix() {
        let idx = FakeIndex::new(vec![hit("cmt_1")]);
        let out = lesson_recall(&idx, None, 10).await.unwrap();
        assert_eq!(idx.query(), "lesson:");
        assert_eq!(out.len(), 1);
    }

    #[tokio::test]
    async fn lesson_recall_with_query_appends_narrowing_term() {
        let idx = FakeIndex::new(vec![]);
        lesson_recall(&idx, Some("  oauth  "), 10).await.unwrap();
        assert_eq!(idx.query(), "lesson: oauth");
    }

    #[tokio::test]
    async fn lesson_recall_blank_query_falls_back_to_prefix() {
        let idx = FakeIndex::new(vec![]);
        lesson_recall(&idx, Some("   "), 10).await.unwrap();
        assert_eq!(idx.query(), "lesson:");
    }

    #[tokio::test]
    async fn semantic_search_trims_and_forwards_query() {
        let idx = FakeIndex::new(vec![hit("task:tsk_1"), hit("task:tsk_2")]);
        let out = semantic_search(&idx, "  rotate tokens  ", 1).await.unwrap();
        assert_eq!(idx.query(), "rotate tokens");
        // limit clamps the replayed hit list.
        assert_eq!(out.len(), 1);
    }

    #[tokio::test]
    async fn semantic_search_rejects_empty_query() {
        let idx = FakeIndex::new(vec![]);
        let err = semantic_search(&idx, "   ", 10).await.unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[tokio::test]
    async fn impact_trims_node_and_clamps_zero_limit_to_default() {
        let graph = FakeGraph::new(vec![hit("plan:pln_1")]);
        let out = impact(&graph, "  task:tsk_1  ", 0).await.unwrap();
        assert_eq!(graph.node(), "task:tsk_1");
        assert_eq!(graph.limit(), DEFAULT_LIMIT);
        assert_eq!(out.len(), 1);
    }

    #[tokio::test]
    async fn impact_rejects_empty_node_id() {
        let graph = FakeGraph::new(vec![]);
        let err = impact(&graph, "  ", 10).await.unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[tokio::test]
    async fn limit_is_clamped_to_max() {
        let many: Vec<IndexHit> = (0..300).map(|i| hit(&format!("cmt_{i}"))).collect();
        let idx = FakeIndex::new(many);
        let out = semantic_search(&idx, "x", 10_000).await.unwrap();
        assert_eq!(out.len(), MAX_LIMIT);
    }
}
