//! Embeddings provider for satori-server: a minimal OpenAI-compatible
//! `/embeddings` client implementing the lib's [`EmbeddingProvider`] seam.
//!
//! Config (env) reuses the ai.rs contract so one key serves the whole
//! co-located platform:
//!   OPENAI_API_KEY         — required; when unset/empty `EmbeddingConfig::from_env`
//!                            returns `None` and semantic methods answer
//!                            `embeddings_not_configured` (503) instead of failing.
//!   OPENAI_BASE_URL        — default `https://api.openai.com/v1` (shared with ai.rs).
//!   OPENAI_EMBEDDING_MODEL — default `text-embedding-3-small`.

use satori::{EmbeddingProvider, SemanticError};
use serde_json::{json, Value};

/// Settings the provider needs to reach the embeddings endpoint.
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl EmbeddingConfig {
    /// Load from env; `None` when `OPENAI_API_KEY` is unset or empty.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())?;
        Some(Self {
            api_key,
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
            model: std::env::var("OPENAI_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".into()),
        })
    }

    fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.base_url)
    }
}

/// [`EmbeddingProvider`] backed by an OpenAI-compatible embeddings API.
/// Clone is cheap (the inner `reqwest::Client` is Arc-backed).
#[derive(Clone)]
pub struct OpenAiEmbeddingProvider {
    http: reqwest::Client,
    cfg: EmbeddingConfig,
}

impl OpenAiEmbeddingProvider {
    pub fn new(cfg: EmbeddingConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            cfg,
        }
    }
}

impl EmbeddingProvider for OpenAiEmbeddingProvider {
    fn model(&self) -> &str {
        &self.cfg.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, SemanticError> {
        let body = json!({ "model": self.cfg.model, "input": texts });
        let resp = self
            .http
            .post(self.cfg.embeddings_url())
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| SemanticError::provider(format!("embeddings request failed: {e}")))?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let message = resp.text().await.unwrap_or_default();
            return Err(SemanticError::provider(format!(
                "embeddings api status {status}: {message}"
            )));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| SemanticError::provider(format!("embeddings decode failed: {e}")))?;
        parse_embeddings(&body, texts.len())
    }
}

/// Parse the `data` array of an embeddings reply into vectors, restored to
/// input order via each item's `index`. Pure — unit-tested without network.
fn parse_embeddings(body: &Value, expected: usize) -> Result<Vec<Vec<f32>>, SemanticError> {
    let items = body["data"]
        .as_array()
        .ok_or_else(|| SemanticError::provider("response missing 'data' array"))?;
    if items.len() != expected {
        return Err(SemanticError::provider(format!(
            "response has {} embeddings for {expected} inputs",
            items.len()
        )));
    }
    let mut out: Vec<Option<Vec<f32>>> = vec![None; expected];
    for item in items {
        let idx = item["index"].as_u64().unwrap_or(0) as usize;
        let vector = item["embedding"]
            .as_array()
            .ok_or_else(|| SemanticError::provider("embedding item missing 'embedding'"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<_>>();
        if idx >= expected {
            return Err(SemanticError::provider(format!(
                "embedding item index {idx} out of range ({expected} inputs)"
            )));
        }
        out[idx] = Some(vector);
    }
    out.into_iter()
        .enumerate()
        .map(|(i, v)| v.ok_or_else(|| SemanticError::provider(format!("missing embedding for input {i}"))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_embeddings_restores_input_order() {
        let body = json!({"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]}
        ]});
        let out = parse_embeddings(&body, 2).unwrap();
        assert_eq!(out[0], vec![1.0, 0.0]);
        assert_eq!(out[1], vec![0.0, 1.0]);
    }

    #[test]
    fn parse_embeddings_rejects_bad_shapes() {
        assert!(parse_embeddings(&json!({"id": "x"}), 1).is_err(), "no data array");
        assert!(
            parse_embeddings(&json!({"data": [{"index": 0, "embedding": [1.0]}]}), 2).is_err(),
            "count mismatch"
        );
        assert!(
            parse_embeddings(&json!({"data": [{"index": 5, "embedding": [1.0]}]}), 1).is_err(),
            "index out of range"
        );
        assert!(
            parse_embeddings(&json!({"data": [{"index": 0}]}), 1).is_err(),
            "missing embedding"
        );
    }
}
