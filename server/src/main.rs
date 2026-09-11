//! satori-server — thin, independently-deployed HTTP/MCP wrapper around the
//! `satori` sensemaking lib. Its own deploy unit (own systemd service, own
//! port). Boundary-clean: no mcpbox dependency; the platform→tool auth
//! contract and the axum/tokio scaffold live in `layer_kit::{auth,serve}`.
//!
//! Routes:
//!   GET  /healthz   — open; liveness + version for the platform registry.
//!   POST /v1/mcp    — requires a valid platform token; sensemaking surface
//!                     (`satori.sense` and `satori.research` run the lib's AI
//!                     sensemaking operations,
//!                     `satori.profiles` process-mines agent profiles from a
//!                     daruma event stream supplied in the params).
//!
//! Env: SATORI_PORT (default 8091), SATORI_PLATFORM_SECRET (HMAC key; if
//! unset, /v1/mcp is closed), SATORI_VERSION (defaults to the crate version).
//! AI methods (`satori.sense`, `satori.research`): OPENAI_API_KEY / OPENAI_BASE_URL /
//! OPENAI_MODEL (see `layer_kit::openai`); without a key they answer
//! `ai_not_configured`. No embedding index or background indexing surface.

use axum::http::StatusCode;
use layer_kit::ai::extract_ai_config;
use layer_kit::auth::Claims;
use layer_kit::openai::{AiConfig, OpenAiProvider};
use layer_kit::serve::{serve, McpHandler, ServeConfig};
use layer_kit::store::Store;
use satori::types::{SensingItem, Source};
use serde_json::json;

const TOOL: &str = "satori";

/// Dispatches satori's MCP methods with the optional AI provider.
struct Handler {
    /// `None` when OPENAI_API_KEY is unset — AI methods then answer
    /// `ai_not_configured` instead of panicking at call time.
    ai: Option<OpenAiProvider>,
    store: Store,
}

impl McpHandler for Handler {
    async fn dispatch(
        &self,
        _claims: &Claims,
        method: &str,
        mut params: serde_json::Value,
    ) -> Result<serde_json::Value, (StatusCode, serde_json::Value)> {
        let request_ai = extract_ai_config(&mut params);
        if let Some(cfg) = request_ai {
            let provider = OpenAiProvider::new(cfg);
            dispatch_with_ai(
                &self.store,
                Some((&provider, provider.model())),
                method,
                params,
            )
            .await
        } else {
            dispatch_with_ai(
                &self.store,
                self.ai
                    .as_ref()
                    .map(|provider| (provider, provider.model())),
                method,
                params,
            )
            .await
        }
    }

    fn tools(&self) -> Vec<serde_json::Value> {
        tools()
    }
}

/// Tool descriptors for `tools/list` — one per method actually handled by
/// [`dispatch_with_ai`] (`satori.search` remains unsupported).
fn tools() -> Vec<serde_json::Value> {
    vec![
        json!({
            "name": "satori_sense",
            "description": "AI sensemaking: build a typed SensingItem from raw material.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "body": {"type": "string"},
                    "source_ref": {"type": "string"}
                },
                "required": ["body"]
            }
        }),
        json!({
            "name": "satori_recall",
            "description": "Get one persisted SensingItem by id, or list recent items.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1}
                }
            }
        }),
        json!({
            "name": "satori_research",
            "description": "AI research operation: answer a free-form query, optionally grounded in task context.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "satori_search",
            "description": "Search sensing items through a host-provided SearchIndex adapter.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "satori_profiles",
            "description": "Process-mine agent capability profiles from a daruma event stream.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "events": {"type": "array", "items": {"type": "object"}},
                    "user_set_overrides": {"type": "array", "items": {"type": "object"}},
                    "as_of": {"type": "string"}
                },
                "required": []
            }
        }),
    ]
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().json().init();

    let ai = AiConfig::from_env().map(OpenAiProvider::new);
    if ai.is_none() {
        tracing::warn!(
            "OPENAI_API_KEY unset — env-backed AI methods will answer ai_not_configured"
        );
    }
    let store = Store::from_env(TOOL).await.unwrap_or_else(|e| {
        tracing::error!(error = %e, "failed to open satori store");
        std::process::exit(1);
    });

    serve(
        ServeConfig {
            tool: TOOL,
            default_port: 8091,
            default_version: env!("CARGO_PKG_VERSION"),
            git_sha: option_env!("GIT_SHA").unwrap_or("dev"),
        },
        Handler { ai, store },
    )
    .await;
}

/// Params for `satori.sense`.
#[derive(serde::Deserialize)]
struct SenseParams {
    body: String,
    /// Optional provenance: the upstream object's id (e.g. a torii RawItem id),
    /// recorded as the sensing item's source so lineage survives the network hop.
    #[serde(default)]
    source_ref: Option<String>,
}

#[derive(serde::Deserialize)]
struct RecallParams {
    #[serde(default)]
    id: Option<String>,
    #[serde(default = "default_limit")]
    limit: i64,
}

fn default_limit() -> i64 {
    100
}

fn storage_error(e: impl std::fmt::Display) -> (StatusCode, serde_json::Value) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": "storage_error", "detail": e.to_string()}),
    )
}

/// Params for `satori.research` — the lib's AI research operation
/// ([`research`](satori::research)): answer a free-form query, optionally
/// grounded in the bodies of existing task summaries.
#[derive(serde::Deserialize)]
struct ResearchParams {
    query: String,
    /// Optional grounding context; each entry maps onto the lib's
    /// `TaskContext` (host-side task mirror).
    #[serde(default)]
    context: Vec<ResearchTaskInput>,
}

#[derive(serde::Deserialize)]
struct ResearchTaskInput {
    id: String,
    title: String,
    #[serde(default)]
    description: String,
}

/// Params for `satori.profiles` — the lib's process-mining operation
/// ([`profiles`](satori::profiles)): fold a daruma event stream (envelope
/// JSON, log order) into per-agent profiles. Stateless: every input arrives
/// in the params, nothing is stored server-side.
#[derive(serde::Deserialize)]
struct ProfilesParams {
    /// `daruma_events::EventEnvelope` JSON values (bare payloads accepted).
    #[serde(default)]
    events: Vec<serde_json::Value>,
    /// Human capability overrides from the daruma core
    /// (`agent_capability_profiles` rows with `source = 'user_set'`).
    #[serde(default)]
    user_set_overrides: Vec<satori::UserSetOverride>,
    /// Optional RFC3339 horizon closing still-open blocked intervals.
    #[serde(default)]
    as_of: Option<satori::Timestamp>,
}

/// Error when no AI provider is configured: an honest 503, not a panic.
fn ai_not_configured() -> (StatusCode, serde_json::Value) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"error": "ai_not_configured", "detail": "OPENAI_API_KEY not set; satori-server has no AI provider"}),
    )
}

/// Map a lib [`SensemakingError`](satori::SensemakingError) onto the wire:
/// caller input problems → 400, provider/upstream problems → 502.
fn ai_error(e: satori::SensemakingError) -> (StatusCode, serde_json::Value) {
    match e {
        satori::SensemakingError::Validation(m) => (
            StatusCode::BAD_REQUEST,
            json!({"error": "validation", "detail": m}),
        ),
        other => (
            StatusCode::BAD_GATEWAY,
            json!({"error": "ai_upstream", "detail": other.to_string()}),
        ),
    }
}

const METHODS: &[&str] = &[
    "satori.sense",
    "satori.recall",
    "satori.research",
    "satori.search",
    "satori.profiles",
];

async fn dispatch_with_ai<P: satori::AiProvider>(
    store: &Store,
    ai: Option<(&P, &str)>,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, (StatusCode, serde_json::Value)> {
    if !METHODS.contains(&method) {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({"error": "unknown_method", "detail": method}),
        ));
    }
    match method {
        "satori.sense" => {
            let p: SenseParams = serde_json::from_value(params).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    json!({"error": "invalid_params", "detail": e.to_string()}),
                )
            })?;
            if p.body.trim().is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    json!({"error": "invalid_params", "detail": "body must be non-empty"}),
                ));
            }
            let (provider, model) = ai.ok_or_else(ai_not_configured)?;
            let (mut item, usage) = satori::sense_ai(provider, &p.body).await.map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    json!({"error": "ai_error", "detail": e.to_string()}),
                )
            })?;
            if let Some(ref_) = p.source_ref {
                // Thread upstream lineage across the hop (torii RawItem → here).
                item.source = Some(Source::External { ref_ });
            }
            store
                .put("sensing_item", &item.id.as_uuid().to_string(), &item)
                .await
                .map_err(storage_error)?;
            let mut out = json!({ "method": "satori.sense", "sensing_item": item });
            let mut meta = json!({"model": model});
            if let Some(usage) = usage {
                meta["usage"] = json!(usage);
            }
            out["_meta"] = meta;
            Ok(out)
        }
        "satori.research" => {
            let p: ResearchParams = serde_json::from_value(params).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    json!({"error": "invalid_params", "detail": e.to_string()}),
                )
            })?;
            let (provider, _) = ai.ok_or_else(ai_not_configured)?;
            let context: Vec<satori::TaskContext> = p
                .context
                .into_iter()
                .map(|t| satori::TaskContext::new(t.id, t.title, t.description))
                .collect();
            // Real AI operation: query (+ optional task grounding) → answer.
            let answer = satori::research(provider, &p.query, &context)
                .await
                .map_err(ai_error)?;
            Ok(json!({ "method": "satori.research", "answer": answer }))
        }
        "satori.profiles" => {
            let p: ProfilesParams = serde_json::from_value(params).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    json!({"error": "invalid_params", "detail": e.to_string()}),
                )
            })?;
            // Pure lib call: events → profiles, no server state involved.
            let report = satori::mine_agent_profiles(&p.events, &p.user_set_overrides, p.as_of);
            Ok(json!({ "method": "satori.profiles", "report": report }))
        }
        "satori.recall" => {
            let p: RecallParams = serde_json::from_value(params).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    json!({"error": "invalid_params", "detail": e.to_string()}),
                )
            })?;
            if let Some(id) = p.id {
                let item: Option<SensingItem> = store
                    .get("sensing_item", &id)
                    .await
                    .map_err(storage_error)?;
                return item
                    .map(|item| json!({"method": "satori.recall", "sensing_item": item}))
                    .ok_or_else(|| {
                        (
                            StatusCode::NOT_FOUND,
                            json!({"error": "not_found", "detail": id}),
                        )
                    });
            }
            let items: Vec<SensingItem> = store
                .list("sensing_item", p.limit)
                .await
                .map_err(storage_error)?;
            Ok(json!({"method": "satori.recall", "sensing_items": items}))
        }
        "satori.search" => Err((
            StatusCode::NOT_IMPLEMENTED,
            json!({"error": "unsupported", "detail": "satori.search needs a SearchIndex adapter"}),
        )),
        other => Err((
            StatusCode::BAD_REQUEST,
            json!({"error": "unknown_method", "detail": other}),
        )),
    }
}

// ── Semantic surface ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use satori::{AiError, AiOutput, AiRequest, AiUsage, ToolCall};
    use std::sync::atomic::{AtomicU64, Ordering};

    static DB_SEQ: AtomicU64 = AtomicU64::new(1);

    fn db_path() -> String {
        std::env::temp_dir()
            .join(format!(
                "satori-server-{}-{}.db",
                std::process::id(),
                DB_SEQ.fetch_add(1, Ordering::Relaxed)
            ))
            .to_string_lossy()
            .into_owned()
    }

    async fn test_store() -> Store {
        Store::open(&db_path()).await.unwrap()
    }

    async fn dispatch<P: satori::AiProvider>(
        ai: Option<&P>,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, (StatusCode, serde_json::Value)> {
        super::dispatch_with_ai(
            &test_store().await,
            ai.map(|provider| (provider, "test")),
            method,
            params,
        )
        .await
    }

    /// Fake provider returning a fixed text answer — lets dispatch tests
    /// exercise `satori.research` without network.
    struct FakeResearch {
        text: String,
    }

    impl satori::AiProvider for FakeResearch {
        async fn respond(&self, _req: AiRequest) -> Result<Vec<AiOutput>, AiError> {
            Ok(vec![AiOutput::Text(self.text.clone())])
        }
    }

    struct FakeSense(Result<Vec<AiOutput>, AiError>);

    fn successful_sense(kind: &str, summary: &str) -> FakeSense {
        FakeSense(Ok(vec![AiOutput::ToolCall(ToolCall {
            name: "sense_material".into(),
            arguments: json!({"kind": kind, "confidence": 0.9, "summary": summary}).to_string(),
        })]))
    }

    impl satori::AiProvider for FakeSense {
        async fn respond(&self, _req: AiRequest) -> Result<Vec<AiOutput>, AiError> {
            self.0.clone()
        }

        async fn respond_with_usage(
            &self,
            _req: AiRequest,
        ) -> Result<(Vec<AiOutput>, Option<AiUsage>), AiError> {
            Ok((
                self.0.clone()?,
                Some(AiUsage {
                    input_tokens: Some(123),
                    output_tokens: Some(45),
                    total_tokens: Some(168),
                }),
            ))
        }
    }

    #[tokio::test]
    async fn sense_builds_typed_sensing_item() {
        let fake = successful_sense("insight", "cache eviction changed the read path");
        let out = dispatch(
            Some(&fake),
            "satori.sense",
            json!({"body": "raw material", "source_ref": "raw_abc"}),
        )
        .await
        .expect("sense must succeed");
        let item = &out["sensing_item"];
        assert_eq!(item["kind"], "insight");
        assert_eq!(item["body"], "cache eviction changed the read path");
        assert!(
            item["id"].as_str().is_some(),
            "SensingItem must carry an id (provenance seed)"
        );
        // Lineage: upstream RawItem id threaded into the sensing item's source.
        assert_eq!(item["source"]["kind"], "external");
        assert_eq!(item["source"]["ref_"], "raw_abc");
        assert_eq!(out["_meta"]["model"], "test");
    }

    #[tokio::test]
    async fn request_ai_senses_without_leaking_secret() {
        let fake = successful_sense("risk", "Token expiry is likely.");
        let mut params = json!({
            "kind": "insight",
            "body": "long raw material",
            "ai": {"api_key": "sk-secret", "base_url": "https://ai.test/v1", "model": "test"}
        });
        assert!(extract_ai_config(&mut params).is_some());
        let store = test_store().await;
        let out = super::dispatch_with_ai(&store, Some((&fake, "test")), "satori.sense", params)
            .await
            .expect("satori.sense must tolerate unknown input fields");
        assert_eq!(out["sensing_item"]["kind"], "risk");
        assert_eq!(out["sensing_item"]["body"], "Token expiry is likely.");
        assert_eq!(out["_meta"]["model"], "test");
        assert_eq!(out["_meta"]["usage"]["total_tokens"], 168);
        let id = out["sensing_item"]["id"].as_str().unwrap();
        let stored: serde_json::Value = store.get("sensing_item", id).await.unwrap().unwrap();
        assert!(stored.get("_meta").is_none());
        assert!(!out.to_string().contains("sk-secret"));
    }

    #[tokio::test]
    async fn request_ai_sense_failure_is_ai_error() {
        let (code, body) = dispatch(
            Some(&FakeSense(Err(AiError::new("boom")))),
            "satori.sense",
            json!({"body": "material"}),
        )
        .await
        .unwrap_err();
        assert_eq!(code, StatusCode::BAD_GATEWAY);
        assert_eq!(body["error"], "ai_error");
    }

    #[tokio::test]
    async fn recall_and_unknown_method_rejected() {
        let out = dispatch(None::<&OpenAiProvider>, "satori.recall", json!({}))
            .await
            .unwrap();
        assert_eq!(out["sensing_items"], json!([]));
        let (code, _) = dispatch(None::<&OpenAiProvider>, "satori.nope", json!({}))
            .await
            .unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn sensing_item_persists_across_restart_and_write_errors_surface() {
        let path = db_path();
        let store = Store::open(&path).await.unwrap();
        let fake = successful_sense("insight", "persist me");
        let created = super::dispatch_with_ai(
            &store,
            Some((&fake, "test")),
            "satori.sense",
            json!({"body": "raw material", "source_ref": "raw_1"}),
        )
        .await
        .unwrap();
        assert_eq!(created["sensing_item"]["kind"], "insight");
        let id = created["sensing_item"]["id"].as_str().unwrap().to_owned();
        drop(store);

        let reopened = Store::open(&path).await.unwrap();
        let recalled = super::dispatch_with_ai(
            &reopened,
            None::<(&OpenAiProvider, &str)>,
            "satori.recall",
            json!({"id": id}),
        )
        .await
        .unwrap();
        assert_eq!(recalled["sensing_item"]["body"], "persist me");

        reopened.pool().close().await;
        let (code, body) = super::dispatch_with_ai(
            &reopened,
            Some((&fake, "test")),
            "satori.sense",
            json!({"body": "fail"}),
        )
        .await
        .unwrap_err();
        assert_eq!(code, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"], "storage_error");
    }

    #[tokio::test]
    async fn sense_rejects_bad_params() {
        for params in [json!({}), json!({"body": "   "})] {
            let (code, body) = dispatch(None::<&OpenAiProvider>, "satori.sense", params)
                .await
                .unwrap_err();
            assert_eq!(code, StatusCode::BAD_REQUEST);
            assert_eq!(body["error"], "invalid_params");
        }
    }

    #[tokio::test]
    async fn sense_without_provider_is_503_and_does_not_persist() {
        let store = test_store().await;
        let (code, body) = super::dispatch_with_ai(
            &store,
            None::<(&OpenAiProvider, &str)>,
            "satori.sense",
            json!({"body": "do not persist"}),
        )
        .await
        .unwrap_err();
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"], "ai_not_configured");

        let listed = super::dispatch_with_ai(
            &store,
            None::<(&OpenAiProvider, &str)>,
            "satori.recall",
            json!({}),
        )
        .await
        .unwrap();
        assert_eq!(listed["sensing_items"], json!([]));
    }

    #[tokio::test]
    async fn research_returns_answer_text() {
        let fake = FakeResearch {
            text: "rotate tokens every 24h".into(),
        };
        let out = dispatch(
            Some(&fake),
            "satori.research",
            json!({
                "query": "how should we rotate tokens?",
                "context": [{"id": "task-1", "title": "Persist tokens", "description": "hashed refresh tokens"}]
            }),
        )
        .await
        .expect("research must succeed");
        assert_eq!(out["method"], "satori.research");
        assert_eq!(out["answer"], "rotate tokens every 24h");
    }

    #[tokio::test]
    async fn research_without_provider_is_honest_503() {
        let (code, body) = dispatch(
            None::<&OpenAiProvider>,
            "satori.research",
            json!({"query": "anything"}),
        )
        .await
        .unwrap_err();
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"], "ai_not_configured");
    }

    #[tokio::test]
    async fn research_rejects_bad_params() {
        let fake = FakeResearch { text: "x".into() };
        let (code, body) = dispatch(Some(&fake), "satori.research", json!({}))
            .await
            .unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "invalid_params");
    }

    #[tokio::test]
    async fn research_empty_query_is_400() {
        let fake = FakeResearch {
            text: "unused".into(),
        };
        let (code, body) = dispatch(Some(&fake), "satori.research", json!({"query": "   "}))
            .await
            .unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "validation");
    }

    #[tokio::test]
    async fn profiles_mines_event_stream() {
        let agent = "11111111-1111-1111-1111-111111111111";
        let unit = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
        let out = dispatch(
            None::<&OpenAiProvider>,
            "satori.profiles",
            json!({
                "events": [
                    { "type": "work_unit_created", "work_unit": { "id": unit, "task_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "capability_tags": ["frontend"] } },
                    { "type": "work_unit_claimed", "work_unit_id": unit, "agent_id": agent },
                    { "type": "work_unit_completed", "work_unit_id": unit, "completed_by": agent, "elapsed_ms": 5000 }
                ],
                "user_set_overrides": [{ "agent_id": agent, "capability": "frontend", "score": 0.9 }]
            }),
        )
        .await
        .expect("profiles must succeed");
        assert_eq!(out["method"], "satori.profiles");
        let report = &out["report"];
        assert_eq!(report["envelopes_parsed"], 3);
        let p = &report["agents"][0];
        assert_eq!(p["agent_id"], agent);
        assert_eq!(p["completed_units"], 1);
        assert_eq!(p["mean_cycle_ms"], 5000.0);
        // The human override promotes the mined pattern to active.
        assert_eq!(p["responsibility"][0]["lifecycle"], "active");
        assert_eq!(p["responsibility"][0]["source"], "user_set");
        let conf = p["workflow_confidence"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&conf));
    }

    #[tokio::test]
    async fn tools_list_names_are_all_dispatchable() {
        for tool in tools() {
            let name = tool["name"].as_str().unwrap();
            assert!(!name.starts_with("satori_semantic_"));
            let method = name.replacen('_', ".", 1);
            let body = match dispatch(None::<&OpenAiProvider>, &method, json!({})).await {
                Ok(_) => continue,
                Err((_, body)) => body,
            };
            assert_ne!(
                body["error"], "unknown_method",
                "{method} must be a real dispatch method"
            );
        }
    }

    #[test]
    fn tools_catalogue_matches_methods() {
        layer_kit::test_support::assert_catalogue_matches(&tools(), METHODS);
    }

    #[tokio::test]
    async fn profiles_rejects_bad_params() {
        let (code, body) = dispatch(
            None::<&OpenAiProvider>,
            "satori.profiles",
            json!({"as_of": "not-a-timestamp"}),
        )
        .await
        .unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "invalid_params");
    }
}
