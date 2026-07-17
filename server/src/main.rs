//! satori-server — thin, independently-deployed HTTP/MCP wrapper around the
//! `satori` sensemaking lib. Its own deploy unit (own systemd service, own
//! port). Boundary-clean: no mcpbox dependency; the platform→tool auth contract
//! is a configured shared key (see `auth`).
//!
//! Routes:
//!   GET  /healthz   — open; liveness + version for the platform registry.
//!   POST /v1/mcp    — requires a valid platform token; sensemaking surface
//!                     (`satori.sense` builds a typed SensingItem via the lib,
//!                     `satori.research` runs the lib's AI research operation,
//!                     `satori.profiles` process-mines agent profiles from a
//!                     daruma event stream supplied in the params).
//!
//! Env: SATORI_PORT (default 8091), SATORI_PLATFORM_SECRET (HMAC key; if
//! unset, /v1/mcp is closed), SATORI_VERSION (defaults to the crate version).
//! AI methods (`satori.research`): OPENAI_API_KEY / OPENAI_BASE_URL /
//! OPENAI_MODEL (see `ai`); without a key they answer `ai_not_configured`.

mod ai;
mod auth;

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use satori::types::{SensingItem, SensingItemKind, Source};
use serde_json::json;

const TOOL: &str = "satori";

struct AppState {
    version: String,
    platform_secret: Option<Vec<u8>>,
    /// Concrete AI provider; `None` when OPENAI_API_KEY is unset — AI methods
    /// then answer `ai_not_configured` instead of panicking at call time.
    ai: Option<ai::OpenAiProvider>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().json().init();

    let version =
        std::env::var("SATORI_VERSION").unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string());
    let platform_secret = std::env::var("SATORI_PLATFORM_SECRET")
        .ok()
        .filter(|s| !s.is_empty())
        .map(String::into_bytes);
    if platform_secret.is_none() {
        tracing::warn!("SATORI_PLATFORM_SECRET unset — /v1/mcp will reject all requests");
    }
    let ai = ai::AiConfig::from_env().map(ai::OpenAiProvider::new);
    if ai.is_none() {
        tracing::warn!("OPENAI_API_KEY unset — AI methods (satori.research) will answer ai_not_configured");
    }
    let state = Arc::new(AppState {
        version,
        platform_secret,
        ai,
    });

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/mcp", post(mcp))
        .with_state(state);

    let port = std::env::var("SATORI_PORT").unwrap_or_else(|_| "8091".to_string());
    // localhost-bound: only the co-located platform reaches it (C3 hardening).
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    tracing::info!(%addr, tool = TOOL, "satori-server listening");
    axum::serve(listener, app).await.expect("server error");
}

async fn healthz(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({ "service": TOOL, "status": "ok", "version": s.version, "git_sha": option_env!("GIT_SHA").unwrap_or("dev") }))
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

async fn mcp(State(s): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> impl IntoResponse {
    let Some(secret) = &s.platform_secret else {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":"auth_disabled"}))).into_response();
    };
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    let Some(claims) = token.and_then(|t| auth::verify(secret, TOOL, now_secs(), t)) else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error":"invalid_platform_token"})),
        )
            .into_response();
    };

    // Auth passed — dispatch the MCP method against the satori sensemaking lib.
    let req: McpRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "bad_request", "detail": e.to_string()})),
            )
                .into_response();
        }
    };
    match dispatch(s.ai.as_ref(), &req.method, req.params).await {
        Ok(mut result) => {
            result["tool"] = json!(TOOL);
            result["version"] = json!(s.version);
            result["workspace"] = json!(claims.workspace);
            result["project"] = json!(claims.project);
            Json(result).into_response()
        }
        Err((code, payload)) => (code, Json(payload)).into_response(),
    }
}

/// One MCP call: `{ "method": "satori.sense", "params": { ... } }`.
#[derive(serde::Deserialize)]
struct McpRequest {
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

/// Params for `satori.sense`. `kind` deserializes from the lib's snake_case
/// enum (`knowledge`/`question`/`hypothesis`/`risk`/`contradiction`/`insight`/
/// `rejected_idea`/`research_gap`).
#[derive(serde::Deserialize)]
struct SenseParams {
    kind: SensingItemKind,
    body: String,
    /// Optional provenance: the upstream object's id (e.g. a torii RawItem id),
    /// recorded as the sensing item's source so lineage survives the network hop.
    #[serde(default)]
    source_ref: Option<String>,
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

/// Pure MCP dispatch over the satori sensemaking lib — no auth, no HTTP, so
/// it is unit-testable directly (AI methods get a fake `AiProvider` in
/// tests). `satori` is a stateless OSS skeleton (no index), so
/// recall/semantic-search (which need a SearchIndex) are unsupported here.
async fn dispatch<P: satori::AiProvider>(
    ai: Option<&P>,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, (StatusCode, serde_json::Value)> {
    match method {
        "satori.sense" => {
            let p: SenseParams = serde_json::from_value(params).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    json!({"error": "invalid_params", "detail": e.to_string()}),
                )
            })?;
            // Real sensemaking: a typed SensingItem with its own id (provenance).
            let mut item = SensingItem::new(p.kind, p.body);
            if let Some(ref_) = p.source_ref {
                // Thread upstream lineage across the hop (torii RawItem → here).
                item.source = Some(Source::External { ref_ });
            }
            Ok(json!({ "method": "satori.sense", "sensing_item": item }))
        }
        "satori.research" => {
            let p: ResearchParams = serde_json::from_value(params).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    json!({"error": "invalid_params", "detail": e.to_string()}),
                )
            })?;
            let Some(provider) = ai else {
                return Err(ai_not_configured());
            };
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
        "satori.recall" | "satori.search" => Err((
            StatusCode::NOT_IMPLEMENTED,
            json!({"error": "unsupported", "detail": "satori-server is stateless (OSS skeleton has no index); recall/search need a SearchIndex adapter"}),
        )),
        other => Err((
            StatusCode::BAD_REQUEST,
            json!({"error": "unknown_method", "detail": other}),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use satori::{AiError, AiOutput, AiRequest};

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

    #[tokio::test]
    async fn sense_builds_typed_sensing_item() {
        let out = dispatch(
            None::<&ai::OpenAiProvider>,
            "satori.sense",
            json!({"kind": "insight", "body": "cache eviction changed the read path", "source_ref": "raw_abc"}),
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
    }

    #[tokio::test]
    async fn recall_unsupported_and_unknown_method_rejected() {
        let (code, _) = dispatch(None::<&ai::OpenAiProvider>, "satori.recall", json!({}))
            .await
            .unwrap_err();
        assert_eq!(code, StatusCode::NOT_IMPLEMENTED);
        let (code, _) = dispatch(None::<&ai::OpenAiProvider>, "satori.nope", json!({}))
            .await
            .unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn sense_rejects_bad_params() {
        let (code, _) = dispatch(
            None::<&ai::OpenAiProvider>,
            "satori.sense",
            json!({"body": "no kind"}),
        )
        .await
        .unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
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
            None::<&ai::OpenAiProvider>,
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
            None::<&ai::OpenAiProvider>,
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
    async fn profiles_rejects_bad_params() {
        let (code, body) = dispatch(
            None::<&ai::OpenAiProvider>,
            "satori.profiles",
            json!({"as_of": "not-a-timestamp"}),
        )
        .await
        .unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "invalid_params");
    }
}
