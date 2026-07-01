//! satori-server — thin, independently-deployed HTTP/MCP wrapper around the
//! `satori` sensemaking lib. Its own deploy unit (own systemd service, own
//! port). Boundary-clean: no mcpbox dependency; the platform→tool auth contract
//! is a configured shared key (see `auth`).
//!
//! Routes:
//!   GET  /healthz   — open; liveness + version for the platform registry.
//!   POST /v1/mcp    — requires a valid platform token; sensemaking surface
//!                     (`satori.sense` builds a typed SensingItem via the lib).
//!
//! Env: SATORI_PORT (default 8091), SATORI_PLATFORM_SECRET (HMAC key; if unset,
//! /v1/mcp is closed), SATORI_VERSION (defaults to the crate version).

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
    let state = Arc::new(AppState {
        version,
        platform_secret,
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
    Json(json!({ "service": TOOL, "status": "ok", "version": s.version }))
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
    match dispatch(&req.method, req.params) {
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

/// Pure MCP dispatch over the satori sensemaking lib — no auth, no HTTP, so it
/// is unit-testable directly. `satori` is a stateless OSS skeleton (no index),
/// so recall/semantic-search (which need a SearchIndex) are unsupported here.
fn dispatch(
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

    #[test]
    fn sense_builds_typed_sensing_item() {
        let out = dispatch(
            "satori.sense",
            json!({"kind": "insight", "body": "cache eviction changed the read path", "source_ref": "raw_abc"}),
        )
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

    #[test]
    fn recall_unsupported_and_unknown_method_rejected() {
        let (code, _) = dispatch("satori.recall", json!({})).unwrap_err();
        assert_eq!(code, StatusCode::NOT_IMPLEMENTED);
        let (code, _) = dispatch("satori.nope", json!({})).unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn sense_rejects_bad_params() {
        let (code, _) = dispatch("satori.sense", json!({"body": "no kind"})).unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
    }
}
