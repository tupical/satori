//! Acceptance: satori's AI methods over `api_protocol=anthropic_messages`.
//!
//! The provider is the real `layer_kit::openai::OpenAiProvider` talking HTTP
//! to a local stub that answers ONLY `POST /v1/messages` and records every
//! hit, so a wrong path, header or body shape cannot pass. Config arrives the
//! way the platform sends it: a request-scoped `params.ai` block that
//! `extract_ai_config` cuts out before dispatch.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::routing::post;
use axum::Json;
use layer_kit::ai::extract_ai_config;
use layer_kit::auth::Claims;
use layer_kit::openai::OpenAiProvider;
use layer_kit::store::Store;
use serde_json::{json, Value};

use super::{dispatch_with_ai, Handler};
use layer_kit::serve::McpHandler;

const KEY: &str = "sk-ant-satori-test";
const REVOKED: &str = "sk-ant-revoked";

#[derive(Debug)]
struct Seen {
    path: String,
    headers: HeaderMap,
    body: Value,
}

type Log = Arc<Mutex<Vec<Seen>>>;

async fn messages(
    State(log): State<Log>,
    uri: Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let revoked = headers["x-api-key"] == REVOKED;
    let has_tools = body.get("tools").is_some();
    let prompt = body["messages"][0]["content"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    log.lock().unwrap().push(Seen {
        path: uri.path().to_owned(),
        headers,
        body,
    });
    if revoked {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"type": "error",
                        "error": {"type": "authentication_error", "message": "invalid x-api-key"}})),
        );
    }
    let reply = if !has_tools {
        // satori.research: plain text answer.
        json!({"stop_reason": "end_turn",
               "content": [{"type": "text", "text": "Rotate the keys quarterly.\ninsight: rotation is cheap"}],
               "usage": {"input_tokens": 30, "output_tokens": 9}})
    } else {
        let questions = if prompt.contains("ambiguous") {
            json!(["Кто владеет ключами?"])
        } else {
            json!([])
        };
        json!({"stop_reason": "tool_use",
               "content": [{"type": "tool_use", "id": "toolu_1", "name": "sense_material",
                            "input": {"kind": "risk", "confidence": 0.8,
                                      "summary": "Keys never rotate.", "open_questions": questions}}],
               "usage": {"input_tokens": 50, "output_tokens": 12}})
    };
    (StatusCode::OK, Json(reply))
}

async fn other(State(log): State<Log>, uri: Uri, headers: HeaderMap) -> StatusCode {
    log.lock().unwrap().push(Seen {
        path: uri.path().to_owned(),
        headers,
        body: Value::Null,
    });
    StatusCode::NOT_FOUND
}

async fn stub() -> (String, Log) {
    let log = Log::default();
    let app = axum::Router::new()
        .route("/v1/messages", post(messages))
        .fallback(other)
        .with_state(log.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), log)
}

/// Cut the `ai` block out of `params` as `Handler::dispatch` does and build
/// the provider without the dev host's HTTP proxy (it would answer loopback).
fn provider(params: &mut Value) -> OpenAiProvider {
    let cfg = extract_ai_config(params).expect("anthropic_messages ai block is a valid config");
    let http = OpenAiProvider::http_client_builder()
        .no_proxy()
        .build()
        .unwrap();
    OpenAiProvider::with_http_client(cfg, http)
}

fn ai_block(base: &str, key: &str) -> Value {
    json!({"api_key": key, "base_url": base, "model": "claude-test",
           "max_output_tokens": 400, "api_protocol": "anthropic_messages"})
}

async fn store() -> Store {
    let path = std::env::temp_dir().join(format!(
        "satori-anthropic-{}-{}.db",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    Store::open(&path.to_string_lossy()).await.unwrap()
}

async fn stored_items(store: &Store) -> Vec<Value> {
    store.list::<Value>("sensing_item", 100).await.unwrap()
}

#[tokio::test]
async fn sense_and_research_reach_anthropic_messages() {
    let (base, log) = stub().await;
    let store = store().await;

    let mut params = json!({"body": "Our API keys never rotate.", "ai": ai_block(&base, KEY)});
    let p = provider(&mut params);
    assert!(params.get("ai").is_none(), "the secret block is cut out");
    let out = dispatch_with_ai(&store, Some((&p, p.model())), "satori.sense", params)
        .await
        .expect("sense over anthropic_messages must succeed");
    assert_eq!(out["sensing_item"]["kind"], "risk");
    assert_eq!(out["sensing_item"]["body"], "Keys never rotate.");
    assert_eq!(out["_meta"]["model"], "claude-test");
    assert_eq!(out["_meta"]["usage"]["input_tokens"], 50);
    assert_eq!(out["_meta"]["usage"]["output_tokens"], 12);
    assert_eq!(out["_meta"]["usage"]["total_tokens"], 62);

    let mut params = json!({"query": "Should we rotate keys?",
                            "context": [{"id": "t1", "title": "Key policy"}],
                            "ai": ai_block(&base, KEY)});
    let p = provider(&mut params);
    let out = dispatch_with_ai(&store, Some((&p, p.model())), "satori.research", params)
        .await
        .expect("research over anthropic_messages must succeed");
    assert_eq!(
        out["answer"],
        "Rotate the keys quarterly.\ninsight: rotation is cheap"
    );
    assert_eq!(out["_meta"]["usage"]["total_tokens"], 39);
    assert!(!out["sensing_items"].as_array().unwrap().is_empty());

    let items = stored_items(&store).await;
    assert_eq!(
        items.len(),
        1 + out["sensing_items"].as_array().unwrap().len()
    );
    assert!(
        !json!(items).to_string().contains(KEY),
        "secret never persisted"
    );

    let log = log.lock().unwrap();
    assert_eq!(log.len(), 2, "one provider call per method: {log:?}");
    for seen in log.iter() {
        assert_eq!(seen.path, "/v1/messages");
        assert_eq!(seen.headers["x-api-key"], KEY);
        assert_eq!(seen.headers["anthropic-version"], "2023-06-01");
        assert!(seen.headers.get("authorization").is_none(), "no Bearer");
        assert_eq!(seen.body["model"], "claude-test");
        assert_eq!(seen.body["max_tokens"], 400);
    }
    let sense = &log[0].body;
    assert_eq!(sense["tools"][0]["name"], "sense_material");
    assert_eq!(
        sense["tools"][0]["input_schema"]["required"][3],
        "open_questions"
    );
    assert_eq!(sense["tool_choice"], json!({"type": "any"}));
    let research = &log[1].body;
    assert!(research.get("tools").is_none() && research.get("tool_choice").is_none());
    assert!(research["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("Should we rotate keys?"));
}

#[tokio::test]
async fn anthropic_open_questions_need_input_and_persist_nothing() {
    let (base, log) = stub().await;
    let store = store().await;
    let mut params = json!({"body": "ambiguous key policy", "ai": ai_block(&base, KEY)});
    let p = provider(&mut params);
    let (code, body) = dispatch_with_ai(&store, Some((&p, p.model())), "satori.sense", params)
        .await
        .unwrap_err();
    assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body,
        json!({"error": "needs_input", "questions": ["Кто владеет ключами?"]})
    );
    assert!(stored_items(&store).await.is_empty());
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn revoked_anthropic_key_is_an_ai_error_without_the_key() {
    let (base, log) = stub().await;
    let store = store().await;

    let mut params = json!({"body": "material", "ai": ai_block(&base, REVOKED)});
    let p = provider(&mut params);
    let (code, body) = dispatch_with_ai(&store, Some((&p, p.model())), "satori.sense", params)
        .await
        .unwrap_err();
    assert_eq!(code, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"], "ai_error");
    let detail = body["detail"].as_str().unwrap();
    assert!(
        detail.contains("401") && detail.contains("authentication_error"),
        "{detail}"
    );
    assert!(!body.to_string().contains(REVOKED));

    let mut params = json!({"query": "q", "ai": ai_block(&base, REVOKED)});
    let p = provider(&mut params);
    let (code, body) = dispatch_with_ai(&store, Some((&p, p.model())), "satori.research", params)
        .await
        .unwrap_err();
    assert_eq!(code, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"], "ai_upstream");
    assert!(body["detail"].as_str().unwrap().contains("401"));
    assert!(!body.to_string().contains(REVOKED));

    assert!(stored_items(&store).await.is_empty());
    assert_eq!(
        log.lock().unwrap().len(),
        2,
        "exactly one call per method, no retry"
    );
}

#[tokio::test]
async fn without_provider_ai_methods_answer_deterministically() {
    let handler = Handler {
        ai: None,
        store: store().await,
    };
    let claims = Claims {
        workspace: "ws".into(),
        project: None,
        tool: "satori".into(),
        exp: i64::MAX,
    };
    let (code, body) = handler
        .dispatch(&claims, "satori.sense", json!({"body": "material"}))
        .await
        .unwrap_err();
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "ai_not_configured");
    let (code, body) = handler
        .dispatch(
            &claims,
            "satori.sense",
            json!({"body": "Plan.\n\nОткрытые вопросы:\n- Кто платит?\n"}),
        )
        .await
        .unwrap_err();
    assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["questions"], json!(["Кто платит?"]));
    let (code, body) = handler
        .dispatch(&claims, "satori.research", json!({"query": "q"}))
        .await
        .unwrap_err();
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "ai_not_configured");
    // An incomplete ai block is no provider either — and its secret is still cut.
    let (code, body) = handler
        .dispatch(
            &claims,
            "satori.sense",
            json!({"body": "material", "ai": {"api_key": KEY, "api_protocol": "anthropic"}}),
        )
        .await
        .unwrap_err();
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "ai_not_configured");
    assert!(!body.to_string().contains(KEY));
    assert!(stored_items(&handler.store.scoped("ws", None))
        .await
        .is_empty());
}
