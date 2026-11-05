---
title: "Satori sensemaking server"
audience: "developer"
intent: "reference"
owner: "MeiSei"
status: "active"
source_of_truth: "README.md"
last_verified: "2026-11-04"
---

# Satori 悟り — sensemaking layer of Meisei

> **Meisei** 明晰 (“clarity”) is an open pipeline that carries raw intent through
> understanding → decision → plan → action to a finished result.

[![Meisei](https://img.shields.io/badge/meisei-明晰-1f2937.svg)](https://meisei.ru)
[![License: Apache-2.0 WITH Commons-Clause](https://img.shields.io/badge/license-Apache--2.0%20WITH%20Commons--Clause-blue.svg)](https://github.com/tupical/satori/blob/main/LICENSE)

<sub>
torii · <b>satori</b> · enma · yatagarasu · fujin · daruma
&nbsp;—&nbsp; intake · <b>sensemaking</b> · decisions · planning · actions · execution (terminal)
</sub>

## What it is

Satori is the **sensemaking** layer of the Meisei pipeline: it turns raw intake
material into understanding. It owns `SensingItem` primitives (with confidence,
sources, links and reconsider triggers), AI sensemaking operations (`sense`,
`research` with task-context annotation). Domain
primitives stay storage-agnostic; the server persists artifacts. The crate has no
dependency on daruma or sibling layers; adapters live only inside the host.

`satori.sense` asks for genuine unresolved questions within the material's scope.
Self-contained material uses an empty `open_questions` array, rather than a
statement such as "None". Explicit and model-reported questions still stop the
request with `needs_input`.

The unused embedding index/provider and semantic MCP methods were removed.
Existing sidecar files are left untouched; old semantic environment flags have
no effect. WorkspaceGraph FTS, lesson recall, downstream impact and capability profiles
are implemented in Daruma. Unwired search/impact adapters and the duplicate
profile miner were removed from Satori. `satori.recall` only reads persisted
SensingItems for the cabinet; it is not Daruma lesson recall.

## Repository layout

- `src/` — the `satori` library: sensing types, `research`, prompt registry, error types.
- `server/` — `satori-server`, a thin, independently-deployed HTTP/MCP wrapper over
  the library (the axum/tokio scaffold comes from `layer-kit`).
- `deploy/` — release `build.sh` (stamps the git SHA into `/healthz`) and a
  systemd user unit.

## Build & run

```sh
cargo run -p satori-server
# GET  /healthz   — open liveness/version probe
# POST /v1/mcp    — platform-token gated MCP surface:
#                   satori.sense, satori.recall, satori.research
```

For production builds use `deploy/build.sh` so `/healthz` reports the real git SHA
instead of `"dev"`.

## Configuration (env)

| Variable | Default | Purpose |
| --- | --- | --- |
| `SATORI_PORT` | `8091` | HTTP listen port |
| `SATORI_PLATFORM_SECRET` | unset | HMAC key; if unset, `/v1/mcp` is closed |
| `SATORI_VERSION` | crate version | Version reported by `/healthz` |
| `SATORI_DB` | `./satori.db` | SQLite store path (`layer_kit::store::Store`) |
| `OPENAI_API_KEY` | unset | Optional AI provider for `satori.sense` / `satori.research`; without a key they fail with `ai_not_configured` (see [Errors over MCP](#errors-over-mcp)) |
| `OPENAI_API_PROTOCOL` | `responses` | Wire protocol: `responses`, `chat_completions` or `anthropic_messages` (Claude, `POST /v1/messages` with `x-api-key`); an unknown value disables AI. The platform may instead send a request-scoped `ai` block with the same `api_protocol` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` (`https://api.anthropic.com` for `anthropic_messages`) | Base URL of the provider API |
| `OPENAI_MODEL` | `gpt-4.1` | Model used by the AI operations |

## Errors over MCP

`POST /v1/mcp` speaks MCP (JSON-RPC 2.0) only. A tool that fails is a tool
result, not an HTTP error: `tools/call` answers HTTP 200 with
`result.isError: true`, the error object in `result.structuredContent`
(`error` is the machine-readable code) and the same JSON as text in
`result.content[0].text`. The status a method carries internally — e.g. 503
for `ai_not_configured` — never reaches the wire. Calling `satori_sense`
without an AI provider returns:

```json
{"jsonrpc": "2.0", "id": 1, "result": {
  "content": [{"type": "text", "text": "{\"detail\":\"…\",\"error\":\"ai_not_configured\"}"}],
  "structuredContent": {"detail": "OPENAI_API_KEY not set; satori-server has no AI provider", "error": "ai_not_configured"},
  "isError": true}}
```

Protocol faults are JSON-RPC `error` objects, also with HTTP 200 (`-32601`
unknown method, `-32602` `tools/call` without `name`). HTTP status codes are
reserved for the transport: 401 for a missing or invalid platform token or a
closed surface (`auth_disabled`), 400 with `{"error":"bad_request","detail":"…"}`
for a body that is not a JSON-RPC 2.0 request, 202 with an empty body for a
notification. A body is a JSON-RPC 2.0 request only if it is valid JSON and its
`jsonrpc` member is exactly the string `"2.0"`: a missing member, another
version or another type gets the 400, notifications included, and never reaches
the layer. The token is checked first, so without a valid one the answer is
401. The envelope is implemented once in `layer-kit` (`src/serve.rs`).

## Docs

Pipeline canon and layer contracts: https://meisei.ru/docs

## License

Apache-2.0 WITH Commons-Clause — see [LICENSE](https://github.com/tupical/satori/blob/main/LICENSE) and
[LICENSE.commons-clause.md](https://github.com/tupical/satori/blob/main/LICENSE.commons-clause.md).
