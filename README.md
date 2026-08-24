# Satori 悟り — sensemaking layer of Meisei

> **Meisei** 明晰 (“clarity”) is an open pipeline that carries raw intent through
> understanding → decision → plan → action to a finished result.

[![Meisei](https://img.shields.io/badge/meisei-明晰-1f2937.svg)](https://meisei.ru)
[![License: Apache-2.0 WITH Commons-Clause](https://img.shields.io/badge/license-Apache--2.0%20WITH%20Commons--Clause-blue.svg)](LICENSE)

<sub>
torii · <b>satori</b> · enma · yatagarasu · fujin · daruma
&nbsp;—&nbsp; intake · <b>sensemaking</b> · decisions · planning · actions · execution (terminal)
</sub>

## What it is

Satori is the **sensemaking** layer of the Meisei pipeline: it turns raw intake
material into understanding. It owns `SensingItem` primitives (with confidence,
sources, links and reconsider triggers), AI sensemaking operations (`sense`,
`research` with task-context annotation), process-mining of agent responsibility
patterns (`profiles`) from a daruma event stream, and an optional semantic
surface (embedding index + hybrid FTS rerank + link suggestions). Domain
primitives stay storage-agnostic; the server persists artifacts. The crate has no
dependency on daruma or sibling layers; adapters live only inside the host.

## Repository layout

- `src/` — the `satori` library: sensing types, `research`, `profiles`, recall,
  semantic primitives, prompt registry, error types.
- `server/` — `satori-server`, a thin, independently-deployed HTTP/MCP wrapper over
  the library (the axum/tokio scaffold comes from [`layer-kit`](../layer-kit)).
- `deploy/` — release `build.sh` (stamps the git SHA into `/healthz`) and a
  systemd user unit.

## Build & run

```sh
cargo run -p satori-server
# GET  /healthz   — open liveness/version probe
# POST /v1/mcp    — platform-token gated MCP surface:
#                   satori.sense, satori.recall, satori.research,
#                   satori.profiles, satori.semantic_index, satori.semantic_search
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
| `OPENAI_API_KEY` | unset | Optional AI provider for `satori.sense` / `satori.research`; without a key they answer `ai_not_configured` (503) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL of the OpenAI-compatible API |
| `OPENAI_MODEL` | `gpt-4.1` | Model used by the AI operations |
| `SATORI_SEMANTIC_ENABLED` | off | Cost gate: `1` enables `satori.semantic_index` / `satori.semantic_search`; when off they answer `semantic_disabled` (403) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for the semantic surface; without a key it answers `embeddings_not_configured` (503) |
| `SATORI_SEMANTIC_DIR` | `./data/semantic` | Per-workspace sidecar index directory |

## Docs

Pipeline canon and layer contracts: https://meisei.ru/docs

## License

Apache-2.0 WITH Commons-Clause — see [LICENSE](LICENSE) and
[LICENSE.commons-clause.md](LICENSE.commons-clause.md).
