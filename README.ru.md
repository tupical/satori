# Satori 悟り — sensemaking-слой Meisei

> **Meisei** 明晰 («ясность») — открытый конвейер, который проводит сырой замысел
> через понимание → решение → план → действие к готовому результату.

[![Meisei](https://img.shields.io/badge/meisei-明晰-1f2937.svg)](https://meisei.ru)
[![License: Apache-2.0 WITH Commons-Clause](https://img.shields.io/badge/license-Apache--2.0%20WITH%20Commons--Clause-blue.svg)](LICENSE)

<sub>
torii · <b>satori</b> · enma · yatagarasu · fujin · daruma
&nbsp;—&nbsp; intake · <b>осмысление</b> · решения · планирование · действия · исполнение (терминальный слой)
</sub>

## Что это

Satori — **sensemaking**-слой конвейера MeiSei: превращает сырье intake в
понимание. Владеет примитивами `SensingItem` (уверенность, источники, связи,
триггеры пересмотра) и AI-операциями осмысления (`sense`, `research` с
аннотированием контекста задачи). Доменные примитивы не зависят от хранилища;
артефакты персистит сервер. Крейт не зависит от daruma и соседних слоёв;
адаптеры живут только внутри host.

`satori.sense` запрашивает только реальные нерешённые вопросы в рамках
материала. Для самодостаточного материала `open_questions` — пустой массив, а
не утверждение вроде «Нет». Явные вопросы и вопросы, о которых сообщила модель,
по-прежнему останавливают запрос с `needs_input`.

Неиспользуемые embedding-индекс/провайдер и семантические MCP-методы удалены.
Существующие sidecar-файлы не трогаются; старые семантические env-флаги ни на
что не влияют. FTS по WorkspaceGraph, recall уроков, downstream impact и
capability profiles реализованы в Daruma. Неподключённые адаптеры
search/impact и дублирующий майнер профилей из Satori удалены. `satori.recall`
только читает сохранённые SensingItem для кабинета; это не recall уроков Daruma.

## Структура репозитория

- `src/` — библиотека `satori`: типы SensingItem, `research`, реестр промптов,
  типы ошибок.
- `server/` — `satori-server`, тонкая независимо развёртываемая HTTP/MCP-обёртка
  над библиотекой (axum/tokio-каркас — из [`layer-kit`](../layer-kit)).
- `deploy/` — release-`build.sh` (прошивает git SHA в `/healthz`) и systemd user unit.

## Сборка и запуск

```sh
cargo run -p satori-server
# GET  /healthz   — открытая проба живости/версии
# POST /v1/mcp    — MCP-поверхность под платформенным токеном:
#                   satori.sense, satori.recall, satori.research
```

Для продовых сборок используйте `deploy/build.sh`, чтобы `/healthz` отдавал
реальный git SHA, а не `"dev"`.

## Конфигурация (env)

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `SATORI_PORT` | `8091` | HTTP-порт |
| `SATORI_PLATFORM_SECRET` | не задан | HMAC-ключ; если не задан, `/v1/mcp` закрыт |
| `SATORI_VERSION` | версия крейта | Версия, отдаваемая `/healthz` |
| `SATORI_DB` | `./satori.db` | Путь к SQLite-хранилищу (`layer_kit::store::Store`) |
| `OPENAI_API_KEY` | не задан | Опциональный AI-провайдер для `satori.sense` / `satori.research`; без ключа — ошибка `ai_not_configured` (см. [Ошибки через MCP](#ошибки-через-mcp)) |
| `OPENAI_API_PROTOCOL` | `responses` | Протокол: `responses`, `chat_completions` или `anthropic_messages` (Claude, `POST /v1/messages` с `x-api-key`); неизвестное значение отключает AI. Платформа может вместо этого прислать request-scoped блок `ai` с тем же `api_protocol` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` (`https://api.anthropic.com` для `anthropic_messages`) | Базовый URL API провайдера |
| `OPENAI_MODEL` | `gpt-4.1` | Модель, используемая AI-операциями |

## Ошибки через MCP

`POST /v1/mcp` принимает только MCP (JSON-RPC 2.0). Упавший инструмент — это
результат инструмента, а не HTTP-ошибка: `tools/call` отвечает HTTP 200 с
`result.isError: true`, объектом ошибки в `result.structuredContent` (поле
`error` — машиночитаемый код) и тем же JSON текстом в
`result.content[0].text`. Внутренний статус метода — например, 503 у
`ai_not_configured` — на провод не попадает. Вызов `satori_sense` без
AI-провайдера возвращает:

```json
{"jsonrpc": "2.0", "id": 1, "result": {
  "content": [{"type": "text", "text": "{\"detail\":\"…\",\"error\":\"ai_not_configured\"}"}],
  "structuredContent": {"detail": "OPENAI_API_KEY not set; satori-server has no AI provider", "error": "ai_not_configured"},
  "isError": true}}
```

Сбои протокола — объекты JSON-RPC `error`, тоже с HTTP 200 (`-32601` —
неизвестный метод, `-32602` — `tools/call` без `name`). HTTP-коды остаются за
транспортом: 401 — нет или неверный платформенный токен либо поверхность
закрыта (`auth_disabled`), 400 с `{"error":"bad_request","detail":"…"}` — тело
не является запросом JSON-RPC 2.0, 202 с пустым телом — notification. Запрос
JSON-RPC 2.0 — это корректный JSON, в котором поле `jsonrpc` равно строке
`"2.0"`: без этого поля, с другой версией или другим типом значения ответ 400,
в том числе для notification, и тело не доходит до слоя. Токен проверяется
первым, поэтому без действительного токена ответ 401. Конверт реализован один
раз в `layer-kit` (`src/serve.rs`).

## Документация

Канон конвейера и контракты слоёв: https://meisei.ru/docs

## Лицензия

Apache-2.0 WITH Commons-Clause — см. [LICENSE](LICENSE) и
[LICENSE.commons-clause.md](LICENSE.commons-clause.md).
