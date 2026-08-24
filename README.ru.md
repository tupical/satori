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
триггеры пересмотра), AI-операциями осмысления (`sense`, `research` с
аннотированием контекста задачи), process-mining'ом паттернов ответственности
агентов (`profiles`) из потока событий daruma и опциональной семантической
поверхностью (embedding-индекс + гибридный FTS-rerank + предложения связей).
Доменные примитивы не зависят от хранилища; артефакты персистит сервер. Крейт
не зависит от daruma и соседних слоёв; адаптеры живут только внутри host.

## Структура репозитория

- `src/` — библиотека `satori`: типы SensingItem, `research`, `profiles`, recall,
  семантические примитивы, реестр промптов, типы ошибок.
- `server/` — `satori-server`, тонкая независимо развёртываемая HTTP/MCP-обёртка
  над библиотекой (axum/tokio-каркас — из [`layer-kit`](../layer-kit)).
- `deploy/` — release-`build.sh` (прошивает git SHA в `/healthz`) и systemd user unit.

## Сборка и запуск

```sh
cargo run -p satori-server
# GET  /healthz   — открытая проба живости/версии
# POST /v1/mcp    — MCP-поверхность под платформенным токеном:
#                   satori.sense, satori.recall, satori.research,
#                   satori.profiles, satori.semantic_index, satori.semantic_search
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
| `OPENAI_API_KEY` | не задан | Опциональный AI-провайдер для `satori.sense` / `satori.research`; без ключа — ответ `ai_not_configured` (503) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Базовый URL OpenAI-совместимого API |
| `OPENAI_MODEL` | `gpt-4.1` | Модель, используемая AI-операциями |
| `SATORI_SEMANTIC_ENABLED` | выключен | Cost gate: `1` включает `satori.semantic_index` / `satori.semantic_search`; когда выключен — ответ `semantic_disabled` (403) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding-модель семантической поверхности; без ключа — ответ `embeddings_not_configured` (503) |
| `SATORI_SEMANTIC_DIR` | `./data/semantic` | Каталог sidecar-индексов по воркспейсам |

## Документация

Канон конвейера и контракты слоёв: https://meisei.ru/docs

## Лицензия

Apache-2.0 WITH Commons-Clause — см. [LICENSE](LICENSE) и
[LICENSE.commons-clause.md](LICENSE.commons-clause.md).
