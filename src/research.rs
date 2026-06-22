//! AI research operation — answer a free-form query, optionally grounded
//! in the bodies of one or more existing task summaries.
//!
//! [`research`] is **pure** in the I/O sense: it assembles the prompt from
//! the supplied query + task context and delegates the LLM call to the
//! provider through [`AiProvider::respond`], returning the answer as a plain
//! `String`. It never writes to storage. If a caller wants to persist the
//! answer (e.g. as a `Research` comment on a task), that is a separate step
//! performed through the ordinary command/comment API — it lives outside
//! this module so the sensemaking layer stays a clean operation with no
//! command-bus dependency.

use serde::Serialize;

use crate::ai::{AiOutput, AiProvider, AiRequest};
use crate::error::SensemakingError;
use crate::prompts::PromptRegistry;
use crate::types::{Confidence, SensingItem, SensingItemKind, Source};

// ── Local task context ───────────────────────────────────────────────────

/// Minimal task representation for grounding a research query.
///
/// Callers (mcpbox) map taskagent's `Task` onto this struct before passing
/// it to [`research`]. Keeping this local means no taskagent dependency
/// leaks into the skeleton.
#[derive(Clone, Debug)]
pub struct TaskContext {
    /// Opaque task identifier (e.g. the taskagent task id string).
    pub id: String,
    pub title: String,
    pub description: String,
}

impl TaskContext {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: description.into(),
        }
    }
}

// ── Prompt helpers ───────────────────────────────────────────────────────

/// Format a task list into a single block suitable for inclusion in
/// the research prompt. Tasks are numbered; descriptions (when
/// non-empty) are indented under the title.
pub fn format_task_context(tasks: &[TaskContext]) -> String {
    let mut s = String::new();
    for (i, t) in tasks.iter().enumerate() {
        s.push_str(&format!("{}. [{}] {}\n", i + 1, t.id, t.title.trim()));
        if !t.description.is_empty() {
            for line in t.description.lines() {
                s.push_str("    ");
                s.push_str(line);
                s.push('\n');
            }
        }
    }
    s
}

/// Build the research prompt. Pure — exposed for tests.
pub fn build_research_prompt(query: &str, context: &[TaskContext]) -> String {
    use crate::ai::wrap_untrusted;

    #[derive(Serialize)]
    struct DefaultCtx<'a> {
        query: &'a str,
    }
    #[derive(Serialize)]
    struct WithCtx<'a> {
        query: &'a str,
        tasks_block: &'a str,
    }

    if context.is_empty() {
        PromptRegistry::load("research", "default", &DefaultCtx { query })
            .expect("bundled research prompt is well-formed")
    } else {
        let tasks_block = wrap_untrusted("task context", &format_task_context(context));
        PromptRegistry::load(
            "research",
            "with_context",
            &WithCtx {
                query,
                tasks_block: &tasks_block,
            },
        )
        .expect("bundled research prompt is well-formed")
    }
}

// ── Annotator ────────────────────────────────────────────────────────────

/// Heuristically classify lines of a research answer into [`SensingItem`]s.
///
/// This is a **best-effort** annotator, not a parser: it scans each
/// non-empty line for leading markers that signal the kind of sensemaking
/// unit, then wraps the rest of the line as the item body.  Lines that
/// carry no recognised marker are emitted as `Knowledge` items (the safest
/// default for plain factual statements).
///
/// Recognised prefixes (case-insensitive):
///
/// | Prefix | Kind |
/// |---|---|
/// | `?` / `q:` | `Question` |
/// | `hypothesis:` / `h:` | `Hypothesis` |
/// | `risk:` / `r:` | `Risk` |
/// | `contradiction:` / `!` | `Contradiction` |
/// | `insight:` / `i:` | `Insight` |
/// | `gap:` / `research_gap:` | `ResearchGap` |
/// | `rejected:` / `rejected_idea:` | `RejectedIdea` (body only; no full provenance) |
/// | anything else | `Knowledge` |
///
/// The `label` is forwarded as [`Source::AiResearch`] on every item.
pub fn annotate_research_output(text: &str, label: Option<String>) -> Vec<SensingItem> {
    let source_proto = label.clone();
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (kind, body) = classify_line(line);
            SensingItem::new(kind, body)
                .with_source(Source::AiResearch {
                    label: source_proto.clone(),
                })
                // Research output is treated as medium-low confidence by
                // default — the caller can adjust per item.
                .with_confidence(Confidence::new(0.4))
        })
        .collect()
}

fn classify_line(line: &str) -> (SensingItemKind, &str) {
    // Try each prefix in order; return on first match.
    let prefixes: &[(&[&str], SensingItemKind)] = &[
        (&["?", "q:"], SensingItemKind::Question),
        (&["hypothesis:", "h:"], SensingItemKind::Hypothesis),
        (&["risk:", "r:"], SensingItemKind::Risk),
        (&["contradiction:", "!"], SensingItemKind::Contradiction),
        (&["insight:", "i:"], SensingItemKind::Insight),
        (&["gap:", "research_gap:"], SensingItemKind::ResearchGap),
        (&["rejected:", "rejected_idea:"], SensingItemKind::RejectedIdea),
    ];

    let lower = line.to_ascii_lowercase();
    for (markers, kind) in prefixes {
        for marker in *markers {
            if lower.starts_with(marker) {
                let body = line[marker.len()..].trim();
                return (*kind, body);
            }
        }
    }
    (SensingItemKind::Knowledge, line)
}

// ── Operation ────────────────────────────────────────────────────────────

/// Run a research query through the provider and return the answer as
/// a plain string. The caller is responsible for any side-effect (e.g.
/// saving the answer as a `Research` comment on a task).
pub async fn research<P: AiProvider>(
    provider: &P,
    query: &str,
    context: &[TaskContext],
) -> Result<String, SensemakingError> {
    if query.trim().is_empty() {
        return Err(SensemakingError::validation("research: query is empty"));
    }
    let prompt = build_research_prompt(query, context);
    let req = AiRequest {
        input: serde_json::Value::String(prompt),
        tools: vec![],
        tool_choice: None,
    };
    let outputs = provider.respond(req).await?;
    // Extract the first text output; fall back to validation error if none.
    for output in outputs {
        if let AiOutput::Text(text) = output {
            return Ok(text);
        }
    }
    Err(SensemakingError::validation(
        "research: provider returned no text output",
    ))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiError, AiOutput, AiProvider, AiRequest};
    use crate::types::SensingItemKind;

    /// Minimal fake provider for unit tests.
    struct FakeProvider {
        response: String,
    }

    impl FakeProvider {
        fn new(response: impl Into<String>) -> Self {
            Self { response: response.into() }
        }
    }

    impl AiProvider for FakeProvider {
        async fn respond(&self, _req: AiRequest) -> Result<Vec<AiOutput>, AiError> {
            Ok(vec![AiOutput::Text(self.response.clone())])
        }
    }

    fn sample_task(title: &str, body: &str) -> TaskContext {
        TaskContext::new("task-001", title, body)
    }

    #[test]
    fn default_prompt_omits_task_block() {
        let p = build_research_prompt("what's the failure mode?", &[]);
        assert!(p.contains("what's the failure mode?"));
        assert!(!p.contains("Task context:"));
    }

    #[test]
    fn with_context_prompt_lists_tasks() {
        let tasks = vec![
            sample_task("Wire OAuth", "Add Google + GitHub providers"),
            sample_task("Persist tokens", "Store hashed refresh tokens"),
        ];
        let p = build_research_prompt("how should we rotate tokens?", &tasks);
        assert!(p.contains("how should we rotate tokens?"));
        assert!(p.contains("Task context:"));
        assert!(p.contains("Wire OAuth"));
        assert!(p.contains("Persist tokens"));
        assert!(p.contains("Add Google + GitHub providers"));
    }

    #[tokio::test]
    async fn empty_query_returns_validation_error() {
        let provider = FakeProvider::new("unused");
        let err = research(&provider, "   ", &[]).await.unwrap_err();
        assert!(matches!(err, SensemakingError::Validation(_)));
    }

    #[tokio::test]
    async fn provider_receives_assembled_prompt_and_returns_text() {
        let provider = FakeProvider::new("answer body");
        let tasks = vec![sample_task("ctx", "")];
        let out = research(&provider, "explain", &tasks).await.unwrap();
        assert_eq!(out, "answer body");
    }

    #[test]
    fn annotate_classifies_prefixed_lines() {
        let text = "\
The sky is blue
? Why is the sky blue?
hypothesis: light scatters at short wavelengths
risk: sensor overheating at noon
contradiction: two sensors disagree on temperature
insight: peak readings cluster at 14:00
gap: no data for winter months
rejected: use infrared only — visible light also needed";

        let items = annotate_research_output(text, Some("test-run".into()));
        assert_eq!(items.len(), 8);
        assert_eq!(items[0].kind, SensingItemKind::Knowledge);
        assert_eq!(items[1].kind, SensingItemKind::Question);
        assert_eq!(items[2].kind, SensingItemKind::Hypothesis);
        assert_eq!(items[3].kind, SensingItemKind::Risk);
        assert_eq!(items[4].kind, SensingItemKind::Contradiction);
        assert_eq!(items[5].kind, SensingItemKind::Insight);
        assert_eq!(items[6].kind, SensingItemKind::ResearchGap);
        assert_eq!(items[7].kind, SensingItemKind::RejectedIdea);
        // Prefixes are stripped from bodies.
        assert_eq!(items[1].body, "Why is the sky blue?");
        assert_eq!(items[2].body, "light scatters at short wavelengths");
    }
}
