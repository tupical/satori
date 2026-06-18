//! AI research operation — answer a free-form query, optionally grounded
//! in the bodies of one or more existing tasks.
//!
//! [`research`] is **pure** in the I/O sense: it assembles the prompt from
//! the supplied query + task context and delegates the LLM call to the
//! provider through [`AiProvider::generate_text`], returning the answer as
//! a plain `String`. It never writes to storage. If a caller wants to
//! persist the answer (e.g. as a `Research` comment on a task), that is a
//! separate step performed through the ordinary command/comment API — it
//! lives outside this module so the sensemaking layer stays a clean
//! operation with no command-bus dependency.

use serde::Serialize;
use taskagent_domain::Task;
use taskagent_shared::CoreError;

use taskagent_ai_infra::{provider::AiProvider, untrusted::wrap_untrusted};

use crate::prompts::PromptRegistry;
use crate::types::{Confidence, SensingItem, SensingItemKind, Source};

/// Format a task list into a single block suitable for inclusion in
/// the research prompt. Tasks are numbered; descriptions (when
/// non-empty) are indented under the title.
pub fn format_task_context(tasks: &[Task]) -> String {
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
pub fn build_research_prompt(query: &str, context: &[Task]) -> String {
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
        let tasks_block =
            wrap_untrusted("task context", &format_task_context(context));
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

/// Run a research query through the provider and return the answer as
/// a plain string. The caller is responsible for any side-effect (e.g.
/// saving the answer as a `Research` comment on a task).
pub async fn research(
    provider: &dyn AiProvider,
    query: &str,
    context: &[Task],
) -> Result<String, CoreError> {
    if query.trim().is_empty() {
        return Err(CoreError::validation("research: query is empty"));
    }
    let prompt = build_research_prompt(query, context);
    provider.generate_text(prompt).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use taskagent_ai_infra::provider::testing::FakeProvider;
    use taskagent_domain::{Priority, Status};
    use taskagent_shared::{time, ProjectId, TaskId};

    fn sample_task(title: &str, body: &str) -> Task {
        let now = time::now();
        Task {
            id: TaskId::new(),
            project_id: Some(ProjectId::new()),
            title: title.into(),
            description: body.into(),
            status: Status::Todo,
            priority: Priority::P2,
            triage_state: None,
            due_at: None,
            created_at: now,
            updated_at: now,
            started_at: None,
            completed_at: None,
            created_by: None,
            completed_by: None,
            updated_by: None,
            updated_event_id: None,
            updated_event_seq: None,
            source_event_id: None,
        }
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
        let provider = FakeProvider::new("unused", serde_json::json!({}));
        let err = research(&provider, "   ", &[]).await.unwrap_err();
        assert_eq!(err.code(), "validation");
    }

    #[tokio::test]
    async fn provider_receives_assembled_prompt_and_returns_text() {
        let provider = FakeProvider::new("answer body", serde_json::json!({}));
        let tasks = vec![sample_task("ctx", "")];
        let out = research(&provider, "explain", &tasks).await.unwrap();
        assert_eq!(out, "answer body");
        let captured = provider.captured_prompts.lock().unwrap();
        assert!(captured[0].contains("explain"));
        assert!(captured[0].contains("ctx"));
    }

    #[test]
    fn annotate_classifies_prefixed_lines() {
        use crate::types::SensingItemKind;
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
