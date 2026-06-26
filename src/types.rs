//! Core sensemaking primitives.
//!
//! This module owns the eight kinds of sensemaking output that the layer
//! can produce from raw input material:
//!
//! | Kind | Meaning |
//! |---|---|
//! | `Knowledge` | A fact that is considered settled in this context. |
//! | `Question` | An open question that blocks or informs a decision. |
//! | `Hypothesis` | A testable claim that has not yet been validated. |
//! | `Risk` | A threat that might materialise and harm an objective. |
//! | `Contradiction` | Two signals that cannot both be true. |
//! | `Insight` | A non-obvious pattern worth acting on. |
//! | `RejectedIdea` | An idea that was considered and explicitly ruled out. |
//! | `ResearchGap` | Missing information that prevents confident reasoning. |
//!
//! [`RejectedIdea`] is a first-class object, not merely a negative label on
//! a `SensingItem`. It carries *why* the idea was rejected, *who* rejected it,
//! *what evidence* grounded the decision, and *when* it may be worth revisiting
//! — so that the record of deliberate rejection is never lost.
//!
//! # Universality
//! The types are domain-agnostic. `SensingTarget` uses a free-form `ExternalRef`
//! variant so that non-software contexts (research, strategy, …) can express
//! links without depending on task-tracker-specific IDs.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::time::{self, Timestamp};

// ── Local actor ────────────────────────────────────────────────────────────

/// Opaque actor reference — who performed an action.
///
/// Kept as a plain string so this crate has no dependency on daruma's
/// domain. mcpbox maps to/from daruma's `Actor` when wiring the layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Actor {
    /// Opaque identifier (user-id, agent-id, service name, …).
    pub id: String,
}

impl Actor {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    /// Convenience: anonymous user actor for tests and defaults.
    #[cfg(test)]
    pub fn user() -> Self {
        Self::new("user")
    }
}

// ── Strongly-typed ID for sensemaking items ────────────────────────────────

/// Opaque UUIDv7 identifier for a [`SensingItem`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SensingItemId(pub Uuid);

impl SensingItemId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for SensingItemId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SensingItemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "si_{}", self.0)
    }
}

impl FromStr for SensingItemId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.strip_prefix("si_").unwrap_or(s);
        Ok(Self(Uuid::parse_str(trimmed)?))
    }
}

/// Opaque UUIDv7 identifier for a [`RejectedIdea`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RejectedIdeaId(pub Uuid);

impl RejectedIdeaId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for RejectedIdeaId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RejectedIdeaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ri_{}", self.0)
    }
}

// ── SensingItemKind ────────────────────────────────────────────────────────

/// The eight kinds of sensemaking output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensingItemKind {
    /// A fact considered settled in this context.
    Knowledge,
    /// An open question that blocks or informs a decision.
    Question,
    /// A testable claim that has not yet been validated.
    Hypothesis,
    /// A threat that might materialise and harm an objective.
    Risk,
    /// Two signals that cannot both be true.
    Contradiction,
    /// A non-obvious pattern worth acting on.
    Insight,
    /// An idea that was considered and explicitly ruled out (first-class;
    /// stored as a [`RejectedIdea`] reference inside the item body).
    RejectedIdea,
    /// Missing information that prevents confident reasoning.
    ResearchGap,
}

impl SensingItemKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Knowledge => "knowledge",
            Self::Question => "question",
            Self::Hypothesis => "hypothesis",
            Self::Risk => "risk",
            Self::Contradiction => "contradiction",
            Self::Insight => "insight",
            Self::RejectedIdea => "rejected_idea",
            Self::ResearchGap => "research_gap",
        }
    }
}

impl fmt::Display for SensingItemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Confidence ────────────────────────────────────────────────────────────

/// Author's confidence in a sensing item, clamped to [0.0, 1.0].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Confidence(f32);

impl Confidence {
    /// Creates a confidence value. Clamps to [0.0, 1.0].
    pub fn new(v: f32) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn value(self) -> f32 {
        self.0
    }
}

impl Default for Confidence {
    /// Default: medium confidence (0.5).
    fn default() -> Self {
        Self(0.5)
    }
}

// ── Source reference (universal) ──────────────────────────────────────────

/// Where a sensing item or rejected idea came from.
///
/// Intentionally open: a plain string covers non-software contexts (a
/// journal article, a meeting transcript, a physical experiment result).
/// Task references use an opaque string ID so this crate has no compile-time
/// dependency on daruma. mcpbox maps to/from typed IDs when wiring.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Source {
    /// A task in the daruma tracker (opaque string ID).
    Task { id: String },
    /// A free-form reference (URL, citation, file path, …).
    External { ref_: String },
    /// Produced by an AI research operation with an optional run label.
    AiResearch { label: Option<String> },
}

// ── SensingItem ────────────────────────────────────────────────────────────

/// A single unit of sensemaking output.
///
/// For `kind == RejectedIdea`, [`SensingItem::rejected_idea`] carries the
/// full first-class record; `body` may be left empty or hold a brief summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensingItem {
    pub id: SensingItemId,
    pub kind: SensingItemKind,
    /// The main textual content of this sensing item.
    pub body: String,
    /// Author's confidence in this item (0 = none, 1 = certain).
    #[serde(default)]
    pub confidence: Confidence,
    /// Where this item originated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    /// Full rejected-idea record — populated only when `kind == RejectedIdea`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_idea: Option<RejectedIdea>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl SensingItem {
    /// Convenience constructor for any non-`RejectedIdea` kind.
    pub fn new(kind: SensingItemKind, body: impl Into<String>) -> Self {
        let now = time::now();
        Self {
            id: SensingItemId::new(),
            kind,
            body: body.into(),
            confidence: Confidence::default(),
            source: None,
            rejected_idea: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Constructor for a `RejectedIdea` item. The `body` can be a short
    /// summary; the full record lives in `rejected_idea`.
    pub fn rejected(idea: RejectedIdea) -> Self {
        let now = time::now();
        Self {
            id: SensingItemId::new(),
            kind: SensingItemKind::RejectedIdea,
            body: idea.what.clone(),
            confidence: Confidence::new(1.0),
            source: None,
            rejected_idea: Some(idea),
            created_at: now,
            updated_at: now,
        }
    }

    /// Builder: set confidence.
    pub fn with_confidence(mut self, c: Confidence) -> Self {
        self.confidence = c;
        self
    }

    /// Builder: attach a source.
    pub fn with_source(mut self, s: Source) -> Self {
        self.source = Some(s);
        self
    }
}

// ── RejectedIdea ─────────────────────────────────────────────────────────

/// Condition under which the rejection decision should be revisited.
///
/// Either a calendar date or a qualitative trigger — "when X happens".
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReconsiderTrigger {
    /// Revisit on or after this date.
    After { at: Timestamp },
    /// Revisit when this condition becomes true (free-form).
    When { condition: String },
}

/// A first-class record of a rejected idea.
///
/// Captures the full deliberation provenance so the decision is never
/// silently forgotten: what was considered, why it was ruled out, who
/// made the call, on what grounds, and under what circumstances it
/// should be reconsidered.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RejectedIdea {
    pub id: RejectedIdeaId,
    /// What the idea was (a concise description).
    pub what: String,
    /// Why it was rejected.
    pub why: String,
    /// Who rejected it.
    pub rejected_by: Actor,
    /// Links to evidence that grounded the rejection (URLs, doc refs, …).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<String>,
    /// Optional condition under which this rejection should be revisited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when_to_reconsider: Option<ReconsiderTrigger>,
    pub created_at: Timestamp,
}

impl RejectedIdea {
    pub fn new(
        what: impl Into<String>,
        why: impl Into<String>,
        rejected_by: Actor,
    ) -> Self {
        Self {
            id: RejectedIdeaId::new(),
            what: what.into(),
            why: why.into(),
            rejected_by,
            evidence: Vec::new(),
            when_to_reconsider: None,
            created_at: time::now(),
        }
    }

    /// Builder: attach evidence references.
    pub fn with_evidence(mut self, ev: impl Into<String>) -> Self {
        self.evidence.push(ev.into());
        self
    }

    /// Builder: set a reconsider trigger.
    pub fn with_reconsider(mut self, trigger: ReconsiderTrigger) -> Self {
        self.when_to_reconsider = Some(trigger);
        self
    }

    /// True when there is a stated condition under which this rejection
    /// should be reconsidered. A plain rejection without a trigger is
    /// considered final for planning purposes.
    pub fn is_reconsiderable(&self) -> bool {
        self.when_to_reconsider.is_some()
    }
}

// ── SensingLink ────────────────────────────────────────────────────────────

/// How a sensing item relates to a piece of intake material or a goal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    /// This item supports or confirms the target.
    Supports,
    /// This item contradicts the target.
    Contradicts,
    /// This item was derived from the target.
    DerivedFrom,
    /// This item addresses or resolves the target.
    Addresses,
}

/// A directed link from a [`SensingItem`] to a [`SensingTarget`].
///
/// `SensingTarget` is intentionally universal: it covers daruma task
/// references as well as opaque external references (a Notion goal, a
/// research objective, a physical project milestone — anything).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SensingLink {
    pub source: SensingItemId,
    pub target: SensingTarget,
    pub kind: LinkKind,
}

impl SensingLink {
    pub fn new(source: SensingItemId, target: SensingTarget, kind: LinkKind) -> Self {
        Self { source, target, kind }
    }
}

/// The referent of a [`SensingLink`].
///
/// Kept universal so sensemaking primitives are not coupled to any single
/// downstream layer (Intake, Decisions, Daruma, …). All IDs are opaque
/// strings so this crate has no compile-time dependency on sibling layers.
/// mcpbox maps to/from typed IDs when wiring the layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SensingTarget {
    /// A raw intake item identified by an opaque string (e.g. the RawItem id
    /// from the Intake layer).
    RawItem { id: String },
    /// A goal or objective in the Decisions layer, identified by an opaque
    /// string.
    Goal { id: String },
    /// A task in the daruma tracker (opaque string ID).
    Task { id: String },
    /// A free-form external reference (URL, document heading, …).
    External { ref_: String },
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensing_item_kind_serde_roundtrip() {
        // All eight kinds must survive a JSON roundtrip with their snake_case
        // wire names.
        let kinds = [
            SensingItemKind::Knowledge,
            SensingItemKind::Question,
            SensingItemKind::Hypothesis,
            SensingItemKind::Risk,
            SensingItemKind::Contradiction,
            SensingItemKind::Insight,
            SensingItemKind::RejectedIdea,
            SensingItemKind::ResearchGap,
        ];
        for kind in kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let back: SensingItemKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, back, "roundtrip failed for {kind}");
        }
    }

    #[test]
    fn rejected_idea_is_reconsiderable_only_when_trigger_set() {
        let actor = Actor::user();
        let plain = RejectedIdea::new("use microservices", "premature complexity", actor.clone());
        assert!(!plain.is_reconsiderable());

        let with_trigger = RejectedIdea::new("use microservices", "premature complexity", actor)
            .with_reconsider(ReconsiderTrigger::When {
                condition: "team grows past 10 engineers".into(),
            });
        assert!(with_trigger.is_reconsiderable());
    }

    #[test]
    fn sensing_item_rejected_wraps_idea() {
        let idea = RejectedIdea::new(
            "rewrite in Go",
            "no bandwidth, Rust already chosen",
            Actor::user(),
        )
        .with_evidence("https://decision-log.example/001");

        let item = SensingItem::rejected(idea.clone());
        assert_eq!(item.kind, SensingItemKind::RejectedIdea);
        assert_eq!(item.body, idea.what);
        assert!(item.rejected_idea.is_some());
        assert_eq!(item.rejected_idea.unwrap().evidence.len(), 1);
    }

    #[test]
    fn sensing_link_new_stores_all_fields() {
        let src = SensingItemId::new();
        let target = SensingTarget::RawItem { id: "raw_001".into() };
        let link = SensingLink::new(src, target.clone(), LinkKind::DerivedFrom);
        assert_eq!(link.source, src);
        assert_eq!(link.target, target);
        assert_eq!(link.kind, LinkKind::DerivedFrom);
    }

    #[test]
    fn confidence_clamps_to_unit_interval() {
        assert_eq!(Confidence::new(1.5).value(), 1.0);
        assert_eq!(Confidence::new(-0.1).value(), 0.0);
        assert_eq!(Confidence::new(0.7).value(), 0.7);
    }

    #[test]
    fn sensing_item_id_display_and_parse_roundtrip() {
        let id = SensingItemId::new();
        let s = id.to_string();
        assert!(s.starts_with("si_"), "got: {s}");
        let back: SensingItemId = s.parse().unwrap();
        assert_eq!(id, back);
    }
}
