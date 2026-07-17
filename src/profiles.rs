//! Process mining of agent profiles from the daruma event stream.
//!
//! Satori is a **stateless calculator**: [`mine_agent_profiles`] folds a slice
//! of daruma event envelopes (their JSON wire form) into per-agent
//! [`AgentProfile`] aggregates. The crate has **no** dependency on daruma —
//! the private `view` structs below mirror the `daruma_events::EventEnvelope`
//! JSON layout (`{ id, seq, occurred_at, actor, payload }` where `payload` is
//! the `#[serde(tag = "type", rename_all = "snake_case")]` `Event` union) for
//! exactly the variants the miner consumes. Any other event type parses as
//! `Other` and is ignored; a bare payload without its envelope is accepted
//! too (timestamps then fall back to the payload's own `at` field).
//!
//! Events are processed in slice order — the caller hands over the log in
//! append order. Envelopes that fail to parse are counted in
//! [`ProfileReport::envelopes_skipped`] and never abort the fold: mining is a
//! best-effort read over a noisy append-only log, not a validation gate.
//!
//! # Mined aggregates
//! - **reopen_count** — `task_reopened` events attributed to the agent that
//!   last completed the task (`task_completed.completion_note.actor`, falling
//!   back to the envelope actor, then to the last claiming agent). A reopen
//!   means "the work this agent closed came back".
//! - **fixup_count** — churn from claim/release/block chains:
//!   claims released unfinished (`work_unit_released` attributed to the
//!   current holder + `agent_released`), re-blocks (2nd and later
//!   `work_unit_blocked` on the same unit), and handoffs rejected with the
//!   agent as producer (`handoff_rejected` on a contract the agent
//!   requested). Each component is a "work had to be redone" signal.
//! - **blocked_ms / open_blocked** — a `work_unit_blocked` opens an interval
//!   at its timestamp; the next resuming event for the same unit
//!   (`work_unit_claimed` / `work_unit_started` / `work_unit_completed` /
//!   `work_unit_released`) closes it. The duration is attributed to the
//!   unit's holder at block time. Intervals still open at the end of the
//!   stream are closed at the `as_of` argument when supplied (and always
//!   counted in `open_blocked`).
//! - **conflict_count** — `task_contested` events naming the agent in their
//!   `actors` list.
//!
//! # Responsibility patterns
//! For every `(agent, capability)` pair the miner folds outcome signals from
//! the unit's `capability_tags` — the same signal values the daruma core
//! projection uses (migration 0044): completed = 1.0, released-unfinished =
//! 0.4, blocked = 0.3, folded as an EWMA with step `1 / min(n+1, 20)` and
//! `confidence = n / (n + 5)`. Mined patterns enter the lifecycle as
//! [`PatternLifecycle::Suggested`].
//!
//! Lifecycle promotion is **human-gated**. The daruma core stores only the
//! `user_set` override (`agent_capability_profiles.source = 'user_set'`);
//! interpretation lives here. The host passes the current overrides as the
//! `user_set` argument:
//! - an override with `score > 0` is a human *accept*: the pattern becomes
//!   [`PatternLifecycle::Active`] with the user's score and full confidence
//!   (the user's word wins, mirroring the core's "mining never overwrites
//!   `user_set`" rule);
//! - an override with `score == 0` is a human *reject*: the pattern becomes
//!   [`PatternLifecycle::Rejected`];
//! - anything without an override stays `Suggested` (source `inferred`).
//!
//! # Workflow confidence
//! `workflow_confidence = w_ev · (0.40·r_comp + 0.25·r_hand + 0.20·p_reopen + 0.15·p_conf)`
//! where
//! - `r_comp` — mean outcome signal over the agent's unit events
//!   (`(1.0·completed + 0.4·released + 0.3·blocked) / n`, n = their count);
//!   the dominant term: direct outcome quality.
//! - `r_hand` — handoff acceptance ratio for contracts the agent produced
//!   (`accepted / (accepted + rejected)`, 1.0 when the agent produced none —
//!   absence of evidence is not penalised); the next most direct
//!   collaboration-quality signal.
//! - `p_reopen = 1 / (1 + reopens)` — rework penalty.
//! - `p_conf = 1 / (1 + conflicts)` — coordination-friction penalty; the
//!   weakest term because contest attribution is the least causal signal.
//! - `w_ev = n / (n + 5)` — evidence saturation, the same shape as the core
//!   projection's confidence, so a single lucky completion cannot yield a
//!   confident profile (cold start at n=1 caps the blend at 1/6).
//!
//! Every factor lies in `[0, 1]`, so the product does too; the weights sum
//! to 1. Raising completions raises both `r_comp` and `w_ev`, adding a
//! rejection lowers `r_hand`, and reopens/conflicts lower their penalties —
//! the score is monotone in each evidence dimension.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::time::Timestamp;

// ── Signal constants (mirror daruma migration 0044 semantics) ───────────────

/// Outcome signal credited when a unit completes.
const SIGNAL_COMPLETED: f64 = 1.0;
/// Outcome signal credited when a claim is released unfinished.
const SIGNAL_RELEASED: f64 = 0.4;
/// Outcome signal credited when a unit blocks.
const SIGNAL_BLOCKED: f64 = 0.3;

/// EWMA step cap: early evidence moves a score fast, long history is stable.
const EWMA_STEP_CAP: f64 = 20.0;
/// Evidence saturation half-point: `confidence = n / (n + 5)`.
const EVIDENCE_HALF: f64 = 5.0;

/// workflow_confidence blend weights (sum = 1.0); see module docs.
const W_COMPLETION: f64 = 0.40;
const W_HANDOFF: f64 = 0.25;
const W_REOPEN: f64 = 0.20;
const W_CONFLICT: f64 = 0.15;

// ── Public input ────────────────────────────────────────────────────────────

/// A human-set capability override, mirroring the daruma core's
/// `agent_capability_profiles` rows with `source = 'user_set'`. The core
/// stores the override; the lifecycle interpretation lives here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserSetOverride {
    /// Agent id (opaque string; daruma `AgentId` serialised as UUID text).
    pub agent_id: String,
    /// Capability tag the human ruled on (matches unit `capability_tags`).
    pub capability: String,
    /// `score > 0` accepts the pattern (→ `active`); `score == 0` rejects it
    /// (→ `rejected`). Clamped to [0, 1].
    pub score: f64,
}

// ── Public output ───────────────────────────────────────────────────────────

/// Lifecycle of a mined responsibility pattern.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternLifecycle {
    /// Mined from the event stream, not yet human-confirmed.
    Suggested,
    /// Promoted by a human accept (`user_set` override with score > 0).
    Active,
    /// Ruled out by a human (`user_set` override with score == 0).
    Rejected,
}

/// Where a responsibility pattern's score came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternSource {
    /// Folded from `WorkUnit*` events.
    Inferred,
    /// Human override from the daruma core (`user_set`).
    UserSet,
}

/// One `(agent, capability)` responsibility pattern with its lifecycle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsibilityPattern {
    pub agent_id: String,
    pub capability: String,
    pub lifecycle: PatternLifecycle,
    pub source: PatternSource,
    /// EWMA outcome signal in [0, 1]; the user's score when `source ==
    /// UserSet`.
    pub score: f64,
    /// Number of mined outcome signals behind the pattern (0 for a bare
    /// override on an unobserved pair).
    pub evidence_count: u64,
    /// Evidence saturation `n / (n + 5)`; 1.0 for `user_set` (human
    /// confidence is taken as certain).
    pub confidence: f64,
}

/// Per-agent process-mined profile.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentProfile {
    pub agent_id: String,
    /// Units completed by this agent (`completed_by`, else holder).
    pub completed_units: u64,
    /// Claims handed back unfinished (`work_unit_released` + `agent_released`).
    pub released_unfinished: u64,
    /// `work_unit_blocked` events while this agent held the unit.
    pub blocked_events: u64,
    /// Total wall-clock milliseconds spent blocked (closed intervals).
    pub blocked_ms: u64,
    /// Blocked intervals still open at the end of the stream.
    pub open_blocked: u64,
    /// Tasks reopened after this agent completed them.
    pub reopen_count: u64,
    /// Churn total: releases + re-blocks + handoffs rejected as producer.
    pub fixup_count: u64,
    /// `task_contested` events naming this agent.
    pub conflict_count: u64,
    /// Handoffs this agent produced that were accepted.
    pub handoffs_accepted: u64,
    /// Handoffs this agent produced that were rejected.
    pub handoffs_rejected: u64,
    /// Mean handoff response latency (request → accept/reject) in ms, when
    /// any response carried the P6 `latency_ms` mining fact.
    pub mean_handoff_latency_ms: Option<f64>,
    /// Mean unit cycle time in ms, when completions carried the P6
    /// `elapsed_ms` mining fact.
    pub mean_cycle_ms: Option<f64>,
    /// Blended workflow confidence in [0, 1]; see module docs for the
    /// formula and weight justification.
    pub workflow_confidence: f64,
    /// Responsibility patterns for this agent, sorted by capability.
    pub responsibility: Vec<ResponsibilityPattern>,
}

/// The mining result over one event stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Per-agent profiles, sorted by agent id.
    pub agents: Vec<AgentProfile>,
    /// Envelopes (or bare payloads) that parsed successfully.
    pub envelopes_parsed: u64,
    /// Inputs that failed to parse and were skipped (best-effort mining).
    pub envelopes_skipped: u64,
}

// ── Event views (private; mirror the daruma JSON wire form) ─────────────────

/// Minimal mirror of `daruma_domain::Actor`: `{ "kind": "user" }` or
/// `{ "kind": "agent", "id": …, "name": … }`.
#[derive(Deserialize)]
struct ActorView {
    kind: String,
    #[serde(default)]
    id: Option<String>,
}

impl ActorView {
    /// The agent id, when this actor is an agent.
    fn agent_id(&self) -> Option<&str> {
        if self.kind == "agent" {
            self.id.as_deref()
        } else {
            None
        }
    }
}

/// The fields of `daruma_domain::WorkUnit` the miner reads.
#[derive(Deserialize)]
struct WorkUnitView {
    id: String,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    capability_tags: Vec<String>,
    #[serde(default)]
    owner_agent_id: Option<String>,
}

/// The fields of `daruma_domain::HandoffContract` the miner reads.
#[derive(Deserialize)]
struct HandoffView {
    id: String,
    #[serde(default)]
    owner_agent_id: Option<String>,
}

/// The fields of `daruma_domain::CompletionNote` the miner reads.
#[derive(Deserialize)]
struct CompletionNoteView {
    #[serde(default)]
    actor: Option<ActorView>,
}

/// `daruma_events::Event` variants the miner consumes; every other variant
/// parses as `Other` and is ignored.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PayloadView {
    WorkUnitCreated {
        work_unit: WorkUnitView,
    },
    WorkUnitClaimed {
        work_unit_id: String,
        agent_id: String,
    },
    WorkUnitStarted {
        work_unit_id: String,
        #[serde(default)]
        at: Option<Timestamp>,
    },
    WorkUnitBlocked {
        work_unit_id: String,
        #[serde(default)]
        at: Option<Timestamp>,
    },
    WorkUnitCompleted {
        work_unit_id: String,
        #[serde(default)]
        completed_by: Option<String>,
        #[serde(default)]
        elapsed_ms: Option<i64>,
        #[serde(default)]
        at: Option<Timestamp>,
    },
    WorkUnitReleased {
        work_unit_id: String,
        #[serde(default)]
        at: Option<Timestamp>,
    },
    HandoffRequested {
        handoff: HandoffView,
    },
    HandoffAccepted {
        handoff_id: String,
        #[serde(default)]
        latency_ms: Option<i64>,
        #[allow(dead_code)]
        #[serde(default)]
        at: Option<Timestamp>,
    },
    HandoffRejected {
        handoff_id: String,
        #[serde(default)]
        latency_ms: Option<i64>,
        #[allow(dead_code)]
        #[serde(default)]
        at: Option<Timestamp>,
    },
    TaskCompleted {
        task_id: String,
        #[serde(default)]
        completion_note: Option<CompletionNoteView>,
    },
    TaskReopened {
        task_id: String,
    },
    TaskContested {
        #[allow(dead_code)]
        task_id: String,
        #[serde(default)]
        actors: Vec<ActorView>,
    },
    AgentClaimed {
        agent_id: String,
        task_id: String,
    },
    AgentReleased {
        agent_id: String,
        #[allow(dead_code)]
        task_id: String,
    },
    #[serde(other)]
    Other,
}

/// Minimal mirror of `daruma_events::EventEnvelope`.
#[derive(Deserialize)]
struct EnvelopeView {
    #[serde(default)]
    occurred_at: Option<Timestamp>,
    #[serde(default)]
    actor: Option<ActorView>,
    payload: PayloadView,
}

// ── Mining state ────────────────────────────────────────────────────────────

/// EWMA fold of outcome signals for one `(agent, capability)` pair.
#[derive(Default)]
struct CapAgg {
    score: f64,
    n: u64,
}

impl CapAgg {
    /// Fold one outcome signal: step `1 / min(n+1, 20)` — fast early, stable
    /// with history (mirrors the core projection's SQL fold).
    fn fold(&mut self, signal: f64) {
        let step = 1.0 / ((self.n + 1) as f64).min(EWMA_STEP_CAP);
        self.score += (signal - self.score) * step;
        self.n += 1;
    }
}

/// Per-agent running totals.
#[derive(Default)]
struct AgentAgg {
    completed: u64,
    released: u64,
    blocked: u64,
    reblocks: u64,
    blocked_ms: u64,
    reopens: u64,
    conflicts: u64,
    handoff_accepted: u64,
    handoff_rejected: u64,
    handoff_latency_ms_total: i64,
    handoff_latency_n: u64,
    cycle_ms_total: i64,
    cycle_n: u64,
    caps: HashMap<String, CapAgg>,
}

#[derive(Default)]
struct MinerState {
    agents: HashMap<String, AgentAgg>,
    /// unit id → current holder (from `work_unit_claimed` / the unit row).
    unit_holder: HashMap<String, String>,
    unit_tags: HashMap<String, Vec<String>>,
    unit_task: HashMap<String, String>,
    /// unit id → blocked events seen so far (2nd+ = re-block).
    unit_blocks: HashMap<String, u64>,
    /// unit id → (blocked since, holder at block time).
    blocked_since: HashMap<String, (Timestamp, String)>,
    /// task id → agent that last completed it (reopen attribution).
    task_last_completer: HashMap<String, String>,
    /// task id → claiming agents in claim order (attribution fallback).
    task_claimers: HashMap<String, Vec<String>>,
    /// handoff id → producer agent (from `handoff_requested.owner_agent_id`).
    handoff_producer: HashMap<String, String>,
}

impl MinerState {
    fn agent(&mut self, id: &str) -> &mut AgentAgg {
        self.agents.entry(id.to_string()).or_default()
    }

    /// Credit `signal` to the holder's `(agent, tag)` EWMA for every
    /// capability tag on the unit.
    fn credit_signal(&mut self, unit_id: &str, holder: &str, signal: f64) {
        if let Some(tags) = self.unit_tags.get(unit_id).cloned() {
            let agg = self.agent(holder);
            for tag in tags {
                agg.caps.entry(tag).or_default().fold(signal);
            }
        }
    }

    /// Close the unit's open blocked interval at `ts`, attributing the
    /// duration to the holder recorded when the interval opened.
    fn close_blocked(&mut self, unit_id: &str, ts: Option<Timestamp>) {
        let Some((since, holder)) = self.blocked_since.remove(unit_id) else {
            return;
        };
        let Some(ts) = ts else { return };
        let ms = (ts - since).num_milliseconds().max(0) as u64;
        self.agent(&holder).blocked_ms += ms;
    }

    fn apply(&mut self, env: EnvelopeView) {
        let EnvelopeView {
            occurred_at,
            actor,
            payload,
        } = env;
        // The envelope stamp wins; fall back to the payload's own `at`.
        let pick = |at: Option<Timestamp>| occurred_at.or(at);
        match payload {
            PayloadView::WorkUnitCreated { work_unit } => {
                self.unit_tags
                    .insert(work_unit.id.clone(), work_unit.capability_tags);
                if let Some(task) = work_unit.task_id {
                    self.unit_task.insert(work_unit.id.clone(), task);
                }
                if let Some(owner) = work_unit.owner_agent_id {
                    self.unit_holder.insert(work_unit.id, owner);
                }
            }
            PayloadView::WorkUnitClaimed {
                work_unit_id,
                agent_id,
            } => {
                self.close_blocked(&work_unit_id, occurred_at);
                self.unit_holder.insert(work_unit_id, agent_id);
            }
            PayloadView::WorkUnitStarted { work_unit_id, at } => {
                self.close_blocked(&work_unit_id, pick(at));
            }
            PayloadView::WorkUnitBlocked { work_unit_id, at } => {
                let ts = pick(at);
                let holder = self.unit_holder.get(&work_unit_id).cloned();
                if let Some(holder) = holder {
                    let agg = self.agent(&holder);
                    agg.blocked += 1;
                    let blocks = self.unit_blocks.entry(work_unit_id.clone()).or_insert(0);
                    *blocks += 1;
                    if *blocks > 1 {
                        self.agent(&holder).reblocks += 1;
                    }
                    self.credit_signal(&work_unit_id, &holder, SIGNAL_BLOCKED);
                    if let Some(ts) = ts {
                        // Re-block while already blocked keeps the earliest
                        // `since` — the unit never actually resumed.
                        self.blocked_since
                            .entry(work_unit_id)
                            .or_insert((ts, holder));
                    }
                }
            }
            PayloadView::WorkUnitCompleted {
                work_unit_id,
                completed_by,
                elapsed_ms,
                at,
            } => {
                let ts = pick(at);
                self.close_blocked(&work_unit_id, ts);
                let agent = completed_by.or_else(|| self.unit_holder.get(&work_unit_id).cloned());
                if let Some(agent) = agent {
                    let agg = self.agent(&agent);
                    agg.completed += 1;
                    if let Some(ms) = elapsed_ms.filter(|ms| *ms >= 0) {
                        agg.cycle_ms_total += ms;
                        agg.cycle_n += 1;
                    }
                    self.credit_signal(&work_unit_id, &agent, SIGNAL_COMPLETED);
                    if let Some(task) = self.unit_task.get(&work_unit_id) {
                        self.task_last_completer.insert(task.clone(), agent);
                    }
                }
                self.unit_holder.remove(&work_unit_id);
            }
            PayloadView::WorkUnitReleased { work_unit_id, at } => {
                let ts = pick(at);
                self.close_blocked(&work_unit_id, ts);
                if let Some(holder) = self.unit_holder.remove(&work_unit_id) {
                    self.agent(&holder).released += 1;
                    self.credit_signal(&work_unit_id, &holder, SIGNAL_RELEASED);
                }
            }
            PayloadView::AgentClaimed { agent_id, task_id } => {
                self.task_claimers
                    .entry(task_id)
                    .or_default()
                    .push(agent_id);
            }
            PayloadView::AgentReleased { agent_id, .. } => {
                self.agent(&agent_id).released += 1;
            }
            PayloadView::HandoffRequested { handoff } => {
                if let Some(owner) = handoff.owner_agent_id {
                    self.handoff_producer.insert(handoff.id, owner);
                }
            }
            PayloadView::HandoffAccepted {
                handoff_id,
                latency_ms,
                ..
            } => {
                if let Some(producer) = self.handoff_producer.get(&handoff_id).cloned() {
                    let agg = self.agent(&producer);
                    agg.handoff_accepted += 1;
                    if let Some(ms) = latency_ms.filter(|ms| *ms >= 0) {
                        agg.handoff_latency_ms_total += ms;
                        agg.handoff_latency_n += 1;
                    }
                }
            }
            PayloadView::HandoffRejected {
                handoff_id,
                latency_ms,
                ..
            } => {
                if let Some(producer) = self.handoff_producer.get(&handoff_id).cloned() {
                    let agg = self.agent(&producer);
                    agg.handoff_rejected += 1;
                    if let Some(ms) = latency_ms.filter(|ms| *ms >= 0) {
                        agg.handoff_latency_ms_total += ms;
                        agg.handoff_latency_n += 1;
                    }
                }
            }
            PayloadView::TaskCompleted {
                task_id,
                completion_note,
            } => {
                let completer = completion_note
                    .and_then(|note| note.actor.and_then(|a| a.agent_id().map(str::to_string)))
                    .or_else(|| actor.and_then(|a| a.agent_id().map(str::to_string)));
                if let Some(agent) = completer {
                    self.task_last_completer.insert(task_id, agent);
                }
            }
            PayloadView::TaskReopened { task_id } => {
                // Attribute to the agent whose completion came back; fall
                // back to the last claiming agent when no completion was
                // seen in this stream.
                let target = self.task_last_completer.get(&task_id).cloned().or_else(|| {
                    self.task_claimers
                        .get(&task_id)
                        .and_then(|v| v.last())
                        .cloned()
                });
                if let Some(agent) = target {
                    self.agent(&agent).reopens += 1;
                }
            }
            PayloadView::TaskContested { actors, .. } => {
                for a in actors {
                    if let Some(agent) = a.agent_id() {
                        self.agent(agent).conflicts += 1;
                    }
                }
            }
            PayloadView::Other => {}
        }
    }
}

// ── Public entry point ──────────────────────────────────────────────────────

/// Fold a stream of daruma event envelopes (JSON wire form, in log order)
/// into per-agent process-mined profiles.
///
/// - `events` — `daruma_events::EventEnvelope` JSON values; bare payloads
///   (without the envelope) are accepted too. Unparseable inputs are skipped
///   and counted, never fatal.
/// - `user_set` — human capability overrides from the daruma core
///   (`agent_capability_profiles.source = 'user_set'`); they drive the
///   responsibility lifecycle (suggested → active/rejected).
/// - `as_of` — closes blocked intervals still open at the end of the stream
///   so their (partial) duration counts toward `blocked_ms`.
pub fn mine_agent_profiles(
    events: &[serde_json::Value],
    user_set: &[UserSetOverride],
    as_of: Option<Timestamp>,
) -> ProfileReport {
    let mut state = MinerState::default();
    let mut envelopes_parsed = 0_u64;
    let mut envelopes_skipped = 0_u64;

    for raw in events {
        let parsed = serde_json::from_value::<EnvelopeView>(raw.clone()).or_else(|_| {
            serde_json::from_value::<PayloadView>(raw.clone()).map(|payload| EnvelopeView {
                occurred_at: None,
                actor: None,
                payload,
            })
        });
        match parsed {
            Ok(env) => {
                envelopes_parsed += 1;
                state.apply(env);
            }
            Err(_) => envelopes_skipped += 1,
        }
    }

    // Close still-open blocked intervals at the horizon and count them.
    let open: Vec<(Timestamp, String)> = state
        .blocked_since
        .values()
        .map(|(since, holder)| (*since, holder.clone()))
        .collect();
    let mut open_blocked: HashMap<String, u64> = HashMap::new();
    for (since, holder) in open {
        *open_blocked.entry(holder.clone()).or_default() += 1;
        if let Some(end) = as_of {
            let ms = (end - since).num_milliseconds().max(0) as u64;
            state.agent(&holder).blocked_ms += ms;
        }
    }

    // Apply human overrides on top of the mined (agent, capability) fold.
    let mut overrides: HashMap<(String, String), f64> = HashMap::new();
    for o in user_set {
        overrides.insert(
            (o.agent_id.clone(), o.capability.clone()),
            o.score.clamp(0.0, 1.0),
        );
        // An override can name an agent/pair the stream never produced —
        // make sure the agent shows up in the report.
        state.agent(&o.agent_id);
    }

    let mut agents: Vec<AgentProfile> = state
        .agents
        .into_iter()
        .map(|(agent_id, agg)| {
            let responsibility = build_responsibility(&agent_id, agg.caps.iter(), &overrides);
            let n = agg.completed + agg.released + agg.blocked;
            let workflow_confidence = workflow_confidence(&agg, n);
            AgentProfile {
                agent_id,
                completed_units: agg.completed,
                released_unfinished: agg.released,
                blocked_events: agg.blocked,
                blocked_ms: agg.blocked_ms,
                open_blocked: 0, // patched below
                reopen_count: agg.reopens,
                fixup_count: agg.released + agg.reblocks + agg.handoff_rejected,
                conflict_count: agg.conflicts,
                handoffs_accepted: agg.handoff_accepted,
                handoffs_rejected: agg.handoff_rejected,
                mean_handoff_latency_ms: (agg.handoff_latency_n > 0)
                    .then(|| agg.handoff_latency_ms_total as f64 / agg.handoff_latency_n as f64),
                mean_cycle_ms: (agg.cycle_n > 0)
                    .then(|| agg.cycle_ms_total as f64 / agg.cycle_n as f64),
                workflow_confidence,
                responsibility,
            }
        })
        .collect();
    for profile in &mut agents {
        profile.open_blocked = open_blocked.get(&profile.agent_id).copied().unwrap_or(0);
    }
    agents.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));

    ProfileReport {
        agents,
        envelopes_parsed,
        envelopes_skipped,
    }
}

/// Assemble the `(agent, capability)` patterns: mined fold → `suggested`,
/// then apply the human overrides (accept → `active`, reject → `rejected`).
fn build_responsibility<'a>(
    agent_id: &str,
    caps: impl Iterator<Item = (&'a String, &'a CapAgg)>,
    overrides: &HashMap<(String, String), f64>,
) -> Vec<ResponsibilityPattern> {
    let mut patterns: Vec<ResponsibilityPattern> = caps
        .map(|(cap, agg)| ResponsibilityPattern {
            agent_id: agent_id.to_string(),
            capability: cap.clone(),
            lifecycle: PatternLifecycle::Suggested,
            source: PatternSource::Inferred,
            score: agg.score,
            evidence_count: agg.n,
            confidence: agg.n as f64 / (agg.n as f64 + EVIDENCE_HALF),
        })
        .collect();

    for ((oid, ocap), score) in overrides {
        if oid != agent_id {
            continue;
        }
        let lifecycle = if *score > 0.0 {
            PatternLifecycle::Active
        } else {
            PatternLifecycle::Rejected
        };
        match patterns.iter_mut().find(|p| p.capability == *ocap) {
            // Human wins: keep the mined evidence count but take the
            // user's score and full confidence.
            Some(p) => {
                p.lifecycle = lifecycle;
                p.source = PatternSource::UserSet;
                p.score = *score;
                p.confidence = 1.0;
            }
            None => patterns.push(ResponsibilityPattern {
                agent_id: agent_id.to_string(),
                capability: ocap.clone(),
                lifecycle,
                source: PatternSource::UserSet,
                score: *score,
                evidence_count: 0,
                confidence: 1.0,
            }),
        }
    }
    patterns.sort_by(|a, b| a.capability.cmp(&b.capability));
    patterns
}

/// Blended workflow confidence in [0, 1]; formula and weight justification
/// in the module docs.
fn workflow_confidence(agg: &AgentAgg, n: u64) -> f64 {
    let mean_signal = if n > 0 {
        (agg.completed as f64 * SIGNAL_COMPLETED
            + agg.released as f64 * SIGNAL_RELEASED
            + agg.blocked as f64 * SIGNAL_BLOCKED)
            / n as f64
    } else {
        // No outcome evidence; `w_ev` zeroes the blend anyway.
        1.0
    };
    let handoffs = agg.handoff_accepted + agg.handoff_rejected;
    let handoff_ratio = if handoffs > 0 {
        agg.handoff_accepted as f64 / handoffs as f64
    } else {
        // Absence of handoff evidence is not penalised.
        1.0
    };
    let reopen_penalty = 1.0 / (1.0 + agg.reopens as f64);
    let conflict_penalty = 1.0 / (1.0 + agg.conflicts as f64);
    let evidence = n as f64 / (n as f64 + EVIDENCE_HALF);

    evidence
        * (W_COMPLETION * mean_signal
            + W_HANDOFF * handoff_ratio
            + W_REOPEN * reopen_penalty
            + W_CONFLICT * conflict_penalty)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use serde_json::json;

    const AGENT: &str = "11111111-1111-1111-1111-111111111111";
    const OTHER: &str = "22222222-2222-2222-2222-222222222222";
    const UNIT: &str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    const TASK: &str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
    const HANDOFF: &str = "cccccccc-cccc-cccc-cccc-cccccccccccc";

    fn t(min: i64) -> Timestamp {
        Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap() + Duration::minutes(min)
    }

    fn ts(min: i64) -> String {
        t(min).to_rfc3339()
    }

    /// Envelope in the daruma wire form.
    fn env(at: i64, actor: serde_json::Value, payload: serde_json::Value) -> serde_json::Value {
        json!({
            "id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
            "seq": 1,
            "occurred_at": ts(at),
            "actor": actor,
            "payload": payload,
        })
    }

    fn agent_actor(id: &str) -> serde_json::Value {
        json!({ "kind": "agent", "id": id, "name": "bot" })
    }

    fn user_actor() -> serde_json::Value {
        json!({ "kind": "user" })
    }

    fn unit_created(at: i64, tags: &[&str]) -> serde_json::Value {
        env(
            at,
            user_actor(),
            json!({
                "type": "work_unit_created",
                "work_unit": {
                    "id": UNIT,
                    "task_id": TASK,
                    "title": "u",
                    "status": "todo",
                    "priority": "p2",
                    "capability_tags": tags,
                    "created_at": ts(at),
                    "updated_at": ts(at)
                }
            }),
        )
    }

    fn claimed(at: i64, agent: &str) -> serde_json::Value {
        env(
            at,
            agent_actor(agent),
            json!({ "type": "work_unit_claimed", "work_unit_id": UNIT, "agent_id": agent, "expires_at": ts(at + 60) }),
        )
    }

    fn blocked(at: i64) -> serde_json::Value {
        env(
            at,
            agent_actor(AGENT),
            json!({ "type": "work_unit_blocked", "work_unit_id": UNIT, "reason": "stuck", "at": ts(at) }),
        )
    }

    fn started(at: i64) -> serde_json::Value {
        env(
            at,
            agent_actor(AGENT),
            json!({ "type": "work_unit_started", "work_unit_id": UNIT, "at": ts(at) }),
        )
    }

    fn completed(at: i64, elapsed_ms: i64) -> serde_json::Value {
        env(
            at,
            user_actor(),
            json!({
                "type": "work_unit_completed",
                "work_unit_id": UNIT,
                "outcome": "ok",
                "completed_by": AGENT,
                "elapsed_ms": elapsed_ms,
                "at": ts(at)
            }),
        )
    }

    fn released(at: i64) -> serde_json::Value {
        env(
            at,
            agent_actor(AGENT),
            json!({ "type": "work_unit_released", "work_unit_id": UNIT, "at": ts(at) }),
        )
    }

    fn profile<'a>(report: &'a ProfileReport, agent: &str) -> &'a AgentProfile {
        report
            .agents
            .iter()
            .find(|a| a.agent_id == agent)
            .expect("agent profile present")
    }

    // ── aggregates ──────────────────────────────────────────────────────────

    #[test]
    fn mines_completion_blocked_time_and_cycle() {
        let events = vec![
            unit_created(0, &["frontend"]),
            claimed(1, AGENT),
            blocked(2),
            started(17), // 15 blocked minutes
            completed(30, 1_800_000),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        assert_eq!(report.envelopes_parsed, 5);
        assert_eq!(report.envelopes_skipped, 0);

        let p = profile(&report, AGENT);
        assert_eq!(p.completed_units, 1);
        assert_eq!(p.blocked_events, 1);
        assert_eq!(p.blocked_ms, 15 * 60 * 1000);
        assert_eq!(p.open_blocked, 0);
        assert_eq!(p.mean_cycle_ms, Some(1_800_000.0));
        assert_eq!(p.reopen_count, 0);
        assert_eq!(p.conflict_count, 0);
    }

    #[test]
    fn open_blocked_interval_closes_at_as_of() {
        let events = vec![unit_created(0, &[]), claimed(1, AGENT), blocked(2)];
        let report = mine_agent_profiles(&events, &[], Some(t(32)));
        let p = profile(&report, AGENT);
        assert_eq!(p.blocked_ms, 30 * 60 * 1000, "closed at as_of horizon");
        assert_eq!(p.open_blocked, 1, "interval never resumed in-stream");

        // Without a horizon the open interval counts only as open.
        let report = mine_agent_profiles(&events, &[], None);
        let p = profile(&report, AGENT);
        assert_eq!(p.blocked_ms, 0);
        assert_eq!(p.open_blocked, 1);
    }

    #[test]
    fn release_and_reblock_count_as_fixup() {
        let events = vec![
            unit_created(0, &[]),
            claimed(1, AGENT),
            blocked(2),
            blocked(3), // re-block on the same unit
            released(4),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        let p = profile(&report, AGENT);
        assert_eq!(p.released_unfinished, 1);
        assert_eq!(p.blocked_events, 2);
        // fixup = 1 release + 1 re-block.
        assert_eq!(p.fixup_count, 2);
    }

    #[test]
    fn reopen_attributes_to_last_completer() {
        let events = vec![
            env(
                10,
                user_actor(),
                json!({
                    "type": "task_completed",
                    "task_id": TASK,
                    "completed_at": ts(10),
                    "completion_note": { "actor": { "kind": "agent", "id": AGENT } }
                }),
            ),
            env(
                20,
                user_actor(),
                json!({ "type": "task_reopened", "task_id": TASK, "by": { "kind": "user" }, "at": ts(20) }),
            ),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        assert_eq!(profile(&report, AGENT).reopen_count, 1);
    }

    #[test]
    fn reopen_falls_back_to_last_claimer() {
        let events = vec![
            env(
                1,
                agent_actor(AGENT),
                json!({ "type": "agent_claimed", "agent_id": AGENT, "task_id": TASK, "expires_at": ts(60) }),
            ),
            env(
                2,
                agent_actor(OTHER),
                json!({ "type": "agent_claimed", "agent_id": OTHER, "task_id": TASK, "expires_at": ts(60) }),
            ),
            env(
                3,
                user_actor(),
                json!({ "type": "task_reopened", "task_id": TASK, "by": { "kind": "user" }, "at": ts(3) }),
            ),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        assert_eq!(profile(&report, OTHER).reopen_count, 1);
    }

    #[test]
    fn contested_counts_agents_only() {
        let events = vec![env(
            5,
            user_actor(),
            json!({
                "type": "task_contested",
                "task_id": TASK,
                "actors": [agent_actor(AGENT), user_actor()]
            }),
        )];
        let report = mine_agent_profiles(&events, &[], None);
        assert_eq!(profile(&report, AGENT).conflict_count, 1);
        // The human actor never produces a profile from a contest.
        assert_eq!(report.agents.len(), 1);
    }

    #[test]
    fn handoff_stats_and_rejection_fixup_for_producer() {
        let events = vec![
            env(
                1,
                agent_actor(AGENT),
                json!({
                    "type": "handoff_requested",
                    "handoff": {
                        "id": HANDOFF,
                        "from_work_unit_id": UNIT,
                        "to_work_unit_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                        "status": "open",
                        "owner_agent_id": AGENT,
                        "created_at": ts(1),
                        "updated_at": ts(1)
                    }
                }),
            ),
            env(
                11,
                agent_actor(OTHER),
                json!({ "type": "handoff_rejected", "handoff_id": HANDOFF, "reason": "needs tests", "latency_ms": 600_000, "at": ts(11) }),
            ),
            env(
                21,
                agent_actor(OTHER),
                json!({ "type": "handoff_accepted", "handoff_id": HANDOFF, "by": OTHER, "latency_ms": 1_200_000, "at": ts(21) }),
            ),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        let p = profile(&report, AGENT);
        assert_eq!(p.handoffs_accepted, 1);
        assert_eq!(p.handoffs_rejected, 1);
        assert_eq!(p.mean_handoff_latency_ms, Some(900_000.0));
        assert_eq!(p.fixup_count, 1, "the rejection is producer-side fixup");
    }

    #[test]
    fn unparseable_inputs_are_skipped_and_counted() {
        let events = vec![json!({"nonsense": true}), json!(42), unit_created(0, &[])];
        let report = mine_agent_profiles(&events, &[], None);
        assert_eq!(report.envelopes_parsed, 1);
        assert_eq!(report.envelopes_skipped, 2);
    }

    #[test]
    fn unrelated_event_types_are_ignored() {
        let events = vec![
            unit_created(0, &[]),
            env(
                1,
                user_actor(),
                json!({ "type": "project_created", "project": { "id": "x", "title": "p" } }),
            ),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        assert_eq!(report.envelopes_parsed, 2);
        assert!(report.agents.is_empty());
    }

    // ── responsibility lifecycle ────────────────────────────────────────────

    #[test]
    fn mined_pattern_is_suggested() {
        let events = vec![
            unit_created(0, &["frontend"]),
            claimed(1, AGENT),
            completed(2, 1000),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        let p = profile(&report, AGENT);
        assert_eq!(p.responsibility.len(), 1);
        let pat = &p.responsibility[0];
        assert_eq!(pat.capability, "frontend");
        assert_eq!(pat.lifecycle, PatternLifecycle::Suggested);
        assert_eq!(pat.source, PatternSource::Inferred);
        assert_eq!(pat.evidence_count, 1);
        assert_eq!(pat.score, 1.0);
    }

    #[test]
    fn user_set_override_promotes_suggested_to_active() {
        let events = vec![
            unit_created(0, &["frontend"]),
            claimed(1, AGENT),
            completed(2, 1000),
        ];
        let overrides = vec![UserSetOverride {
            agent_id: AGENT.into(),
            capability: "frontend".into(),
            score: 0.9,
        }];
        let report = mine_agent_profiles(&events, &overrides, None);
        let pat = &profile(&report, AGENT).responsibility[0];
        assert_eq!(pat.lifecycle, PatternLifecycle::Active);
        assert_eq!(pat.source, PatternSource::UserSet);
        assert_eq!(pat.score, 0.9, "the user's score wins over mining");
        assert_eq!(pat.confidence, 1.0);
        assert_eq!(pat.evidence_count, 1, "mined evidence is retained");
    }

    #[test]
    fn zero_score_override_marks_pattern_rejected() {
        let events = vec![
            unit_created(0, &["db"]),
            claimed(1, AGENT),
            completed(2, 1000),
        ];
        let overrides = vec![UserSetOverride {
            agent_id: AGENT.into(),
            capability: "db".into(),
            score: 0.0,
        }];
        let report = mine_agent_profiles(&events, &overrides, None);
        let pat = &profile(&report, AGENT).responsibility[0];
        assert_eq!(pat.lifecycle, PatternLifecycle::Rejected);
    }

    #[test]
    fn override_on_unobserved_pair_creates_active_pattern() {
        let overrides = vec![UserSetOverride {
            agent_id: AGENT.into(),
            capability: "db".into(),
            score: 0.8,
        }];
        let report = mine_agent_profiles(&[], &overrides, None);
        let pat = &profile(&report, AGENT).responsibility[0];
        assert_eq!(pat.lifecycle, PatternLifecycle::Active);
        assert_eq!(pat.source, PatternSource::UserSet);
        assert_eq!(pat.evidence_count, 0);
    }

    #[test]
    fn weak_signals_lower_mined_pattern_score() {
        // completed (1.0) then blocked (0.3): EWMA with n=2 step 1/2 → 0.65.
        let events = vec![
            unit_created(0, &["frontend"]),
            claimed(1, AGENT),
            completed(2, 1000),
            // second claim + block on the same unit id (synthetic stream).
            claimed(3, AGENT),
            blocked(4),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        let pat = &profile(&report, AGENT).responsibility[0];
        assert!((pat.score - 0.65).abs() < 1e-9, "got {}", pat.score);
        assert_eq!(pat.evidence_count, 2);
    }

    // ── workflow_confidence ─────────────────────────────────────────────────

    #[test]
    fn confidence_stays_within_unit_interval() {
        // All completions, no negatives → still ≤ 1.
        let good = vec![unit_created(0, &[]), claimed(1, AGENT), completed(2, 1)];
        let c = profile(&mine_agent_profiles(&good, &[], None), AGENT).workflow_confidence;
        assert!((0.0..=1.0).contains(&c), "good stream: {c}");

        // Only blocked + reblocks + conflicts + reopen → ≥ 0 and low.
        let bad = vec![
            unit_created(0, &[]),
            claimed(1, AGENT),
            blocked(2),
            blocked(3),
            env(
                4,
                user_actor(),
                json!({ "type": "task_contested", "task_id": TASK, "actors": [agent_actor(AGENT)] }),
            ),
            env(
                5,
                user_actor(),
                json!({ "type": "task_completed", "task_id": TASK, "completed_at": ts(5), "completion_note": { "actor": { "kind": "agent", "id": AGENT } } }),
            ),
            env(
                6,
                user_actor(),
                json!({ "type": "task_reopened", "task_id": TASK, "by": { "kind": "user" }, "at": ts(6) }),
            ),
        ];
        let c = profile(&mine_agent_profiles(&bad, &[], None), AGENT).workflow_confidence;
        assert!((0.0..=1.0).contains(&c), "bad stream: {c}");
        assert!(c < 0.2, "pathological stream should score low, got {c}");
    }

    #[test]
    fn confidence_is_monotone_in_evidence_quality() {
        // More clean completions → non-decreasing confidence.
        let one = vec![unit_created(0, &[]), claimed(1, AGENT), completed(2, 1)];
        let mut many = one.clone();
        for i in 0..4 {
            let base = 10 + i * 3;
            many.extend([
                unit_created(base, &[]),
                claimed(base + 1, AGENT),
                completed(base + 2, 1),
            ]);
        }
        let c1 = profile(&mine_agent_profiles(&one, &[], None), AGENT).workflow_confidence;
        let c5 = profile(&mine_agent_profiles(&many, &[], None), AGENT).workflow_confidence;
        assert!(
            c5 > c1,
            "clean history should raise confidence: {c1} → {c5}"
        );

        // Adding a reopen to the same stream lowers confidence.
        let mut with_reopen = many.clone();
        with_reopen.push(env(
            40,
            user_actor(),
            json!({
                "type": "task_completed",
                "task_id": TASK,
                "completed_at": ts(40),
                "completion_note": { "actor": { "kind": "agent", "id": AGENT } }
            }),
        ));
        with_reopen.push(env(
            41,
            user_actor(),
            json!({ "type": "task_reopened", "task_id": TASK, "by": { "kind": "user" }, "at": ts(41) }),
        ));
        let c_re =
            profile(&mine_agent_profiles(&with_reopen, &[], None), AGENT).workflow_confidence;
        assert!(c_re < c5, "a reopen should lower confidence: {c5} → {c_re}");
    }

    #[test]
    fn single_completion_is_discounted_by_evidence_saturation() {
        let events = vec![unit_created(0, &[]), claimed(1, AGENT), completed(2, 1)];
        let c = profile(&mine_agent_profiles(&events, &[], None), AGENT).workflow_confidence;
        // n=1 → w_ev = 1/6; blend of perfect terms = 1.0 → exactly 1/6.
        assert!((c - 1.0 / 6.0).abs() < 1e-9, "got {c}");
    }

    #[test]
    fn bare_payloads_without_envelope_are_accepted() {
        let events = vec![
            json!({ "type": "work_unit_created", "work_unit": { "id": UNIT, "task_id": TASK, "capability_tags": [] } }),
            json!({ "type": "work_unit_claimed", "work_unit_id": UNIT, "agent_id": AGENT }),
            json!({ "type": "work_unit_completed", "work_unit_id": UNIT, "completed_by": AGENT }),
        ];
        let report = mine_agent_profiles(&events, &[], None);
        assert_eq!(report.envelopes_parsed, 3);
        assert_eq!(profile(&report, AGENT).completed_units, 1);
    }
}
