//! # Enforcement Session
//!
//! Tracks the complete lifecycle of a single intent evaluation.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique identifier for an enforcement session
pub type SessionId = String;

/// Complete record of an intent evaluation session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementSession {
    /// Unique session ID (UUID v4)
    pub session_id: SessionId,

    /// Timestamp when intent arrived (Unix milliseconds)
    pub timestamp_ms: u64,

    /// Layer being enforced (L0-L6)
    pub layer: String,

    /// Agent ID (if available)
    pub agent_id: Option<String>,

    /// Tenant ID (if available)
    pub tenant_id: Option<String>,

    /// Complete intent JSON
    pub intent_json: String,

    /// Encoded intent vector (128-d)
    pub intent_vector: Option<Vec<f32>>,

    /// Timeline of events during enforcement
    pub events: Vec<SessionEvent>,

    /// Rules evaluated during this session
    pub rules_evaluated: Vec<RuleEvaluationEvent>,

    /// Final enforcement decision (0 = BLOCK, 1 = ALLOW)
    pub final_decision: u8,

    /// Final slice similarities [action, resource, data, risk]
    pub final_similarities: Option<[f32; 4]>,

    /// Total duration in microseconds
    pub duration_us: u64,

    /// Performance breakdown
    pub performance: PerformanceMetrics,

    /// Error message if enforcement failed
    pub error: Option<String>,
}

/// Events that occur during enforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    /// Intent arrived and parsed
    IntentReceived {
        timestamp_us: u64,
        intent_id: String,
        layer: String,
    },

    /// Intent encoding started
    EncodingStarted { timestamp_us: u64 },

    /// Intent encoding completed
    EncodingCompleted {
        timestamp_us: u64,
        duration_us: u64,
        vector_norm: f32,
    },

    /// Intent encoding failed
    EncodingFailed { timestamp_us: u64, error: String },

    /// Rules queried from bridge
    RulesQueried {
        timestamp_us: u64,
        layer: String,
        rule_count: usize,
        query_duration_us: u64,
    },

    /// No rules found (fail-closed)
    NoRulesFound { timestamp_us: u64, layer: String },

    /// Rule evaluation started
    RuleEvaluationStarted {
        timestamp_us: u64,
        rule_id: String,
        rule_priority: u32,
    },

    /// Rule evaluation completed
    RuleEvaluationCompleted {
        timestamp_us: u64,
        rule_id: String,
        decision: u8,
        similarities: [f32; 4],
        duration_us: u64,
    },

    /// Short-circuit triggered (first BLOCK)
    ShortCircuit {
        timestamp_us: u64,
        rule_id: String,
        rules_remaining: usize,
    },

    /// Final decision reached
    FinalDecision {
        timestamp_us: u64,
        decision: u8,
        rules_evaluated: usize,
        total_duration_us: u64,
    },

    /// Enforcement error
    Error {
        timestamp_us: u64,
        error: String,
        fail_closed: bool,
    },
}

/// Detailed record of a single rule evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleEvaluationEvent {
    /// Rule unique identifier
    pub rule_id: String,

    /// Rule family (e.g., "tool_whitelist")
    pub rule_family: String,

    /// Rule priority
    pub priority: u32,

    /// Rule description
    pub description: Option<String>,

    /// Timestamp when evaluation started
    pub started_at_us: u64,

    /// Evaluation duration in microseconds
    pub duration_us: u64,

    /// Rule decision (0 = BLOCK, 1 = ALLOW)
    pub decision: u8,

    /// Per-slice similarity scores
    pub slice_similarities: [f32; 4],

    /// Per-slice thresholds used
    pub thresholds: [f32; 4],

    /// Number of anchors per slice
    pub anchor_counts: [usize; 4],

    /// Whether this rule caused short-circuit
    pub short_circuited: bool,

    /// Detailed comparison results per slice
    pub slice_details: Vec<SliceComparisonDetail>,
}

/// Detailed comparison for a single slice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceComparisonDetail {
    /// Slice name (action/resource/data/risk)
    pub slice_name: String,

    /// Similarity score (max across anchors)
    pub similarity: f32,

    /// Threshold for this slice
    pub threshold: f32,

    /// Whether this slice passed
    pub passed: bool,

    /// Number of anchors compared
    pub anchor_count: usize,

    /// Max similarity was from which anchor index
    pub best_anchor_idx: Option<usize>,
}

/// Performance metrics for the session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Time spent encoding intent (microseconds)
    pub encoding_duration_us: u64,

    /// Time spent querying rules from bridge (microseconds)
    pub rule_query_duration_us: u64,

    /// Time spent in rule evaluations (microseconds)
    pub evaluation_duration_us: u64,

    /// Total end-to-end duration (microseconds)
    pub total_duration_us: u64,

    /// Number of rules queried
    pub rules_queried: usize,

    /// Number of rules actually evaluated
    pub rules_evaluated: usize,

    /// Whether short-circuit occurred
    pub short_circuited: bool,
}

impl EnforcementSession {
    /// Create a new enforcement session
    pub fn new(session_id: String, layer: String, intent_json: String) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        EnforcementSession {
            session_id,
            timestamp_ms,
            layer,
            agent_id: None,
            tenant_id: None,
            intent_json,
            intent_vector: None,
            events: Vec::new(),
            rules_evaluated: Vec::new(),
            final_decision: 0,
            final_similarities: None,
            duration_us: 0,
            performance: PerformanceMetrics::default(),
            error: None,
        }
    }

    /// Add an event to the session timeline
    pub fn add_event(&mut self, event: SessionEvent) {
        self.events.push(event);
    }

    /// Add a rule evaluation record
    pub fn add_rule_evaluation(&mut self, evaluation: RuleEvaluationEvent) {
        self.rules_evaluated.push(evaluation);
    }

    /// Mark session as complete
    pub fn finalize(&mut self, final_decision: u8, total_duration_us: u64) {
        self.final_decision = final_decision;
        self.duration_us = total_duration_us;
        self.performance.total_duration_us = total_duration_us;
        self.performance.rules_evaluated = self.rules_evaluated.len();
    }

    /// Get timestamp in microseconds
    pub fn timestamp_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        PerformanceMetrics {
            encoding_duration_us: 0,
            rule_query_duration_us: 0,
            evaluation_duration_us: 0,
            total_duration_us: 0,
            rules_queried: 0,
            rules_evaluated: 0,
            short_circuited: false,
        }
    }
}

impl SessionEvent {
    /// Get event timestamp
    pub fn timestamp_us(&self) -> u64 {
        match self {
            SessionEvent::IntentReceived { timestamp_us, .. } => *timestamp_us,
            SessionEvent::EncodingStarted { timestamp_us } => *timestamp_us,
            SessionEvent::EncodingCompleted { timestamp_us, .. } => *timestamp_us,
            SessionEvent::EncodingFailed { timestamp_us, .. } => *timestamp_us,
            SessionEvent::RulesQueried { timestamp_us, .. } => *timestamp_us,
            SessionEvent::NoRulesFound { timestamp_us, .. } => *timestamp_us,
            SessionEvent::RuleEvaluationStarted { timestamp_us, .. } => *timestamp_us,
            SessionEvent::RuleEvaluationCompleted { timestamp_us, .. } => *timestamp_us,
            SessionEvent::ShortCircuit { timestamp_us, .. } => *timestamp_us,
            SessionEvent::FinalDecision { timestamp_us, .. } => *timestamp_us,
            SessionEvent::Error { timestamp_us, .. } => *timestamp_us,
        }
    }

    /// Get event type name
    pub fn event_type(&self) -> &'static str {
        match self {
            SessionEvent::IntentReceived { .. } => "intent_received",
            SessionEvent::EncodingStarted { .. } => "encoding_started",
            SessionEvent::EncodingCompleted { .. } => "encoding_completed",
            SessionEvent::EncodingFailed { .. } => "encoding_failed",
            SessionEvent::RulesQueried { .. } => "rules_queried",
            SessionEvent::NoRulesFound { .. } => "no_rules_found",
            SessionEvent::RuleEvaluationStarted { .. } => "rule_evaluation_started",
            SessionEvent::RuleEvaluationCompleted { .. } => "rule_evaluation_completed",
            SessionEvent::ShortCircuit { .. } => "short_circuit",
            SessionEvent::FinalDecision { .. } => "final_decision",
            SessionEvent::Error { .. } => "error",
        }
    }
}
