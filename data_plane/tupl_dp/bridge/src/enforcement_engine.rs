//! # Enforcement Engine for Layer-Based Rule Enforcement (v1.3)
//!
//! Orchestrates the enforcement pipeline:
//! 1. Receives IntentEvent from SDK via gRPC
//! 2. Calls Management Plane to encode intent to 128d vector
//! 3. Queries rules from Bridge for the specified layer
//! 4. Compares intent vector directly against rule anchors using in-process comparison
//! 5. Implements short-circuit evaluation (first BLOCK stops evaluation)
//! 6. Returns enforcement decision with evidence
//! 7. Records complete telemetry to /var/hitlogs for audit trail

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::api_types::IntentEvent;
use reqwest::{header::CONTENT_TYPE, Client};
use serde::Deserialize;
use serde_json::Value;

use crate::bridge::Bridge;
use crate::rule_vector::RuleVector;
use crate::telemetry::session::SliceComparisonDetail;
use crate::telemetry::{EnforcementSession, RuleEvaluationEvent, SessionEvent, TelemetryRecorder};
use crate::types::RuleInstance;
use crate::vector_comparison::{compare_intent_vs_rule, ComparisonResult, DecisionMode};

const CONNECT_TIMEOUT_MS: u64 = 500;
const REQUEST_TIMEOUT_MS: u64 = 1_500;

// Per-slot thresholds for ToolWhitelist family
// [Action, Resource, Data, Risk]
// Resource slot (0.88) is most critical for tool identity matching
// Calibrated to distinguish exact tool matches from semantic similarities
// Action threshold lowered to 0.60 to account for semantic variation (read/query/search)

// Default thresholds for other rule families
const DEFAULT_THRESHOLDS: [f32; 4] = [0.75, 0.75, 0.75, 0.75];

#[derive(Debug, Deserialize)]
struct SliceThresholdsPayload {
    action: f32,
    resource: f32,
    data: f32,
    risk: f32,
}

// ============================================================================
// Data Structures
// ============================================================================

/// Enforcement engine that coordinates intent evaluation against rules
pub struct EnforcementEngine {
    /// Reference to the Bridge for querying rules
    bridge: Arc<Bridge>,

    /// Management Plane encoding endpoint
    encoding_endpoint: String,

    /// Shared HTTP client (reqwest + rustls)
    http_client: Client,

    /// Telemetry recorder (optional - can be disabled)
    telemetry: Option<Arc<TelemetryRecorder>>,
}

/// Result of enforcement evaluation
#[derive(Debug, Clone)]
pub struct EnforcementResult {
    /// 0 = BLOCK, 1 = ALLOW
    pub decision: u8,

    /// Per-slot similarity scores [action, resource, data, risk]
    pub slice_similarities: [f32; 4],

    /// Number of rules evaluated before decision
    pub rules_evaluated: usize,

    /// Evidence from each rule evaluation
    pub evidence: Vec<RuleEvidence>,
}

/// Evidence from a single rule evaluation
#[derive(Debug, Clone)]
pub struct RuleEvidence {
    pub rule_id: String,
    pub rule_name: String,
    pub decision: u8, // 0 = blocked, 1 = passed
    pub similarities: [f32; 4],
}

#[derive(Debug, Deserialize)]
struct IntentEncodingResponse {
    vector: Vec<f32>,
}

// ============================================================================
// EnforcementEngine Implementation
// ============================================================================

impl EnforcementEngine {
    fn endpoint(&self, path: &str) -> String {
        let trimmed = path.trim_start_matches('/');
        format!("{}/{}", self.encoding_endpoint, trimmed)
    }

    /// Create a new enforcement engine
    pub fn new(bridge: Arc<Bridge>, encoding_endpoint: String) -> Self {
        Self::with_telemetry(bridge, encoding_endpoint, None).unwrap()
    }

    /// Create enforcement engine with telemetry enabled
    pub fn with_telemetry(
        bridge: Arc<Bridge>,
        encoding_endpoint: String,
        telemetry: Option<Arc<TelemetryRecorder>>,
    ) -> Result<Self, String> {
        let sanitized_endpoint = encoding_endpoint.trim_end_matches('/').to_string();

        let http_client = Client::builder()
            .connect_timeout(Duration::from_millis(CONNECT_TIMEOUT_MS))
            .timeout(Duration::from_millis(REQUEST_TIMEOUT_MS))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        Ok(EnforcementEngine {
            bridge,
            encoding_endpoint: sanitized_endpoint,
            http_client,
            telemetry,
        })
    }

    /// Enforce rules against an IntentEvent
    ///
    /// This is the main entry point for enforcement. It:
    /// 1. Encodes the intent to 128d vector (via Management Plane)
    /// 2. Queries rules for the specified layer from Bridge
    /// 3. Evaluates each rule with short-circuit (first BLOCK stops)
    /// 4. Records complete telemetry to hitlog
    /// 5. Returns enforcement decision with evidence
    pub async fn enforce(
        &self,
        intent_json: &str,
        vector_override: Option<[f32; 128]>,
    ) -> Result<EnforcementResult, String> {
        let session_start = Instant::now();

        // Parse IntentEvent JSON
        let intent: IntentEvent = serde_json::from_str(intent_json)
            .map_err(|e| format!("Failed to parse IntentEvent: {}", e))?;

        // Extract layer from intent
        let layer = intent
            .layer_str()
            .ok_or("Missing 'layer' field in IntentEvent")?;

        println!("Enforcing intent for layer: {}", layer);

        // Start telemetry session
        let session_id = self
            .telemetry
            .as_ref()
            .and_then(|t| t.start_session(layer.to_string(), intent_json.to_string()));

        // Populate agent_id and tenant_id from IntentEvent
        if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
            telemetry.with_session(sid, |session| {
                // Set tenant_id from intent
                session.tenant_id = Some(intent.tenant_id.clone());

                // Set agent_id from rate_limit_context if available
                if let Some(ref rate_limit) = intent.rate_limit_context {
                    session.agent_id = Some(rate_limit.agent_id.clone());
                }
            });
        }

        // Record intent received event
        if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
            telemetry.with_session(sid, |session| {
                session.add_event(SessionEvent::IntentReceived {
                    timestamp_us: EnforcementSession::timestamp_us(),
                    intent_id: intent.id.clone(),
                    layer: layer.to_string(),
                });
            });
        }

        // 1. Encode intent to 128d vector (or reuse override)
        let encoding_start = Instant::now();

        if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
            telemetry.with_session(sid, |session| {
                session.add_event(SessionEvent::EncodingStarted {
                    timestamp_us: EnforcementSession::timestamp_us(),
                });
            });
        }

        let (intent_vector, encoding_duration, vector_norm) = if let Some(vector) = vector_override
        {
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            (vector, 0u64, norm)
        } else {
            match self.encode_intent(intent_json).await {
                Ok(vector) => {
                    let duration = encoding_start.elapsed().as_micros() as u64;
                    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                    (vector, duration, norm)
                }
                Err(err) => {
                    println!(
                        "Intent encoding failed: {}. Blocking intent (fail-closed).",
                        err
                    );

                    if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
                        telemetry.with_session(sid, |session| {
                            session.add_event(SessionEvent::EncodingFailed {
                                timestamp_us: EnforcementSession::timestamp_us(),
                                error: err.clone(),
                            });
                            session.error = Some(err.clone());
                        });

                        let total_duration = session_start.elapsed().as_micros() as u64;
                        telemetry.complete_session(sid, 0, total_duration).ok();
                    }

                    return Ok(EnforcementResult {
                        decision: 0,
                        slice_similarities: [0.0; 4],
                        rules_evaluated: 0,
                        evidence: vec![],
                    });
                }
            }
        };

        if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
            telemetry.with_session(sid, |session| {
                session.add_event(SessionEvent::EncodingCompleted {
                    timestamp_us: EnforcementSession::timestamp_us(),
                    duration_us: encoding_duration,
                    vector_norm,
                });
                session.intent_vector = Some(intent_vector.to_vec());
                session.performance.encoding_duration_us = encoding_duration;
            });
        }

        // 2. Query rules for this layer from Bridge
        let query_start = Instant::now();
        let rules = self.get_rules_for_layer(layer)?;
        let query_duration = query_start.elapsed().as_micros() as u64;

        if rules.is_empty() {
            // No rules = fail-closed (BLOCK)
            println!(
                "No rules configured for layer {}, blocking by default",
                layer
            );

            // Record no rules found
            if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
                telemetry.with_session(sid, |session| {
                    session.add_event(SessionEvent::NoRulesFound {
                        timestamp_us: EnforcementSession::timestamp_us(),
                        layer: layer.to_string(),
                    });
                });

                let total_duration = session_start.elapsed().as_micros() as u64;
                telemetry.complete_session(sid, 0, total_duration).ok();
            }

            return Ok(EnforcementResult {
                decision: 0,
                slice_similarities: [0.0; 4],
                rules_evaluated: 0,
                evidence: vec![],
            });
        }

        let rules_count = rules.len();
        println!("Found {} rules for layer {}", rules_count, layer);

        // Record rules queried
        if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
            telemetry.with_session(sid, |session| {
                session.add_event(SessionEvent::RulesQueried {
                    timestamp_us: EnforcementSession::timestamp_us(),
                    layer: layer.to_string(),
                    rule_count: rules_count,
                    query_duration_us: query_duration,
                });
                session.performance.rule_query_duration_us = query_duration;
                session.performance.rules_queried = rules_count;
            });
        }

        // 3. Evaluate rules with SHORT-CIRCUIT
        let mut evidence = Vec::new();
        let evaluation_start = Instant::now();

        for rule in &rules {
            let rule_eval_start = Instant::now();

            // Record rule evaluation started
            if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
                telemetry.with_session(sid, |session| {
                    session.add_event(SessionEvent::RuleEvaluationStarted {
                        timestamp_us: EnforcementSession::timestamp_us(),
                        rule_id: rule.rule_id().to_string(),
                        rule_priority: rule.priority(),
                    });
                });
            }

            // Retrieve pre-encoded anchors from bridge
            let rule_vector = if let Some(vector) = self.bridge.get_rule_anchors(rule.rule_id()) {
                vector
            } else {
                return Err(format!(
                    "Rule '{}' missing pre-encoded anchors (install-time encoding incomplete)",
                    rule.rule_id()
                ));
            };

            // Compare using semantic sandbox FFI
            let result = self.compare_with_sandbox(&intent_vector, &rule_vector, &rule)?;
            let rule_eval_duration = rule_eval_start.elapsed().as_micros() as u64;

            // Mark rule as recently evaluated (updates LRU timestamp in hot cache)
            // This prevents it from being evicted, as it's actively being used
            let _ = self.bridge.hot_cache.get_and_mark(rule.rule_id());

            // Record evidence
            evidence.push(RuleEvidence {
                rule_id: rule.rule_id().to_string(),
                rule_name: rule.description().unwrap_or("").to_string(),
                decision: result.decision,
                similarities: result.slice_similarities,
            });

            // Record rule evaluation completed
            if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
                let thresholds = self.get_thresholds(&rule);
                let slice_details = self.build_slice_details(&result, &thresholds);

                let payload = rule.management_plane_payload();
                let rule_family = payload
                    .get("rule_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("design_boundary")
                    .to_string();

                telemetry.with_session(sid, |session| {
                    session.add_event(SessionEvent::RuleEvaluationCompleted {
                        timestamp_us: EnforcementSession::timestamp_us(),
                        rule_id: rule.rule_id().to_string(),
                        decision: result.decision,
                        similarities: result.slice_similarities,
                        duration_us: rule_eval_duration,
                    });

                    // Add detailed rule evaluation record
                    session.add_rule_evaluation(RuleEvaluationEvent {
                        rule_id: rule.rule_id().to_string(),
                        rule_family,
                        priority: rule.priority(),
                        description: rule.description().map(|s| s.to_string()),
                        started_at_us: EnforcementSession::timestamp_us() - rule_eval_duration,
                        duration_us: rule_eval_duration,
                        decision: result.decision,
                        slice_similarities: result.slice_similarities,
                        thresholds,
                        anchor_counts: [
                            rule_vector.action_count,
                            rule_vector.resource_count,
                            rule_vector.data_count,
                            rule_vector.risk_count,
                        ],
                        short_circuited: false,
                        slice_details,
                    });
                });
            }

            // SHORT CIRCUIT: First failure = immediate BLOCK
            if result.decision == 0 {
                println!(
                    "BLOCKED by rule '{}' (priority {}). Short-circuiting.",
                    rule.rule_id(),
                    rule.priority()
                );

                let remaining_rules = rules.len() - evidence.len();

                // Record short-circuit
                if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
                    telemetry.with_session(sid, |session| {
                        session.add_event(SessionEvent::ShortCircuit {
                            timestamp_us: EnforcementSession::timestamp_us(),
                            rule_id: rule.rule_id().to_string(),
                            rules_remaining: remaining_rules,
                        });

                        // Mark last rule as short-circuited
                        if let Some(last_eval) = session.rules_evaluated.last_mut() {
                            last_eval.short_circuited = true;
                        }

                        session.performance.short_circuited = true;
                    });
                }

                let evaluation_duration = evaluation_start.elapsed().as_micros() as u64;
                let total_duration = session_start.elapsed().as_micros() as u64;

                // Record final decision
                if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
                    telemetry.with_session(sid, |session| {
                        session.add_event(SessionEvent::FinalDecision {
                            timestamp_us: EnforcementSession::timestamp_us(),
                            decision: 0,
                            rules_evaluated: evidence.len(),
                            total_duration_us: total_duration,
                        });
                        session.performance.evaluation_duration_us = evaluation_duration;
                        session.final_similarities = Some(result.slice_similarities);
                    });

                    // Complete session
                    telemetry.complete_session(sid, 0, total_duration).ok();
                }

                return Ok(EnforcementResult {
                    decision: 0,
                    slice_similarities: result.slice_similarities,
                    rules_evaluated: evidence.len(),
                    evidence,
                });
            }
        }

        // All rules passed - ALLOW
        println!("ALLOWED: All {} rules passed", rules_count);

        let evaluation_duration = evaluation_start.elapsed().as_micros() as u64;
        let total_duration = session_start.elapsed().as_micros() as u64;
        let avg_similarities = Self::average_similarities(&evidence);

        // Record final decision
        if let (Some(ref telemetry), Some(ref sid)) = (&self.telemetry, &session_id) {
            telemetry.with_session(sid, |session| {
                session.add_event(SessionEvent::FinalDecision {
                    timestamp_us: EnforcementSession::timestamp_us(),
                    decision: 1,
                    rules_evaluated: evidence.len(),
                    total_duration_us: total_duration,
                });
                session.performance.evaluation_duration_us = evaluation_duration;
                session.final_similarities = Some(avg_similarities);
            });

            // Complete session
            telemetry.complete_session(sid, 1, total_duration).ok();
        }

        Ok(EnforcementResult {
            decision: 1,
            slice_similarities: avg_similarities,
            rules_evaluated: evidence.len(),
            evidence,
        })
    }

    /// Encode intent to 128d vector by calling Management Plane
    async fn encode_intent(&self, intent_json: &str) -> Result<[f32; 128], String> {
        let url = self.endpoint("/encode/intent");

        let response = self
            .http_client
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .body(intent_json.to_owned())
            .send()
            .await
            .map_err(|e| format!("Failed to call Management Plane /encode/intent: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unavailable>".to_string());
            return Err(format!(
                "/encode/intent returned {} (fail-closed): {}",
                status, body
            ));
        }

        let payload: IntentEncodingResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse /encode/intent response: {e}"))?;

        if payload.vector.len() != 128 {
            return Err(format!(
                "Management Plane returned {}-dim vector, expected 128",
                payload.vector.len()
            ));
        }

        let mut vector = [0f32; 128];
        vector.copy_from_slice(&payload.vector);
        Ok(vector)
    }

    /// Query rules for a specific layer from Bridge
    fn get_rules_for_layer(&self, layer: &str) -> Result<Vec<Arc<dyn RuleInstance>>, String> {
        println!("Querying rules for layer: {}", layer);

        let requested_layer = if layer.is_empty() { None } else { Some(layer) };

        let mut filtered: Vec<_> = self
            .bridge
            .all_rules()
            .into_iter()
            .filter(|rule| match (rule.layer(), requested_layer) {
                (None, _) => true,
                (Some(rule_layer), Some(requested)) => rule_layer == requested,
                (Some(_), None) => false,
            })
            .collect();

        filtered.sort_by(|a, b| b.priority().cmp(&a.priority()));

        println!("Found {} rules for layer {}", filtered.len(), layer);
        Ok(filtered)
    }

    /// Compare intent vector against rule anchors using direct in-process comparison
    fn compare_with_sandbox(
        &self,
        intent_vector: &[f32; 128],
        rule_vector: &RuleVector,
        rule: &Arc<dyn RuleInstance>,
    ) -> Result<ComparisonResult, String> {
        let (thresholds, decision_mode) = self.get_rule_thresholds(rule);

        Ok(compare_intent_vs_rule(
            intent_vector,
            rule_vector,
            thresholds,
            decision_mode,
        ))
    }

    /// Calculate average similarities across all evidence
    fn average_similarities(evidence: &[RuleEvidence]) -> [f32; 4] {
        if evidence.is_empty() {
            return [0.0; 4];
        }

        let mut sums = [0.0; 4];
        for ev in evidence {
            for i in 0..4 {
                sums[i] += ev.similarities[i];
            }
        }

        let count = evidence.len() as f32;
        [
            sums[0] / count,
            sums[1] / count,
            sums[2] / count,
            sums[3] / count,
        ]
    }

    /// Get thresholds for a rule family
    fn get_thresholds(&self, rule: &Arc<dyn RuleInstance>) -> [f32; 4] {
        self.get_rule_thresholds(rule).0
    }

    /// Build detailed slice comparison data for telemetry
    fn build_slice_details(
        &self,
        result: &ComparisonResult,
        thresholds: &[f32; 4],
    ) -> Vec<SliceComparisonDetail> {
        let slice_names = ["action", "resource", "data", "risk"];

        slice_names
            .iter()
            .enumerate()
            .map(|(i, &name)| SliceComparisonDetail {
                slice_name: name.to_string(),
                similarity: result.slice_similarities[i],
                threshold: thresholds[i],
                passed: result.slice_similarities[i] >= thresholds[i],
                anchor_count: 0, // Could be populated from rule_vector if needed
                best_anchor_idx: None,
            })
            .collect()
    }

    /// Flush telemetry to disk
    pub fn flush_telemetry(&self) -> Result<(), String> {
        if let Some(ref telemetry) = self.telemetry {
            telemetry.flush()
        } else {
            Ok(())
        }
    }

    /// Get telemetry statistics
    pub fn telemetry_stats(&self) -> Option<crate::telemetry::recorder::TelemetryStats> {
        self.telemetry.as_ref().map(|t| t.stats())
    }

    fn get_rule_thresholds(&self, rule: &Arc<dyn RuleInstance>) -> ([f32; 4], DecisionMode) {
        let payload = rule.management_plane_payload();

        if let Value::Object(map) = payload {
            let mut thresholds = DEFAULT_THRESHOLDS;
            if let Some(Value::String(threshold_str)) = map.get("thresholds") {
                if let Ok(decoded) = serde_json::from_str::<SliceThresholdsPayload>(threshold_str) {
                    thresholds = [
                        decoded.action,
                        decoded.resource,
                        decoded.data,
                        decoded.risk,
                    ];
                }
            }

            let decision = map
                .get("rule_decision")
                .and_then(Value::as_str)
                .map(|s| match s {
                    "weighted-avg" => DecisionMode::WeightedAvgMode,
                    _ => DecisionMode::MinMode,
                })
                .unwrap_or(DecisionMode::MinMode);

            return (thresholds, decision);
        }

        (DEFAULT_THRESHOLDS, DecisionMode::MinMode)
    }
}

// ============================================================================
// Helper Types
// ============================================================================
