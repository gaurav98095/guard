//! # Bridge Types Module
//!
//! Core type definitions, enums, and traits for the Bridge rule system.
//!
//! This module provides:
//! - Common trait definitions for rule instances
//! - Action and match type definitions
//! - Scope and constraint types

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ================================================================================================
// RULE INSTANCE TRAIT
// ================================================================================================

/// Common trait that all rule instances must implement
///
/// This trait provides a unified interface for accessing rule metadata.
pub trait RuleInstance: Send + Sync {
    /// Unique identifier for this rule instance
    fn rule_id(&self) -> &str;

    /// Priority value (higher = evaluated first)
    fn priority(&self) -> u32;

    /// Scope definition for this rule
    fn scope(&self) -> &RuleScope;

    /// Optional layer metadata hint (None = no layer metadata)
    fn layer(&self) -> Option<&str> {
        None
    }

    /// Timestamp when rule was created
    fn created_at(&self) -> u64;

    /// Optional description for this rule
    fn description(&self) -> Option<&str> {
        None
    }

    /// Whether this rule is currently enabled
    fn is_enabled(&self) -> bool {
        true
    }

    /// Returns a Management Plane payload for encoding APIs (default empty for unsupported families)
    fn management_plane_payload(&self) -> Value {
        json!({})
    }
}

impl fmt::Debug for dyn RuleInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuleInstance")
            .field("rule_id", &self.rule_id())
            .field("layer", &self.layer())
            .field("priority", &self.priority())
            .field("enabled", &self.is_enabled())
            .field("created_at", &self.created_at())
            .finish()
    }
}

// ================================================================================================
// SCOPE DEFINITION
// ================================================================================================

/// Defines the scope/applicability of a rule
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleScope {
    /// Agent IDs this rule applies to (empty = all agents)
    pub agent_ids: Vec<String>,

    /// Tags for additional scoping
    pub tags: HashMap<String, String>,

    /// Whether this is a global rule (applies to all)
    pub is_global: bool,
}

impl RuleScope {
    /// Creates a new global scope
    pub fn global() -> Self {
        RuleScope {
            agent_ids: vec![],
            tags: HashMap::new(),
            is_global: true,
        }
    }

    /// Creates a scope for specific agents
    pub fn for_agents(agent_ids: Vec<String>) -> Self {
        RuleScope {
            agent_ids,
            tags: HashMap::new(),
            is_global: false,
        }
    }

    /// Creates a scope for a single agent
    pub fn for_agent(agent_id: String) -> Self {
        RuleScope {
            agent_ids: vec![agent_id],
            tags: HashMap::new(),
            is_global: false,
        }
    }

    /// Checks if this scope applies to a given agent
    pub fn applies_to(&self, agent_id: &str) -> bool {
        self.is_global || self.agent_ids.iter().any(|id| id == agent_id)
    }

    /// Adds a tag to this scope
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }
}

impl Default for RuleScope {
    fn default() -> Self {
        RuleScope::global()
    }
}

// ================================================================================================
// RULE METADATA
// ================================================================================================

/// Serializable rule metadata for cold storage persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleMetadata {
    pub rule_id: String,
    pub priority: u32,
    pub scope: RuleScope,
    pub layer: Option<String>,
    pub created_at_ms: u64,
    pub enabled: bool,
    pub description: Option<String>,
    pub params: Value,
}

impl RuleMetadata {
    pub fn from_rule(rule: &dyn RuleInstance) -> Self {
        Self {
            rule_id: rule.rule_id().to_string(),
            priority: rule.priority(),
            scope: rule.scope().clone(),
            layer: rule.layer().map(|layer| layer.to_string()),
            created_at_ms: rule.created_at(),
            enabled: rule.is_enabled(),
            description: rule
                .description()
                .map(|description| description.to_string()),
            params: rule.management_plane_payload(),
        }
    }
}

// ================================================================================================
// ACTION TYPES
// ================================================================================================

/// Common action types across bridge rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleAction {
    /// Allow the operation
    Allow,
    /// Deny/block the operation
    Deny,
    /// Redirect to alternative target
    Redirect,
    /// Rewrite/modify the payload
    Rewrite,
    /// Redact sensitive information
    Redact,
    /// Escalate to human review
    Escalate,
    /// Truncate to fit constraints
    Truncate,
    /// Log but don't enforce
    Audit,
    /// Drop context from memory
    DropContext,
}

impl fmt::Display for RuleAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuleAction::Allow => write!(f, "ALLOW"),
            RuleAction::Deny => write!(f, "DENY"),
            RuleAction::Redirect => write!(f, "REDIRECT"),
            RuleAction::Rewrite => write!(f, "REWRITE"),
            RuleAction::Redact => write!(f, "REDACT"),
            RuleAction::Escalate => write!(f, "ESCALATE"),
            RuleAction::Truncate => write!(f, "TRUNCATE"),
            RuleAction::Audit => write!(f, "AUDIT"),
            RuleAction::DropContext => write!(f, "DROP_CONTEXT"),
        }
    }
}

// ================================================================================================
// MATCH TYPES
// ================================================================================================

/// Network protocols for L0 network egress rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkProtocol {
    TCP,
    UDP,
    HTTP,
    HTTPS,
}

impl Default for NetworkProtocol {
    fn default() -> Self {
        NetworkProtocol::HTTPS
    }
}

/// Parameter types for tool constraint validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    String,
    Int,
    Float,
    Bool,
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

/// Returns current timestamp in milliseconds
pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
