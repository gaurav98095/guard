use serde_json::Value;

use crate::types::{RuleInstance, RuleScope};

/// Lightweight rule instance representing a DesignBoundary-derived rule.
#[derive(Debug)]
pub struct DesignBoundaryRule {
    rule_id: String,
    priority: u32,
    scope: RuleScope,
    layer: Option<String>,
    created_at_ms: u64,
    description: Option<String>,
    enabled: bool,
    params: Value,
}

impl DesignBoundaryRule {
    pub fn new(
        rule_id: String,
        priority: u32,
        scope: RuleScope,
        layer: Option<String>,
        created_at_ms: u64,
        enabled: bool,
        description: Option<String>,
        params: Value,
    ) -> Self {
        Self {
            rule_id,
            priority,
            scope,
            layer,
            created_at_ms,
            description,
            enabled,
            params,
        }
    }
}

impl RuleInstance for DesignBoundaryRule {
    fn rule_id(&self) -> &str {
        &self.rule_id
    }

    fn priority(&self) -> u32 {
        self.priority
    }

    fn scope(&self) -> &RuleScope {
        &self.scope
    }

    fn layer(&self) -> Option<&str> {
        self.layer.as_deref()
    }

    fn created_at(&self) -> u64 {
        self.created_at_ms
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn management_plane_payload(&self) -> Value {
        self.params.clone()
    }
}
