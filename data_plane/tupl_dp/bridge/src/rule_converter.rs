//! Lightweight representations of control plane rules used by the gRPC server.

use std::collections::HashMap;

/// Represents a rule sent from the management plane.
#[derive(Debug, Clone)]
pub struct ControlPlaneRule {
    pub rule_id: String,
    pub family_id: String,
    pub layer: String,
    pub agent_id: String,
    pub priority: i32,
    pub enabled: bool,
    pub created_at_ms: i64,
    pub params: HashMap<String, ParamValue>,
}

/// Parameter value from the control plane payload.
#[derive(Debug, Clone)]
pub enum ParamValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    StringList(Vec<String>),
}

impl ParamValue {
    pub fn as_string(&self) -> Option<String> {
        match self {
            ParamValue::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_string_or_default(&self, default: &str) -> String {
        self.as_string().unwrap_or_else(|| default.to_string())
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_bool_or_default(&self, default: bool) -> bool {
        self.as_bool().unwrap_or(default)
    }

    pub fn as_string_list(&self) -> Option<Vec<String>> {
        match self {
            ParamValue::StringList(list) => Some(list.clone()),
            _ => None,
        }
    }

    pub fn as_string_list_or_default(&self) -> Vec<String> {
        self.as_string_list().unwrap_or_default()
    }
}
