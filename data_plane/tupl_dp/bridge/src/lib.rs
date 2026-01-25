//! # Rule Bridge Library
//!
//! High-performance rule storage and query engine for multi-layer enforcement.

// Core modules
pub mod api_types;
pub mod bridge;
pub mod enforcement_engine;
pub mod families;
pub mod grpc_server;
pub mod indices;
pub mod refresh;
pub mod rule_converter;
pub mod rule_vector;
pub mod storage;
pub mod telemetry;
pub mod types;
pub mod vector_comparison;

// Re-export commonly used types
pub use bridge::Bridge;
pub use refresh::{RefreshService, RefreshStats};
pub use rule_vector::RuleVector;
pub use storage::{CachedRuleVector, ColdStorage, StorageStats, StorageTier, WarmStorage};
pub use types::{RuleInstance, RuleScope};
pub use vector_comparison::{compare_intent_vs_rule, ComparisonResult, DecisionMode};
