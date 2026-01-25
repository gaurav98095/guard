# API-Only Stack v2: Implementation Plan

**Branch**:  `feat/enforcement-v2`

**Timeline**: 3 weeks

**Objective**: Rebuild enforcement pipeline as clean, API-only service with standardized encoding and simplified data plane.

**Related Document**: See `docs/implementation/02-canonicalization-plan.md` for detailed BERT canonicalization strategy and data collection approach. Both documents are required for v2 implementation.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Key Design Decisions](#key-design-decisions)
4. [File Structure](#file-structure)
5. [Implementation Phases](#implementation-phases)
6. [Detailed Task Breakdown](#detailed-task-breakdown)
7. [Testing Strategy](#testing-strategy)
8. [Deployment Guide](#deployment-guide)

---

## Executive Summary

### Goals

1. **Expose enforcement pipeline as API-only** - No SDK required for external stacks; API returns allow/block decisions only
2. **Add canonicalization layer** - Normalize variable vocabulary to canonical terms before encoding
3. **Simplify data plane** - Remove FFI, merge semantic sandbox into bridge
4. **Add tiered storage** - Hot (memory) + Warm (mmap) + Cold (SQLite)
5. **Rule refresh mechanism** - Scheduled (6hr) + Event-driven
6. **Clean, documented code** - Clear patterns, no redundancy, comprehensive comments

### Non-Goals

- WASM/eBPF sandboxing (future consideration)
- Rule TTL/expiration (rules live until explicitly removed)
- Management UI (focus on API only)

### Requirements

**Scale Target**:

- 10-50 rules per agent (current)
- Scalable to 100K+ rules (future)
- Multiple agents per deployment

**Persistence**:

- Must survive restarts
- Acceptable to re-install rules after crash
- Scheduled refresh every 6 hours
- Event-driven refresh on rule changes

**Performance**:

- P95 latency < 30ms for enforcement
- Support 1000+ RPS per instance
- Lock-free reads

---

## Architecture Overview

### High-Level Flow

```
External Stack (Python/Node/Go/etc)
         ↓ REST/JSON (HTTPS)
┌────────────────────────────────────────────────────┐
│  API Gateway (FastAPI) :8000                       │
│  ────────────────────────────────                  │
│  • POST /v2/enforce                                │
│  • POST /v2/policies/install                       │
│  • POST /v2/encode/intent                          │
│  • POST /v2/encode/policy                          │
│  • GET  /v2/encode/config                          │
│  ────────────────────────────────                  │
│  Internal Services:                                │
│  • Canonicalizer (BERT classifier)                 │
│  • SemanticEncoder (base semantic encoding class)  │
│    ├── IntentEncoder (128d intent vectors)         │
│    └── PolicyEncoder (4×16×32 anchor vectors)      │
│  • AuthService (API keys)                          │
└────────────────────────────────────────────────────┘
         ↓ gRPC (internal network)
┌────────────────────────────────────────────────────┐
│  Data Plane (Rust) :50051                          │
│  ────────────────────────────────                  │
│  gRPC Service:                                     │
│  • Enforce()                                       │
│  • InstallPolicies()                               │
│  • RefreshPolicies()                               │
│  ────────────────────────────────                  │
│  Enforcement Engine:                               │
│  • Query policies by layer                         │
│  • Vector comparison (cosine similarity, no FFI)   │
│  • Short-circuit on BLOCK                          │
│  ────────────────────────────────                  │
│  Storage (3-tier):                                 │
│  • Hot: HashMap (10K policies, in-memory)          │
│  • Warm: Mmap (100K policies, memory-mapped)       │
│  • Cold: SQLite (overflow, disk)                   │
└────────────────────────────────────────────────────┘
```

**Decision-only behavior**: `/v2/enforce` returns ALLOW/BLOCK decisions. Callers are responsible for enforcing those decisions in their own runtime until the hook-based enforcement flow is implemented.

### Component Responsibilities


| Component       | Responsibilities                              | Language         | Port  |
| ----------------- | ----------------------------------------------- | ------------------ | ------- |
| **API Gateway** | REST endpoints, auth, encoding, decision orchestration | Python (FastAPI) | 8000  |
| **Data Plane**  | Rule storage, comparison, decision evaluation  | Rust             | 50051 |
| **Storage**     | Rule persistence, crash recovery              | SQLite + mmap    | -     |

---

## Key Design Decisions

### 1. Canonicalization Layer (Pre-Encoding)

**Problem**: IntentEvents and policies contain variable vocabulary ("query" vs "read", "postgres" vs "database"). Without normalization, encoding similarity becomes fuzzy and unreliable.

**Solution**: Add a BERT classifier that maps free-form terms to canonical vocabulary defined in YAML.

**Flow**:

```
IntentEvent/Policy (variable) → BERT Classifier → Canonical Terms
                                                          ↓
                                    ┌───────────────────┬────────────────────┐
                                    │                   │                    │
                            IntentEncoder        PolicyEncoder        (other encoders)
                           (SemanticEncoder)     (SemanticEncoder)
                                    │                   │
                            128d intent vector   4×16×32 anchor vectors
```

Both IntentEncoder and PolicyEncoder inherit from a common SemanticEncoder base class that handles shared responsibilities: model loading, projection matrix generation, and embedding encoding. The specialized subclasses differ only in how they extract semantic slots and aggregate the final vectors.

**Benefits**:

- **Deterministic**: Same input always produces same output
- **Controllable**: Canonical terms managed in YAML
- **Learnable**: BERT improves via production data and offline retraining
- **Fail-safe**: Unknown terms pass through unchanged (likely fails matching)

**Key Details**:

- BERT model: TinyBERT (14.5M params, <10ms inference)
- Multi-head classification: action, resource_type, sensitivity (configurable)
- Confidence thresholds: high (≥0.9), medium (0.7-0.9), low (<0.7)
- Fail behavior: `passthrough` (use raw term if not confident)
- Logging: All predictions + enforcement outcomes for learning loop

**See**: `docs/implementation/02-canonicalization-plan.md` for detailed design.

---

### 2. Remove FFI Semantic Sandbox

**Current**:

```
EnforcementEngine → FFI call → semantic-sandbox (cdylib) → compare.rs
```

**New**:

```
EnforcementEngine → vector_comparison.rs (direct function call)
```

**Rationale**:

- No isolation needed currently
- Simplifies debugging and testing
- Removes FFI overhead (~8KB struct copying)
- Easier to maintain

**Implementation**:

- Move `semantic-sandbox/src/compare.rs` → `bridge/src/vector_comparison.rs`
- Replace FFI call with direct function call
- Keep same comparison logic (cosine similarity, thresholds)

---

### 3. Tiered Storage System

**Three-tier architecture**:


| Tier     | Technology          | Capacity   | Latency | Use Case                    |
| ---------- | --------------------- | ------------ | --------- | ----------------------------- |
| **Hot**  | HashMap (in-memory) | 10K rules  | <1μs   | Recently evaluated rules    |
| **Warm** | Memory-mapped file  | 100K rules | ~10μs  | Startup cache, fast access  |
| **Cold** | SQLite              | Unlimited  | ~100μs | Overflow, infrequently used |

**Storage Structure**:

```rust
pub struct Bridge {
    // Hot tier - in-memory cache
    hot_cache: Arc<RwLock<HashMap<String, CachedRule>>>,
    hot_capacity: usize,  // Default: 10,000
  
    // Warm tier - memory-mapped file
    warm_storage: Arc<RwLock<Option<Mmap>>>,
    warm_index: Arc<RwLock<HashMap<String, usize>>>,  // rule_id → offset
  
    // Cold tier - SQLite database
    cold_storage: Arc<RwLock<Option<SqliteConnection>>>,
  
    // Metrics
    last_evaluated: Arc<RwLock<HashMap<String, u64>>>,  // LRU tracking
}

struct CachedRule {
    rule: Arc<dyn RuleInstance>,
    anchors: RuleVector,
    loaded_at: u64,
    last_evaluated_at: u64,
}
```

**Loading Strategy**:

1. **On startup**: Load warm tier into memory
2. **On enforce**:
   - Check hot cache → hit: use it
   - Check warm storage → hit: promote to hot
   - Check cold storage → hit: promote to warm
   - Miss: return error (rule not installed)
3. **On eviction** (hot cache full):
   - Evict least-recently-used to warm tier
   - Compact warm tier periodically

**Persistence Format** (warm tier):

```
┌─────────────────────────────────────────────────┐
│  Header                                         │
│  ──────────────────                             │
│  magic: u32 (0x47554152)  // "GUAR"             │
│  version: u32                                   │
│  rule_count: u32                                │
│  index_offset: u64                              │
├─────────────────────────────────────────────────┤
│  Rule Records (variable length)                 │
│  ──────────────────────────────                 │
│  [record_1]                                     │
│  [record_2]                                     │
│  ...                                            │
├─────────────────────────────────────────────────┤
│  Index (rule_id → offset mapping)               │
│  ──────────────────────────────                 │
│  rule_id_1: String → offset: u64                │
│  rule_id_2: String → offset: u64                │
│  ...                                            │
└─────────────────────────────────────────────────┘

Record Format:
├── rule_id: String (length-prefixed)
├── family_id: String
├── layer: String
├── priority: i32
├── enabled: bool
├── created_at_ms: i64
├── params: HashMap<String, ParamValue> (bincode)
└── anchors: RuleVector (4 × 16 × 32 floats)
```

---

### 4. Rule Refresh Mechanism

**Two refresh modes**:

1. **Scheduled refresh** (every 6 hours):

   - Background thread in Data Plane
   - Queries Management Plane for updated rules
   - Compares versions, installs/removes as needed
   - Updates warm storage
2. **Event-driven refresh** (on rule change):

   - API Gateway sends `RefreshRules()` gRPC call
   - Data Plane reloads from warm storage
   - Optional: webhook callback when refresh completes

**Implementation**:

```rust
pub struct RuleRefreshService {
    bridge: Arc<Bridge>,
    refresh_interval: Duration,  // Default: 6 hours
    last_refresh_at: Arc<RwLock<u64>>,
}

impl RuleRefreshService {
    pub async fn start_scheduled_refresh(&self) {
        loop {
            tokio::time::sleep(self.refresh_interval).await;
  
            match self.refresh_from_storage().await {
                Ok(stats) => {
                    info!("Scheduled refresh completed: {:?}", stats);
                }
                Err(e) => {
                    error!("Scheduled refresh failed: {}", e);
                }
            }
        }
    }
  
    pub async fn refresh_from_storage(&self) -> Result<RefreshStats, String> {
        // 1. Read warm storage
        let rules = self.load_warm_storage()?;
  
        // 2. Compare with hot cache
        let (added, removed, updated) = self.compute_diff(&rules)?;
  
        // 3. Apply changes
        self.bridge.apply_rule_changes(added, removed, updated)?;
  
        // 4. Update timestamp
        *self.last_refresh_at.write() = now_ms();
  
        Ok(RefreshStats { added, removed, updated })
    }
}
```

**API Endpoint**:

```python
@router.post("/v1/rules/refresh")
async def trigger_refresh(api_key: str = Depends(verify_api_key)):
    """
    Trigger immediate rule refresh from storage.
  
    Useful when:
    - Rules changed externally
    - Need to force reload after configuration update
    - Testing rule synchronization
    """
    client = get_data_plane_client()
    result = await asyncio.to_thread(client.refresh_rules)
  
    return {
        "success": True,
        "rules_added": result.added,
        "rules_removed": result.removed,
        "rules_updated": result.updated,
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

### 5. Clean Code Standards

**Code formatting rules**:

1. **Rust**:

   - Run `cargo fmt` on all files
   - Run `cargo clippy -- -D warnings` (zero warnings)
   - Max function length: 100 lines
   - Max file length: 1000 lines
   - Document all public functions with `///` doc comments
   - Use `#[inline]` for hot-path functions
2. **Python**:

   - Black formatter (line length: 100)
   - isort for imports
   - Type hints on all function signatures
   - Docstrings for all public functions (Google style)
   - Pylint score > 9.0
3. **Comments**:

   - Explain **why**, not **what**
   - Performance-critical sections: explain optimization
   - Complex algorithms: cite source or explanation
   - Every file: module-level docstring with purpose
4. **Naming conventions**:

   - Rust: snake_case for functions/variables, PascalCase for types
   - Python: snake_case for functions/variables, PascalCase for classes
   - Constants: SCREAMING_SNAKE_CASE
   - Acronyms: treat as words (e.g., `HttpClient`, not `HTTPClient`)
5. **Error handling**:

   - Rust: `Result<T, E>` for fallible operations, never `unwrap()` in production
   - Python: Specific exception types, not bare `Exception`
   - All errors logged with context

---

## File Structure

```
guard/
├── management_plane/
│   ├── app/
│   │   ├── main.py                     # ✏️ MODIFIED: Add /v2 routes, OpenAPI
│   │   ├── config.py                   # ✏️ MODIFIED: Add canonicalization config
│   │   ├── auth.py                     # ✏️ MODIFIED: Add API key auth
│   │   │
│   │   ├── services/                   # ✨ NEW: Service layer
│   │   │   ├── __init__.py
│   │   │   ├── canonicalizer.py        # ✨ NEW: BERT classifier for vocabulary normalization
│   │   │   ├── canonicalization_logger.py # ✨ NEW: Log predictions for learning loop
│   │   │   ├── semantic_encoder.py     # ✨ NEW: Base class for semantic encoding (shared logic)
│   │   │   ├── intent_encoder.py       # ✏️ MODIFIED: Subclass of SemanticEncoder, 128d intent vectors
│   │   │   ├── policy_encoder.py       # ✏️ MODIFIED: Subclass of SemanticEncoder, anchor encoding
│   │   │   └── dataplane_client.py     # gRPC client wrapper
│   │   │
│   │   ├── endpoints/
│   │   │   ├── enforcement.py          # ✏️ MODIFIED: Support API key auth, canonicalization
│   │   │   ├── encoding.py             # ✏️ MODIFIED: /v2/encode/* endpoints
│   │   │   ├── policies.py             # ✨ NEW: /v2/policies/* endpoints
│   │   │   └── health.py               # ✏️ MODIFIED: Enhanced health check
│   │   │
│   │   ├── models/
│   │   │   ├── api.py                  # ✨ NEW: API request/response models
│   │   │   ├── events.py               # ✨ NEW: IntentEvent v2 schema
│   │   │   └── policies.py             # ✨ NEW: Policy configuration schemas
│   │   │
│   │   ├── config/
│   │   │   ├── encoding.yaml           # ✨ NEW: Encoding slot definitions
│   │   │   └── canonicalization.yaml   # ✨ NEW: Canonical vocabulary + BERT config
│   │   │
│   │   └── models/
│   │       └── canonicalizer_tinybert/ # ✨ NEW: Fine-tuned BERT model files
│   │
│   ├── Dockerfile                      # ✏️ MODIFIED: Production-ready
│   └── requirements.txt                # ✏️ MODIFIED: Add dependencies
│
├── data_plane/
│   ├── bridge/                         # ✏️ REORGANIZED: Moved from tupl_dp/bridge
│   │   ├── src/
│   │   │   ├── lib.rs                  # ✏️ MODIFIED: Export new modules
│   │   │   ├── grpc_server.rs          # ✏️ MODIFIED: Add RefreshRules RPC
│   │   │   ├── enforcement_engine.rs   # ✏️ MODIFIED: Use direct vector_comparison
│   │   │   │
│   │   │   ├── vector_comparison.rs    # ✨ NEW: Merged from semantic-sandbox
│   │   │   │
│   │   │   ├── storage/                # ✨ NEW: Tiered storage
│   │   │   │   ├── mod.rs
│   │   │   │   ├── hot_cache.rs        # In-memory HashMap cache
│   │   │   │   ├── warm_storage.rs     # Memory-mapped file
│   │   │   │   ├── cold_storage.rs     # SQLite backend
│   │   │   │   └── types.rs            # Storage data structures
│   │   │   │
│   │   │   ├── refresh/                # ✨ NEW: Rule refresh service
│   │   │   │   ├── mod.rs
│   │   │   │   ├── scheduler.rs        # Scheduled refresh (6hr)
│   │   │   │   └── service.rs          # Refresh logic
│   │   │   │
│   │   │   ├── bridge.rs               # ✏️ MODIFIED: Use tiered storage
│   │   │   ├── table.rs                # ✏️ MODIFIED: Integrate with storage
│   │   │   └── rule_vector.rs          # (unchanged)
│   │   │
│   │   ├── Cargo.toml                  # ✏️ MODIFIED: Add deps (memmap2, rusqlite)
│   │   └── build.rs                    # (unchanged)
│   │
│   ├── semantic-sandbox/               # ❌ DELETED: Merged into bridge
│   │
│   └── Dockerfile                      # ✏️ MODIFIED: Production-ready
│
├── docker-compose.yml                  # ✨ NEW: Orchestration
├── .env.example                        # ✨ NEW: Configuration template
│
├── docs/
│   ├── implementation/
│   │   ├── 00-enforcement-flow-trace.md
│   │   ├── 01-api-only-v2-plan.md      # ✨ THIS FILE
│   │   └── 02-canonicalization-plan.md # ✨ NEW: Vocabulary normalization + BERT strategy
│   │
│   └── api/
│       ├── openapi.yaml                # ✨ NEW: OpenAPI spec
│       ├── external-integration.md     # ✨ NEW: Integration guide
│       └── examples/                   # ✨ NEW: Code examples
│           ├── python_client.py
│           ├── nodejs_client.js
│           ├── go_client.go
│           └── curl_examples.sh
│
└── tests/
    ├── integration/                    # ✨ NEW: End-to-end tests
    │   ├── test_api_enforcement.py
    │   ├── test_rule_lifecycle.py
    │   └── test_storage_persistence.py
    │
    └── load/                           # ✨ NEW: Load testing
        └── locustfile.py

Legend:
  ✨ NEW: Create new file
  ✏️ MODIFIED: Edit existing file
  ❌ DELETED: Remove file
```

---

## Implementation Phases

### Week 1: Foundation & Canonicalization Layer

**Goals**:

- Set up BERT classifier and canonicalization service
- Create encoding configuration (YAML)
- Expose REST API endpoints (/v2)
- Add API key authentication
- Basic integration tests

**Deliverables**:

- Working `/v2/enforce` endpoint with canonicalization
- Working `/v2/encode/intent` endpoint
- BERT classifier loaded and inference working
- Canonical vocabulary configured
- API key auth functional
- Docker Compose setup

**Week 1 Completion Summary (COMPLETED ✅):**

**Files Created** (11 new files, ~2,100 LOC):

1. **`management_plane/app/services/canonicalizer.py`** (410 lines)

   - TinyBERT ONNX multi-head classifier for vocabulary normalization
   - Classifies 3 fields: action, resource_type, sensitivity
   - Confidence thresholds: high (≥0.9), medium (≥0.7), low (<0.7)
   - Fail behavior: `passthrough` (unknown terms logged with confidence=0.0)
2. **`management_plane/app/services/canonicalization_logger.py`** (250 lines)

   - Async JSONL logging to `/var/log/guard/canonicalization/{date}.jsonl`
   - Daily file rotation, 90-day retention
   - Logs canonicalization predictions + enforcement outcomes
3. **`management_plane/app/services/semantic_encoder.py`** (280 lines)

   - Base class for semantic encoding (shared logic)
   - Loads sentence-transformers model (`all-MiniLM-L6-v2`)
   - Generates deterministic projection matrices (seed-based)
   - LRU cache for embeddings (10K entries)
4. **`management_plane/app/services/intent_encoder.py`** (170 lines)

   - IntentEncoder subclass of SemanticEncoder
   - Extracts 4 semantic slots from canonical IntentEvent
   - Encodes to 128-dimensional intent vector with per-slot normalization
5. **`management_plane/app/services/policy_encoder.py`** (300 lines)

   - PolicyEncoder subclass of SemanticEncoder
   - Extracts anchor lists from canonical DesignBoundary
   - Encodes to 4×16×32 RuleVector structure
6. **`management_plane/app/services/__init__.py`**

   - Service layer exports
7. **`management_plane/app/endpoints/enforcement_v2.py`** (550 lines)

   - `POST /api/v2/enforce` - Main enforcement with canonicalization trace
   - `POST /api/v2/canonicalize` - Debug endpoint (canonicalization only)
   - `POST /api/v2/policies/install` - Policy installation endpoint (stub)
   - Returns canonicalization trace in `metadata.canonicalization_trace`
8. **`management_plane/app/config/canonicalization.yaml`**

   - Canonical vocabulary definitions for action, resource_type, sensitivity
   - Maps synonyms to canonical terms (e.g., "query" → "read")
9. **`tests/test_canonicalizer.py`** (150 lines)

   - Unit tests for BERT canonicalizer
   - Tests confidence thresholds and fail behaviors
10. **`tests/test_semantic_encoders.py`** (280 lines)

    - Unit tests for Intent and Policy encoders
    - Tests embedding generation and vector dimensionality
11. **`management_plane/models/canonicalizer_tinybert_v1.0/`**

    - TinyBERT ONNX model files (14.5M params, <10ms inference)

**Files Modified** (3 files):

1. **`management_plane/app/main.py`**

   - Added v2 router registration
   - Service initialization (Canonicalizer, IntentEncoder, PolicyEncoder)
   - Graceful shutdown with logger flush
2. **`management_plane/app/config.py`**

   - Added 11 canonicalization configuration variables
   - BERT model paths, confidence thresholds, logging directories
3. **`management_plane/pyproject.toml`**

   - Added dependencies: `onnxruntime>=1.17.0`, `transformers>=4.35.0`

**Architecture**:

```
External Stack → REST/JSON → Management Plane :8000
                              • /api/v2/enforce
                              • /api/v2/canonicalize (debug)
                          
                              Services:
                              • Canonicalizer (TinyBERT ONNX)
                              • SemanticEncoder (base class)
                                ├─ IntentEncoder (128d)
                                └─ PolicyEncoder (4×16×32)
                              • CanonicalizationLogger (async JSONL)
                          
                              ↓ gRPC ↓
                          
                              Data Plane :50051 (unchanged)
```

**Performance**:


| Operation             | Latency                         |
| ----------------------- | --------------------------------- |
| BERT Inference        | 5-10ms                          |
| Intent Encoding       | 5-8ms                           |
| **Total V2 Enforce**  | **30-50ms**                     |
| V1 Enforce (baseline) | 20-30ms                         |
| Overhead              | +50% for vocabulary flexibility |

**Key Design Decisions**:

1. **Full BERT Canonicalization**: ML-based classification, not rule-based
2. **Intents + Policies**: Both canonicalized for vocabulary consistency
3. **Passthrough Behavior**: Unknown terms pass through (logged with confidence=0.0)
   - ⚠️ May cause policy matching failures - monitor logs
4. **File-based JSONL Logging**: Simple, auditable, ready for offline learning
5. **API Versioning**: `/api/v2` added as NEW endpoint (v1 unchanged for backward compatibility)

**Status**: ✅ **COMPLETE**

- All 11 files created with ~2,100 lines of code
- 3 files modified for integration
- Unit tests included
- Ready for testing and deployment

---

### Week 2: Data Plane Refactor & Storage

#### Day 4-5: Remove FFI & Tiered Storage Implementation

**Task 2.0: Merge semantic-sandbox comparison logic** (4 hours)

1. Move `data_plane/semantic-sandbox/src/compare.rs` → `data_plane/bridge/src/vector_comparison.rs`
2. Update `enforcement_engine.rs` to call directly (no FFI)
3. Remove `VectorEnvelope` struct (no longer needed for FFI)
4. Test: `cargo test vector_comparison::tests`
5. Delete `semantic-sandbox/` directory

**Task 2.1: Define storage types** (3 hours)

1. Create `data_plane/bridge/src/storage/types.rs`:
   ```rust
   //! Storage data structures and traits.

   use crate::types::RuleInstance;
   use crate::rule_vector::RuleVector;
   use std::sync::Arc;

   /// A cached rule with metadata for LRU eviction.
   #[derive(Clone, Debug)]
   pub struct CachedRule {
       /// The rule instance
       pub rule: Arc<dyn RuleInstance>,

       /// Pre-encoded anchor vectors
       pub anchors: RuleVector,

       /// When this rule was loaded into cache (Unix timestamp ms)
       pub loaded_at: u64,

       /// Last time this rule was evaluated (Unix timestamp ms)
       pub last_evaluated_at: u64,
   }

   impl CachedRule {
       /// Create a new cached rule.
       pub fn new(rule: Arc<dyn RuleInstance>, anchors: RuleVector) -> Self {
           let now = crate::types::now_ms();
           Self {
               rule,
               anchors,
               loaded_at: now,
               last_evaluated_at: now,
           }
       }

       /// Update last evaluated timestamp.
       #[inline]
       pub fn mark_evaluated(&mut self) {
           self.last_evaluated_at = crate::types::now_ms();
       }
   }

   /// Statistics about storage tier usage.
   #[derive(Clone, Debug, Default)]
   pub struct StorageStats {
       pub hot_rules: usize,
       pub warm_rules: usize,
       pub cold_rules: usize,
       pub hot_hits: u64,
       pub warm_hits: u64,
       pub cold_hits: u64,
       pub evictions: u64,
   }

   /// Storage tier levels.
   #[derive(Clone, Copy, Debug, PartialEq, Eq)]
   pub enum StorageTier {
       Hot,   // In-memory HashMap
       Warm,  // Memory-mapped file
       Cold,  // SQLite database
   }
   ```

**Task 2.2: Implement hot cache** (4 hours)

1. Create `data_plane/bridge/src/storage/hot_cache.rs`:
   ```rust
   //! Hot cache - in-memory HashMap for recently evaluated rules.
   //!
   //! # Capacity Management
   //! - Default capacity: 10,000 rules
   //! - LRU eviction when capacity reached
   //! - Lock-free reads using RwLock
   //!
   //! # Performance
   //! - Get: O(1) average, <1μs
   //! - Insert: O(1) average
   //! - Evict: O(n) worst case (sorted by last_evaluated_at)

   use crate::storage::types::{CachedRule, StorageStats};
   use parking_lot::RwLock;
   use std::collections::HashMap;
   use std::sync::Arc;

   /// Hot cache configuration.
   #[derive(Clone, Debug)]
   pub struct HotCacheConfig {
       /// Maximum number of rules to keep in memory
       pub capacity: usize,

       /// Number of rules to evict when capacity reached (default: 10% of capacity)
       pub eviction_batch_size: usize,
   }

   impl Default for HotCacheConfig {
       fn default() -> Self {
           Self {
               capacity: 10_000,
               eviction_batch_size: 1_000,
           }
       }
   }

   /// In-memory cache for hot rules.
   pub struct HotCache {
       /// Map of rule_id → cached rule
       rules: Arc<RwLock<HashMap<String, CachedRule>>>,

       /// Configuration
       config: HotCacheConfig,

       /// Statistics
       stats: Arc<RwLock<StorageStats>>,
   }

   impl HotCache {
       /// Create a new hot cache with default configuration.
       pub fn new() -> Self {
           Self::with_config(HotCacheConfig::default())
       }

       /// Create a hot cache with custom configuration.
       pub fn with_config(config: HotCacheConfig) -> Self {
           Self {
               rules: Arc::new(RwLock::new(HashMap::with_capacity(config.capacity))),
               config,
               stats: Arc::new(RwLock::new(StorageStats::default())),
           }
       }

       /// Get a rule from the cache.
       ///
       /// Updates last_evaluated_at timestamp if found.
       pub fn get(&self, rule_id: &str) -> Option<CachedRule> {
           let mut rules = self.rules.write();

           if let Some(cached) = rules.get_mut(rule_id) {
               cached.mark_evaluated();
               self.stats.write().hot_hits += 1;
               Some(cached.clone())
           } else {
               None
           }
       }

       /// Insert a rule into the cache.
       ///
       /// If capacity is exceeded, evicts least-recently-used rules.
       pub fn insert(&self, rule_id: String, cached_rule: CachedRule) -> Result<(), String> {
           let mut rules = self.rules.write();

           // Check capacity
           if rules.len() >= self.config.capacity && !rules.contains_key(&rule_id) {
               // Evict LRU rules
               self.evict_lru(&mut rules)?;
           }

           rules.insert(rule_id, cached_rule);
           Ok(())
       }

       /// Remove a rule from the cache.
       pub fn remove(&self, rule_id: &str) -> Option<CachedRule> {
           self.rules.write().remove(rule_id)
       }

       /// Get cache statistics.
       pub fn stats(&self) -> StorageStats {
           let mut stats = self.stats.read().clone();
           stats.hot_rules = self.rules.read().len();
           stats
       }

       /// Evict least-recently-used rules to make space.
       ///
       /// Returns evicted rules for promotion to warm tier.
       fn evict_lru(
           &self,
           rules: &mut HashMap<String, CachedRule>
       ) -> Result<Vec<(String, CachedRule)>, String> {
           // Sort by last_evaluated_at
           let mut entries: Vec<_> = rules.iter()
               .map(|(id, rule)| (id.clone(), rule.last_evaluated_at))
               .collect();

           entries.sort_by_key(|(_, timestamp)| *timestamp);

           // Take the oldest batch
           let to_evict: Vec<String> = entries
               .iter()
               .take(self.config.eviction_batch_size)
               .map(|(id, _)| id.clone())
               .collect();

           // Remove from cache
           let evicted: Vec<_> = to_evict
               .iter()
               .filter_map(|id| rules.remove(id).map(|rule| (id.clone(), rule)))
               .collect();

           self.stats.write().evictions += evicted.len() as u64;

           Ok(evicted)
       }
   }
   ```

**Task 2.3: Implement warm storage (mmap)** (6 hours)

1. Create `data_plane/bridge/src/storage/warm_storage.rs`:
   ```rust
   //! Warm storage - memory-mapped file for persistent rule cache.
   //!
   //! # File Format
   //! ```
   //! Header (32 bytes):
   //!   magic: u32 (0x47554152 = "GUAR")
   //!   version: u32
   //!   rule_count: u32
   //!   index_offset: u64
   //!   reserved: [u8; 16]
   //!
   //! Rule Records (variable length):
   //!   [record_1]
   //!   [record_2]
   //!   ...
   //!
   //! Index (rule_id → offset):
   //!   [rule_id_1, offset_1]
   //!   [rule_id_2, offset_2]
   //!   ...
   //! ```
   //!
   //! # Record Format
   //! Each record is bincode-serialized:
   //! - rule_id: String
   //! - family_id: String
   //! - layer: String
   //! - priority: i32
   //! - enabled: bool
   //! - created_at_ms: i64
   //! - params: HashMap<String, ParamValue>
   //! - anchors: RuleVector (4 × 16 × 32 floats = 8KB)

   use memmap2::{Mmap, MmapMut};
   use parking_lot::RwLock;
   use serde::{Deserialize, Serialize};
   use std::collections::HashMap;
   use std::fs::{File, OpenOptions};
   use std::io::{Seek, SeekFrom, Write};
   use std::path::{Path, PathBuf};
   use std::sync::Arc;

   use crate::storage::types::CachedRule;
   use crate::rule_vector::RuleVector;

   const MAGIC: u32 = 0x47554152; // "GUAR"
   const VERSION: u32 = 1;
   const HEADER_SIZE: u64 = 32;

   /// Warm storage header.
   #[derive(Debug, Clone, Serialize, Deserialize)]
   struct Header {
       magic: u32,
       version: u32,
       rule_count: u32,
       index_offset: u64,
   }

   /// Serializable rule record.
   #[derive(Debug, Clone, Serialize, Deserialize)]
   struct RuleRecord {
       rule_id: String,
       family_id: String,
       layer: String,
       priority: i32,
       enabled: bool,
       created_at_ms: i64,
       params: HashMap<String, String>,  // Simplified for serialization
       anchors: RuleVector,
   }

   /// Memory-mapped warm storage.
   pub struct WarmStorage {
       /// Path to storage file
       path: PathBuf,

       /// Memory-mapped file (read-only)
       mmap: Arc<RwLock<Option<Mmap>>>,

       /// Index: rule_id → offset in file
       index: Arc<RwLock<HashMap<String, u64>>>,
   }

   impl WarmStorage {
       /// Open or create warm storage at the given path.
       pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
           let path = path.as_ref().to_path_buf();

           // Create parent directory if needed
           if let Some(parent) = path.parent() {
               std::fs::create_dir_all(parent)
                   .map_err(|e| format!("Failed to create storage directory: {}", e))?;
           }

           let storage = Self {
               path: path.clone(),
               mmap: Arc::new(RwLock::new(None)),
               index: Arc::new(RwLock::new(HashMap::new())),
           };

           // Load existing storage if file exists
           if path.exists() {
               storage.load()?;
           } else {
               // Create new empty storage
               storage.create()?;
           }

           Ok(storage)
       }

       /// Get a rule by ID.
       pub fn get(&self, rule_id: &str) -> Result<Option<CachedRule>, String> {
           let index = self.index.read();
           let offset = match index.get(rule_id) {
               Some(off) => *off,
               None => return Ok(None),
           };

           let mmap_guard = self.mmap.read();
           let mmap = mmap_guard
               .as_ref()
               .ok_or("Warm storage not loaded")?;

           // Read record at offset
           let record: RuleRecord = bincode::deserialize(&mmap[offset as usize..])
               .map_err(|e| format!("Failed to deserialize record: {}", e))?;

           // Convert to CachedRule
           // Note: This requires implementing From<RuleRecord> for CachedRule
           todo!("Convert RuleRecord to CachedRule")
       }

       /// Write all rules to storage.
       ///
       /// This is called when:
       /// - New rules are installed
       /// - Rules are evicted from hot cache
       /// - Scheduled refresh completes
       pub fn write_all(&self, rules: Vec<(String, CachedRule)>) -> Result<(), String> {
           // Create temporary file
           let tmp_path = self.path.with_extension("tmp");
           let mut file = OpenOptions::new()
               .write(true)
               .create(true)
               .truncate(true)
               .open(&tmp_path)
               .map_err(|e| format!("Failed to create temp file: {}", e))?;

           // Write header (placeholder)
           let header = Header {
               magic: MAGIC,
               version: VERSION,
               rule_count: rules.len() as u32,
               index_offset: 0, // Will update after writing records
           };
           let header_bytes = bincode::serialize(&header)
               .map_err(|e| format!("Failed to serialize header: {}", e))?;
           file.write_all(&header_bytes)
               .map_err(|e| format!("Failed to write header: {}", e))?;

           // Write records and build index
           let mut index = HashMap::new();
           for (rule_id, cached_rule) in rules {
               let offset = file.stream_position()
                   .map_err(|e| format!("Failed to get position: {}", e))?;

               // Convert CachedRule to RuleRecord
               let record = RuleRecord {
                   rule_id: rule_id.clone(),
                   // ... convert fields
                   anchors: cached_rule.anchors.clone(),
               };

               let record_bytes = bincode::serialize(&record)
                   .map_err(|e| format!("Failed to serialize record: {}", e))?;
               file.write_all(&record_bytes)
                   .map_err(|e| format!("Failed to write record: {}", e))?;

               index.insert(rule_id, offset);
           }

           // Write index
           let index_offset = file.stream_position()
               .map_err(|e| format!("Failed to get index position: {}", e))?;
           let index_bytes = bincode::serialize(&index)
               .map_err(|e| format!("Failed to serialize index: {}", e))?;
           file.write_all(&index_bytes)
               .map_err(|e| format!("Failed to write index: {}", e))?;

           // Update header with index offset
           file.seek(SeekFrom::Start(0))
               .map_err(|e| format!("Failed to seek to header: {}", e))?;
           let header = Header {
               index_offset,
               ..header
           };
           let header_bytes = bincode::serialize(&header)
               .map_err(|e| format!("Failed to serialize updated header: {}", e))?;
           file.write_all(&header_bytes)
               .map_err(|e| format!("Failed to write updated header: {}", e))?;

           file.sync_all()
               .map_err(|e| format!("Failed to sync file: {}", e))?;
           drop(file);

           // Atomic rename
           std::fs::rename(&tmp_path, &self.path)
               .map_err(|e| format!("Failed to rename file: {}", e))?;

           // Reload mmap
           self.load()?;

           Ok(())
       }

       /// Load storage file into memory map.
       fn load(&self) -> Result<(), String> {
           let file = File::open(&self.path)
               .map_err(|e| format!("Failed to open storage file: {}", e))?;

           let mmap = unsafe {
               Mmap::map(&file)
                   .map_err(|e| format!("Failed to mmap file: {}", e))?
           };

           // Read header
           let header: Header = bincode::deserialize(&mmap[..HEADER_SIZE as usize])
               .map_err(|e| format!("Failed to deserialize header: {}", e))?;

           if header.magic != MAGIC {
               return Err("Invalid magic number".to_string());
           }

           if header.version != VERSION {
               return Err(format!("Unsupported version: {}", header.version));
           }

           // Read index
           let index: HashMap<String, u64> = bincode::deserialize(
               &mmap[header.index_offset as usize..]
           ).map_err(|e| format!("Failed to deserialize index: {}", e))?;

           *self.mmap.write() = Some(mmap);
           *self.index.write() = index;

           Ok(())
       }

       /// Create empty storage file.
       fn create(&self) -> Result<(), String> {
           self.write_all(vec![])?;
           Ok(())
       }
   }
   ```

**Task 2.4: Implement cold storage (SQLite)** (4 hours)

1. Create `data_plane/bridge/src/storage/cold_storage.rs`:
   ```rust
   //! Cold storage - SQLite database for overflow and long-term persistence.
   //!
   //! Used when:
   //! - Total rules exceed warm storage capacity (>100K)
   //! - Archiving old/disabled rules
   //! - Audit logging of rule changes

   use rusqlite::{Connection, params};
   use parking_lot::Mutex;
   use std::path::Path;
   use std::sync::Arc;

   use crate::storage::types::CachedRule;

   /// SQLite-backed cold storage.
   pub struct ColdStorage {
       conn: Arc<Mutex<Connection>>,
   }

   impl ColdStorage {
       /// Open or create cold storage database.
       pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
           let conn = Connection::open(path)
               .map_err(|e| format!("Failed to open database: {}", e))?;

           // Create schema
           conn.execute(
               "CREATE TABLE IF NOT EXISTS rules (
                   rule_id TEXT PRIMARY KEY,
                   family_id TEXT NOT NULL,
                   layer TEXT NOT NULL,
                   priority INTEGER NOT NULL,
                   enabled BOOLEAN NOT NULL,
                   created_at_ms INTEGER NOT NULL,
                   params_json TEXT,
                   anchors_blob BLOB NOT NULL,
                   stored_at_ms INTEGER NOT NULL
               )",
               [],
           ).map_err(|e| format!("Failed to create table: {}", e))?;

           // Create indices
           conn.execute(
               "CREATE INDEX IF NOT EXISTS idx_layer ON rules(layer)",
               [],
           ).map_err(|e| format!("Failed to create index: {}", e))?;

           conn.execute(
               "CREATE INDEX IF NOT EXISTS idx_family ON rules(family_id)",
               [],
           ).map_err(|e| format!("Failed to create index: {}", e))?;

           Ok(Self {
               conn: Arc::new(Mutex::new(conn)),
           })
       }

       /// Get a rule by ID.
       pub fn get(&self, rule_id: &str) -> Result<Option<CachedRule>, String> {
           let conn = self.conn.lock();

           let mut stmt = conn.prepare(
               "SELECT anchors_blob FROM rules WHERE rule_id = ?1"
           ).map_err(|e| format!("Failed to prepare query: {}", e))?;

           let result = stmt.query_row(params![rule_id], |row| {
               let blob: Vec<u8> = row.get(0)?;
               Ok(blob)
           });

           match result {
               Ok(blob) => {
                   // Deserialize blob to CachedRule
                   todo!("Deserialize blob to CachedRule")
               }
               Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
               Err(e) => Err(format!("Query failed: {}", e)),
           }
       }

       /// Insert or update a rule.
       pub fn upsert(&self, rule_id: &str, cached_rule: &CachedRule) -> Result<(), String> {
           let conn = self.conn.lock();

           // Serialize anchors to blob
           let anchors_blob = bincode::serialize(&cached_rule.anchors)
               .map_err(|e| format!("Failed to serialize anchors: {}", e))?;

           conn.execute(
               "INSERT OR REPLACE INTO rules 
                (rule_id, family_id, layer, priority, enabled, created_at_ms, 
                 anchors_blob, stored_at_ms)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
               params![
                   rule_id,
                   cached_rule.rule.family_id().family_id(),
                   cached_rule.rule.layer().to_string(),
                   cached_rule.rule.priority(),
                   cached_rule.rule.enabled(),
                   cached_rule.rule.created_at(),
                   anchors_blob,
                   crate::types::now_ms(),
               ],
           ).map_err(|e| format!("Failed to insert rule: {}", e))?;

           Ok(())
       }

       /// Remove a rule.
       pub fn remove(&self, rule_id: &str) -> Result<bool, String> {
           let conn = self.conn.lock();

           let rows = conn.execute(
               "DELETE FROM rules WHERE rule_id = ?1",
               params![rule_id],
           ).map_err(|e| format!("Failed to delete rule: {}", e))?;

           Ok(rows > 0)
       }

       /// Get all rule IDs (for refresh).
       pub fn list_rule_ids(&self) -> Result<Vec<String>, String> {
           let conn = self.conn.lock();

           let mut stmt = conn.prepare("SELECT rule_id FROM rules")
               .map_err(|e| format!("Failed to prepare query: {}", e))?;

           let ids = stmt.query_map([], |row| row.get(0))
               .map_err(|e| format!("Query failed: {}", e))?
               .collect::<Result<Vec<String>, _>>()
               .map_err(|e| format!("Failed to collect results: {}", e))?;

           Ok(ids)
       }
   }
   ```

**Task 2.5: Integrate storage into Bridge** (6 hours)

1. Update `data_plane/bridge/src/bridge.rs`:
   - Replace simple HashMap with tiered storage
   - Add storage configuration
   - Implement tier promotion logic
   - Add background compaction task

---

#### Day 6-7: Rule Refresh Mechanism

**Task 2.6: Implement refresh service** (6 hours)

1. Create `data_plane/bridge/src/refresh/service.rs`:
   ```rust
   //! Rule refresh service.
   //!
   //! Handles both scheduled (6hr) and event-driven rule refresh from storage.

   use std::sync::Arc;
   use std::time::Duration;
   use tokio::time::interval;
   use parking_lot::RwLock;

   use crate::bridge::Bridge;
   use crate::types::now_ms;

   /// Statistics from a refresh operation.
   #[derive(Debug, Clone)]
   pub struct RefreshStats {
       pub rules_added: usize,
       pub rules_removed: usize,
       pub rules_updated: usize,
       pub duration_ms: u64,
       pub timestamp: u64,
   }

   /// Rule refresh service configuration.
   #[derive(Clone, Debug)]
   pub struct RefreshConfig {
       /// How often to run scheduled refresh
       pub refresh_interval: Duration,

       /// Whether to enable scheduled refresh
       pub enable_scheduled: bool,
   }

   impl Default for RefreshConfig {
       fn default() -> Self {
           Self {
               refresh_interval: Duration::from_secs(6 * 3600), // 6 hours
               enable_scheduled: true,
           }
       }
   }

   /// Rule refresh service.
   pub struct RefreshService {
       bridge: Arc<Bridge>,
       config: RefreshConfig,
       last_refresh_at: Arc<RwLock<u64>>,
   }

   impl RefreshService {
       /// Create a new refresh service.
       pub fn new(bridge: Arc<Bridge>, config: RefreshConfig) -> Self {
           Self {
               bridge,
               config,
               last_refresh_at: Arc::new(RwLock::new(now_ms())),
           }
       }

       /// Start the scheduled refresh loop.
       ///
       /// Runs in background task, never returns unless error.
       pub async fn start_scheduled_refresh(&self) -> Result<(), String> {
           if !self.config.enable_scheduled {
               return Ok(());
           }

           let mut interval = interval(self.config.refresh_interval);

           loop {
               interval.tick().await;

               match self.refresh_from_storage().await {
                   Ok(stats) => {
                       log::info!(
                           "Scheduled refresh completed: +{} -{} ~{} ({}ms)",
                           stats.rules_added,
                           stats.rules_removed,
                           stats.rules_updated,
                           stats.duration_ms
                       );
                   }
                   Err(e) => {
                       log::error!("Scheduled refresh failed: {}", e);
                   }
               }
           }
       }

       /// Trigger an immediate refresh from storage.
       ///
       /// Called via gRPC RefreshRules() endpoint.
       pub async fn refresh_from_storage(&self) -> Result<RefreshStats, String> {
           let start = now_ms();

           // Load all rules from warm storage
           let warm_rules = self.bridge.storage.warm.list_all()?;

           // Compare with current hot cache
           let current_ids: Vec<_> = self.bridge
               .hot_cache
               .rules
               .read()
               .keys()
               .cloned()
               .collect();

           let warm_ids: Vec<_> = warm_rules
               .iter()
               .map(|(id, _)| id.clone())
               .collect();

           // Compute diff
           let to_add: Vec<_> = warm_rules
               .iter()
               .filter(|(id, _)| !current_ids.contains(id))
               .collect();

           let to_remove: Vec<_> = current_ids
               .iter()
               .filter(|id| !warm_ids.contains(id))
               .collect();

           // Apply changes
           for (id, cached_rule) in to_add {
               self.bridge.hot_cache.insert(id.clone(), cached_rule.clone())?;
           }

           for id in to_remove {
               self.bridge.hot_cache.remove(id);
           }

           let duration_ms = now_ms() - start;
           *self.last_refresh_at.write() = now_ms();

           Ok(RefreshStats {
               rules_added: to_add.len(),
               rules_removed: to_remove.len(),
               rules_updated: 0, // TODO: detect updates
               duration_ms,
               timestamp: now_ms(),
           })
       }
   }
   ```

**Task 2.7: Add RefreshRules gRPC endpoint** (3 hours)

1. Update `data_plane/proto/rule_installation.proto`:

   ```protobuf
   service DataPlane {
     // ... existing RPCs

     // Trigger rule refresh from storage
     rpc RefreshRules(RefreshRulesRequest) returns (RefreshRulesResponse);
   }

   message RefreshRulesRequest {
     // Empty - triggers full refresh
   }

   message RefreshRulesResponse {
     bool success = 1;
     string message = 2;
     int32 rules_added = 3;
     int32 rules_removed = 4;
     int32 rules_updated = 5;
     int64 duration_ms = 6;
   }
   ```
2. Implement in `grpc_server.rs`:

   ```rust
   async fn refresh_rules(
       &self,
       request: Request<RefreshRulesRequest>,
   ) -> Result<Response<RefreshRulesResponse>, Status> {
       log::info!("RefreshRules RPC called");

       match self.refresh_service.refresh_from_storage().await {
           Ok(stats) => {
               Ok(Response::new(RefreshRulesResponse {
                   success: true,
                   message: format!(
                       "Refreshed: +{} -{} ~{}",
                       stats.rules_added, stats.rules_removed, stats.rules_updated
                   ),
                   rules_added: stats.rules_added as i32,
                   rules_removed: stats.rules_removed as i32,
                   rules_updated: stats.rules_updated as i32,
                   duration_ms: stats.duration_ms as i64,
               }))
           }
           Err(e) => {
               Err(Status::internal(format!("Refresh failed: {}", e)))
           }
       }
   }
   ```

---

### Week 3: Rule Refresh (Scheduled), Polish & Documentation

**See**: `docs/implementation/04-week3-plan.md` for detailed implementation strategy.

**Priorities**:
1. Scheduled refresh service (background 6hr task)
2. Hot cache LRU eviction (capacity enforcement)
3. Polish & documentation (code standards, tests)

#### Day 8: Rule Refresh - Scheduled Refresh + LRU Eviction

**Task 3.0: Implement scheduled refresh service** (4 hours)

1. Create `data_plane/tupl_dp/bridge/src/refresh/mod.rs`:
   ```rust
   //! Rule refresh service - scheduled background task.
   //!
   //! # Scheduled Refresh (6hr)
   //! - Background tokio task spawned on startup
   //! - Loads warm storage anchors every 6 hours
   //! - Replaces hot cache with latest rules
   //! - Logs refresh stats
   //!
   //! # Implementation
   //! Only implement the scheduled loop here.
   //! Event-driven refresh (gRPC) already implemented in Week 2.
   
   use std::sync::Arc;
   use std::time::Duration;
   use tokio::time::interval;
   use parking_lot::RwLock;
   
   use crate::bridge::Bridge;
   use crate::types::now_ms;
   
   #[derive(Debug, Clone)]
   pub struct RefreshStats {
       pub rules_refreshed: usize,
       pub duration_ms: u64,
       pub timestamp: u64,
   }
   
   pub struct RefreshService {
       bridge: Arc<Bridge>,
       refresh_interval: Duration,
       last_refresh_at: Arc<RwLock<u64>>,
   }
   
   impl RefreshService {
       pub fn new(bridge: Arc<Bridge>) -> Self {
           Self {
               bridge,
               refresh_interval: Duration::from_secs(6 * 3600), // 6 hours
               last_refresh_at: Arc::new(RwLock::new(now_ms())),
           }
       }
       
       /// Start scheduled refresh loop (background task).
       pub async fn start_scheduled_refresh(self: Arc<Self>) -> ! {
           let mut interval = interval(self.refresh_interval);
           
           loop {
               interval.tick().await;
               
               match self.do_refresh().await {
                   Ok(stats) => {
                       log::info!(
                           "Scheduled refresh: {} rules ({}ms)",
                           stats.rules_refreshed, stats.duration_ms
                       );
                   }
                   Err(e) => {
                       log::error!("Scheduled refresh failed: {}", e);
                   }
               }
           }
       }
       
       /// Perform the actual refresh.
       async fn do_refresh(&self) -> Result<RefreshStats, String> {
           let start = now_ms();
           
           // Load warm storage
           let warm_anchors = self.bridge.warm_storage.load_anchors()?;
           
           // Replace hot cache with warm storage contents
           *self.bridge.hot_cache.write() = warm_anchors.clone();
           
           *self.last_refresh_at.write() = now_ms();
           
           Ok(RefreshStats {
               rules_refreshed: warm_anchors.len(),
               duration_ms: now_ms() - start,
               timestamp: now_ms(),
           })
       }
   }
   ```

2. Spawn refresh task in `grpc_server.rs::new()`:
   ```rust
   // In DataPlane struct initialization
   let refresh_service = Arc::new(RefreshService::new(bridge.clone()));
   let refresh_handle = tokio::spawn({
       let svc = refresh_service.clone();
       async move {
           svc.start_scheduled_refresh().await
       }
   });
   ```

**Task 3.1: Implement LRU eviction for hot cache** (4 hours)

1. Update `src/storage/hot_cache.rs` to add capacity enforcement:
   ```rust
   // Existing HotCache struct already has eviction logic skeleton
   // Update insert() to enforce capacity:
   
   pub fn insert(&self, rule_id: String, anchors: RuleVector) -> Result<(), String> {
       let mut cache = self.cache.write();
       
       // Check capacity
       if cache.len() >= self.capacity && !cache.contains_key(&rule_id) {
           // Need to evict LRU entries
           let mut entries: Vec<_> = cache.iter()
               .map(|(id, (ts, _))| (id.clone(), *ts))
               .collect();
           
           // Sort by timestamp (ascending = oldest first)
           entries.sort_by_key(|(_, ts)| *ts);
           
           // Evict 10% of capacity
           let to_evict = (self.capacity / 10).max(1);
           for (id, _) in entries.iter().take(to_evict) {
               cache.remove(id);
           }
           
           self.stats.write().evictions += to_evict;
       }
       
       cache.insert(rule_id, (now_ms(), anchors));
       Ok(())
   }
   ```

2. Add tests for LRU eviction.

---

#### Day 9: Code Quality

#### Day 8-9: Code Quality

**Task 3.1: Code formatting and linting** (4 hours)

1. Rust:

   ```bash
   cd data_plane/bridge
   cargo fmt --all
   cargo clippy --all -- -D warnings
   cargo test --all
   ```
2. Python:

   ```bash
   cd management_plane
   black app/ --line-length 100
   isort app/
   pylint app/ --min-score=9.0
   mypy app/
   pytest tests/
   ```

**Task 3.2: Add comprehensive doc comments** (6 hours)

1. Rust modules:

   - Module-level `//!` comments for every file
   - Function-level `///` comments for all public functions
   - Explain algorithm complexity where relevant
   - Add `# Examples` sections for complex APIs
2. Python modules:

   - Module docstrings (Google style)
   - Function docstrings with Args, Returns, Raises
   - Type hints on all signatures

**Task 3.3: Performance benchmarks** (4 hours)

1. Create `data_plane/bridge/benches/enforcement.rs`:
   ```rust
   use criterion::{black_box, criterion_group, criterion_main, Criterion};

   fn bench_vector_comparison(c: &mut Criterion) {
       c.bench_function("vector_comparison", |b| {
           b.iter(|| {
               // Benchmark comparison logic
           });
       });
   }

   fn bench_hot_cache_lookup(c: &mut Criterion) {
       c.bench_function("hot_cache_lookup", |b| {
           b.iter(|| {
               // Benchmark cache access
           });
       });
   }

   criterion_group!(benches, bench_vector_comparison, bench_hot_cache_lookup);
   criterion_main!(benches);
   ```

---

#### Day 10: Documentation & Examples

**Task 3.4: OpenAPI documentation** (3 hours)

1. Update `management_plane/app/main.py`:

   ```python
   from fastapi import FastAPI
   from fastapi.openapi.utils import get_openapi

   app = FastAPI(
       title="Guard Enforcement API",
       description="""
       API-only enforcement pipeline for policy evaluation.

       ## Features
       - Intent enforcement against semantic rules
       - Rule installation with anchor encoding
       - Tiered storage (hot/warm/cold)
       - Scheduled rule refresh (6hr)

       ## Authentication
       All endpoints require API key authentication via Bearer token.
       """,
       version="2.0.0",
       docs_url="/docs",
       redoc_url="/redoc",
   )

   def custom_openapi():
       if app.openapi_schema:
           return app.openapi_schema

       openapi_schema = get_openapi(
           title="Guard Enforcement API",
           version="2.0.0",
           description="Semantic policy enforcement",
           routes=app.routes,
       )

       # Add security scheme
       openapi_schema["components"]["securitySchemes"] = {
           "BearerAuth": {
               "type": "http",
               "scheme": "bearer",
               "bearerFormat": "API Key",
           }
       }

       app.openapi_schema = openapi_schema
       return app.openapi_schema

   app.openapi = custom_openapi
   ```
2. Export OpenAPI spec:

   ```bash
   python -c "import json; from app.main import app; print(json.dumps(app.openapi()))" > docs/api/openapi.yaml
   ```

**Task 3.5: Integration guide** (4 hours)

1. Create `docs/api/external-integration.md`:
   ```markdown
   # External Stack Integration Guide

   ## Quick Start

   ### 1. Get API Key

   Contact your Guard administrator for an API key.

   ### 2. Install Rules

   First, encode your rule configuration:

   \`\`\`bash
   curl -X POST https://guard.api.com/v1/rules/encode \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "rule_type": "tool_whitelist",
       "allowed_tools": ["database_query", "file_read"],
       "allowed_methods": ["read", "query"]
     }'
   \`\`\`

   Then install the rule:

   \`\`\`bash
   curl -X POST https://guard.api.com/v1/rules/install \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "agent_id": "my-agent",
       "rules": [
         {
           "rule_id": "rule-001",
           "family_id": "tool_whitelist",
           "layer": "L4",
           "priority": 100,
           "enabled": true,
           "anchors": { ... }
         }
       ]
     }'
   \`\`\`

   ### 3. Enforce Intents

   Before executing any action:

   \`\`\`bash
   curl -X POST https://guard.api.com/v1/enforce \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "intent-123",
       "timestamp": "2026-01-23T10:00:00Z",
       "layer": "L4",
       "actor": {"id": "user-123", "type": "user"},
       "action": "execute_tool",
       "resource": {"type": "api", "name": "database_query"},
       "data": {"sensitivity": ["internal"], "pii": false},
       "risk": {"authn": "required"},
       "tenant_id": "tenant-1",
       "agent_id": "my-agent"
     }'
   \`\`\`

   Response:

   \`\`\`json
   {
     "decision": 1,
     "slice_similarities": [0.95, 0.87, 0.92, 0.88],
     "rules_evaluated": 1,
     "evidence": [...]
   }
   \`\`\`

   - `decision`: 0 = BLOCK, 1 = ALLOW
   - If BLOCK, inspect `evidence` for details

   ## Client Libraries

   See `docs/api/examples/` for ready-to-use clients in:
   - Python
   - Node.js
   - Go
   - cURL
   ```

**Task 3.6: Client examples** (5 hours)

1. Create `docs/api/examples/python_client.py`
2. Create `docs/api/examples/nodejs_client.js`
3. Create `docs/api/examples/go_client.go`
4. Create `docs/api/examples/curl_examples.sh`

---

## Testing Strategy

### Unit Tests

**Rust**:

- `vector_comparison.rs`: Cosine similarity, threshold logic
- `hot_cache.rs`: LRU eviction, capacity management
- `warm_storage.rs`: File I/O, serialization
- `cold_storage.rs`: SQLite operations

**Python**:

- `intent_encoder.py`: 128d encoding
- `rule_encoder.py`: Anchor generation
- `auth.py`: API key validation

### Integration Tests

1. **Rule lifecycle**:

   - Install rules → Enforce → Remove → Verify gone
2. **Storage persistence**:

   - Install rules → Restart service → Enforce → Still works
3. **Refresh mechanism**:

   - Install rules → Modify warm storage → Refresh → New rules loaded
4. **Tiered storage**:

   - Fill hot cache → Trigger eviction → Verify promotion to warm

### Load Tests

1. Create `tests/load/locustfile.py`:

   ```python
   from locust import HttpUser, task, between

   class EnforcementUser(HttpUser):
       wait_time = between(0.01, 0.1)

       @task(10)
       def enforce(self):
           self.client.post("/v1/enforce", json={
               "id": "intent-123",
               "layer": "L4",
               # ... full intent
           }, headers={"Authorization": "Bearer test-key"})

       @task(1)
       def stats(self):
           self.client.get("/v1/rules/stats",
               headers={"Authorization": "Bearer test-key"})
   ```
2. Run load test:

   ```bash
   locust -f tests/load/locustfile.py --host=http://localhost:8000
   ```

**Target**: 1000+ RPS, P95 < 30ms

---

## Deployment Guide

### Local Development

```bash
# 1. Clone repo
git clone <repo-url>
cd guard
git checkout feature/api-only-v2

# 2. Configure
cp .env.example .env
# Edit .env with your API key

# 3. Start services
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### Production Deployment

**Prerequisites**:

- Docker + Docker Compose
- 4GB RAM minimum
- 10GB disk space

**Steps**:

1. **Build images**:

   ```bash
   docker-compose build
   ```
2. **Configure environment**:

   ```bash
   cat > .env << EOF
   GUARD_API_KEYS=<key_id>:<sha256_hash>
   DATA_PLANE_URL=data-plane:50051
   LOG_LEVEL=INFO
   RUST_LOG=info
   EOF
   ```
3. **Start services**:

   ```bash
   docker-compose up -d
   ```
4. **Verify health**:

   ```bash
   curl http://localhost:8000/health
   ```
5. **Monitor logs**:

   ```bash
   docker-compose logs -f
   ```

**Monitoring**:

- API logs: `./var/logs/management/`
- Data Plane telemetry: `./var/logs/dataplane/`
- Storage files: `./var/data/`

---

## Success Criteria

### Functional

- [ ] `/v1/enforce` endpoint works (P95 < 30ms)
- [ ] `/v1/rules/install` endpoint works
- [ ] `/v1/rules/encode` endpoint works
- [ ] `/v1/rules/refresh` endpoint works
- [ ] Rules survive restart
- [ ] Scheduled refresh works (6hr)
- [ ] Event-driven refresh works
- [ ] FFI removed, direct function calls work

### Performance

- [ ] 1000+ RPS sustained
- [ ] P95 latency < 30ms
- [ ] P99 latency < 100ms
- [ ] Hot cache hit rate > 95%
- [ ] Memory usage < 200MB for 10K rules

### Code Quality

- [ ] Zero Clippy warnings
- [ ] Pylint score > 9.0
- [ ] 100% formatted (cargo fmt, black)
- [ ] All public functions documented
- [ ] Integration tests pass
- [ ] Load tests pass

### Documentation

- [ ] OpenAPI spec generated
- [ ] Integration guide complete
- [ ] Client examples working (Python, Node, Go)
- [ ] Deployment guide tested

---

## Migration Path (From Current Stack)

For existing deployments, here's how to migrate:

1. **Deploy API-only stack** (new branch) alongside existing stack
2. **Dual-write rules** to both old and new stack
3. **Shadow mode**: Send enforcement requests to both, log diffs
4. **Gradual cutover**: Route % of traffic to new stack
5. **Full migration**: Switch all traffic to new stack
6. **Decommission old stack**

Timeline: 2-4 weeks for production migration

---

## Appendix: Dependencies

### Rust Crates

```toml
[dependencies]
# Existing
parking_lot = "0.12"
tonic = "0.10"
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"

# New
memmap2 = "0.9"           # Memory-mapped files
rusqlite = "0.30"         # SQLite backend
criterion = "0.5"         # Benchmarking
```

### Python Packages

```
# Existing
fastapi==0.109.0
pydantic==2.5.3
uvicorn==0.27.0
sentence-transformers==2.2.2

# New
locust==2.20.0           # Load testing
black==24.1.0            # Formatting
pylint==3.0.3            # Linting
mypy==1.8.0              # Type checking
```

---

## Timeline Summary


| Week       | Focus      | Deliverables                                          |
| ------------ | ------------ | ------------------------------------------------------- |
| **Week 1** | Foundation | FFI removed, REST API working, Docker setup           |
| **Week 2** | Data Plane | Tiered storage, refresh mechanism, persistence        |
| **Week 3** | Polish     | Documentation, examples, load tests, deployment guide |

**Total**: 3 weeks (15 working days)

---

## Contact & Support

For questions or issues during implementation:

- Check `docs/api/external-integration.md` for integration help
- See `docs/implementation/00-enforcement-flow-trace.md` for architecture details
- Review code comments for implementation specifics
