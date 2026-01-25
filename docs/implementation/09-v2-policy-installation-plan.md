# Phase 3: V2 Policy Installation with Tiered Storage Integration

**Status**: Ready for Implementation  
**Date**: January 25, 2026  
**Estimated Duration**: 6-10 hours  
**Dependency**: All previous phases complete  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Implementation Phases](#implementation-phases)
4. [Technical Details by Phase](#technical-details-by-phase)
5. [Integration Points](#integration-points)
6. [Risks & Mitigations](#risks--mitigations)

---

## Executive Summary

### Objective

Complete the v2 enforcement pipeline by implementing policy installation with full tiered storage integration. This enables policies (via LooseDesignBoundary API) to be installed as rules in the Data Plane with optional layer filtering metadata, leveraging the three-tier storage system (hot/warm/cold) already implemented in Phase 2.

### Scope

- Implement missing `DataPlaneClient.install_policies()` method (Python)
- Create `DesignBoundaryConverter` to map Python policies to Rust rule instances
- Enable and test `/api/v2/policies/install` endpoint
- Verify end-to-end flow: Policy → Canonicalization → Encoding → Installation → Storage

### Success Criteria

1. ✅ `/api/v2/policies/install` accepts LooseDesignBoundary + installed rules
2. ✅ Rules stored across all 3 tiers (hot → warm → cold)
3. ✅ Enforcement can read installed policies and evaluate intents
4. ✅ Storage stats show rules distributed across tiers
5. ✅ E2E test: Install → Enforce → Get evidence

### Decision: Storage Architecture & Layer Model

**Tiered Storage Only** (Recommended):
- **ALL Rules**: Tiered storage exclusively
- **Deprecate**: RuleFamilyTable and layer hierarchy
- **Layer**: Optional filtering metadata (not type classification)
- **Policy Model**: All policies are rules, optionally scoped by layer

**Rationale**:
- ✅ Single source of truth (eliminates dual-write complexity)
- ✅ RuleFamilyTable has 0 active rules (nothing to migrate)
- ✅ Layer is metadata for enforcement filtering, not architecture
- ✅ Simplifies rule model (one rule type, optional layer scoping)
- ✅ Reduces code surface: 2500+ lines of unused infrastructure

---

## Current State Analysis

### ✅ Already Implemented (Phases 1-2)

| Component | Location | Status |
|-----------|----------|--------|
| **Canonicalizer (BERT)** | `management_plane/app/services/canonicalizer.py` | ✅ 410 lines, working |
| **PolicyEncoder** | `management_plane/app/services/policy_encoder.py:102-300` | ✅ Converts DesignBoundary → 4×16×32 RuleVector |
| **SemanticEncoder** | `management_plane/app/services/semantic_encoder.py` | ✅ Base class for encoding |
| **Tiered Storage** | `data_plane/tupl_dp/bridge/src/storage/` | ✅ Hot/warm/cold fully implemented |
| **Bridge Structure** | `data_plane/tupl_dp/bridge/src/bridge.rs:89-300` | ✅ add_rule_with_anchors() exists |
| **gRPC InstallRules** | `data_plane/tupl_dp/bridge/src/grpc_server.rs:269-425` | ✅ Endpoint functional |
| **RuleConverter** | `data_plane/tupl_dp/bridge/src/rule_converter.rs` | ✅ All 14 families supported |
| **V2 Enforce Endpoint** | `management_plane/app/endpoints/enforcement_v2.py:252` | ✅ Working |
| **V2 Canonicalize Endpoint** | `management_plane/app/endpoints/enforcement_v2.py:385` | ✅ Working |

### ❌ Missing (Critical Path)

| Component | Location | Gap | Impact |
|-----------|----------|-----|--------|
| **DesignBoundaryConverter** | Does not exist | No Python→Rust conversion | 🔴 Blocks installation |
| **DataPlaneClient.install_policies()** | `dataplane_client.py:53-150` | Method not implemented | 🔴 Blocks endpoint |
| **V2 Install Endpoint** | `enforcement_v2.py:448` | Returns 501 error | 🔴 API unusable |

---

## Implementation Phases

### Phase 1: DataPlaneClient Enhancement (1-2 hours)
**Objective**: Enable gRPC communication for policy installation  
**Output**: `install_policies()` method in DataPlaneClient

**Files Modified**:
1. `management_plane/app/services/dataplane_client.py`

**Key Tasks**:
- [ ] Add `install_policies()` method to wrap gRPC InstallRules call
- [ ] Handle RuleInstance protobuf serialization
- [ ] Handle RuleAnchorsPayload conversion from numpy arrays
- [ ] Proper error handling for gRPC failures
- [ ] Return installation response with stats

---

### Phase 2: Policy Converter Implementation (2-3 hours)
**Objective**: Convert DesignBoundary objects to installable RuleInstance format  
**Output**: Complete policy-to-rule conversion pipeline

**Files Created**:
1. `management_plane/app/services/policy_converter.py` (NEW)

**Files Modified**:
1. `management_plane/app/services/__init__.py` (add import)

**Key Components**:

#### 2.1 PolicyConverter Class
```python
class PolicyConverter:
    """Convert canonical DesignBoundary to gRPC RuleInstance format."""
    
    @staticmethod
    def boundary_to_rule_instance(
        boundary: DesignBoundary,
        rule_vector: RuleVector,
        tenant_id: str,
    ) -> dict:
        """
        Maps DesignBoundary fields to RuleInstance protobuf structure:
        
        - boundary.id → rule_id
        - boundary.layer (if present) → layer (optional filtering metadata)
        - tenant_id → agent_id (tenant scoping)
        - boundary.type → priority (mandatory=100, optional=50)
        - boundary.status → enabled flag
        - boundary constraints + rules → params (serialized JSON)
        - rule_vector → RuleAnchorsPayload (4 slots × 16 anchors × 32 dims)
        
        NOTE: 
        - No family_id field needed - tiered storage uses rule_id as primary key
        - layer is OPTIONAL - if not specified, rule is globally applicable
        - Layer is metadata for enforcement filtering, not a type classification
        """
```

#### 2.2 Parameters Serialization
DesignBoundary fields serialized as JSON params for Rust access:
```python
params = {
    "boundary_id": boundary.id,
    "boundary_name": boundary.name,
    "boundary_type": boundary.type,  # "mandatory" | "optional"
    "rule_effect": boundary.rules.effect,  # "allow" | "deny"
    "rule_decision": boundary.rules.decision,  # "min" | "weighted-avg"
    "thresholds": json.dumps({
        "action": boundary.rules.thresholds.action,
        "resource": boundary.rules.thresholds.resource,
        "data": boundary.rules.thresholds.data,
        "risk": boundary.rules.thresholds.risk,
    }),
    "constraints": json.dumps(boundary.constraints.dict()),
}
```

#### 2.3 RuleVector → RuleAnchorsPayload Conversion
Map encoded numpy arrays to protobuf structure:
```python
@staticmethod
def rule_vector_to_proto(rule_vector: np.ndarray) -> RuleAnchorsPayload:
    """
    Convert 4×16×32 RuleVector to protobuf RuleAnchorsPayload.
    
    Input: rule_vector shape (4, 16, 32) from PolicyEncoder
    Output: RuleAnchorsPayload with 4 anchor blocks
    """
```

---

### Phase 3: V2 Endpoint Implementation (1-2 hours)
**Objective**: Complete `/api/v2/policies/install` endpoint  
**Output**: Functional policy installation API

**Files Modified**:
1. `management_plane/app/endpoints/enforcement_v2.py` (lines 448-550)

**Key Changes**:

#### 3.1 Remove 501 Error (Line 475-478)
```python
# DELETE these lines:
raise HTTPException(
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    detail="Policy installation not implemented yet",
)
```

#### 3.2 Implement Full Flow
```python
@router.post("/policies/install", status_code=status.HTTP_201_CREATED)
async def install_policies_v2(
    boundary: LooseDesignBoundary,
    current_user: User = Depends(get_current_tenant),
) -> dict:
    """
    Install policy with automatic canonicalization.
    
    Flow:
    1. Canonicalize LooseDesignBoundary → canonical DesignBoundary
    2. Encode canonical boundary → PolicyEncoder → 4×16×32 RuleVector
    3. Convert to RuleInstance format via PolicyConverter
    4. Install via DataPlaneClient.install_policies()
    5. Log canonicalization trace
    6. Return installation status
    """
    
    # 1. Get services
    canonicalizer = get_canonicalizer()
    policy_encoder = get_policy_encoder()
    client = get_data_plane_client()
    
    # 2. Canonicalize
    canonicalized = canonicalizer.canonicalize_boundary(boundary)
    canonical_boundary = canonicalized.canonical_boundary
    
    # 3. Encode
    rule_vector = policy_encoder.encode(canonical_boundary)
    
    # 4. Convert & Install
    rule_instance = PolicyConverter.boundary_to_rule_instance(
        canonical_boundary,
        rule_vector,
        current_user.id,
    )
    
    # 5. Call Data Plane
    result = await asyncio.to_thread(
        client.install_policies,
        [canonical_boundary],
        [rule_vector.tolist()],
    )
    
    # 6. Return response
    return {
        "status": "installed",
        "boundary_id": boundary.id,
        "request_id": str(uuid.uuid4()),
        "canonicalization_trace": canonicalized.to_trace_dict(),
        "installation_stats": result,
    }
```

#### 3.3 Error Handling
```python
try:
    # ... implementation ...
except HTTPException:
    raise
except DataPlaneError as e:
    raise HTTPException(status_code=502, detail=f"Data Plane error: {e}")
except Exception as e:
    logger.error(f"Policy installation failed: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Installation failed")
```

---

### Phase 4: Tiered Storage Integration (1-2 hours)
**Objective**: Verify policies are properly stored and retrieved from all tiers  
**Output**: Verified end-to-end storage and retrieval

**Files Modified**:
1. `data_plane/tupl_dp/bridge/src/bridge.rs` (simplify add_rule_with_anchors)
2. `data_plane/tupl_dp/bridge/src/enforcement_engine.rs` (query tiered storage directly)

**Key Tasks**:

#### 4.1 Bridge: Simplify add_rule_with_anchors
**File**: `bridge.rs:200-250`

```rust
pub fn add_rule_with_anchors(
    &self,
    rule: Arc<dyn RuleInstance>,
    anchors: RuleVector,
) -> Result<(), String> {
    let rule_id = rule.rule_id().to_string();
    
    // ALL rules use tiered storage (no RuleFamilyTable writes)
    // 1. Insert into hot cache
    let cached_rule = CachedRule::new(
        rule.clone(),
        anchors.clone(),
    );
    self.hot_cache.insert(rule_id.clone(), cached_rule)?;
    
    // 2. Persist to warm storage
    self.warm_storage.insert(rule_id.clone(), &rule, &anchors)?;
    
    // 3. Update version
    *self.active_version.write() += 1;
    
    Ok(())
}
```

#### 4.2 Enforcement Engine: Query Tiered Storage Only
**File**: `enforcement_engine.rs:150-200`

```rust
fn get_rules_for_layer(&self, layer: Option<&str>) -> Result<Vec<Arc<dyn RuleInstance>>, String> {
    // Query tiered storage directly (hot cache as primary)
    let all_rules = self.bridge.hot_cache.get_all_rules()?;
    
    // Filter by layer (if specified) + include global rules (layer=None)
    let mut filtered: Vec<_> = all_rules.into_iter()
        .filter(|r| {
            match (r.layer(), layer) {
                (None, _) => true,           // Global rules always included
                (Some(rule_layer), Some(requested_layer)) => rule_layer == requested_layer,
                (Some(_), None) => false,    // Skip layer-specific rules when no layer requested
            }
        })
        .collect();
    
    // Sort by priority (higher priority first)
    filtered.sort_by(|a, b| b.priority().cmp(&a.priority()));
    
    Ok(filtered)
}
```

#### 4.3 Verify Storage Persistence
**Tasks**:
- [ ] Verify warm storage file created: `./var/data/warm_storage.bin`
- [ ] Verify cold storage DB created: `./var/data/cold_storage.db`
- [ ] Check hot cache has installed rules
- [ ] Verify storage_stats() reflects rule distribution
- [ ] Confirm enforcement queries hot cache (no RuleFamilyTable access)

---

## Technical Details by Phase

### Phase 1 Implementation Details

#### DataPlaneClient.install_policies()

**Location**: `management_plane/app/services/dataplane_client.py:160-200` (NEW)

**Signature**:
```python
def install_policies(
    self,
    boundaries: List[DesignBoundary],
    rule_vectors: List[List[float]],
) -> dict:
```

**Implementation Steps**:

1. **Validate inputs**:
   ```python
   if not boundaries or len(boundaries) != len(rule_vectors):
       raise ValueError("Boundaries and rule_vectors must have same length")
   ```

2. **Convert to RuleInstance protobuf**:
   - Use PolicyConverter for each boundary
   - Collect into list of RuleInstance messages

3. **Build InstallRulesRequest**:
   ```python
   request = InstallRulesRequest(
       agent_id=boundaries[0].scope.tenantId,
       rules=[rule_instance_1, rule_instance_2, ...],
       config_id="design_boundary_v2",
       owner="management_plane",
   )
   ```

4. **Call gRPC**:
   ```python
   try:
       response = self.stub.InstallRules(
           request,
           timeout=self.timeout,
           metadata=metadata,
       )
       return {
           "success": response.success,
           "message": response.message,
           "rules_installed": response.rules_installed,
           "rules_by_layer": dict(response.rules_by_layer),
           "bridge_version": response.bridge_version,
       }
   except grpc.RpcError as e:
       raise DataPlaneError(f"InstallRules failed: {e.details()}", e.code())
   ```

---

### Phase 2 Implementation Details

#### PolicyConverter Structure

**Location**: `management_plane/app/services/policy_converter.py` (NEW, ~150 lines)

**Key Methods**:

1. **`boundary_to_rule_instance(boundary, rule_vector, tenant_id)`**
   - Input: DesignBoundary (canonical), RuleVector (numpy 4×16×32), tenant_id
   - Output: dict matching RuleInstance protobuf schema
   - Maps all constraint fields to serialized params

2. **`constraints_to_params(constraints)`**
   - Extract action, resource, data, risk from LooseBoundaryConstraints
   - Serialize to JSON for Rust deserialization
   - Include thresholds and weights if present

3. **`rule_vector_to_anchor_payload(rule_vector)`**
   - Convert numpy array to RuleAnchorsPayload protobuf
   - Split into 4 anchor blocks (action, resource, data, risk)
   - Each block: 16 anchors × 32 dims
   - Return properly formatted protobuf dict

**Example Conversion**:
```python
LooseDesignBoundary {
    id: "boundary-001",
    name: "Allow Database Reads",
    type: "mandatory",
    rules: {
        effect: "allow",
        thresholds: {action: 0.85, ...},
    },
    constraints: {
        action: {actions: ["read"], actor_types: ["user"]},
        resource: {types: ["database"], ...},
        ...
    }
}

↓ (conversion)

RuleInstance {
    rule_id: "boundary-001",
    layer: "layer_from_request_or_None",  // Optional: from LooseDesignBoundary.layer
    agent_id: "tenant-123",
    priority: 100,
    enabled: true,
    created_at_ms: 1704980000000,
    params: {
        "boundary_id": "boundary-001",
        "boundary_name": "Allow Database Reads",
        "rule_effect": "allow",
        "thresholds": "{\"action\": 0.85, ...}",
        ...
    },
    anchors: RuleAnchorsPayload {
        action_anchors: [[...], [...], ...],  // 16 × 32
        action_count: 4,
        ...
    }
}
```

---

### Phase 3 Implementation Details

#### Endpoint Implementation

**Location**: `management_plane/app/endpoints/enforcement_v2.py:448-550`

**Request Model** (with optional layer):
```python
class LooseDesignBoundary(BaseModel):
    id: str
    name: str
    status: Literal["active", "disabled"]
    type: Literal["mandatory", "optional"]
    boundarySchemaVersion: Literal["v1.1", "v1.2"]
    scope: BoundaryScope
    rules: BoundaryRules
    constraints: LooseBoundaryConstraints
    notes: Optional[str]
    layer: Optional[str] = None  # Optional: for filtering during enforcement
    createdAt: float
    updatedAt: float
```

**Response Model**:
```python
class InstallPoliciesResponse(BaseModel):
    status: str  # "installed"
    boundary_id: str
    request_id: str
    canonicalization_trace: list[CanonicalizedField]
    installation_stats: dict  # From gRPC response
```

**Error Cases**:
- ❌ 400: Invalid DesignBoundary (missing required fields)
- ❌ 401: Unauthorized (invalid auth)
- ❌ 500: Canonicalization failed
- ❌ 500: Encoding failed
- ❌ 502: Data Plane error (gRPC failure)
- ✅ 201: Installation successful

---

### Phase 4 Implementation Details

#### Bridge L4 Rule Handling

**Current Structure** (bridge.rs:69-87):
```rust
pub struct Bridge {
    tables: HashMap<RuleFamilyId, Arc<RwLock<RuleFamilyTable>>>,
    active_version: Arc<RwLock<u64>>,
    hot_cache: Arc<HotCache>,
    warm_storage: Arc<WarmStorage>,
    cold_storage: Arc<ColdStorage>,
}
```

**Required Changes**:
1. Add method to distinguish L4 rules from L0-L3
2. Route L4 rules directly to tiered storage
3. Keep L0-L3 routing to RuleFamilyTable (backward compatible)

**Storage Layout**:
```
Bridge
└── Tiered Storage (ALL rules)
    ├── Hot: HashMap<rule_id, (rule_metadata, RuleVector)>
    │   ├── Rules with layer=<specific_layer> (for filtering)
    │   ├── Rules with layer=None (global, always evaluated)
    ├── Warm: Mmap file + index
    └── Cold: SQLite

(DEPRECATED: RuleFamilyTable infrastructure removed in Phase 5)
```

#### Query Flow

```
Enforce Intent (with optional layer)
    ↓
Find matching rules
    └─→ Query Tiered Storage (all rules)
        ├─→ Include global rules (layer=None)
        └─→ Include layer-specific rules (if intent specifies layer)
    ↓
Evaluate similarity
    ↓
Return decision + evidence
```

---

## Integration Points

### Data Flow Diagram

```
┌────────────────────────────────┐
│ Client                         │
│ POST /api/v2/policies/install  │
└────────────┬───────────────────┘
             │
             ↓ (LooseDesignBoundary)
┌────────────────────────────────┐
│ Management Plane               │
│ enforcement_v2.install_endpoint│
└────────────┬───────────────────┘
             │
             ├─→ (1) Canonicalizer
             │   Input: LooseDesignBoundary
             │   Output: DesignBoundary (canonical)
             │
             ├─→ (2) PolicyEncoder
             │   Input: DesignBoundary
             │   Output: RuleVector (4×16×32)
             │
             ├─→ (3) PolicyConverter
             │   Input: (DesignBoundary, RuleVector)
             │   Output: RuleInstance dict
             │
             ↓ (gRPC InstallRulesRequest)
┌────────────────────────────────┐
│ Data Plane                     │
│ gRPC InstallRules Handler      │
└────────────┬───────────────────┘
             │
             ├─→ (4) RuleConverter
             │   Input: RuleInstance dict
             │   Output: Arc<dyn RuleInstance>
             │
             ├─→ (5) Bridge.add_rule_with_anchors()
             │   Input: (Arc<RuleInstance>, RuleVector)
             │   Output: () [stored]
             │
             └─→ (6) Tiered Storage
                 Hot Cache → Warm Storage → Cold Storage
```

### Service Dependencies

| Service | Phase | Requires | Provides |
|---------|-------|----------|----------|
| **Canonicalizer** | Phase 1 | BERT model | Canonical DesignBoundary |
| **PolicyEncoder** | Phase 1 | SemanticEncoder | RuleVector (4×16×32) |
| **PolicyConverter** | Phase 2 | - | RuleInstance dict |
| **DataPlaneClient** | Phase 2 | gRPC stubs | InstallRulesRequest |
| **V2 Endpoint** | Phase 3 | All above | REST API |
| **Bridge** | Phase 4 | Tiered Storage | Rule persistence |

---

## Risks & Mitigations

### Risk 1: Canonicalization Produces Low-Confidence Results
**Impact**: Unknown constraints may cause policy mismatches  
**Mitigation**:
- Log all predictions with confidence scores
- Monitor medium/low confidence boundaries
- Alert on boundaries with >10% low-confidence fields
- Implement fallback: reject installation if any field <0.5 confidence

### Risk 2: Tiered Storage Eviction Loses Rules
**Impact**: Critical policies evicted from hot cache, slow to retrieve  
**Mitigation**:
- Pin L4 rules in hot cache (no LRU eviction)
- Verify warm storage persistence on every add_rule_with_anchors()
- Test: Verify rule accessible after cache clear

### Risk 3: RuleVector Encoding Differs Between Phases
**Impact**: Same policy produces different vectors, breaks matching  
**Mitigation**:
- Use identical SemanticEncoder instance across phases
- Version RuleVector format (current: 4×16×32)
- Test: Verify policy encoder output matches gRPC received anchors
- Compare hashes of encoded vectors

### Risk 4: Protobuf Version Mismatch
**Impact**: gRPC serialization fails, rules not installed  
**Mitigation**:
- Verify protobuf version in both management_plane and data_plane
- Test: Round-trip RuleInstance serialization
- Check git history for proto changes

### Risk 5: RuleFamilyTable Removal Breaks Queries
**Impact**: Enforcement engine fails to retrieve rules  
**Mitigation**:
- Update get_rules_for_layer() before removing family tables
- Query hot_cache directly instead of RuleFamilyTable
- Test: Verify all enforcement paths work with tiered storage only
- Keep RuleFamilyTable code until new query logic verified

---

## Architecture Decision: Why Tiered Storage Only?

### Background

The codebase currently has two parallel systems:

1. **RuleFamilyTable** (14 families across L0-L6 layers)
   - ~2500 lines of infrastructure code
   - **0 active rules** in production
   - Used only in infrastructure tests
   - Supports L0-L3 (System, Input, Planner, ModelIO) and L5-L6 (RAG, Egress) rules

2. **Tiered Storage** (Hot/Warm/Cold)
   - ~800 lines of implementation
   - **Already storing all active rules** (v1 ToolWhitelist rules)
   - Single source of truth for rule vectors and metadata
   - Optimized query performance: hot → warm → cold

### Key Finding: v1 Already Bypasses Family Tables

Current v1 installation flow:
```
PolicyRules → RuleInstaller → gRPC InstallRules → Bridge.add_rule_with_anchors()
    ↓
Writes to BOTH:
  - RuleFamilyTable[L4][ToolWhitelist] (unused)
  - Tiered Storage hot_cache (active)
```

Current v1 enforcement flow:
```
Intent → EnforcementEngine.get_rules_for_layer() → RuleFamilyTable.query_all()
    ↓
Queries ONLY:
  - RuleFamilyTable (mostly empty)
  - Falls back to tiered storage for anchors
```

**Result**: Even v1 primarily uses tiered storage; RuleFamilyTable is write-heavy, read-light redundancy.

### Why Tiered Storage Only is Better

**Current Hybrid Complexity**:
- Dual writes on installation (redundancy)
- Dual query paths on enforcement (confusion)
- L0-L3 family table infrastructure unused (dead code)
- Two mental models: "families" vs "layers"
- 2500+ lines of unused code

**Proposed Tiered Storage Single Path**:
- Single write: rule_id → (rule_metadata, RuleVector)
- Single query: hot_cache.get_rules_for_layer(layer)
- Clear mental model: "rules indexed by ID and layer"
- Remove 2500+ lines of unused infrastructure
- L4 ToolWhitelist + L4 DesignBoundary coexist naturally

### No Breaking Changes

**Why this is safe**:
1. **No v1 rules in L0-L3**: Nothing to migrate from family tables
2. **v1 ToolWhitelist already uses tiered storage**: Functionally unchanged
3. **v2 clean slate**: Starts directly on tiered storage
4. **Fully testable**: Can verify enforcement works before removal

### Implementation Timeline

- **Phases 1-3**: Build v2 installation (no family table changes)
- **Phase 4**: Update enforcement to query tiered storage directly
- **Phase 5**: Remove family table infrastructure (2500 lines cleanup)

---

## Pre-Implementation Checklist

- [ ] Review Phase 1-2 code for any breaking changes
- [ ] Verify warm storage file paths are writable: `./var/data/`
- [ ] Confirm protobuf version compatibility between planes
- [ ] Check DataPlaneClient gRPC channel initialization
- [ ] Test existing enforce endpoint to verify base functionality
- [ ] Verify canonicalizer + encoder are initialized in main.py
- [ ] Create test DesignBoundary fixtures for testing

---

## References

- **Canonicalization**: `docs/implementation/02-canonicalization-plan.md`
- **Tiered Storage**: `docs/implementation/10-tiered-storage-bridge-integration.md`
- **gRPC InstallRules**: `data_plane/proto/rule_installation.proto:7-8`
- **Bridge Structure**: `data_plane/tupl_dp/bridge/src/bridge.rs:69-150`
- **RuleConverter**: `data_plane/tupl_dp/bridge/src/rule_converter.rs`
- **PolicyEncoder**: `management_plane/app/services/policy_encoder.py:102-300`

---

## Completion Criteria

**Phase 1 Complete**:
- ✅ `DataPlaneClient.install_policies()` method exists and handles gRPC serialization
- ✅ All RuleInstance fields properly mapped from Python
- ✅ Error handling for gRPC failures
- ✅ No family_id field in RuleInstance (semantic type in params only)

**Phase 2 Complete**:
- ✅ `PolicyConverter` class created with all conversion methods
- ✅ DesignBoundary → RuleInstance mapping verified (no family_id)
- ✅ RuleVector → RuleAnchorsPayload conversion working
- ✅ Semantic type ("design_boundary") in params, not as family

**Phase 3 Complete**:
- ✅ V2 endpoint implementation complete (no 501 error)
- ✅ Full canonicalization + encoding + conversion flow working
- ✅ Response includes canonicalization trace
- ✅ Bypass RuleFamilyTable completely (tiered storage only)

**Phase 4 Complete**:
- ✅ Rules stored in hot cache after installation
- ✅ Warm storage file persists rules
- ✅ Cold storage DB contains rules
- ✅ Enforcement engine queries tiered storage (no family tables)
- ✅ v1 ToolWhitelist and v2 DesignBoundary coexist in hot cache
