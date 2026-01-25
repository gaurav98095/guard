# V2 Policy Installation - Quick Start Guide

**Document**: `docs/implementation/09-v2-policy-installation-plan.md` (707 lines)  
**Estimated Duration**: 6-10 hours  
**Status**: Ready for Implementation  

---

## Quick Overview

Implement the missing `/api/v2/policies/install` endpoint to complete the v2 enforcement pipeline with full tiered storage integration.

### What's Missing? (3 Critical Pieces)

| Component | Where | What to Do |
|-----------|-------|-----------|
| **DesignBoundaryConverter** | Create NEW file | Convert Python DesignBoundary → Rust RuleInstance |
| **DataPlaneClient.install_policies()** | `dataplane_client.py:160` | Add gRPC wrapper method |
| **V2 Install Endpoint** | `enforcement_v2.py:448` | Remove 501, implement flow |

### What's Already Built? (8 Components ✅)

- ✅ Canonicalizer (BERT) - `canonicalizer.py:410 lines`
- ✅ PolicyEncoder - `policy_encoder.py:300 lines`
- ✅ Tiered Storage - `bridge/src/storage/` (hot/warm/cold)
- ✅ gRPC InstallRules - `grpc_server.rs:269-425`
- ✅ RuleConverter - Rust implementation
- ✅ Bridge Structure - `bridge.rs:add_rule_with_anchors()`
- ✅ V2 Enforce Endpoint - `enforcement_v2.py:252`
- ✅ V2 Canonicalize Endpoint - `enforcement_v2.py:385`

---

## 4 Implementation Phases

### Phase 1: DataPlaneClient Enhancement (1-2 hours)
**What**: Add `install_policies()` method to Python gRPC client  
**File**: `management_plane/app/services/dataplane_client.py:160-200` (NEW)  
**Deliverable**: Method accepts policies + vectors, returns installation stats

**Key Steps**:
1. Create method signature: `install_policies(boundaries, rule_vectors)`
2. Convert numpy arrays → RuleAnchorsPayload protobuf
3. Build InstallRulesRequest
4. Call gRPC `stub.InstallRules()`
5. Handle errors (gRPC failures)

---

### Phase 2: Policy Converter Implementation (2-3 hours)
**What**: Convert DesignBoundary objects to RuleInstance format  
**File**: `management_plane/app/services/policy_converter.py` (NEW, ~150 lines)  
**Deliverable**: Complete conversion pipeline

**Key Methods**:
1. `boundary_to_rule_instance()` - Maps all fields
2. `constraints_to_params()` - Serializes constraints to JSON
3. `rule_vector_to_anchor_payload()` - Converts numpy → protobuf

**Example Conversion**:
```
LooseDesignBoundary
  ↓ (canonicalize)
DesignBoundary (canonical)
  ↓ (encode)
RuleVector (4×16×32 numpy array)
  ↓ (convert)
RuleInstance dict (protobuf-ready)
  ↓ (gRPC send)
Data Plane → Tiered Storage
```

---

### Phase 3: V2 Endpoint Implementation (1-2 hours)
**What**: Complete `/api/v2/policies/install` endpoint  
**File**: `management_plane/app/endpoints/enforcement_v2.py:448-550`  
**Deliverable**: Functional REST API endpoint

**Changes**:
1. **Line 475-478**: DELETE the 501 error
2. **Implement flow**:
   ```python
   1. Canonicalize LooseDesignBoundary
   2. Encode to RuleVector
   3. Convert via PolicyConverter
   4. Call client.install_policies()
   5. Return installation response
   ```

**Request/Response**:
- **IN**: LooseDesignBoundary (non-canonical)
- **OUT**: Installation status + canonicalization trace

---

### Phase 4: Tiered Storage Integration (2-3 hours)
**What**: Verify L4 rules flow through all storage tiers  
**Files**: `bridge.rs`, `enforcement_engine.rs`, `grpc_server.rs`  
**Deliverable**: Verified end-to-end storage pipeline

**Changes**:
1. **Bridge**: Distinguish L4 rules from L0-L3, route to tiered storage
2. **Enforcement**: Query tiered storage for L4 rules
3. **Verification**: Check hot/warm/cold storage after installation

**Storage Layout**:
```
Hot Cache (memory)
  ↓ (eviction or sync)
Warm Storage (mmap file)
  ↓ (overflow)
Cold Storage (SQLite)
```

---

## File Checklist

### Create (3 new files):
- [ ] `management_plane/app/services/policy_converter.py`

### Modify (5 files):
- [ ] `management_plane/app/services/dataplane_client.py` (add method)
- [ ] `management_plane/app/services/__init__.py` (add import)
- [ ] `management_plane/app/endpoints/enforcement_v2.py` (remove 501, implement)
- [ ] `data_plane/tupl_dp/bridge/src/bridge.rs` (L4 rule handling)
- [ ] `data_plane/tupl_dp/bridge/src/enforcement_engine.rs` (L4 rule lookup)

---

## Data Structures Reference

### Input: LooseDesignBoundary
```python
{
  "id": "boundary-001",
  "name": "Allow Database Reads",
  "status": "active",
  "type": "mandatory",
  "boundarySchemaVersion": "v1.2",
  "scope": {"tenantId": "tenant-123"},
  "layer": null,  # Optional: if null, rule is globally applicable
  "rules": {
    "effect": "allow",
    "thresholds": {"action": 0.85, "resource": 0.80, ...},
    "decision": "min"
  },
  "constraints": {
    "action": {"actions": ["read"], "actor_types": ["user"]},
    "resource": {"types": ["database"], ...},
    "data": {"sensitivity": ["internal"], ...},
    "risk": {"authn": "required"}
  },
  "createdAt": 1704980000.0,
  "updatedAt": 1704980000.0
}
```

### Intermediate: DesignBoundary (canonical)
```python
# Same structure but with canonicalized values:
{
  "constraints": {
    "action": {"actions": ["read"], "actor_types": ["user"]},  # canonical
    "resource": {"types": ["database"], ...},  # canonical
    "data": {"sensitivity": ["internal"], ...},  # canonical
    ...
  }
}
```

### Output: RuleInstance (protobuf dict)
```python
{
  "rule_id": "boundary-001",
  "layer": null,  # Optional: from request.layer (or None if not specified)
  "agent_id": "tenant-123",
  "priority": 100,  # mandatory=100, optional=50
  "enabled": true,
  "created_at_ms": 1704980000000,
  "params": {
    "boundary_id": "boundary-001",
    "rule_effect": "allow",
    "thresholds": "{...}",  # JSON
    "constraints": "{...}",  # JSON
    ...
  },
  "anchors": {
    "action_anchors": [[...], [...], ...],  # 16 × 32
    "action_count": 4,
    "resource_anchors": [[...], ...],
    "resource_count": 8,
    ...
  }
}
```

---

## Integration Points

```
Management Plane (Python)
│
├─→ Phase 1: DataPlaneClient.install_policies()
│   └─→ gRPC call to Data Plane
│
├─→ Phase 2: PolicyConverter
│   └─→ DesignBoundary → RuleInstance (no family_id)
│
├─→ Phase 3: V2 Endpoint
│   └─→ Orchestrates canonicalization + encoding + conversion
│
└─→ Phase 4: Tiered Storage (Data Plane side)
    └─→ Hot → Warm → Cold persistence
    
(DEPRECATED: RuleFamilyTable removed in Phase 5)
```

---

## Key Design Decisions

### ✅ Tiered Storage Only (Recommended)
- **ALL Rules**: Tiered storage exclusively
- **Deprecate**: RuleFamilyTable and layer hierarchy
- **Rationale**:
  - Single source of truth (no dual-write complexity)
  - RuleFamilyTable has 0 active rules (nothing to migrate)
  - Layer is metadata, not architecture
  - Eliminates 2500+ lines of unused infrastructure

### ✅ Layer as Optional Filtering Metadata
- Layer is OPTIONAL - extracted from request if present
- If present: used during enforcement to filter rules
- If absent (None): rule is globally applicable, always evaluated
- Layer is NOT a type classification, just filtering scope

### ✅ Simple Rule Model (No family_id)
- Tiered storage uses rule_id as primary key
- No family classification needed
- No semantic type distinctions (all are policies)
- Request structure (LooseDesignBoundary) is API concern, not rule property

---

## Risk Mitigation Summary

| Risk | Mitigation |
|------|-----------|
| **Low-confidence canonicalization** | Log predictions, monitor, reject if <0.5 confidence |
| **Rules evicted from hot cache** | Pin L4 rules (no LRU eviction) |
| **Vector encoding drift** | Version format, test round-trip serialization |
| **Protobuf mismatch** | Verify versions, test serialization |
| **Family table removal breaks enforcement** | Update enforcement query logic before removal |

---

## Expected Outcomes After Implementation

### API
```bash
POST /api/v2/policies/install HTTP/1.1
Content-Type: application/json
X-Tenant-Id: tenant-123

{
  "id": "boundary-001",
  "name": "...",
  ...
}

← HTTP 201 Created
{
  "status": "installed",
  "boundary_id": "boundary-001",
  "canonicalization_trace": [...],
  "installation_stats": {
    "rules_installed": 1,
    "rules_by_layer": {"L4": 1},
    "bridge_version": 42
  }
}
```

### Storage
```
./var/data/warm_storage.bin  ← 1+ rules stored
./var/data/cold_storage.db   ← Backup copy
```

### Enforcement
```bash
POST /api/v2/enforce HTTP/1.1
...
← HTTP 200 OK
{
  "decision": "ALLOW",
  "evidence": [
    {
      "boundary_id": "boundary-001",
      "effect": "allow",
      "decision": 1,
      "similarities": [0.92, 0.88, 0.85, 0.90]
    }
  ]
}
```

---

## References

- **Full Plan**: `docs/implementation/09-v2-policy-installation-plan.md`
- **Canonicalization**: `docs/implementation/02-canonicalization-plan.md`
- **Tiered Storage**: `docs/implementation/10-tiered-storage-bridge-integration.md`
- **Protobuf Schema**: `data_plane/proto/rule_installation.proto`
- **Bridge Code**: `data_plane/tupl_dp/bridge/src/bridge.rs`

---

**Ready to start? Begin with Phase 1!**
