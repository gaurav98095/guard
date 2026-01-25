# Layer Model Correction - Documentation Update

**Status**: ✅ COMPLETE  
**Date**: January 25, 2026  
**Scope**: Updated all implementation docs with correct layer model

---

## The Correction

### Before (Incorrect Model)
```
Layer = Type Classification
├── L4 = DesignBoundaries (v2 policies)
├── v1 ToolWhitelist = Another type on L4
└── Need family_id + rule_type to distinguish
```

### After (Correct Model)
```
Layer = Optional Filtering Metadata
├── All rules are just rules
├── Layer (if present) = filter scope during enforcement
├── If layer=None = rule is globally applicable
└── No type classification needed
```

---

## Key Changes Made

### 1. LooseDesignBoundary API Structure

**Added**: Optional `layer` field

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
    layer: Optional[str] = None  # NEW: Optional filtering metadata
    createdAt: float
    updatedAt: float
```

### 2. RuleInstance Format

**Removed**: 
- `family_id` field (no family classification)
- `rule_type` in params (not needed)
- Hardcoded `"L4"` assignment

**Kept**:
- `layer: Optional[str]` extracted from request
- All constraint/rule data in params

**Example**:
```python
RuleInstance {
    rule_id: "boundary-001",
    layer: None,  # Optional, from LooseDesignBoundary.layer
    agent_id: "tenant-123",
    priority: 100,
    enabled: true,
    created_at_ms: 1704980000000,
    params: {
        "boundary_id": "boundary-001",
        "boundary_name": "Allow Database Reads",
        "rule_effect": "allow",
        "thresholds": "{...}",
        "constraints": "{...}",
    },
    anchors: RuleAnchorsPayload { ... }
}
```

### 3. Enforcement Query Logic

**Before** (hard enforcement):
```rust
fn get_rules_for_layer(&self, layer: &str) -> Result<Vec<...>> {
    // Parse layer string to LayerId enum
    // Query RuleFamilyTable by layer
    // Return only rules matching that layer
}
```

**After** (optional filtering):
```rust
fn get_rules_for_layer(&self, layer: Option<&str>) -> Result<Vec<...>> {
    let all_rules = self.bridge.hot_cache.get_all_rules()?;
    
    let filtered = all_rules.into_iter()
        .filter(|r| {
            match (r.layer(), layer) {
                (None, _) => true,  // Global rules always included
                (Some(rule_layer), Some(requested)) => rule_layer == requested,
                (Some(_), None) => false,  // Skip layer-specific if no filter
            }
        })
        .collect();
    
    Ok(filtered)
}
```

### 4. Storage Layout

**Before** (type-based):
```
Hot Cache
├── v1 ToolWhitelist rules (layer=L4)
└── v2 DesignBoundary rules (layer=L4)
```

**After** (metadata-based):
```
Hot Cache
├── Rules with layer="L4" (for filtering)
├── Rules with layer=None (global)
└── Rules with layer="L2" or any other value
```

### 5. LooseIntentEvent Integration

Enforcement API already has optional layer:
```python
class LooseIntentEvent(BaseModel):
    # ... other fields ...
    layer: Optional[str] = None  # For filtering rules
```

Query logic matches:
```rust
// Intent comes with optional layer
// get_rules_for_layer(intent.layer) 
// Returns global rules (layer=None) + matching layer-specific rules
```

---

## Enforcement Behavior

### Global Rules (layer=None)
- **Included in**: All enforcement queries
- **Use case**: Cross-layer policies that apply everywhere
- **Example**: "Require authentication" (applies to all intents)

### Layer-Specific Rules (layer="L4")
- **Included in**: Only when intent.layer="L4"
- **Use case**: Rules specific to tool gateway operations
- **Example**: "Tool whitelist for database operations"

### Query Examples

**Scenario 1: Intent with layer="L4"**
```
Intent: {action: "execute_tool", layer: "L4", ...}
Rules returned:
  - Global rules (layer=None)
  - Layer-specific rules (layer="L4")
Excluded:
  - Rules with layer="L2" or other values
```

**Scenario 2: Intent with layer=None**
```
Intent: {action: "read_file", layer: None, ...}
Rules returned:
  - Global rules (layer=None) ONLY
Excluded:
  - All layer-specific rules (layer="L4", "L2", etc.)
```

---

## Documentation Files Updated

### 1. `09-v2-policy-installation-plan.md`
- ✅ Updated Executive Summary (storage architecture, layer model)
- ✅ Updated PolicyConverter docstring (layer is optional)
- ✅ Removed "rule_type" from params
- ✅ Updated example RuleInstance (layer from request or None)
- ✅ Updated LooseDesignBoundary model (added layer field)
- ✅ Updated enforcement query logic (optional layer filtering)
- ✅ Updated storage layout (metadata-based not type-based)
- ✅ Updated Query Flow diagram (simpler)

**Lines changed**: ~50 edits across multiple sections

### 2. `09-QUICK_START.md`
- ✅ Updated key design decisions (layer as metadata, not type)
- ✅ Updated data structure examples (added layer field to input)
- ✅ Removed "semantic type" language
- ✅ Updated RuleInstance output example

**Lines changed**: ~25 edits

### 3. `README-DEPRECATION.md`
- ✅ Updated key changes summary
- ✅ Updated critical code changes (removed family_id, hardcoded layer)
- ✅ Updated enforcement change example

**Lines changed**: ~15 edits

---

## Summary of Mental Model Shift

| Aspect | Before | After |
|--------|--------|-------|
| **What is a rule?** | Type-specific (v1 vs v2, L4 vs L0-L3) | Just a rule |
| **Layer purpose** | Type classification | Optional filtering scope |
| **layer field** | Hardcoded "L4" for v2 | From request, or None |
| **Storage model** | Families + layers + types | Simple rule_id → rule mapping |
| **Enforcement** | Query by layer + type | Query rules, filter by layer |
| **Global vs scoped** | All rules scoped to layer | Some global (layer=None), some scoped |
| **API request format** | Implies rule type | Just request structure |

---

## Implementation Impact

### PolicyConverter (Phase 2)
```python
# Extract layer from request (optional)
rule_instance['layer'] = boundary.get('layer')  # None if not present

# Don't add family_id or rule_type
# All constraint/rule data goes to params
```

### DataPlaneClient (Phase 1)
```python
# Serialize layer as-is (optional string or None)
# No special handling needed
```

### Enforcement Engine (Phase 4)
```rust
// Accept optional layer from intent
fn get_rules_for_layer(&self, layer: Option<&str>) -> ... {
    // Include global rules (layer=None)
    // Include matching layer-specific rules
    // Exclude non-matching layer-specific rules
}
```

---

## Testing Checklist

### Unit Tests
- [ ] LooseDesignBoundary with layer="L4" → RuleInstance with layer="L4"
- [ ] LooseDesignBoundary without layer → RuleInstance with layer=None
- [ ] PolicyConverter doesn't add family_id or rule_type
- [ ] RuleVector conversion unchanged

### Integration Tests
- [ ] Install policy with layer="L4"
- [ ] Install policy without layer (globally applicable)
- [ ] Enforce intent with layer="L4" → includes global + L4 rules
- [ ] Enforce intent without layer → includes global rules only
- [ ] Enforce intent with layer="L2" → includes global rules only (no L4 rules)

### End-to-End Tests
- [ ] API: POST /api/v2/policies/install with layer
- [ ] API: POST /api/v2/policies/install without layer
- [ ] Storage: Rules persisted with correct layer metadata
- [ ] Enforcement: Rules retrieved based on intent layer

---

## Future Simplifications

Once this model is in place:

1. **Remove RuleFamilyTable** (Phase 5)
   - No more family classification
   - Just rules indexed by rule_id
   - Layer is a simple optional field

2. **Remove LayerId enum** (Phase 5)
   - Layer is just a string, not an enum
   - Or keep for documentation, make runtime value

3. **Simplify Bridge** (Phase 4)
   - No per-layer table logic
   - Just query hot_cache with optional filter

---

## Validation

✅ **Aligns with enforcement API**: LooseIntentEvent already has optional layer  
✅ **Simpler than before**: No type classification, just metadata  
✅ **Backward compatible**: v1 rules can have layer=None (global)  
✅ **Forward compatible**: Any future policy format can use same model  
✅ **Clear semantics**: Global vs scoped is explicit  

---

## Status

**ALL DOCUMENTATION UPDATED** ✅

Ready to implement with correct mental model:
- Layer = optional filtering metadata
- All policies are rules
- No type classification needed
- Simple, clean architecture
