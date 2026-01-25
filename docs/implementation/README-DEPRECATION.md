# Family Table Deprecation - Quick Reference

## What Changed?

Your intuition was correct: **RuleFamilyTable and L0-L6 layers should be deprecated completely.**

The implementation docs have been updated to reflect this simpler, cleaner architecture.

---

## Three Document to Read

### 1. 📋 **DEPRECATION_SUMMARY.md** (Read First - 5 min)
   - **What**: Overview of all changes made
   - **When**: Read to understand what was updated
   - **Use for**: Quick understanding of the changes

### 2. 🔍 **08-family-table-deprecation-analysis.md** (Read Second - 15 min)
   - **What**: Comprehensive validation that deprecation is safe
   - **When**: Read to understand the "why"
   - **Use for**: Justifying architecture decisions, risk assessment

### 3. 🛠️ **09-v2-policy-installation-plan.md** (Reference During Build - 30 min)
   - **What**: Updated implementation plan with corrected architecture
   - **When**: Reference while implementing Phases 1-5
   - **Use for**: Technical specs, code examples, integration details

### 4. ⚡ **09-QUICK_START.md** (Quick Reference - 10 min)
   - **What**: Condensed version of main plan
   - **When**: Reference for high-level overview
   - **Use for**: Quick phase descriptions, data structures, integration points

---

## Key Changes Summary

### Before (Wrong) ❌
```
Complex Model: Type Hierarchy + Family Classification
├── L4 Rules (DesignBoundaries): Special v2 type
├── v1 ToolWhitelist: Another type
└── Layer = type classification
    
Problem: Unnecessary complexity, no actual type differences
```

### After (Correct) ✅
```
Simple Model: Layer as Optional Filtering Metadata
├── ALL Rules: Just rules, stored in tiered storage
├── Layer: Optional metadata (for enforcement filtering)
│   ├── If present: filters rules during enforcement
│   └── If absent (None): rule is globally applicable
└── Result: Simpler model, 2500+ lines of code removed
    
Benefit: Single rule type, optional layer scoping, clean architecture
```

---

## What's NOT Changing

- ✅ gRPC API (InstallRules endpoint)
- ✅ REST endpoints (/api/v2/policies/install)
- ✅ Rule storage format
- ✅ v1 policies still work
- ✅ Enforcement semantics

---

## Implementation Roadmap

### Phase 1-3: Build v2 Installation (This Week)
```
✅ No family table changes
✅ No family_id field needed
✅ Write directly to tiered storage
✅ Follow updated plan as-is
```

### Phase 4: Update Enforcement (Next Week)
```
- Update get_rules_for_layer() to query hot_cache directly
- Remove RuleFamilyTable query logic
- Test enforcement with tiered storage queries
```

### Phase 5: Remove Infrastructure (Week After)
```
- Delete L0-L3, L5-L6 family files
- Remove RuleFamilyTable completely
- Clean up 2500+ lines of dead code
```

---

## Key Findings

| Finding | Details |
|---------|---------|
| **RuleFamilyTable Status** | 428 lines, 0 active production rules |
| **L0-L3, L5-L6 Status** | 1500+ lines code, 0 active rules |
| **Only Active Family** | L4 ToolWhitelist (~10 rules, already in tiered storage) |
| **Secondary Indices** | 300 lines code, no active usage |
| **Risk of Deprecation** | VERY LOW (nothing depends on it) |
| **Code to Remove** | ~2500 lines |

---

## Evidence

✅ **Code Analysis**:
- Cold storage: 0 L0-L3, L5-L6 rules found
- Git history: Family files never modified after initial creation
- Tests: Only infrastructure tests, no integration tests using these families

✅ **v1 Already Proven**:
- v1 writes to both RuleFamilyTable and tiered storage
- Could just query tiered storage directly
- Simpler enforcement logic

✅ **v2 Clean Slate**:
- No prior dependencies on family tables
- No migration data needed
- Can start directly on tiered storage

---

## For Different Roles

### 👨‍💻 Implementing Phase 1-3?
1. Read `DEPRECATION_SUMMARY.md` (understand changes)
2. Reference `09-QUICK_START.md` (overview)
3. Use `09-v2-policy-installation-plan.md` (detailed specs)
4. Skip family_id entirely, write directly to tiered storage

### 👨‍🏫 Reviewing Architecture?
1. Read `08-family-table-deprecation-analysis.md` (full analysis)
2. Check "Architecture Decision" section in `09-v2-policy-installation-plan.md`
3. Review evidence and risk assessment

### 📊 Making Decisions?
1. Read `08-family-table-deprecation-analysis.md` (validate readiness)
2. Check "Risk Assessment" section
3. Confirm with "Recommendations" section

### 🧪 Writing Tests?
1. Use tiered storage APIs directly (no RuleFamilyTable)
2. Reference `09-QUICK_START.md` data structures
3. Test both v1 ToolWhitelist and v2 DesignBoundary rules coexisting

---

## Critical Code Changes

### Before Implementation (Wrong)
```python
# Over-engineered type system
RuleInstance {
    family_id: "design_boundary",  # ❌ Creates unnecessary family
    layer: "L4",  # ❌ Hardcoded
    rule_type: "design_boundary",  # ❌ Redundant
    ...
}
```

### After Implementation (Correct)
```python
# Simple filtering metadata
RuleInstance {
    rule_id: "boundary-001",
    layer: None,  # ✅ Optional, from request
    params: { ... }  # ✅ All constraints/rules here
}
```

---

## Phase 4 Enforcement Change (Critical)

### Before (Query Family Tables)
```rust
fn get_rules_for_layer(&self, layer: &str) -> Result<Vec<Arc<dyn RuleInstance>>, String> {
    let layer_id = match layer { ... };  // Complex mapping
    let tables = self.bridge.get_tables_by_layer(&layer_id);
    for (_, table) in tables {
        let rules = table.read().query_all();
    }
}
```

### After (Query Tiered Storage with Optional Layer)
```rust
fn get_rules_for_layer(&self, layer: Option<&str>) -> Result<Vec<Arc<dyn RuleInstance>>, String> {
    let all_rules = self.bridge.hot_cache.get_all_rules()?;
    let filtered = all_rules.into_iter()
        .filter(|r| {
            match (r.layer(), layer) {
                (None, _) => true,  // Global rules always included
                (Some(rule_layer), Some(requested)) => rule_layer == requested,
                (Some(_), None) => false,  // Skip layer-specific if no layer requested
            }
        })
        .collect();
    Ok(filtered)
}
```

---

## Deprecation Timeline

```
Week 1: Implement v2 installation (Phases 1-3)
  └─ No family table changes, just build v2

Week 2: Update enforcement (Phase 4)
  └─ Query tiered storage directly instead of family tables

Week 3: Remove infrastructure (Phase 5)
  └─ Delete 2500+ lines of unused code

Week 4: Verify and document
  └─ All tests passing, no regressions
```

---

## Questions?

- **Why now?** RuleFamilyTable has 0 active rules. It's dead code.
- **Why not hybrid?** L0-L3 don't exist, so hybrid solves nothing.
- **Why no family_id?** Tiered storage uses rule_id as primary key.
- **Why risk so low?** Nothing depends on family tables.
- **Why remove 2500 lines?** Reduce complexity, single source of truth.

---

## Success Criteria

✅ v2 DesignBoundary rules install and enforce
✅ v1 ToolWhitelist rules still work
✅ Enforcement queries tiered storage directly
✅ All tests passing with new query logic
✅ 2500+ lines of dead code removed
✅ Zero performance regression
✅ Clear, simple architecture documented

---

**Status**: READY TO BUILD 🚀

Follow the updated plan in `09-v2-policy-installation-plan.md` with confidence that:
- You're building the right architecture
- Risk is very low
- Code will be cleaner and simpler
- No confusing hybrid approaches
