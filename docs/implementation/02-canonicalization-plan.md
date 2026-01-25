# Canonicalization Pipeline v2: Pre-Encoding Normalization

**Objective**: Normalize variable vocabulary in IntentEvents to canonical terms before encoding, ensuring policies and intents align semantically without requiring strict input validation or fuzzy matching.

**Core Principle**: A lightweight BERT classifier maps free-form intent vocabulary to canonical terms defined in configuration. Both policies and intents use the same canonical terms, ensuring deterministic, high-fidelity encoding and comparison.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Canonicalization Architecture](#canonicalization-architecture)
3. [Agent-Agnostic Hook System](#agent-agnostic-hook-system)
4. [BERT Classifier Design](#bert-classifier-design)
5. [Configuration Schema](#configuration-schema)
6. [Data Collection & Training](#data-collection--training)
7. [Learning Loop](#learning-loop)
8. [Integration Points](#integration-points)
9. [Implementation Tasks](#implementation-tasks)

---

## Executive Summary

### Problem

IntentEvents and policies contain variable vocabulary:

- Intent: "query the users table"
- Policy: expects "read" for allowed action

Without normalization, encoding similarity becomes fuzzy and unreliable. We cannot guarantee high-recall matching (policies must match intended intents).

### Solution

Insert a **canonicalization layer** before encoding:

```
IntentEvent (variable) → BERT Classifier → Canonical Terms → IntentEncoder (SemanticEncoder)
                                                                  ↓
                                                          128d intent vector
                                                        
Policy (variable) → BERT Classifier → Canonical Terms → PolicyEncoder (SemanticEncoder)
                                                              ↓
                                                      4×16×32 anchor vectors
```

Both use an identical pipeline up to canonicalization, then diverge via specialized encoder subclasses (IntentEncoder and PolicyEncoder) that share a common **SemanticEncoder** base class. This ensures:

- **Deterministic**: Same input always produces same output
- **Controllable**: Canonical terms defined in YAML
- **Learnable**: BERT model improves via production data
- **Fail-safe**: Unknown terms pass through unchanged (likely blocks matching)
- **DRY**: Shared encoding logic in SemanticEncoder base class, no duplication

---

## Canonicalization Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  CANONICALIZATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Event/Policy                                             │
│       ↓                                                         │
│  ┌───────────────────────────────────────┐                     │
│  │  BERT Classifier (Fine-tuned)          │                     │
│  │  ──────────────────────────            │                     │
│  │  Model: TinyBERT / DistilBERT         │                     │
│  │  Input: Full context (128 tokens)     │                     │
│  │  Output: Multi-head predictions       │                     │
│  │  Latency: <10ms CPU inference         │                     │
│  └───────────────────────────────────────┘                     │
│       ↓                                                         │
│  ┌───────────────────────────────────────┐                     │
│  │  Confidence Thresholds                 │                     │
│  │  ──────────────────────────            │                     │
│  │  • High (≥0.9): Use prediction        │                     │
│  │  • Medium (0.7-0.9): Use + log        │                     │
│  │  • Low (<0.7): Passthrough (unsafe)   │                     │
│  └───────────────────────────────────────┘                     │
│       ↓                                                         │
│  ┌───────────────────────────────────────┐                     │
│  │  Production Logging                    │                     │
│  │  ──────────────────────────            │                     │
│  │  • All predictions + confidence       │                     │
│  │  • Enforcement outcome (match/block)  │                     │
│  │  • Used for offline learning loop     │                     │
│  └───────────────────────────────────────┘                     │
│       ↓                                                         │
│  Canonical IntentEvent/Policy                                   │
│  (ready for slot builder + encoding)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Canonicalization Scope

**Fields to canonicalize** (configurable per deployment):

```yaml
canonicalization:
  fields:
    - name: action
      source: "event.action"
      context_fields: ["event.tool_name", "event.tool_method"]
      required: true
  
    - name: resource_type
      source: "event.resource.type"
      context_fields: ["event.tool_name", "event.resource.location"]
      required: true
  
    - name: sensitivity
      source: "event.data.sensitivity"
      context_fields: []
      required: true
  
    # Optional: canonicalize tool names too
    - name: tool_category
      source: "event.tool_name"
      context_fields: ["event.tool_method"]
      required: false
```

**Fail behavior** (configurable):

- `passthrough`: Use raw term if not confidently mapped (safe, may fail matching)
- `reject`: Return error to client (fail-fast, prevents bad data)
- `default`: Map unknown terms to "other" category (risky, may over-block)

**Recommendation**: Use `passthrough` with high-confidence threshold to maximize safety.

---

## Agent-Agnostic Hook System

### Problem: SDK Coupling

The current enforcement system uses SDK callbacks (`AgentCallback` in LangChain/LangGraph), coupling Guard to specific agent frameworks. This prevents adoption across diverse environments:

- Custom agents (home-grown orchestration)
- n8n workflows (visual workflow builder, no code)
- OpenAI API calls (direct SDK usage)
- Anthropic SDK calls
- Any future agent framework

### Solution: HTTP Hook Endpoint

Replace SDK coupling with a single **HTTP endpoint** that any agent can call at configurable points in their execution flow. The endpoint is **framework and vocabulary-agnostic**:

```
┌────────────────────────────────────────────────────────────────────────┐
│              AGENT-AGNOSTIC GUARD ENFORCEMENT                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ Any Agent Framework                                     │           │
│  │ (LangGraph, LangChain, OpenAI SDK, n8n, custom, etc)    │           │
│  └──────────────────────────┬──────────────────────────────┘           │
│                             │                                          │
│                 HTTP POST /v2/guard/enforce                            │
│                             │                                          │
│        ┌────────────────────▼───────────────────────┐                  │
│        │       HOOK REQUEST (Flexible Format)       │                  │
│        ├────────────────────────────────────────────┤                  │
│        │ {                                          │                  │
│        │   "hook": "pre_tool_call",                 │                  │
│        │   "intent": {                              │                  │
│        │     "tool_name": "database_query",         │                  │
│        │     "action": "query",    ← non-canonical  │                  │
│        │     "resource": "users",                   │                  │
│        │     "description": "fetch active accounts" │                  │
│        │   },                                       │                  │
│        │   "context": { ... }                       │                  │
│        │ }                                          │                  │
│        └────────────────────┬───────────────────────┘                  │
│                             │                                          │
│        ┌────────────────────▼───────────────────────┐                  │
│        │    HOOK PROCESSOR (Framework Agnostic)     │                  │
│        ├────────────────────────────────────────────┤                  │
│        │                                            │                  │
│        │  1. Hook Validator (hooks.yaml)            │                  │
│        │     ├─ Validate hook type                  │                  │
│        │     └─ Check required fields               │                  │
│        │                                            │                  │
│        │  2. Field Extractor (extraction.yaml)      │                  │
│        │     ├─ Detect input format                 │                  │
│        │     └─ Apply field mappings                │                  │
│        │                                            │                  │
│        │  3. BERT Canonicalizer                     │                  │
│        │     ├─ "query" → "read"                    │                  │
│        │     ├─ Infer resource_type from context    │                  │
│        │     └─ Apply confidence thresholds         │                  │
│        │                                            │                  │
│        │  4. Intent Builder                         │                  │
│        │     └─ Assemble canonical IntentEvent      │                  │
│        │                                            │                  │
│        └────────────────────┬───────────────────────┘                  │
│                             │                                          │
│        ┌────────────────────▼───────────────────────┐                  │
│        │   EXISTING ENFORCEMENT PIPELINE            │                  │
│        │   (encode_to_128d → data_plane.enforce)    │                  │
│        └────────────────────┬───────────────────────┘                  │
│                             │                                          │
│        ┌────────────────────▼───────────────────────┐                  │
│        │      RESPONSE (HookEnforcementResult)      │                  │
│        ├────────────────────────────────────────────┤                  │
│        │ {                                          │                  │
│        │   "decision": 1,                           │                  │
│        │   "hook": "pre_tool_call",                 │                  │
│        │   "canonical_intent": {                    │                  │
│        │     "action": "read",    ← canonicalized   │                  │
│        │     "resource_type": "database",           │                  │
│        │     ...                                    │                  │
│        │   },                                       │                  │
│        │   "evidence": [ ... ],                     │                  │
│        │   "trace": { ... }       ← debug info      │                  │
│        │ }                                          │                  │
│        └────────────────────────────────────────────┘                  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Hook Types (Configurable)

Agents call the endpoint at specific points in their execution flow, identified by hook type:


| Hook Type        | Description                    | Requires Decision | Use Case                                  |
| ------------------ | -------------------------------- | ------------------- | ------------------------------------------- |
| `pre_tool_call`  | Before agent executes a tool   | Yes               | Most common - block unsafe operations     |
| `checkpoint`     | At user-defined workflow gates | Yes               | Multi-step workflows with approval points |
| `audit`          | After action completes         | No                | Logging only, no blocking                 |
| `post_execution` | Log completed actions          | No                | Audit trail / forensics                   |

### Who Calls The Hook?

**The agent code** (not Guard) calls the hook. Guard only provides the endpoint.

**Example: LangGraph Agent**

```python
# In the agent's tool execution handler
async def execute_tool(state, config):
    tool_name = state.tool_name
    tool_input = state.tool_input
  
    # Agent calls Guard hook (new, framework-agnostic)
    response = await httpx.post(
        "https://guard.example.com/v2/guard/enforce",
        json={
            "hook": "pre_tool_call",
            "intent": {
                "tool_name": tool_name,
                "action": state.action,           # May be non-canonical
                "resource": state.resource,
                "description": state.description
            }
        },
        headers={"Authorization": f"Bearer {api_key}"}
    )
  
    if response.json()["decision"] == 0:  # Blocked
        raise PermissionError(f"Guard blocked: {response.json()['evidence']}")
  
    # Proceed with tool execution
    return tool.execute(tool_input)
```

**Example: n8n Workflow**

```
HTTP Request Node
├─ URL: POST /v2/guard/enforce
├─ Headers: Authorization: Bearer $API_KEY
├─ Body:
│  {
│    "hook": "pre_tool_call",
│    "intent": {
│      "tool_name": "{{ $node.PreviousNode.json.tool_name }}",
│      "action": "{{ $node.PreviousNode.json.action }}",
│      ...
│    }
│  }
└─ If response.decision == 0: Stop workflow
```

**Example: Direct OpenAI SDK Call**

```python
# Wrapper function in agent codebase
async def call_with_guard(model, messages, tools, **kwargs):
    # Guard hook call
    guard_response = await check_guard(
        hook="pre_tool_call",
        intent={
            "tool_name": tools[0].name,
            "action": "execute",
            "resource": "openai-api"
        }
    )
  
    if not guard_response.allowed:
        raise PermissionError(guard_response.evidence)
  
    # Proceed with OpenAI call
    return model.create(messages=messages, tools=tools, **kwargs)
```

### Key Properties

1. **Framework Agnostic**: Works with any orchestration tool via HTTP
2. **Vocabulary Flexible**: Accepts non-canonical terms, normalizes via BERT
3. **Input Format Flexible**: Supports structured fields, natural language descriptions, or hybrid
4. **Fail-Safe**: Unknown terms default to safe values and are flagged
5. **Observable**: Returns canonicalization trace for debugging

---

## BERT Classifier Design

### Model Selection


| Model           | Size  | Inference | Training  | Recommendation    |
| ----------------- | ------- | ----------- | ----------- | ------------------- |
| **TinyBERT**    | 14.5M | <5ms      | Fast      | ✓ Preferred      |
| **DistilBERT**  | 66M   | <10ms     | Moderate  | Alternative       |
| **MiniLM**      | 22M   | <8ms      | Moderate  | Alternative       |
| **ALBERT-tiny** | 5M    | <3ms      | Very fast | Ultra-lightweight |

**Choice**: **TinyBERT** - Good balance of accuracy, speed, and training time.

### Architecture

```
Input: "query the users table for active accounts" (128 tokens max)
       ↓
┌──────────────────────────────────────────┐
│  TinyBERT Encoder                        │
│  (2-layer, 312 hidden units)             │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Classification Heads (one per field)    │
├──────────────────────────────────────────┤
│  • action_head: softmax(6 actions)       │
│  • resource_type_head: softmax(5 types) │
│  • sensitivity_head: softmax(3 levels)  │
│  • (optional) tool_category_head: ...   │
└──────────────────────────────────────────┘
       ↓
Output: {
  "action": { "label": "read", "confidence": 0.94 },
  "resource_type": { "label": "database", "confidence": 0.89 },
  "sensitivity": { "label": "internal", "confidence": 0.76 }
}
```

### Training Strategy

**Phase 1: Initialization (This Release)**

Pre-train TinyBERT classifier using seed dataset before going to production:

1. **Data Preparation**

   - Collect ~50K-100K labeled examples (see Data Collection section)
   - Stratify by canonical category to avoid bias
   - Create 80/10/10 train/val/test split
   - Tokenize to max 128 tokens per example
2. **Model Architecture**

   ```python
   # TinyBERT backbone: 2-layer, 312 hidden units
   bert = TinyBertModel.from_pretrained("huawei-noah/TinyBERT-4L-312D")

   # Multi-head classification (one head per field)
   classification_heads = {
       "action": LinearLayer(312 → 6),        # [read, write, update, delete, execute, export]
       "resource_type": LinearLayer(312 → 5), # [database, file, api, queue, cache]
       "sensitivity": LinearLayer(312 → 3)    # [public, internal, secret]
   }
   ```
3. **Training Pipeline**

   - Loss: Cross-entropy (per-field independent heads)
   - Optimizer: AdamW (lr=2e-5, warmup 10% of steps)
   - Epochs: 3-5 (early stopping on val loss)
   - Batch size: 32
   - Device: GPU (training), CPU (inference)
4. **Validation & Metrics**

   - Per-field accuracy: Target ≥95%
   - Confidence calibration: Expected calibration error (ECE) <0.05
   - Precision/Recall per category
   - Inference latency: Measure on CPU (<10ms target)
5. **Checkpoint Management**

   - Save best checkpoint (based on validation accuracy)
   - Store in: `management_plane/models/canonicalizer_tinybert_v1.0/`
   - Include tokenizer config, model weights, vocabulary

**Phase 2: Production Learning Loop (Post-Launch)**

After deployment, collect production data to improve the model:

1. **Logging Collection**

   - Every canonicalization attempt is logged (see Phase 2 in Data Collection)
   - Store in `/var/log/guard/canonicalization/` (JSON lines format)
   - Retention: 90 days
   - Async logging to avoid latency impact
2. **Signal Detection** (Weekly)

   - Low confidence predictions (< 0.8): Uncertain BERT predictions
   - Enforcement mismatches: High confidence but policy didn't match
   - New rare terms: Vocabulary not in seed dataset
   - Distribution shifts: Input patterns changing over time
3. **Curation** (Monthly)

   - Human team reviews flagged examples
   - Correct mislabeled predictions
   - Add new categories if needed
   - Target: 500-1000 curated examples per cycle
4. **Retraining** (Monthly)

   ```python
   # Combine seed dataset + curated production examples
   train_data = seed_dataset + monthly_curated_examples

   # Fine-tune from previous checkpoint
   bert = load_model("canonicalizer_tinybert_v1.X")

   # Continue training for 1-2 epochs (avoid catastrophic forgetting)
   for epoch in range(1, 3):
       for batch in train_loader:
           loss = compute_loss(bert(batch), batch.labels)
           optimizer.step()

   # Validate on held-out test set
   accuracy = validate(bert, test_loader)

   # A/B test before rollout
   if accuracy >= baseline:
       save_model(bert, "canonicalizer_tinybert_v1.X+1")
       deploy_to_prod()
   else:
       log_failure_metrics()
   ```
5. **Version Management**

   - Model versions: `canonicalizer_tinybert_v1.0`, `v1.1`, etc.
   - Track in git with model checkpoints
   - Backward compatible: Old vocab categories never removed
   - New categories added with versioning

**Phase 3: Continuous Improvement**

- Quarterly vocabulary reviews (add new categories if patterns emerge)
- Track accuracy per field over time
- Monitor confidence calibration drift
- Adjust thresholds if needed

---

## Configuration Schema

### Hook Configuration (hooks.yaml)

```yaml
# config/hooks.yaml
version: "1.0"

hooks:
  # Pre-tool-call enforcement (most common)
  pre_tool_call:
    description: "Enforce before agent executes a tool"
    required_fields: ["intent"]
    optional_fields: ["context", "session_id"]
    decision_required: true
    default_effect: "deny"  # Deny if no policy matches
  
  # Checkpoint enforcement (workflow gates)
  checkpoint:
    description: "Enforce at workflow checkpoint"
    required_fields: ["intent", "checkpoint_id"]
    optional_fields: ["context"]
    decision_required: true
    default_effect: "deny"
  
  # Audit-only mode (logging, no blocking)
  audit:
    description: "Log intent without blocking"
    required_fields: ["intent"]
    decision_required: false
  
  # Post-execution logging
  post_execution:
    description: "Log completed actions for audit trail"
    required_fields: ["intent", "result"]
    decision_required: false

# Default hook if not specified
default_hook: "pre_tool_call"
```

### Field Extraction Configuration (extraction.yaml)

```yaml
# config/extraction.yaml
version: "1.0"

# Detect input format
format_detection:
  structured_indicators:
    - "intent.action"
    - "intent.tool_name"
    - "intent.resource"
  natural_language_field: "intent"  # If intent is string, treat as NL

# Extraction mappings for structured input
extraction:
  # Action field - WILL BE CANONICALIZED via BERT
  action:
    sources:  # Try in order
      - "intent.action"
      - "intent.verb"
      - "intent.operation"
    infer_from:
      - field: "intent.tool_method"
        rules:
          - match: "^(get|fetch|list|describe|select|query).*"
            value: "read"
          - match: "^(post|create|insert|add|save).*"
            value: "write"
          - match: "^(put|patch|update|modify|change).*"
            value: "update"
          - match: "^(delete|remove|drop|purge).*"
            value: "delete"
    fallback: "execute"
    canonicalize: true   # Pass through BERT classifier

  # Resource type - WILL BE CANONICALIZED via BERT
  resource_type:
    sources:
      - "intent.resource.type"
      - "intent.resource_type"
    infer_from:
      - field: "intent.tool_name"
        rules:
          - match: ".*(database|db|sql|postgres|mysql|mongo).*"
            value: "database"
          - match: ".*(file|storage|s3|blob|gcs).*"
            value: "file"
          - match: ".*(api|http|rest|graphql|webhook).*"
            value: "api"
          - match: ".*(queue|sqs|kafka|rabbitmq).*"
            value: "queue"
          - match: ".*(cache|redis|memcached).*"
            value: "cache"
    fallback: "api"
    canonicalize: true

  # Resource name (identifier) - NOT canonicalized
  resource_name:
    sources:
      - "intent.resource.name"
      - "intent.resource"
      - "intent.target"
    fallback: null
    canonicalize: false

  # Resource location
  resource_location:
    sources:
      - "intent.resource.location"
    infer_from:
      - field: "intent.tool_name"
        rules:
          - match: ".*(s3|gcs|azure|cloud|aws|gcp).*"
            value: "cloud"
          - match: ".*(local|file|disk|on-premise).*"
            value: "local"
    fallback: "cloud"
    canonicalize: false

  # Data sensitivity - WILL BE CANONICALIZED via BERT
  sensitivity:
    sources:
      - "intent.data.sensitivity"
      - "intent.sensitivity"
    infer_from:
      - field: "intent.resource.name"
        rules:
          - match: ".*(user|customer|pii|personal|private|secret|credential|password|token|key).*"
            value: ["internal"]
          - match: ".*(public|open|external).*"
            value: ["public"]
    fallback: ["internal"]  # Fail-safe: default to restrictive
    canonicalize: true

  # Volume
  volume:
    sources:
      - "intent.data.volume"
    infer_from:
      - field: "intent.tool_method"
        rules:
          - match: ".*(list|all|batch|bulk|export|dump).*"
            value: "bulk"
    fallback: "single"
    canonicalize: false

  # Actor context (passed through)
  actor_id:
    sources:
      - "context.actor_id"
      - "context.agent_id"
      - "context.user_id"
    fallback: "unknown"
  
  actor_type:
    sources:
      - "context.actor_type"
    fallback: "agent"

  # Authentication requirement
  authn:
    sources:
      - "intent.risk.authn"
    infer_from:
      - field: "context.authenticated"
        rules:
          - match: "true"
            value: "required"
          - match: "false"
            value: "not_required"
    fallback: "required"
  
  # Tool context (used by BERT canonicalizer)
  tool_name:
    sources:
      - "intent.tool_name"
      - "context.tool_name"
    fallback: null
  
  tool_method:
    sources:
      - "intent.tool_method"
      - "intent.method"
    fallback: null

# Defaults for IntentEvent fields not extracted
defaults:
  schemaVersion: "v1.3"
  pii: false
```

### Canonical Vocabulary (YAML)

```yaml
# config/canonicalization.yaml
version: "2.0"

canonicalization:
  enabled: true
  model:
    name: "guard/canonicalizer-tinybert"
    type: "tinybert"
    max_input_length: 128
    device: "cpu"  # cpu | cuda

  # Which fields to canonicalize
  fields:
    - name: action
      source: "event.action"
      context_fields: ["event.tool_name", "event.tool_method"]
      required: true
  
    - name: resource_type
      source: "event.resource.type"
      context_fields: ["event.tool_name", "event.resource.location"]
      required: true
  
    - name: sensitivity
      source: "event.data.sensitivity"
      context_fields: []
      required: true

  # Confidence thresholds
  thresholds:
    high_confidence: 0.9   # Use prediction directly
    low_confidence: 0.7    # Flag for review but use
    reject: 0.5            # Below this: fail-safe behavior
  
  # Behavior when confidence is below threshold
  fail_behavior: "passthrough"  # passthrough | reject | default

  # Production logging for learning loop
  logging:
    enabled: true
    log_all_predictions: true
    log_path: "/var/log/guard/canonicalization/"
    retention_days: 90

# ============================================
# CANONICAL VOCABULARY (Source of Truth)
# ============================================
vocabulary:
  version: "1.0"
  last_updated: "2025-01-24"

  action:
    read:
      description: "Retrieve or access data without modification"
      examples:
        - read
        - query
        - fetch
        - get
        - retrieve
        - select
        - lookup
        - find
        - search
        - load
        - list
        - describe
    write:
      description: "Create new data"
      examples:
        - write
        - insert
        - create
        - add
        - put
        - save
        - store
        - post
        - append
        - upload
    update:
      description: "Modify existing data"
      examples:
        - update
        - modify
        - change
        - edit
        - patch
        - alter
        - set
        - replace
    delete:
      description: "Remove data"
      examples:
        - delete
        - remove
        - drop
        - truncate
        - purge
        - clear
        - destroy
        - unlink
    execute:
      description: "Run a function, process, or command"
      examples:
        - execute
        - run
        - invoke
        - call
        - trigger
        - start
        - launch
        - spawn
    export:
      description: "Extract data to external destination"
      examples:
        - export
        - download
        - extract
        - dump
        - backup
        - copy
        - transfer
        - send

  resource_type:
    database:
      description: "Structured data storage systems"
      examples:
        - database
        - db
        - postgres
        - postgresql
        - mysql
        - sqlite
        - mongodb
        - dynamodb
        - rds
        - sql
    storage:
      description: "File and object storage systems"
      examples:
        - storage
        - s3
        - blob
        - bucket
        - file
        - filesystem
        - gcs
        - azure-blob
        - minio
    api:
      description: "External service endpoints"
      examples:
        - api
        - endpoint
        - rest
        - graphql
        - webhook
        - service
        - http
    queue:
      description: "Message queuing systems"
      examples:
        - queue
        - sqs
        - kafka
        - rabbitmq
        - pubsub
        - sns
    cache:
      description: "Caching systems"
      examples:
        - cache
        - redis
        - memcached

  sensitivity:
    public:
      description: "Data that can be freely shared"
      examples:
        - public
        - open
        - unrestricted
        - external
    internal:
      description: "Data restricted to organization"
      examples:
        - internal
        - private
        - confidential
        - restricted
    secret:
      description: "Highly sensitive data"
      examples:
        - secret
        - sensitive
        - pii
        - phi
        - credential
        - password
        - key
        - token
```

---

## Data Collection & Training

### Phase 1: Seed Dataset (Pre-Training)

**Goal**: Build unbiased baseline covering diverse agent tool patterns.

**Sources**:

1. **Public API Specifications** (30% of data)

   - OpenAPI specs: Stripe, Twilio, GitHub, AWS, GCP, Slack, etc.
   - Extract: verbs, nouns, parameters
   - Benefit: Real-world, diverse, standardized
2. **LLM Tool-Use Datasets** (20% of data)

   - ToolBench (1.6M tool instructions)
   - API-Bank (645 APIs, 9K tool-use examples)
   - ToolAlpaca (3.9K tool-use instructions)
   - Benefit: Agent-specific patterns
3. **Database & Storage Patterns** (20% of data)

   - SQL operation verbs (SELECT, INSERT, UPDATE, DELETE)
   - NoSQL operations (put, get, delete, scan)
   - File operations (read, write, append, truncate)
   - Benefit: Core data operations
4. **Synthetic Variations** (20% of data)

   - Template-based generation: "Generate 20 ways to express 'read from database'"
   - Human review before inclusion
   - Benefit: Handles synonyms, context variations
5. **Manual Curation** (10% of data)

   - Domain experts define high-confidence examples per category
   - Edge cases and ambiguities
   - Benefit: Ensures coverage of corner cases

**Bias Mitigation**:

- Stratified sampling: Equal examples per canonical category
- Diverse sources: No single API style dominates
- Human review: Flag and remove outliers
- Version control: Track data provenance

**Seed Dataset Structure**:

```json
{
  "id": "seed-001",
  "raw_text": "fetch all users from postgres",
  "context": {
    "tool_name": "database_query",
    "tool_method": "query",
    "resource_location": "users"
  },
  "labels": {
    "action": "read",
    "resource_type": "database",
    "sensitivity": null
  },
  "source": "openapi-spec",
  "reviewed": true,
  "reviewer": "alice@guard.ai"
}
```

**Target Size**: 5,000-10,000 examples per canonical category (~50K total).

### Phase 2: Production Logging

**What to log** on every canonicalization:

```json
{
  "timestamp": "2025-01-24T10:30:00Z",
  "request_id": "req-abc123",
  "tenant_id": "acme-corp",
  "field": "action",
  
  "raw_input": "query",
  "full_context": "query the users table for active accounts",
  "context_fields": {
    "tool_name": "database_query",
    "tool_method": "query"
  },
  
  "prediction": {
    "canonical": "read",
    "confidence": 0.94,
    "top_3": [
      { "label": "read", "confidence": 0.94 },
      { "label": "execute", "confidence": 0.04 },
      { "label": "query", "confidence": 0.02 }
    ]
  },
  
  "enforcement_outcome": {
    "intent_encoded": true,
    "policy_matched": true,
    "decision": "allow",
    "matched_policy_id": "allow-db-read",
    "similarity_score": 0.91
  },
  
  "model_version": "canonicalizer-tinybert-v1.2"
}
```

**Key Signals for Curation**:

1. **Low confidence** (confidence < 0.8): Uncertain predictions
2. **Enforcement mismatch**: High confidence but no policy matched
3. **New terms**: Rare/unseen vocabulary
4. **Distribution changes**: Shift in input patterns over time

### Phase 3: Curation & Retraining

```
Production Logs (daily)
      ↓
┌─────────────────────────────────────┐
│  AUTOMATED FILTERING                │
│  • Low confidence (< 0.8)           │
│  • Enforcement mismatches           │
│  • New/rare terms                   │
│  • High disagreement (top-2 close)  │
└─────────────────────────────────────┘
      ↓
Candidate Pool (flagged)
      ↓
┌─────────────────────────────────────┐
│  HUMAN REVIEW INTERFACE             │
│  • Show raw input + full context    │
│  • Show model prediction            │
│  • Reviewer selects correct label   │
│  • Option: "New category needed"    │
└─────────────────────────────────────┘
      ↓
Curated Examples (~500-1000/month)
      ↓
┌─────────────────────────────────────┐
│  PERIODIC RETRAINING (Monthly)      │
│  • Combine: seed + curated data     │
│  • 80/10/10 split (train/val/test)  │
│  • Track metrics per field          │
│  • A/B test before rollout          │
└─────────────────────────────────────┘
      ↓
Updated BERT Model
(deployed with confidence)
```

**Retraining Schedule**:

- **Weekly**: Analyze logs, identify candidates
- **Monthly**: Curation sprint + retraining
- **Quarterly**: Vocabulary review + updates

---

## Learning Loop

### Metrics Tracked

Per classification head:

- **Accuracy**: % of correct predictions
- **Confidence calibration**: Do confidence scores match reality?
- **Precision/Recall**: Per canonical category
- **Latency**: Average inference time

### Feedback Loop

1. **Model predicts**: "action" = "read" (confidence 0.94)
2. **Enforcement runs**: Policy matched or not
3. **Outcome logged**: Match / no-match
4. **Mismatch analysis**: Why did policy not match?
   - Policy missing? (add new policy)
   - Wrong canonicalization? (retrain BERT)
   - Threshold too strict? (adjust)
5. **Curate**: Human review of low-confidence / mismatches
6. **Retrain**: Update model weights monthly

### Vocabulary Evolution

Canonical vocabulary can evolve as new patterns emerge:

```yaml
vocabulary:
  version: "1.1"  # Bumped when vocabulary changes
  action:
    stream:  # NEW - added in v1.1
      description: "Continuous data flow operations"
      examples:
        - subscribe
        - listen
        - watch
        - tail
        - follow
      added: "2025-01-20"
      justification: "High volume of streaming ops not fitting existing categories"
```

---

## Integration Points

### 1. Agent Hook Endpoint (NEW - Framework Agnostic Entry Point)

**Route**: `POST /v2/guard/enforce`

**Decision-only behavior**: The endpoint returns ALLOW/BLOCK decisions. The caller must enforce the decision in its own execution flow until the hook-based enforcement is implemented.

**Request Model**:

```python
class HookRequest(BaseModel):
    """Flexible input from any agent framework."""
    hook: str = "pre_tool_call"      # Hook type
    intent: Union[str, dict]          # NL string or structured dict
    context: Optional[dict] = None    # Additional context (tool_name, etc)
    session_id: Optional[str] = None
```

**Implementation**:

```python
# management_plane/app/endpoints/hooks.py

@router.post("", response_model=HookEnforcementResult)
async def enforce_hook(
    request: HookRequest,
    current_user: User = Depends(get_current_tenant),
) -> HookEnforcementResult:
    """
    Agent-agnostic enforcement endpoint.
  
    1. Validates hook type (hooks.yaml)
    2. Extracts fields from flexible input (extraction.yaml)
    3. Canonicalizes via BERT classifier
    4. Builds IntentEvent
    5. Encodes and enforces
    """
    # Validate hook
    hook_config = validate_hook(request.hook, hooks_yaml_config)
  
    # Extract fields from flexible input
    extracted = extract_fields(
        request.intent,
        request.context,
        extraction_yaml_config
    )
  
    # Canonicalize vocabulary via BERT
    canonicalized = canonicalizer.canonicalize(
        extracted,
        confidence_thresholds=canonicalization_yaml_config.thresholds,
        fail_behavior="default"  # Use defaults for unknown terms
    )
  
    # Build complete IntentEvent
    intent_event = build_intent_event(
        canonicalized,
        defaults_from_extraction_yaml
    )
    intent_event.tenantId = current_user.id
  
    # Encode and enforce (existing flow)
    vector = encode_to_128d(intent_event)
    enforcement_result = await client.enforce(intent_event, vector.tolist())
  
    # Return enriched result
    return HookEnforcementResult(
        decision=enforcement_result.decision,
        hook=request.hook,
        canonical_intent=canonicalized,  # Show canonicalization
        evidence=enforcement_result.evidence,
        trace=canonicalization_trace,  # Debug: BERT confidence, inferred fields
    )
```

**Response Model**:

```python
class CanonicalizedFields(BaseModel):
    """Result of extraction + canonicalization."""
    action: str
    resource_type: str
    resource_name: Optional[str]
    resource_location: Optional[str]
    sensitivity: list[str]
    volume: str
    actor_id: str
    actor_type: str
    authn: str
    tool_name: Optional[str]
    tool_method: Optional[str]
  
    # Canonicalization metadata
    confidence_scores: dict[str, float]  # {field: confidence}
    inferred_fields: list[str]           # Fields inferred vs explicit
    fallback_fields: list[str]           # Fields that used defaults

class HookEnforcementResult(BaseModel):
    """Response with full trace."""
    decision: int  # 0 = block, 1 = allow
    hook: str
    canonical_intent: CanonicalizedFields
    evidence: list[BoundaryEvidence]
    trace: Optional[dict] = None  # {field: {raw, predicted, confidence, source}}
```

**Example Usage** (from Agent):

```bash
curl -X POST https://guard.example.com/v2/guard/enforce \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "hook": "pre_tool_call",
    "intent": {
      "tool_name": "database_query",
      "action": "query",
      "resource": "users",
      "description": "fetch active accounts"
    },
    "context": {
      "actor_id": "agent-123",
      "actor_type": "agent"
    }
  }'

# Response:
{
  "decision": 1,
  "hook": "pre_tool_call",
  "canonical_intent": {
    "action": "read",                    # Canonicalized from "query"
    "resource_type": "database",         # Inferred from tool_name
    "resource_name": "users",
    "resource_location": "cloud",        # Inferred from tool_name
    "sensitivity": ["internal"],         # Inferred from resource_name
    "volume": "single",
    "actor_id": "agent-123",
    "actor_type": "agent",
    "authn": "required",
    "tool_name": "database_query",
    "tool_method": null,
    "confidence_scores": {
      "action": 0.94,
      "resource_type": 0.89,
      "sensitivity": 0.76
    },
    "inferred_fields": ["resource_type", "resource_location", "sensitivity"],
    "fallback_fields": []
  },
  "evidence": [
    {
      "boundary_id": "allow-db-read",
      "boundary_name": "Allow Database Read Operations",
      "effect": "allow",
      "decision": 1,
      "similarities": [0.92, 0.88, 0.85, 0.90]
    }
  ],
  "trace": {
    "action": {
      "raw": "query",
      "predicted": "read",
      "confidence": 0.94,
      "source": "bert_classifier"
    },
    "resource_type": {
      "raw": null,
      "predicted": "database",
      "confidence": 0.89,
      "source": "bert_infer_from_tool_name"
    }
  }
}
```

### 2. Intent Canonicalization (Enforcement Path)

```python
# management_plane/app/services/canonicalizer.py

class Canonicalizer:
    def __init__(self, config: CanonicalizationConfig):
        self.bert_model = load_tinybert(config.model_name)
        self.config = config
    
    def canonicalize_intent(self, intent: IntentEvent) -> CanonicalizedIntent:
        """Canonicalize all configured fields in intent.
    
        After canonicalization, the intent is passed to IntentEncoder
        (a subclass of SemanticEncoder) for semantic encoding.
        """
        canonicalized = intent.copy()
        logs = []
    
        for field_config in self.config.fields:
            raw_value = self._extract_field(intent, field_config.source)
            context = self._build_context(intent, field_config)
        
            prediction = self.bert_model.predict(
                text=raw_value,
                context=context,
                field=field_config.name
            )
        
            # Log for learning loop
            logs.append(CanonicalLog(
                field=field_config.name,
                raw_input=raw_value,
                prediction=prediction,
                timestamp=now()
            ))
        
            # Apply confidence threshold
            if prediction.confidence >= self.config.thresholds.high_confidence:
                self._set_field(canonicalized, field_config.source, prediction.label)
            elif prediction.confidence >= self.config.thresholds.low_confidence:
                self._set_field(canonicalized, field_config.source, prediction.label)
                logs[-1].flagged = True
            else:
                # Passthrough - use raw value
                pass
    
        # Persist logs asynchronously
        asyncio.create_task(self._log_predictions(logs, intent))
    
        return canonicalized
```

**Integration with IntentEncoder:**
After canonicalization, the `CanonicalizedIntent` is passed to `IntentEncoder` (which inherits from `SemanticEncoder`):

```python
# In enforcement endpoint
canonical_intent = canonicalizer.canonicalize_intent(raw_intent)
intent_vector = intent_encoder.encode(canonical_intent)  # Returns 128d vector
enforcement_result = data_plane.enforce(intent_vector, policies)
```

### 2. Policy Canonicalization (Installation Path)

```python
# When policies are installed
@router.post("/v2/policies/install")
async def install_policies(
    request: InstallPoliciesRequest,
    api_key: str = Depends(verify_api_key)
):
    """Install policies with validation and canonicalization.
  
    After canonicalization, policies are passed to PolicyEncoder
    (a subclass of SemanticEncoder) for anchor vector encoding.
    """
    canonicalizer = get_canonicalizer()
  
    # Validate policy anchors use canonical terms
    for policy in request.policies:
        for slot_name, anchor_terms in policy.anchors.items():
            for term in anchor_terms:
                # Check if term is canonical
                if not is_canonical(term, slot_name):
                    # Try to canonicalize
                    canonical = canonicalizer.canonicalize_single(term, slot_name)
                    # Log + continue (silent canonicalization)
  
    # Encode policies using PolicyEncoder (inherits from SemanticEncoder)
    policy_encoder = get_policy_encoder()
    for policy in request.policies:
        policy.anchors = policy_encoder.encode(policy)  # Returns RuleVector
  
    # Install canonicalized, encoded policies
    return install_rules(request)
```

### 3. Unified Semantic Encoding (Post-Canonicalization)

After canonicalization, both intents and policies are encoded using a shared **SemanticEncoder** infrastructure. This base class handles all common encoding tasks:

**SemanticEncoder (Base Class) - Shared Responsibilities:**

- Load and cache sentence-transformers model (`all-MiniLM-L6-v2`, 384d)
- Generate deterministic projection matrix from seed
- Encode text inputs to 384d embeddings
- Project embeddings to target dimensions
- Configuration management and error handling

**Encoding Pipeline:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Canonicalized Input (Intent or Policy)                             │
├─────────────────────────────────────────────────────────────────────┤
│                   SemanticEncoder Base Class                        │
│  ────────────────────────────────────────────────────────────────   │
│  • Load sentence-transformers model                                │
│  • Initialize projection matrices (seed-based, deterministic)      │
│  • Encode extracted text → 384d embeddings                         │
│  • Project embeddings to target dimension                          │
└─────────────────────────────────────────────────────────────────────┘
        ↓                                        ↓
┌──────────────────────────────┐    ┌──────────────────────────────┐
│  IntentEncoder Subclass       │    │ PolicyEncoder Subclass       │
│ (Specialization Logic)        │    │ (Specialization Logic)       │
├──────────────────────────────┤    ├──────────────────────────────┤
│ • Extract 4 semantic slots    │    │ • Extract anchor lists       │
│ • Build slot text strings     │    │ • Build anchor term groups   │
│ • Aggregate to 128d vector    │    │ • Aggregate to 4×16×32      │
│                              │    │   RuleVector                 │
└──────────────────────────────┘    └──────────────────────────────┘
        ↓                                    ↓
   128d Vector                         RuleVector (anchors)
(intent representation)            (policy constraints)
```

**Intent Flow:**

```
IntentEvent (raw)
    ↓ Canonicalize
Canonical Intent (semantic slots filled with canonical terms)
    ↓ IntentEncoder.extract_slots()
4 slot text strings: ["action_text", "resource_text", "data_text", "risk_text"]
    ↓ SemanticEncoder.encode_text() [inherited]
4 × 384d embeddings
    ↓ SemanticEncoder.project_embedding() [inherited]
4 × 128d vectors
    ↓ IntentEncoder.aggregate()
Single 128d intent vector
```

**Policy Flow:**

```
Policy (raw)
    ↓ Canonicalize
Canonical Policy (anchor terms organized per dimension)
    ↓ PolicyEncoder.extract_slots()
4 anchor groups: [["anchor1", "anchor2", ...], ...]
    ↓ SemanticEncoder.encode_text() [inherited, called per anchor]
4 × 16 × 384d embeddings
    ↓ SemanticEncoder.project_embedding() [inherited]
4 × 16 × 32d vectors
    ↓ PolicyEncoder.aggregate()
RuleVector: 4 slots × 16 anchors × 32d
```

---

## Implementation Tasks

### Week 1: BERT Model, Hook System & Configuration

**Task 1: Seed Dataset Collection & Preparation** (12 hours)

- Collect public API specs (Stripe, GitHub, AWS, etc)
- Download ToolBench, API-Bank datasets
- Synthesize variations (template-based generation)
- Stratified sampling by canonical category
- Create 80/10/10 train/val/test split
- Output: ~50K-100K labeled examples

**Task 2: Build & Train BERT Classifier** (12 hours)

- Download TinyBERT from HuggingFace
- Create multi-head classification architecture
  - action_head: softmax(6 classes)
  - resource_type_head: softmax(5 classes)
  - sensitivity_head: softmax(3 classes)
- Implement training pipeline (AdamW, cross-entropy loss, 3-5 epochs)
- Validate: ≥95% accuracy per field, <10ms inference latency
- Save best checkpoint to `management_plane/models/canonicalizer_tinybert_v1.0/`

**Task 3: Create YAML Configurations** (3 hours)

- Create `config/hooks.yaml` (hook types, required fields)
- Create `config/extraction.yaml` (field mappings, inference rules)
- Create `config/canonicalization.yaml` (thresholds, logging config)
- Create `config/vocabulary.yaml` (canonical terms + examples)
- Load configs at startup with validation

**Task 4: Implement Field Extractor** (6 hours)

- Create `management_plane/app/services/field_extractor.py`
- Detect input format (structured vs natural language)
- Apply extraction mappings from extraction.yaml
- Handle fallbacks and defaults
- Support inference rules (regex patterns for tool_name → resource_type)

**Task 5: Implement Canonicalizer Service** (6 hours)

- Create `management_plane/app/services/canonicalizer.py`
- Load BERT model at startup (cached)
- Implement `canonicalize()` method
- Apply confidence thresholds
- Implement fail-behavior (passthrough for unknown terms)
- Logging: track predictions + confidence

**Task 6: Implement Hook Endpoint** (6 hours)

- Create `management_plane/app/endpoints/hooks.py`
- Implement `POST /v2/guard/enforce` endpoint
- Hook validation (hooks.yaml)
- Integration with field extractor + canonicalizer
- Integration with existing encoding pipeline
- Return HookEnforcementResult with trace info

**Task 7: Implement SemanticEncoder base class** (6 hours)

- Create `management_plane/app/services/semantic_encoder.py`
- Implement abstract `SemanticEncoder` class with:
  - Shared sentence-transformers model loading and caching
  - Seed-based deterministic projection matrix generation
  - `encode_text(text: str)` method for 384d embeddings
  - `project_embedding(embedding)` for dimension reduction
  - Configuration management (model_name, seed, target dimensions)
  - Abstract `build_slots()` method (to be overridden by subclasses)
  - Abstract `encode()` method (to be overridden by subclasses)
- Add comprehensive docstrings explaining shared responsibilities

**Task 8: Logging Infrastructure** (4 hours)

- Create `management_plane/app/services/canonicalization_logger.py`
- Async logging: predictions + confidence + outcomes
- Store in `/var/log/guard/canonicalization/` (JSON lines format)
- Log schema: timestamp, field, raw_input, prediction, confidence, enforcement_outcome
- Add retention policy (90 days)

### Week 2: Integration Testing & Encoder Implementation

**Task 9: Unit Tests for Canonicalizer & Extractor** (6 hours)

- Test field extraction (all source priorities)
- Test inference rules (regex matching)
- Test fallback behavior
- Test BERT predictions (mocked initially)
- Test confidence threshold logic

**Task 10: Integration Testing** (8 hours)

- End-to-end hook request flow
- Test canonicalization on diverse inputs
- Verify canonical fields are valid per IntentEvent schema
- Test error handling (invalid hook, missing required fields)
- Test response trace data
- Verify no regressions in existing enforcement

**Task 11: Implement IntentEncoder subclass** (3 hours)

- Create `IntentEncoder` class inheriting from `SemanticEncoder`
- Implement `extract_slots(canonical_intent)` to extract 4 semantic slots
- Implement `encode(canonical_intent)` to return 128d intent vector
- Add configuration for 128d target dimension
- Add unit tests for slot extraction and vector aggregation

**Task 12: Implement PolicyEncoder subclass** (3 hours)

- Create `PolicyEncoder` class inheriting from `SemanticEncoder`
- Implement `extract_slots(canonical_policy)` to extract anchor groups
- Implement `encode(canonical_policy)` to return RuleVector (4×16×32)
- Add configuration for 32d per-anchor dimension
- Add unit tests for anchor extraction and RuleVector generation

### Week 3: Production Logging & Learning Loop Setup

**Task 13: Production Logging Collection** (4 hours)

- Ensure canonicalization logger captures all signals
- Log enforcement outcomes alongside predictions
- Implement query API for logs (by field, date, confidence, tenant)
- Verify async logging doesn't impact latency

**Task 14: Curation Pipeline Setup** (6 hours)

- Create schema for curated examples (approved by human)
- Create admin interface for human review (web or CLI tool)
- Query low-confidence predictions from logs
- Flag enforcement mismatches
- Export approved examples as new training data

**Task 15: Retraining Automation** (6 hours)

- Create monthly retraining script
  - Load seed dataset + curated examples
  - Prepare train/val/test splits (80/10/10)
  - Fine-tune BERT from previous checkpoint (1-2 epochs)
  - Validate accuracy per field
  - A/B test: compare to baseline
  - Deploy new model if accuracy maintained/improved
- Create model versioning system (`v1.0` → `v1.1`, etc)

**Task 16: Monitoring & Observability** (6 hours)

- Dashboard: canonicalization accuracy per field over time
- Dashboard: confidence distribution across fields
- Dashboard: coverage (% of real-world vocabulary handled)
- Alerts: accuracy drops below threshold
- Logging: track model version in every prediction
- Metrics: count of inferred vs explicit fields
- Metrics: count of fallback fields used

**Task 17: Documentation & Examples** (4 hours)

- API documentation for `/v2/guard/enforce` endpoint
- Example curl commands for different hook types
- Example agent integration code (LangGraph, n8n, OpenAI SDK)
- Configuration guide for hooks.yaml and extraction.yaml
- BERT training pipeline documentation
- Troubleshooting guide for canonicalization issues

---

## Success Criteria

### Canonicalization Performance

- **Accuracy**: ≥95% on test set per field (action, resource_type, sensitivity)
- **Latency**: <10ms per canonicalization (CPU)
- **Inference scalability**: Handle 1000+ requests/sec on modest hardware

### Hook System

- **Framework agnostic**: Works with any agent (LangGraph, LangChain, n8n, custom)
- **Input flexibility**: Accepts structured, natural language, or hybrid intent format
- **Fail-safe defaults**: Unknown terms default to safe values, flagged in response
- **Observable**: Trace data shows canonicalization decisions for debugging

### Data Quality

- **Zero enforcement regressions**: No increase in false blocks vs baseline
- **Coverage**: Handles ≥95% of real-world vocabulary (from production)
- **Precision per category**: ≥90% precision for each canonical class

### Production Learning Loop

- **Data collection**: Log 100% of predictions with confidence + outcomes
- **Curation**: Monthly retraining with ≥500 curated examples per cycle
- **Model versioning**: Track versions with backward compatibility

### Encoder Architecture

- **Inheritance**: SemanticEncoder base class handles 100% of shared logic
- **Code reuse**: ≥60% reduction in code duplication (IntentEncoder + PolicyEncoder)
- **Consistency**: Both encoder subclasses produce deterministic outputs for identical canonical inputs
- **Testing**: 100% code coverage of SemanticEncoder base class, inheritance test suite passes

---

## Appendix: Data Collection Sources

### Public APIs (OpenAPI specs)


| Source       | Count          | Notes               |
| -------------- | ---------------- | --------------------- |
| Stripe API   | ~200 endpoints | Payment operations  |
| AWS API      | ~10K endpoints | Infrastructure      |
| GitHub API   | ~150 endpoints | Developer tools     |
| Google Cloud | ~1K endpoints  | Cloud services      |
| Twilio       | ~50 endpoints  | Communications      |
| Notion API   | ~30 endpoints  | Document management |
| Slack API    | ~100 endpoints | Chat/collaboration  |

**Collection**: Parse OpenAPI specs → extract verbs, nouns, parameters

### Public Datasets


| Dataset    | Size | Focus                 | License    |
| ------------ | ------ | ----------------------- | ------------ |
| ToolBench  | 1.6M | Tool use instructions | Apache 2.0 |
| API-Bank   | 9K   | API calling patterns  | MIT        |
| ToolAlpaca | 3.9K | LLM tool use          | CC-BY-NC   |

### Internal Sources (Post-Launch)

- Production IntentEvent logs (anonymized)
- Policy definitions from customers
- Error logs (failed canonicalizations)
