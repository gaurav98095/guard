# Guard MCP Server & Client Implementation Plan

**Created:** 2025-01-26
**Status:** Draft (Validated against FastMCP 2.x API)
**Owner:** Sid
**Priority:** High
**SDK:** FastMCP 2.x (production-ready, stable)
**Last Validated:** 2025-01-26

---

## 1. Executive Summary

This plan describes the implementation of a **Model Context Protocol (MCP) server and client** that integrates with Guard's existing `/api/v2/enforce` endpoint. The goal is to enable agents (built with any framework) to perform **prompt-driven, intent-based enforcement** without requiring framework-specific integrations.

### Key Design Principles

1. **Framework-agnostic**: No dependency on LangChain, OpenAI ADK, or other agent frameworks
2. **Prompt-driven enforcement**: The agent's own LLM extracts and validates intent before tool execution
3. **Horizontal scalability**: Single MCP server serves unlimited agents/customers via standard HTTP transport
4. **Non-invasive integration**: Customers only need to:
   - Connect to the MCP server
   - Inject a system prompt (provided by Guard)
   - Call `send_intent()` before tools (guided by the prompt)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Customer's Agent Runtime                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Agent (any framework)                                        │   │
│  │  • LangChain, OpenAI ADK, Anthropic, n8n, custom loop, etc.   │   │
│  │  • System prompt guides intent-first execution                │   │
│  │  • Calls send_intent() before any tool invocation             │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │ MCP (Streamable HTTP)                 │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                    Guard MCP Server (Python)                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  MCP Tools:                                                   │   │
│  │  • send_intent(action, resource, data, risk, context)        │   │
│  │  • explain_denial(request_id)                                │   │
│  │                                                               │   │
│  │  MCP Prompts:                                                │   │
│  │  • governed_agent_instructions (system prompt)               │   │
│  │                                                               │   │
│  │  MCP Resources:                                              │   │
│  │  • policy://summary (optional, for reference)                │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │ HTTP to Management Plane               │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│               Management Plane (/api/v2/enforce)                     │
│  • Canonicalization (BERT)                                           │
│  • Intent Encoding                                                   │
│  • Data Plane Enforcement                                            │
│  • Decision: ALLOW / DENY                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Objectives & Scope

### In Scope

1. **MCP Server** (Python, `management_plane/mcp_server/`)

   - Expose `send_intent()` tool
   - Expose `explain_denial()` tool (optional)
   - Expose `governed_agent_instructions` prompt
   - Streamable HTTP transport for remote agents
   - Integration with existing `/api/v2/enforce` endpoint
   - Multi-tenant support via API keys
2. **MCP Client SDK** (Python, separate package)

   - Standard MCP client (zero-config tool discovery)
   - Works with any MCP-compatible agent framework
   - Tools auto-appear when agent connects (like Claude Code/Cursor)
   - For custom agents: thin connection wrapper for manual tool discovery
3. **Documentation & Examples**

   - System prompt template for agents
   - Integration examples (OpenAI, Anthropic, LangChain, custom)
   - Deployment guide (docker-compose, environment variables)

### Out of Scope

1. MCP clients for other languages (TypeScript, Go, etc.) in this phase
2. Tool interception/wrapping (agents must call `send_intent()` explicitly via prompts)
3. Data Plane modifications
4. New canonicalization models or changes to BERT pipeline
5. MCP UI components

### Success Criteria

- [ ] MCP server starts alongside Management Plane
- [ ] Agents connect to MCP server and call `send_intent()`
- [ ] Intent is sent to `/api/v2/enforce`, canonicalized, and enforced
- [ ] Decision (ALLOW/DENY) is returned to agent
- [ ] Agent responds appropriately (proceed, deny, or retry)
- [ ] No framework-specific code required
- [ ] Multi-tenant support via API key authentication
- [ ] Integration examples for 3+ frameworks

---

## 2.5 SDK Selection: FastMCP 2.x

**Decision**: Use **FastMCP 2.14.4** instead of official MCP Python SDK v2.

**Rationale**:

- Official MCP SDK v2 is pre-alpha (not recommended for production until Q1 2026)
- FastMCP 2.x is production-ready (1M+ downloads/day, powers 70% of MCP servers)
- Nearly identical decorator-based API
- Built-in enterprise authentication (JWT, OAuth, API keys)
- Superior tooling (testing utilities, CLI, server composition)

**API Differences**:

- Import: `from fastmcp import FastMCP` (vs `from mcp.server.mcpserver import MCPServer`)
- Transport: `transport="http"` (vs `transport="streamable-http"`)
- Tool decorator: `@mcp.tool` (vs `@mcp.tool()` - parentheses optional)

**Migration Risk**: Low. If official SDK v2 becomes preferred, migration is ~2-4 hours of import/decorator changes.

---

## 3. Architecture Details

### 3.1 MCP Server Components

#### Tool: `send_intent()`

**Purpose**: Primary enforcement point. Agent submits intent before tool execution.

**Input Schema** (Pydantic):

```python
class SendIntentRequest(BaseModel):
    action: str                          # "read", "write", "query", "send", etc.
    resource: Dict[str, str]             # {"type": "database|file|api", "name": "...", "location": "local|cloud"}
    data: Dict[str, Any]                 # {"sensitivity": [...], "pii": bool, "volume": "single|bulk"}
    risk: Dict[str, str]                 # {"authn": "required|not_required"}
    context: Optional[Dict[str, Any]]    # Metadata (tool_name, tool_method, tool_params, etc.)
```

**Output Schema** (Pydantic):

```python
class SendIntentResponse(BaseModel):
    decision: Literal["ALLOW", "DENY"]
    request_id: str                      # UUID for tracing/audit
    rationale: str                       # Human-friendly reason (always included)
    enforcement_latency_ms: float
    metadata: Dict[str, Any]             # Canonicalization trace, evidence, etc.
```

**Flow**:

1. MCP server receives `send_intent()` call
2. Extracts tenant from MCP client context (via API key in headers)
3. Constructs `LooseIntentEvent` from input
4. Calls `POST /api/v2/enforce` on Management Plane (colocated)
5. Returns `{decision, request_id, rationale, metadata}` to agent
6. Agent logs the decision and acts accordingly

#### Tool: `explain_denial()` (Optional)

**Purpose**: Help agent understand why an intent was denied.

**Input Schema**:

```python
class ExplainDenialRequest(BaseModel):
    request_id: str
```

**Output Schema**:

```python
class ExplainDenialResponse(BaseModel):
    rationale: str
    suggested_alternatives: List[str]
    policy_reference: Optional[str]
```

#### Prompt: `governed_agent_instructions`

**Purpose**: System prompt that guides agent to call `send_intent()` before tools.

**Content** (template):

```
You are an AI agent with strict security policies.

IMPORTANT: Before calling ANY tool or taking any action that accesses data,
systems, or performs side effects, you MUST:

1. Analyze what you're about to do:
   - WHAT action? (read, write, delete, export, execute, update)
   - WHAT resource? (database, file, api, etc.)
   - WHAT data sensitivity? (internal, public, has PII, volume, etc.)
   - WHAT risk level? (authentication required or not)

2. Call the send_intent() tool with these parameters:
   {
     "action": "<action>",
     "resource": {"type": "<type>", "name": "<name>", "location": "cloud"},
     "data": {"sensitivity": ["<level>"], "pii": <bool>, "volume": "<volume>"},
     "risk": {"authn": "required"}
   }

3. Wait for the response. It will be either:
   - ALLOW: Safe to proceed with your tool call
   - DENY: You cannot perform this action. Explain to the user and suggest alternatives

Example:
  I want to query the customer database for email addresses (has PII).
  → Call send_intent(action="read", resource={type: "database", name: "customers"},
                     data={sensitivity: ["internal"], pii: true, volume: "bulk"})
  → Response: DENY (policy only allows singular access for email containing PII)
  → Tell user: "I cannot access bulk customer email. I can help you contact our support team."

RULE: Never skip send_intent(). If you do, your action may be audited/blocked later.
```

#### Resource: `policy://summary` (Optional)

**Purpose**: Return human-readable policy summary for current tenant. All policies are only ALLOW (whitelisted) policies. Guard's stack has fail-closed behavior by default.

**Output**:

```
Policies for Tenant: acme-corp
- Allow: Employees can read internal databases (non-PII, single records)
- Allow: Service accounts can write logs to S3
```

### 3.2 Authentication & Tenancy

**Auth Model**: API Key per tenant

1. **API Key provisioning** (via Management Plane):
   Note: The API Keys will be generated on the client-facing UI/application. That UI/application will call management plane APIs to store the API keys in the `api_keys` table and this gets used for MCP Server auth.

   - Customer gets API key during onboarding
   - Stored securely in Supabase (`api_keys` table)
   - Can be revoked or rotated
2. **MCP Server auth** (per request):

   - MCP client sends API key via HTTP header: `Authorization: Bearer <api_key>`
   - MCP server validates key against Management Plane's auth endpoint (or local cache)
   - Extracts `tenant_id` from API key metadata
   - Sets `event.tenantId = tenant_id` in all `send_intent()` calls
3. **Rate limiting** (optional):

   - Per-tenant rate limit on `send_intent()` calls (e.g., 200/min)
   - Track in Management Plane or locally with TTL cache

### 3.3 Deployment

**Location**: Colocated with Management Plane

```
┌─────────────────────────────────────────────────────────┐
│  Single Container/Process                               │
│  ┌──────────────────────┐     ┌──────────────────────┐  │
│  │  Management Plane    │     │  MCP Server          │  │
│  │  • /api/v1/*         │────▶│  • MCP HTTP (port 3001)
│  │ • /api/v2/enforce    │     │  • Shared auth       │  │
│  │  • Port :8000        │     │  • Shared DB client  │  │
│  └──────────────────────┘     └──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Or**: Separate container in same docker-compose with internal HTTP link.

**Benefits**:

- Simplified deployment
- Shared Supabase connection, BERT model cache, Data Plane gRPC client
- No additional infrastructure

### 3.4 Error Handling

**Scenarios**:

1. **Invalid API key**: Return 401 Unauthorized
2. **Tenant not found**: Return 404 or 403 Forbidden
3. **Malformed intent**: Return 400 Bad Request with field errors
4. **Management Plane unavailable**: Return 503 Service Unavailable (fail closed)
5. **Rate limit exceeded**: Return 429 Too Many Requests

### 3.5 Client Integration Model

#### The "Claude Code Experience"

Guard MCP integration works **exactly like adding an MCP server to Claude Code or Cursor**:

1. Business client adds Guard MCP server URL to their agent config
2. Tools (`send_intent`, `explain_denial`) **auto-appear** in the agent via MCP protocol
3. System prompt guides agent to call `send_intent` before business tools
4. **Zero code changes required**

#### Integration Paths

**Path A: MCP-Native Agents** (Recommended - Zero Code)

For agents with built-in MCP support (Claude Desktop, Cursor, LangChain with MCP plugin, etc.):

**Step 1:** Add to MCP config file:

```json
{
  "mcpServers": {
    "guard": {
      "url": "https://guard.company.com/mcp",
      "headers": {"Authorization": "Bearer YOUR_API_KEY"}
    }
  }
}
```

**Step 2:** Restart agent. Tools now available automatically.

**That's it!** No Python code, no framework integration, no tool registration.

**Path B: Custom Agents** (Manual MCP Connection)

For custom agents without native MCP support:

```python
from guard_mcp_client import GuardMCPClient

async def my_custom_agent():
    async with GuardMCPClient(api_key="sk-...") as mcp:
        # Discover tools automatically
        tools = await mcp.list_tools()
        # Returns: [{"name": "send_intent", "description": "...", ...}]
  
        # Get system prompt
        prompt = await mcp.get_prompt("governed_agent_instructions")
  
        # Call tools
        result = await mcp.call_tool("send_intent", {...})
```

#### What Business Clients Must Do

**Minimal Requirements:**

1. **Get API Key** from Guard console
2. **Add MCP Config**:
   - MCP-native agents: Add Guard URL to config file
   - Custom agents: `pip install guard-mcp-client`
3. **Restart Agent** - tools auto-discover

**No Code Changes:**

- ❌ No manual tool registration
- ❌ No framework-specific adapters
- ❌ No schema definitions
- ✅ Just connect - tools appear via MCP protocol

---

## 4. Implementation Plan

### Phase 1: MCP Server Core (Weeks 1-2)

#### Implementation Notes

- Dependencies live in `management_plane/pyproject.toml`: `fastmcp<3`, `PyYAML>=6.0` (repo uses `uv`, no `requirements.txt`).
- MCP entrypoint is `management_plane/mcp_server/server.py` (`transport="http"`, env-configurable host/port/path).
- Auth uses `app.auth.validate_api_key` with `Authorization: Bearer <api_key>`.
- `send_intent` builds `LooseIntentEvent` with `tenantId` from API key, `actor=llm-agent`, and optional `layer` from `context`.
- Enforcement call: `POST /api/v2/enforce` via `MANAGEMENT_PLANE_URL`, forwarding the same Authorization header.
- Prompt content is stored in YAML under `management_plane/mcp_server/prompts/` and loaded by `prompts.py`.

#### Task 1.1: Set up MCP server project structure

**Files**:

- Create `management_plane/mcp_server/` directory
- Create `management_plane/mcp_server/__init__.py`
- Create `management_plane/mcp_server/server.py` (main MCP server)
- Create `management_plane/mcp_server/tools.py` (tool implementations)
- Create `management_plane/mcp_server/prompts.py` (prompt templates)
- Create `management_plane/mcp_server/auth.py` (authentication logic)
- Update `management_plane/requirements.txt` to include `mcp` package

**What it does**:

- Adds MCP SDK as dependency
- Scaffolds directory structure

**Dependencies**:

- `fastmcp<3` (FastMCP 2.x - production-ready, pin to avoid v3 beta)
- Existing: `fastapi`, `pydantic`, `supabase-py`

**Estimate**: 4 hours

---

#### Task 1.2: Implement `send_intent()` tool

**Files**:

- Modify `management_plane/mcp_server/tools.py`
- Modify `management_plane/mcp_server/server.py`

**What it does**:

1. Define MCP tool schema for `send_intent()`
2. Implement tool handler:
   - Extract tenant from MCP client context (API key validation)
   - Build `LooseIntentEvent` from input
   - Call Management Plane's `/api/v2/enforce` endpoint
   - Format response as `SendIntentResponse`
3. Add comprehensive logging for debugging

**Pseudocode**:

```python
async def handle_send_intent(action: str, resource: Dict, data: Dict, 
                             risk: Dict, context: Optional[Dict] = None) -> Dict:
    # 1. Get tenant from request context
    tenant_id = get_current_tenant_from_mcp_context()
  
    # 2. Build LooseIntentEvent
    loose_event = LooseIntentEvent(
        id=str(uuid.uuid4()),
        tenantId=tenant_id,
        timestamp=time.time(),
        actor=Actor(id="llm-agent", type="agent"),
        action=action,
        resource=LooseResource(**resource),
        data=LooseData(**data),
        risk=Risk(**risk),
        context=context,
        layer="L4"  # Agent tool use layer
    )
  
    # 3. Call /api/v2/enforce (same process or HTTP)
    response = await call_enforce_endpoint(loose_event)
  
    # 4. Format and return
    return {
        "decision": "ALLOW" if response.decision == 1 else "DENY",
        "request_id": response.metadata.get("request_id"),
        "rationale": build_human_rationale(response),
        "enforcement_latency_ms": response.enforcement_latency_ms,
        "metadata": response.metadata
    }
```

**Tests**:

- Unit test: Valid input → ALLOW response
- Unit test: Valid input → DENY response
- Unit test: Invalid API key → 401
- Unit test: Malformed input → 400

**Estimate**: 12 hours

---

#### Task 1.3: Implement `explain_denial()` tool (optional, can defer)

**Files**:

- Modify `management_plane/mcp_server/tools.py`

**What it does**:

- Takes `request_id` from a previous denial
- Looks up decision in audit logs or cache
- Returns human-friendly explanation + suggested alternatives

**Estimate**: 6 hours (defer to Phase 3)

---

#### Task 1.4: Implement `governed_agent_instructions` prompt

**Files**:

- Modify `management_plane/mcp_server/prompts.py`

**What it does**:

1. Define MCP prompt template
2. Customize per tenant (if needed)
3. Return markdown-formatted prompt text

**Estimate**: 4 hours

---

#### Task 1.5: Implement MCP server main loop

**Files**:

- Modify `management_plane/mcp_server/server.py`

**What it does**:

1. Initialize MCP server with Streamable HTTP transport
2. Register tools (send_intent, explain_denial)
3. Register prompts (governed_agent_instructions)
4. Add request/response logging middleware
5. Add authentication interceptor

**Pseudocode**:

```python
from fastmcp import FastMCP, Context
from fastmcp.server.auth.providers.jwt import JWTVerifier

# Set up authentication
auth = JWTVerifier(
    jwks_uri="https://guard-auth.com/.well-known/jwks.json",
    issuer="https://guard-auth.com",
    audience="guard-mcp-server"
)

# Initialize MCP server
mcp = FastMCP(name="guard-mcp-server", auth=auth)

# Register tools via decorators
@mcp.tool
async def send_intent(
    action: str,
    resource: dict,
    data: dict,
    risk: dict,
    ctx: Context,  # Auto-injected by FastMCP
    context: Optional[dict] = None
) -> dict:
    """Send intent for enforcement checking."""
    # Extract tenant from authenticated JWT claims
    tenant_id = ctx.request_context.user.get("tenant_id")
  
    # Build LooseIntentEvent
    loose_event = LooseIntentEvent(...)
  
    # Call /api/v2/enforce
    response = await call_enforce_endpoint(loose_event)
  
    # Return decision
    return {
        "decision": "ALLOW" if response.decision == 1 else "DENY",
        "request_id": response.metadata["request_id"],
        "rationale": build_human_rationale(response),
        "enforcement_latency_ms": response.enforcement_latency_ms,
        "metadata": response.metadata
    }

# Register prompts via decorators
@mcp.prompt()
def governed_agent_instructions() -> str:
    """System prompt for governed agents."""
    return """
    You are an AI agent with strict security policies...
    """

# Run server with HTTP transport
if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=3001, path="/mcp")
```

**Estimate**: 10 hours

---

#### Task 1.6: Integration test - MCP server with Management Plane

**Files**:

- Create `tests/test_mcp_server_integration.py`

**What it does**:

1. Start Management Plane (if not running)
2. Start MCP server
3. Connect as MCP client
4. Send test `send_intent()` calls
5. Verify ALLOW/DENY responses

**Estimate**: 6 hours

---

### Phase 2: MCP Client SDK (Weeks 2-3)

#### Task 2.1: Create Guard MCP Client package

**Files**:

- Create `guard_mcp_client/` (pip installable package)
- Create `guard_mcp_client/__init__.py`
- Create `guard_mcp_client/client.py`
- Create `guard_mcp_client/config_templates/` (MCP config examples)
- Create setup.py / pyproject.toml

**What it does**:

- Standard MCP client wrapper (NOT framework-specific)
- Connection management only - tools auto-discover via MCP protocol
- Config templates for popular MCP hosts (Claude Desktop, Cursor, Continue.dev)
- Zero custom tool logic - everything comes from server

**Key Design Principle**:
Client is like adding a URL to Claude Code settings - connect and tools appear.
No manual tool registration, no framework adapters.

**Estimate**: 4 hours

---

#### Task 2.2: Implement client connect & auth

**Files**:

- Modify `guard_mcp_client/client.py`

**What it does**:

```python
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import asynccontextmanager

class GuardMCPClient:
    """
    Standard MCP client for Guard enforcement.
  
    Works like adding Guard to Claude Code/Cursor - just connect and 
    tools appear automatically in any MCP-compatible agent.
  
    For agents with native MCP support (LangChain, AutoGen, etc.):
      - Add Guard URL to agent's MCP config
      - Tools auto-discover (send_intent, explain_denial, etc.)
      - No code changes needed
  
    For custom agents without MCP support:
      - Use this client to connect manually
      - Discover tools via list_tools()
      - Call tools via call_tool()
    """
  
    def __init__(self, server_url: str = "https://guard.company.com/mcp", api_key: str = None):
        self.server_url = server_url
        self.api_key = api_key
  
    @asynccontextmanager
    async def connect(self):
        """
        Connect to Guard MCP server.
  
        Returns MCP session with auto-discovered tools.
        """
        # For HTTP transport with auth
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
  
        async with sse_client(self.server_url, extra_headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
  
    async def __aenter__(self):
        """Context manager support."""
        self._context = self.connect()
        self.session = await self._context.__aenter__()
        return self.session
  
    async def __aexit__(self, *args):
        await self._context.__aexit__(*args)
```

**Estimate**: 4 hours

---

#### Task 2.3: Create MCP integration examples

**Files**:

- Create `examples/mcp_config/claude_desktop.json`
- Create `examples/mcp_config/cursor.json`
- Create `examples/mcp_config/langgraph.yaml`
- Create `examples/custom_agent.py` (for non-MCP frameworks)
- Create `examples/README.md`

**What each example shows**:

1. **MCP Config Examples** (preferred for most users):

   - Claude Desktop / Cursor / Continue.dev config
   - LangChain MCP config
   - AutoGen MCP config
   - **No code needed** - just config!
2. **Custom Agent Example** (for non-MCP frameworks):

   - Manual MCP connection
   - Tool discovery via `list_tools()`
   - Tool invocation via `call_tool()`

**Example (Claude Desktop config)**:

```json
{
  "mcpServers": {
    "guard-enforcement": {
      "url": "http://<GUARD_URL>:<PORT>/mcp",
      "headers": {
        "Authorization": "Bearer sk-guard-YOUR_API_KEY"
      }
    }
  }
}
```

**That's it!** Tools appear automatically. No Python code needed.

**Estimate**: 6 hours

---

### Phase 3: Documentation & Deployment (Week 4)

#### Task 3.1: Write deployment guide

**Files**:

- Create `docs/plans/MCP_DEPLOYMENT.md`

**Content**:

- Architecture overview
- Environment variables
- Docker-compose configuration
- Kubernetes manifests (optional)
- Troubleshooting

**Estimate**: 6 hours

---

#### Task 3.2: Create comprehensive examples README

**Files**:

- Enhance `examples/README.md`

**Content**:

- Table of supported frameworks
- Quick start for each
- Common integration patterns
- Troubleshooting

**Estimate**: 4 hours

---

#### Task 3.3: Update main README

**Files**:

- Update `README.md` in root

**What to add**:

- MCP integration section
- Link to MCP deployment docs
- Link to examples

**Estimate**: 2 hours

---

#### Task 3.4: End-to-end integration test

**Files**:

- Create `tests/test_e2e_mcp_agent.py`

**What it does**:

1. Start all services (Management Plane, MCP Server, Data Plane, Chroma)
2. Connect as agent via Guard MCP Client
3. Send `send_intent()` for allowed action → verify ALLOW
4. Send `send_intent()` for denied action → verify DENY
5. Verify audit logs are created

**Estimate**: 8 hours

---

### Phase 4: Hardening & Production (Week 5, optional)

#### Task 4.1: Add `explain_denial()` tool

**Estimate**: 6 hours

---

#### Task 4.2: Add MCP resources (`policy://summary`)

**Estimate**: 4 hours

---

#### Task 4.3: Add comprehensive error handling

**Estimate**: 6 hours

---

#### Task 4.4: Performance testing & optimization

**Estimate**: 8 hours

---

## 5. Technical Specifications

### 5.1 Data Models

#### SendIntentRequest

```python
class SendIntentRequest(BaseModel):
    action: str                          # Flexible to allow non-canonical
    resource: Dict[str, str]             # {"type": str, "name": str, "location": str}
    data: Dict[str, Any]                 # {"sensitivity": list, "pii": bool, "volume": str}
    risk: Dict[str, str]                 # {"authn": str}
    context: Optional[Dict[str, Any]]    # Optional metadata
```

#### SendIntentResponse

```python
class SendIntentResponse(BaseModel):
    decision: Literal["ALLOW", "DENY"]
    request_id: str                      # UUID
    rationale: str                       # Human-readable explanation
    enforcement_latency_ms: float
    metadata: Dict[str, Any]             # Canonicalization trace, evidence
```

### 5.2 Environment Variables

```bash
# MCP Server
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=3001
MCP_SERVER_TRANSPORT=streamable-http  # or stdio for testing

# Connection to Management Plane
MANAGEMENT_PLANE_URL=http://localhost:8000
MANAGEMENT_PLANE_API_KEY=...          # Service-to-service auth (optional)

# MCP Authentication
MCP_API_KEY_VALIDATION_ENABLED=true   # Validate against Supabase api_keys table

# Logging
LOG_LEVEL=INFO
MCP_LOG_REQUESTS=false                # Log all request/response
```

### 5.3 API Endpoints

All endpoints are MCP tool calls, but for reference:

**HTTP Calls Made by MCP Server**:

```
POST /api/v2/enforce
  Headers: X-Tenant-Id: <tenant_id>
  Body: LooseIntentEvent
  Response: EnforcementResponse { decision, enforcement_latency_ms, metadata }
```

---

## 6. Testing Strategy

### Unit Tests

- `test_send_intent_tool.py`: Test tool schema, input validation, output formatting
- `test_auth.py`: Test API key validation
- `test_prompts.py`: Test prompt rendering

### Integration Tests

- `test_mcp_server_integration.py`: MCP server + Management Plane
- `test_mcp_client_integration.py`: Client connecting to server

### End-to-End Tests

- `test_e2e_mcp_agent.py`: Full stack (agent → MCP client → MCP server → Management Plane → Data Plane)

### Performance Tests

- Load test with 100 concurrent agents
- Measure `send_intent()` latency (target: <100ms p99)

---

## 7. Risks & Mitigations


| Risk                         | Impact | Mitigation                                                          |
| ------------------------------ | -------- | --------------------------------------------------------------------- |
| MCP SDK instability          | High   | Use official SDK, pin to stable version, monitor updates            |
| Agent prompt compliance      | Medium | Strong prompt engineering, add examples, document fallback behavior |
| Multi-tenant auth bypass     | High   | Strict API key validation, audit logging, security review           |
| Performance under load       | Medium | Load testing, caching, rate limiting                                |
| Management Plane unavailable | High   | Fail closed (deny all intents), health checks, circuit breaker      |

---

## 8. Success Metrics

1. **Functional**:

   - [ ] MCP server starts and serves `send_intent()` tool
   - [ ] Agent calls `send_intent()`, receives ALLOW/DENY decision
   - [ ] Decision is enforced (agent respects DENY)
   - [ ] Multi-tenant isolation verified
2. **Performance**:

   - [ ] `send_intent()` latency < 100ms p99
   - [ ] Support 1000+ concurrent agents
   - [ ] Memory footprint < 500MB
3. **UX**:

   - [ ] Integration examples work out-of-box
   - [ ] System prompt is clear and followed by agents
   - [ ] Documentation is complete and clear
4. **Security**:

   - [ ] All requests authenticated via API key
   - [ ] Tenant isolation verified
   - [ ] Audit logs for all enforcement decisions
   - [ ] No secrets in logs or examples

---

## 9. Deliverables

### Phase 1 (Weeks 1-2)

- [ ] MCP server running colocated with Management Plane
- [ ] `send_intent()` tool fully functional
- [ ] Initial integration tests passing
- [ ] Logging and error handling working

### Phase 2 (Weeks 2-3)

- [ ] Guard MCP Client package created and published
- [ ] Integration examples for 3+ frameworks
- [ ] Client documentation complete
- [ ] All tests passing

### Phase 3 (Week 4)

- [ ] Deployment guide complete
- [ ] Docker-compose configurations updated
- [ ] End-to-end tests passing
- [ ] README updated with MCP section

### Phase 4 (Week 5, optional)

- [ ] `explain_denial()` tool implemented
- [ ] MCP resources added
- [ ] Comprehensive error handling
- [ ] Performance tests passing

---

## 10. Timeline


| Phase                   | Duration       | Owner | Status      |
| ------------------------- | ---------------- | ------- | ------------- |
| 1: MCP Server Core      | 2 weeks        | TBD   | Not Started |
| 2: Client SDK           | 1 week         | TBD   | Not Started |
| 3: Documentation        | 1 week         | TBD   | Not Started |
| 4: Hardening (optional) | 1 week         | TBD   | Not Started |
| **Total**               | **~4-5 weeks** |       |             |

---

## 11. Next Steps

1. **Review & Approval**: Get stakeholder sign-off on this plan
2. **Assign Owner**: Designate engineering lead
3. **Create Subtasks**: Break Phase 1 into JIRA/GitHub Issues
4. **Begin Phase 1, Task 1.1**: Set up MCP server project structure

---

## Appendix A: SDK & Specification References

### FastMCP (Chosen Implementation)

- [FastMCP Official Documentation](https://gofastmcp.com)
- [FastMCP GitHub Repository](https://github.com/jlowin/fastmcp)
- [FastMCP Examples](https://github.com/jlowin/fastmcp/tree/main/examples)
- [Authentication Guide](https://gofastmcp.com/docs/authentication)
- Version: 2.14.4 (stable, production-ready)

### MCP Protocol Specification

- [Model Context Protocol Overview](https://modelcontextprotocol.io)
- [MCP Specification (2025-06-18)](https://modelcontextprotocol.io/specification/2025-06-18)
- [Official Python SDK (v2 pre-alpha - not used)](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Clients (Claude Desktop, Cursor, etc.)](https://modelcontextprotocol.io/clients)

---

## Appendix B: Guard-Specific References

- Existing `/api/v2/enforce` endpoint: `management_plane/app/endpoints/enforcement_v2.py:337`
- Authentication: `management_plane/app/auth.py:229`
- Models: `management_plane/app/models.py:163` (LooseIntentEvent)
- Intent encoding: `management_plane/app/services/intent_encoder.py`
