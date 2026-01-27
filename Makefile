.PHONY: help install test test-mgmt test-sdk clean run-mgmt run-data run-all run-mcp build-rust build-data lint format no-mcp

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LOG_DIR := $(ROOT)/data/logs
MCP_HEALTH_CHECK_TIMEOUT := 30
DATA_PLANE_HEALTH_CHECK_TIMEOUT := 30

ifneq (,$(filter no-mcp,$(MAKECMDGOALS)))
NO_MCP=1
endif

# Health check helpers
define wait_for_port
	@echo "⏳ Waiting for service on port $(1) (timeout: $(2)s)..."
	@for i in $$(seq 1 $(2)); do \
		if nc -z localhost $(1) 2>/dev/null; then \
			echo "✅ Service on port $(1) is ready"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "❌ Timeout waiting for port $(1) after $(2)s"; \
	exit 1
endef

define wait_for_mcp_server
	@echo "⏳ Waiting for MCP server on port 3001 (timeout: $(1)s)..."
	@for i in $$(seq 1 $(1)); do \
		if curl -s -H "Accept: text/event-stream" http://localhost:3001/mcp > /dev/null 2>&1; then \
			echo "✅ MCP server on port 3001 is ready"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "❌ Timeout waiting for MCP server after $(1)s"; \
	exit 1
endef

help:
	@echo "Semantic Security MVP - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install all dependencies (Python + Rust)"
	@echo "  make clean            Remove build artifacts and cache"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-mgmt        Run management-plane tests"
	@echo "  make test-sdk         Run Python SDK tests"
	@echo "  make test-rust        Run Rust tests"
	@echo ""
	@echo "Running:"
	@echo "  make run-mgmt         Run management-plane server (dev mode, port 8000)"
	@echo "  make run-mgmt PORT=9000  Run with custom port"
	@echo "  make run-data         Run data-plane server (port 50051)"
	@echo "  make run-mcp          Run MCP server (port 3001)"
	@echo "  make run-all          Run both management-plane AND data-plane"
	@echo "  make run-all PORT=9000   Run both with custom mgmt port"
	@echo "  make run-mgmt no-mcp   Run management-plane only (skip MCP)"
	@echo ""
	@echo "Building:"
	@echo "  make build-rust       Build Rust semantic-sandbox library"
	@echo "  make build-data       Build data-plane (bridge-server)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters (when configured)"
	@echo "  make format           Format code (when configured)"

install:
	@echo "Installing Python dependencies..."
	uv sync --all-packages
	@echo "Building Rust component..."
	cd semantic-sandbox && cargo build --release
	@echo "Building data-plane..."
	cd data_plane/tupl_dp/bridge && cargo build --release
	@echo "✅ Setup complete!"

test:
	@echo "Running all tests..."
	uv run pytest management-plane/tests/ -v
	cd semantic-sandbox && cargo test

test-mgmt:
	@echo "Running management-plane tests..."
	cd management-plane && uv run pytest tests/ -v

test-sdk:
	@echo "Running Python SDK tests..."
	cd tupl_sdk/python && uv run pytest tests/ -v

test-rust:
	@echo "Running Rust tests..."
	cd semantic-sandbox && cargo test

clean:
	@echo "Cleaning build artifacts..."
	rm -rf .venv
	rm -rf **/__pycache__
	rm -rf **/.pytest_cache
	rm -rf semantic-sandbox/target
	rm -rf data_plane/tupl_dp/bridge/target
	rm -rf **/*.egg-info
	rm -rf .uv
	rm -f uv.lock
	@echo "✅ Cleaned!"

run-mgmt:
	@echo "🚀 Starting management-plane server on port $(or $(PORT),8000)..."
	@mkdir -p $(LOG_DIR)
	@if [ "$(NO_MCP)" = "1" ]; then \
		echo "⏭️  MCP server disabled (no-mcp flag set)"; \
	else \
		echo "📍 Starting MCP server on port 3001..."; \
		(cd management_plane && uv run python -m mcp_server >> $(LOG_DIR)/mcp-server.log 2>&1) & \
		MCP_PID=$$!; \
		echo "   MCP server PID: $$MCP_PID"; \
		echo "   Logs: $(LOG_DIR)/mcp-server.log"; \
		$(call wait_for_mcp_server,$(MCP_HEALTH_CHECK_TIMEOUT)) || { kill $$MCP_PID 2>/dev/null; exit 1; }; \
	fi
	@echo "📍 Starting management-plane server..."
	cd management_plane && MGMT_PLANE_PORT=$(or $(PORT),8000) uv run uvicorn app.main:app --reload --host 0.0.0.0 --port $(or $(PORT),8000)

run-data:
	@echo "🚀 Starting data-plane server on port 50051..."
	@mkdir -p $(LOG_DIR)
	cd data_plane/tupl_dp/bridge && cargo run --bin bridge-server

run-mcp:
	@echo "🚀 Starting MCP server on port 3001..."
	@mkdir -p $(LOG_DIR)
	cd management_plane && uv run python -m mcp_server

run-all:
	@echo "🚀 Starting all services..."
	@echo "   - Data Plane:       port 50051"
	@echo "   - MCP Server:       port 3001"
	@echo "   - Management Plane: port $(or $(PORT),8000)"
	@echo ""
	@echo "📝 Logs will be written to:"
	@echo "   - Data Plane:       $(LOG_DIR)/data-plane.log"
	@echo "   - MCP Server:       $(LOG_DIR)/mcp-server.log"
	@echo "   - Management Plane: $(LOG_DIR)/management-plane.log"
	@echo ""
	@mkdir -p $(LOG_DIR)
	@trap 'echo "🛑 Shutting down all services..."; kill 0' EXIT; \
	echo "📍 Step 1/3: Starting data-plane on port 50051..."; \
	(cd data_plane/tupl_dp/bridge && MANAGEMENT_PLANE_URL=http://localhost:$(or $(PORT),8000)/api/v2 cargo run --bin bridge-server > $(LOG_DIR)/data-plane.log 2>&1) & \
	DATA_PLANE_PID=$$!; \
	$(call wait_for_port,50051,$(DATA_PLANE_HEALTH_CHECK_TIMEOUT)) || { kill $$DATA_PLANE_PID 2>/dev/null; exit 1; }; \
	echo ""; \
	echo "📍 Step 2/3: Starting MCP server on port 3001..."; \
	(cd management_plane && uv run python -m mcp_server >> $(LOG_DIR)/mcp-server.log 2>&1) & \
	MCP_PID=$$!; \
	$(call wait_for_mcp_server,$(MCP_HEALTH_CHECK_TIMEOUT)) || { kill $$MCP_PID 2>/dev/null; kill $$DATA_PLANE_PID 2>/dev/null; exit 1; }; \
	echo ""; \
	echo "📍 Step 3/3: Starting management-plane on port $(or $(PORT),8000)..."; \
	(cd management_plane && MGMT_PLANE_PORT=$(or $(PORT),8000) uv run uvicorn app.main:app --reload --host 0.0.0.0 --port $(or $(PORT),8000) >> $(LOG_DIR)/management-plane.log 2>&1) & \
	MGMT_PID=$$!; \
	echo ""; \
	echo "✅ All services started! Press Ctrl+C to stop."; \
	echo ""; \
	wait

build-rust:
	@echo "Building Rust semantic-sandbox..."
	cd semantic-sandbox && cargo build --release
	@echo "✅ Built: semantic-sandbox/target/release/libsemantic_sandbox.dylib"

build-data:
	@echo "Building data-plane (bridge-server)..."
	cd data_plane/tupl_dp/bridge && cargo build --release
	@echo "✅ Built: data_plane/tupl_dp/bridge/target/release/bridge-server"

lint:
	@echo "Linting (ruff not configured yet)..."
	# uv run ruff check .

format:
	@echo "Formatting (ruff not configured yet)..."
	# uv run ruff format .

# Convenience aliases
t: test
tm: test-mgmt
ts: test-sdk
tr: test-rust
i: install
c: clean
r: run-mgmt
rd: run-data
ra: run-all

no-mcp:
	@:
