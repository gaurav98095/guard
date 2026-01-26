.PHONY: help install test test-mgmt test-sdk clean run-mgmt run-data run-all run-mcp build-rust build-data lint format no-mcp

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LOG_DIR := $(ROOT)/data/logs

ifneq (,$(filter no-mcp,$(MAKECMDGOALS)))
NO_MCP=1
endif

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
	@echo "Starting management-plane server on port $(or $(PORT),8000)..."
	@mkdir -p $(LOG_DIR)
	@if [ "$(NO_MCP)" = "1" ]; then \
		echo "MCP server disabled"; \
	else \
		echo "Starting MCP server on port 3001..."; \
		(cd management_plane && uv run python -m mcp_server.server >> $(LOG_DIR)/mcp-server.log 2>&1) & \
	fi
	cd management_plane && MGMT_PLANE_PORT=$(or $(PORT),8000) uv run uvicorn app.main:app --reload --host 0.0.0.0 --port $(or $(PORT),8000)

run-data:
	@echo "Starting data-plane server on port 50051..."
	@mkdir -p $(LOG_DIR)
	cd data_plane/tupl_dp/bridge && cargo run --bin bridge-server

run-mcp:
	@echo "Starting MCP server on port 3001..."
	@mkdir -p $(LOG_DIR)
	cd management_plane && uv run python -m mcp_server

run-all:
	@echo "Starting management-plane ($(or $(PORT),8000)), data-plane (50051), and MCP (3001)..."
	@echo "Logs will be written to:"
	@echo "  - Management Plane: $(LOG_DIR)/management-plane.log"
	@echo "  - Data Plane:       $(LOG_DIR)/data-plane.log"
	@echo "  - MCP Server:       $(LOG_DIR)/mcp-server.log"
	@echo ""
	@mkdir -p $(LOG_DIR)
	@trap 'kill 0' EXIT; \
	(cd data_plane/tupl_dp/bridge && MANAGEMENT_PLANE_URL=http://localhost:$(or $(PORT),8000)/api/v2 cargo run --bin bridge-server > $(LOG_DIR)/data-plane.log 2>&1) & \
	sleep 3; \
	(cd management_plane && uv run python -m mcp_server >> $(LOG_DIR)/mcp-server.log 2>&1) & \
	sleep 1; \
	(cd management_plane && MGMT_PLANE_PORT=$(or $(PORT),8000) uv run uvicorn app.main:app --reload --host 0.0.0.0 --port $(or $(PORT),8000) >> $(LOG_DIR)/management-plane.log 2>&1) & \
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
