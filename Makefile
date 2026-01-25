.PHONY: help install test test-mgmt test-sdk clean run-mgmt build-rust lint format

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
	@echo "  make run-mgmt         Run management-plane server (dev mode)"
	@echo ""
	@echo "Building:"
	@echo "  make build-rust       Build Rust semantic-sandbox library"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters (when configured)"
	@echo "  make format           Format code (when configured)"

install:
	@echo "Installing Python dependencies..."
	uv sync --all-packages
	@echo "Building Rust component..."
	cd semantic-sandbox && cargo build --release
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
	rm -rf **/*.egg-info
	rm -rf .uv
	rm -f uv.lock
	@echo "✅ Cleaned!"

run-mgmt:
	@echo "Starting management-plane server..."
	cd management_plane && uv run uvicorn app.main:app --reload

build-rust:
	@echo "Building Rust semantic-sandbox..."
	cd semantic-sandbox && cargo build --release
	@echo "✅ Built: semantic-sandbox/target/release/libsemantic_sandbox.dylib"

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
