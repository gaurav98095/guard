#!/bin/bash
# MCP Gateway - Local Development Deployment Script
# Purpose: Deploy the full stack locally for development
# Prerequisites: Docker, Docker Compose
# Run from: deployment/gateway/ directory
# Usage: bash deploy-local.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect docker-compose command
if docker compose version &>/dev/null; then
  DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &>/dev/null; then
  DOCKER_COMPOSE="docker-compose"
else
  echo "❌ Error: Neither 'docker compose' nor 'docker-compose' found"
  echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
  exit 1
fi

echo "🚀 Deploying MCP Gateway Stack (Local Development)"
echo "Using: $DOCKER_COMPOSE"
echo "================================================"

# Check if .env.local file exists, otherwise use .env
if [ -f ".env.local" ]; then
  ENV_FILE=".env.local"
  echo "📄 Using .env.local"
elif [ -f ".env" ]; then
  ENV_FILE=".env"
  echo "📄 Using .env"
else
  echo "❌ Error: No .env.local or .env file found"
  echo "Please create .env.local from .env.local.example and add your credentials"
  exit 1
fi

# Source environment variables from gateway
set -a
source "$ENV_FILE"
set +a

# Source environment variables from Console if available
CONSOLE_ENV_FILE="../console/.env.local"
if [ -f "$CONSOLE_ENV_FILE" ]; then
  echo "📄 Loading Console environment variables from $CONSOLE_ENV_FILE..."
  set -a
  source "$CONSOLE_ENV_FILE"
  set +a
  echo "✅ Console environment variables loaded"
else
  echo "⚠️  Warning: $CONSOLE_ENV_FILE not found. Using default localhost URLs for Console"
  # Set default local URLs
  export VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://localhost:8000/api}"
fi

# Validate required environment variables
REQUIRED_VARS=(
  "GEMINI_API_KEY"
  "SUPABASE_URL"
  "SUPABASE_SERVICE_KEY"
  "SUPABASE_JWT_SECRET"
)

# VITE variables for Console build
VITE_VARS=(
  "VITE_SUPABASE_URL"
  "VITE_SUPABASE_ANON_KEY"
)

echo "🔍 Validating environment variables..."

# Check required variables
MISSING_REQUIRED=0
for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var}" ]; then
    echo "❌ Error: $var is not set in $ENV_FILE"
    MISSING_REQUIRED=1
  fi
done

if [ $MISSING_REQUIRED -eq 1 ]; then
  echo ""
  echo "Please add the missing variables to $ENV_FILE"
  exit 1
fi

# Check VITE variables (warn but don't fail)
MISSING_VITE=0
for var in "${VITE_VARS[@]}"; do
  if [ -z "${!var}" ]; then
    echo "⚠️  Warning: $var is not set (Console may not work correctly)"
    MISSING_VITE=1
  fi
done

if [ $MISSING_VITE -eq 0 ]; then
  echo "✅ All environment variables validated"
else
  echo "⚠️  Some VITE_* variables are missing - Console functionality may be limited"
fi


# Stop any existing containers
echo "🛑 Stopping any existing containers..."
$DOCKER_COMPOSE -f docker-compose.local.yml down || true

# Build and start Docker containers
echo "🐳 Building and starting Docker containers..."
echo "   This may take several minutes on first run..."
$DOCKER_COMPOSE -f docker-compose.local.yml up

# Wait for all services health check
echo "⏳ Waiting for services to become healthy..."
echo "   (This can take up to 2 minutes for first-time builds)"
for i in {1..60}; do
  SECURITY_STACK_HEALTHY=false
  CONSOLE_HEALTHY=false

  if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    SECURITY_STACK_HEALTHY=true
  fi

  if curl -f -s http://localhost:8080/ >/dev/null 2>&1; then
    CONSOLE_HEALTHY=true
  fi

  if [ "$SECURITY_STACK_HEALTHY" = true ] && [ "$CONSOLE_HEALTHY" = true ]; then
    echo "✅ All services are healthy!"
    break
  fi

  if [ $i -eq 60 ]; then
    echo "❌ Services failed to become healthy within 120 seconds (2 minutes)"
    echo "Security Stack: $SECURITY_STACK_HEALTHY, Console: $CONSOLE_HEALTHY"
    echo ""
    echo "Checking container status:"
    $DOCKER_COMPOSE -f docker-compose.local.yml ps
    echo ""
    echo "Recent logs from failing services:"
    if [ "$SECURITY_STACK_HEALTHY" != true ]; then
      echo "--- Security Stack Logs ---"
      $DOCKER_COMPOSE -f docker-compose.local.yml logs --tail=30 ai-security-stack
    fi
    if [ "$CONSOLE_HEALTHY" != true ]; then
      echo "--- Console Logs ---"
      $DOCKER_COMPOSE -f docker-compose.local.yml logs --tail=30 console-ui
    fi
    echo ""
    echo "Check full logs with: $DOCKER_COMPOSE -f docker-compose.local.yml logs"
    exit 1
  fi

  # Show progress every 10 iterations
  if [ $((i % 10)) -eq 0 ]; then
    echo "   Still waiting... (Security: $SECURITY_STACK_HEALTHY, Console: $CONSOLE_HEALTHY) - ${i}s elapsed"
  fi

  sleep 2
done

# Show service status
echo ""
echo "================================================"
echo "✅ Local Deployment Complete!"
echo "================================================"
echo ""
echo "Service Status:"
$DOCKER_COMPOSE -f docker-compose.local.yml ps
echo ""
echo "🌐 Service URLs (all accessible from your browser):"
echo "  Console:             http://localhost:8080"
echo "  Management Plane:    http://localhost:8000/api/v2"
echo "  ChromaDB:            http://localhost:8002"
echo "  Data Plane gRPC:     localhost:50051"
echo ""
echo "📊 Health Endpoints:"
echo "  Management Plane:    http://localhost:8000/health"
echo ""
echo "🔧 Useful commands:"
echo "  View logs (all):     $DOCKER_COMPOSE -f docker-compose.local.yml logs -f"
echo "  View logs (service): $DOCKER_COMPOSE -f docker-compose.local.yml logs -f ai-security-stack"
echo "  Stop services:       $DOCKER_COMPOSE -f docker-compose.local.yml down"
echo "  Restart:             bash deploy-local.sh"
echo "  Shell into service:  docker exec -it ai-security-stack-local bash"
echo ""
echo "🐛 Debugging:"
echo "  Check all logs:      $DOCKER_COMPOSE -f docker-compose.local.yml logs"
echo "  Check specific:      $DOCKER_COMPOSE -f docker-compose.local.yml logs ai-security-stack"
echo "  Container status:    $DOCKER_COMPOSE -f docker-compose.local.yml ps"
echo ""
echo "================================================"
echo "🎉 Your local stack is ready! Visit http://localhost:8080 to get started"
echo "================================================"
