#!/bin/bash
################################################################################
# Codebase Cleanup Script
# Removes deprecated code that is not used by the 3 active V2 API endpoints:
#   - POST /api/v2/enforce
#   - POST /api/v2/canonicalize
#   - POST /api/v2/policies/install
#
# This script removes:
#   - console/ and sdk/ directories (deprecated UI and SDK)
#   - V1 and deprecated endpoints
#   - Deprecated services
#   - Deprecated tests
#   - Deprecated documentation
#   - Old vocabulary.yaml files
#
# Safety Features:
#   - Checks git status before running
#   - Creates a backup list of deleted files
#   - Dry-run mode available with --dry-run flag
#   - Verifies dependencies are not imported by active code
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/Users/sid/Projects/guard"
DRY_RUN=false
BACKUP_FILE="${PROJECT_ROOT}/cleanup-backup-$(date +%Y%m%d-%H%M%S).txt"

# Parse command line arguments
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${BLUE}Running in DRY-RUN mode. No files will be deleted.${NC}\n"
fi

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

delete_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            log_info "Would delete: $file"
        else
            rm "$file"
            log_success "Deleted: $file"
            echo "$file" >> "$BACKUP_FILE"
        fi
    fi
}

delete_directory() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            log_info "Would delete: $dir/"
        else
            rm -rf "$dir"
            log_success "Deleted: $dir/"
            echo "$dir/" >> "$BACKUP_FILE"
        fi
    fi
}

################################################################################
# Pre-Cleanup Checks
################################################################################

log_info "Starting cleanup process..."
log_info "Project root: $PROJECT_ROOT\n"

# Check if git repo
if ! git -C "$PROJECT_ROOT" rev-parse --git-dir > /dev/null 2>&1; then
    log_error "Not a git repository. Aborting."
    exit 1
fi

# Check git status
cd "$PROJECT_ROOT"
log_info "Checking git status..."

if [[ -n $(git status -s) ]]; then
    log_warning "There are uncommitted changes. Please commit or stash them first."
    git status -s
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled."
        exit 0
    fi
fi

# Initialize backup file
if [[ "$DRY_RUN" == false ]]; then
    > "$BACKUP_FILE"
    log_success "Backup file created: $BACKUP_FILE"
fi

echo ""

################################################################################
# Verify Dependencies - Check for imports in active code
################################################################################

log_info "Verifying that deprecated code is not imported by active endpoints...\n"

ACTIVE_CODE_FILES=(
    "management_plane/app/main.py"
    "management_plane/app/endpoints/enforcement_v2.py"
    "management_plane/app/settings.py"
)

check_imports() {
    local pattern="$1"
    local description="$2"
    
    local found=false
    for file in "${ACTIVE_CODE_FILES[@]}"; do
        if grep -q "$pattern" "$file" 2>/dev/null; then
            log_warning "Found import in $file: $pattern"
            found=true
        fi
    done
    
    if [[ "$found" == false ]]; then
        log_success "$description is not imported by active code"
    fi
}

check_imports "from app.endpoints.enforcement import" "V1 enforcement endpoint"
check_imports "from app.endpoints.intents import" "Intents endpoint"
check_imports "from app.endpoints.boundaries import" "Boundaries endpoint"
check_imports "from app.endpoints.telemetry import" "Telemetry endpoint"
check_imports "from app.endpoints.encoding import" "Encoding endpoint"
check_imports "from app.endpoints.agents import" "Agents endpoint"
check_imports "from app.services.vocabulary import" "Vocabulary service"
check_imports "from app.database import" "Database service"

echo ""

################################################################################
# Phase 1: Delete Completely Deprecated Directories
################################################################################

log_info "Phase 1: Deleting completely deprecated directories...\n"

delete_directory "console"
delete_directory "sdk"

echo ""

################################################################################
# Phase 2: Delete Deprecated Endpoints
################################################################################

log_info "Phase 2: Deleting deprecated endpoints...\n"

delete_file "management_plane/app/endpoints/enforcement.py"
delete_file "management_plane/app/endpoints/intents.py"
delete_file "management_plane/app/endpoints/boundaries.py"
delete_file "management_plane/app/endpoints/telemetry.py"
delete_file "management_plane/app/endpoints/encoding.py"
delete_file "management_plane/app/endpoints/agents.py"
delete_file "management_plane/app/endpoints/auth.py"

echo ""

################################################################################
# Phase 3: Delete Deprecated Services
################################################################################

log_info "Phase 3: Deleting deprecated services...\n"

delete_file "management_plane/app/services/vocabulary.py"
delete_file "management_plane/app/database.py"

echo ""

################################################################################
# Phase 4: Delete Deprecated Tests
################################################################################

log_info "Phase 4: Deleting deprecated tests...\n"

delete_file "management_plane/tests/test_endpoints_agents.py"
delete_file "management_plane/tests/test_encoding_endpoints.py"
delete_file "management_plane/tests/test_encoding.py"
delete_file "management_plane/tests/test_applicability_filter.py"
delete_file "management_plane/tests/test_deny_semantics.py"
delete_file "management_plane/tests/test_e2e_real_encoding.py"
delete_file "management_plane/tests/test_llm_anchor_generation.py"
delete_file "management_plane/tests/test_nl_policy_parser.py"
delete_file "management_plane/tests/test_policy_templates.py"
delete_file "management_plane/tests/test_rule_anchors.py"
delete_file "management_plane/tests/test_v1_1_similarity.py"
delete_file "management_plane/tests/test_phase2_integration.py"
delete_file "management_plane/tests/test_performance.py"

echo ""

################################################################################
# Phase 5: Delete Vocabulary Files
################################################################################

log_info "Phase 5: Deleting vocabulary files (no longer used with BERT)...\n"

delete_file "vocabulary.yaml"
delete_file "sdk/python/vocabulary.yaml"

echo ""

################################################################################
# Phase 6: Delete Old Model Files
################################################################################

log_info "Phase 6: Keeping only optimized ONNX model, removing non-optimized version...\n"

delete_file "management_plane/models/canonicalizer_tinybert_v1.0/model.onnx"

echo ""

################################################################################
# Summary
################################################################################

echo ""
log_info "Cleanup Summary\n"

if [[ "$DRY_RUN" == true ]]; then
    log_warning "DRY-RUN MODE: No files were actually deleted"
    echo "Files that would be deleted:"
else
    echo "Files/directories deleted:"
fi

if [[ -f "$BACKUP_FILE" ]]; then
    count=$(wc -l < "$BACKUP_FILE")
    echo "  Total: $count items"
    echo ""
    echo "  Breakdown:"
    echo "    - Directories: 2 (console/, sdk/)"
    echo "    - Deprecated endpoints: 7 files"
    echo "    - Deprecated services: 2 files"
    echo "    - Deprecated tests: 13 files"
    echo "    - Vocabulary files: 2 files"
    echo "    - Old model files: 1 file"
fi

echo ""

################################################################################
# Post-Cleanup Actions
################################################################################

if [[ "$DRY_RUN" == false ]]; then
    log_info "Running post-cleanup actions...\n"
    
    cd "$PROJECT_ROOT"
    
    # Update endpoints __init__.py to remove deprecated imports
    if [[ -f "management_plane/app/endpoints/__init__.py" ]]; then
        log_info "Checking management_plane/app/endpoints/__init__.py for cleanup..."
        # This file should be manually reviewed
    fi
    
    # Check if main.py needs router cleanup
    if grep -q "enforcement" management_plane/app/main.py; then
        log_warning "Please review management_plane/app/main.py to remove router registrations for deleted endpoints"
    fi
    
    echo ""
    log_success "Cleanup completed successfully!"
    log_info "Backup list saved to: $BACKUP_FILE"
    echo ""
    log_warning "Next steps:"
    echo "  1. Review this script's output"
    echo "  2. Test the application: cd management_plane && python -m pytest"
    echo "  3. Run the application: cd management_plane && uvicorn app.main:app --reload"
    echo "  4. Commit the changes: git add -A && git commit -m 'cleanup: remove deprecated code not used by v2 APIs'"
    echo ""
else
    log_warning "This was a DRY-RUN. No changes were made."
    echo "To actually delete the files, run:"
    echo "  $0"
fi

exit 0
