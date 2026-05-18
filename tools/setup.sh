#!/bin/bash

# ==============================================================================
# CONFIGURATION & REGISTRY
# ==============================================================================

# strictly fail on errors, unset variables, and pipe failures
set -euo pipefail

# Define colors for readable output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 1. REGISTER FEATURES HERE
# Format: [feature_name]="Description"
declare -A FEATURES_REGISTRY
FEATURES_REGISTRY["cuext"]="Build CUDA extension (co_me_cuext)"
FEATURES_REGISTRY["flash"]="Build Co-Me customized flash attention"
FEATURES_REGISTRY["zedloader"]="Install ZedLoader"
# ==============================================================================
# INSTALLATION HANDLERS
# ==============================================================================

# Define a function named "install_<feature_name>" for each entry in the registry.

setup_git_hooks() {
    log_info "Setting up git hooks..."
    mkdir -p .git/hooks/
    cp .githooks/post-checkout.sh .git/hooks/post-checkout
    chmod +x .git/hooks/post-checkout
    log_success "Git hooks configured"
}

install_cuext() {
    log_info "Building CUDA extension (token merge/split)..."
    (
        cd src/cuda_extension/co_me
        bash install.sh
    )
}

install_flash() {
    log_info "Building flash attention (co-me customized)..."
    (
        cd src/cuda_extension/flash_attn
        bash install.sh
    )
}

install_zedloader() {
    log_info "Installing ZedLoader..."
    cd src/cuda_extension/zed_loader
    bash install.sh --no-exec
    log_success "ZedLoader installed successfully."
}

# ==============================================================================
# CORE LOGIC (Do not edit often)
# ==============================================================================

log_info() { echo -e "${BLUE}${BOLD}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}${BOLD}[OK]${NC} $1"; }
log_error() { echo -e "${RED}${BOLD}[ERROR]${NC} $1"; }

show_usage() {
    echo -e "Usage: $0 [OPTIONS] [FEATURES...]"
    echo ""
    echo "Setup script for MAC-SLAM with optional features"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "AVAILABLE FEATURES:"
    
    # Dynamically print features from the registry
    for key in "${!FEATURES_REGISTRY[@]}"; do
        printf "  %-15s %s\n" "$key" "${FEATURES_REGISTRY[$key]}"
    done

    echo "  all             Install all optional features"
    echo ""
}

main() {
    local requested_features=()
    local install_queue=()

    # 1. Parse Arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            all)
                # Add all registered features.
                for feature_name in "${!FEATURES_REGISTRY[@]}"; do
                    requested_features+=("$feature_name")
                done
                shift
                ;;
            *)
                # Check if the argument is a valid key in our registry
                if [[ -n "${FEATURES_REGISTRY[$1]+found}" ]]; then
                    requested_features+=("$1")
                else
                    log_error "Unknown feature: '$1'"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # 2. Deduplicate features (in case user types 'all sparse_ba')
    # and sort them allows for consistent install order if needed
    eval "install_queue=($(printf "%s\n" "${requested_features[@]}" | sort -u | xargs))"

    echo "=== MAC-SLAM Setup Script ==="
    
    # 3. Run Base Setup
    setup_git_hooks
    log_info "Detecting platform and writing Docker .env..."
    bash .devcontainer/detect-config.sh
    log_success "Docker .env configured"

    # 4. Run Feature Installations
    if [[ ${#install_queue[@]} -eq 0 ]]; then
        echo "No optional features selected."
    else
        echo "Features to install: ${install_queue[*]}"
        echo ""
        
        # Save current directory to return to it after each install
        local root_dir
        root_dir=$(pwd)

        for feature in "${install_queue[@]}"; do
            # Dynamically call the function named install_<feature>
            func_name="install_${feature}"
            
            if declare -f "$func_name" > /dev/null; then
                # Run in a subshell () to prevent directory changes affecting main script
                ( "$func_name" ) 
                
                if [ $? -eq 0 ]; then
                    log_success "$feature installed successfully."
                else
                    log_error "Failed to install $feature"
                    exit 1
                fi
            else
                log_error "Implementation missing for function: $func_name"
                exit 1
            fi
            echo ""
        done
    fi

    echo "=== Setup Complete ==="
}

# Run Main
main "$@"
