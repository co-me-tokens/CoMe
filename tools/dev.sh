#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVICE="${FORCE_SERVICE:-dev-unified-entry}"

exec docker compose \
    -f "${REPO_ROOT}/docker/docker-compose.yaml" \
    -f "${SCRIPT_DIR}/docker-compose.user.yaml" \
    run --rm "${SERVICE}" bash "$@"
