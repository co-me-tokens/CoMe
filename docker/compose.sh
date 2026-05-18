#!/usr/bin/env bash
# Parallel Docker builds on one host multiply compiler processes; cap jobs to avoid OOM.
# Override: BUILD_MAX_JOBS=8 ./compose.sh build
export BUILD_MAX_JOBS="${BUILD_MAX_JOBS:-4}"

ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    PROFILE="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    PROFILE="arm64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

ARGS=("$@")
if [ "${ARGS[0]:-}" = "build" ]; then
    ARGS=(build --build-arg "BUILD_MAX_JOBS=${BUILD_MAX_JOBS}" "${ARGS[@]:1}")
fi

docker compose -f docker-compose.yaml --profile "$PROFILE" "${ARGS[@]}" && \
docker compose -f docker-compose-type.yaml --profile "$PROFILE" "${ARGS[@]}"
