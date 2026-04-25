#!/usr/bin/env bash
#
# Build + push the quanty/modelopt image via depot.
# Usage:
#   quanty/docker/build.sh                # builds :dev
#   quanty/docker/build.sh sha            # also tags :sha-<short>
#   quanty/docker/build.sh pr 1234        # also tags :pr-1234

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# DEPOT_PROJECT_ID defaults to the org's `default` project; override if you
# want builds in a different project (e.g. `lumen`).  DEPOT_TOKEN is not
# required when the depot CLI is already logged in via `depot login`.
: "${DEPOT_PROJECT_ID:=3w53ndbslf}"

REGISTRY="docker.cloudsmith.io/coreweave/infr-dev"
IMAGE="${REGISTRY}/quanty-modelopt"

EXTRA_TAGS=()
case "${1:-}" in
    sha)
        EXTRA_TAGS+=("${IMAGE}:sha-$(git rev-parse --short HEAD)")
        ;;
    pr)
        : "${2:?Pass a PR number after 'pr'}"
        EXTRA_TAGS+=("${IMAGE}:pr-$2")
        ;;
    "")
        ;;
    *)
        echo "unknown mode: $1 (expected: sha | pr <n> | empty)" >&2
        exit 2
        ;;
esac

TAG_FLAGS=("--tag" "${IMAGE}:dev")
for t in "${EXTRA_TAGS[@]}"; do
    TAG_FLAGS+=("--tag" "${t}")
done

depot build \
    --project "${DEPOT_PROJECT_ID}" \
    --platform linux/amd64 \
    --file quanty/docker/Dockerfile \
    --push \
    "${TAG_FLAGS[@]}" \
    .
