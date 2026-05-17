#!/usr/bin/env bash
# =============================================================================
# deploy_hetzner.sh — bring up the Spine Segmentation stack on a fresh server.
#
# Idempotent: safe to re-run.  Verifies prereqs, builds the image, starts the
# stack, and waits until the healthcheck reports OK.
#
# Run on the SERVER (not on your laptop):
#   cd ~/spine-segmentation
#   bash scripts/deploy_hetzner.sh
#
# Requires: docker, docker compose v2, a valid .env (see .env.example).
# See docs/DEPLOYMENT.md for the full runbook.
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log()  { printf "\033[1;36m[deploy]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
fail() { printf "\033[1;31m[fail]\033[0m %s\n" "$*"; exit 1; }

# ----- 1. Prereqs ------------------------------------------------------------
log "Checking prerequisites..."
command -v docker >/dev/null            || fail "docker is not installed"
docker compose version >/dev/null 2>&1  || fail "docker compose v2 is not installed"

if [ ! -f .env ]; then
    fail "no .env file in $ROOT. Copy .env.example -> .env and edit it first."
fi

# Source .env in a safe subshell so we can validate required vars without
# polluting the current shell
required_vars=(HF_REPO_ID DOMAIN)
missing=()
for v in "${required_vars[@]}"; do
    val=$(grep -E "^${v}=" .env | head -n1 | cut -d= -f2- | tr -d '"' | tr -d "'")
    if [ -z "$val" ] || [ "$val" = "replace-me.nip.io" ]; then
        missing+=("$v")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    fail "missing or placeholder values in .env for: ${missing[*]}"
fi

log "Prereqs OK."

# ----- 2. Build the image ----------------------------------------------------
log "Building Docker image (this may take 5-10 min the first time)..."
docker compose build

# ----- 3. Bring the stack up -------------------------------------------------
log "Starting stack..."
docker compose up -d

# ----- 4. Wait for healthcheck -----------------------------------------------
log "Waiting for the app to become healthy (up to 5 min for first HF download)..."
deadline=$(( $(date +%s) + 300 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    status=$(docker inspect --format='{{.State.Health.Status}}' spine_app 2>/dev/null || echo "starting")
    case "$status" in
        healthy)
            log "App is healthy."
            break
            ;;
        unhealthy)
            warn "App reported unhealthy. Logs:"
            docker compose logs --tail=80 app || true
            fail "Healthcheck failed."
            ;;
        *)
            sleep 5
            ;;
    esac
done

if [ "$status" != "healthy" ]; then
    warn "Timeout waiting for healthy state. Current logs:"
    docker compose logs --tail=80 app || true
    fail "App did not become healthy in 5 min."
fi

# ----- 5. Final status -------------------------------------------------------
log "Stack status:"
docker compose ps

DOMAIN=$(grep -E "^DOMAIN=" .env | head -n1 | cut -d= -f2- | tr -d '"' | tr -d "'")
log "Done. The app should be reachable at: https://$DOMAIN"
log "Caddy will request a Let's Encrypt certificate on the first request to that hostname."
log "If the cert hasn't been issued yet, give it ~30 s and reload the page."
