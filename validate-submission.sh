#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-}"

if [[ -z "${BASE_URL}" ]]; then
  echo "Usage: ./validate-submission.sh https://your-space-name.hf.space"
  exit 1
fi

echo "== Checking local project files =="
[[ -f "Dockerfile" ]] || { echo "Missing Dockerfile"; exit 1; }
[[ -f "openenv.yaml" ]] || { echo "Missing openenv.yaml"; exit 1; }
[[ -f "inference.py" ]] || { echo "Missing inference.py"; exit 1; }

echo "== Building Docker image =="
docker build -t traffic-env-validate . >/dev/null
echo "Docker build: OK"

echo "== Remote endpoint checks against ${BASE_URL} =="

reset_json="$(curl -fsS "${BASE_URL}/reset")"
echo "reset: OK"

step_json="$(curl -fsS -X POST "${BASE_URL}/step" -H "Content-Type: application/json" -d '{"action":"SWITCH"}')"
echo "step: OK"

state_json="$(curl -fsS "${BASE_URL}/state")"
echo "state: OK"

echo "== Contract checks =="
echo "${reset_json}" | grep -q '"observation"' || { echo "reset missing observation"; exit 1; }
echo "${step_json}" | grep -q '"reward"' || { echo "step missing reward"; exit 1; }
echo "${step_json}" | grep -q '"done"' || { echo "step missing done"; exit 1; }
echo "${step_json}" | grep -q '"info"' || { echo "step missing info"; exit 1; }
echo "${state_json}" | grep -q '"observation"' || { echo "state missing observation"; exit 1; }

echo "All validation checks passed."