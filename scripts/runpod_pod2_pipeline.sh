#!/usr/bin/env bash
# Run on the RunPod host (Linux). See scripts/pod2_orchestrate.py for details.
set -euo pipefail

REPO_ROOT="${BTP_REPO:-/workspace/btp_2_compass}"
SIM_DIR="${BTP_SIM_DIR:-${REPO_ROOT}/outputs/btp_equil/A8Q2}"
TICA_DIR="${BTP_TICA_ARTIFACTS_DIR:-}"

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1

ARGS=(--repo-root "${REPO_ROOT}" --sim-output-dir "${SIM_DIR}")
if [[ -n "${TICA_DIR}" ]]; then
  ARGS+=(--tica-artifacts-dir "${TICA_DIR}")
fi

exec python -u scripts/pod2_orchestrate.py "${ARGS[@]}" "$@"
