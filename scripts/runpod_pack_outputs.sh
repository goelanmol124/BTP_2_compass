#!/usr/bin/env bash
# Run ON the RunPod host (bash). Packs repo outputs/ and splits for reliable download.
set -euo pipefail
REPO_ROOT="${1:-/workspace/BTP_2_Calm}"
cd "$REPO_ROOT"
echo "[1/3] Creating single tarball (uncompressed — DCDs barely compress) …"
rm -f /tmp/btp_outputs_bundle.tar
tar -cf /tmp/btp_outputs_bundle.tar outputs
ls -lh /tmp/btp_outputs_bundle.tar
echo "[2/3] Splitting into 400 MiB chunks …"
rm -rf /tmp/btp_parts && mkdir -p /tmp/btp_parts
split -b 400M /tmp/btp_outputs_bundle.tar /tmp/btp_parts/p_
echo "[3/3] Done. List:"
ls -lh /tmp/btp_parts/
echo "From your laptop (PowerShell):"
echo "  .\\scripts\\sync_runpod_outputs.ps1 -HostName <ip> -Port <port> -RepoRoot <path>"
