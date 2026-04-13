#!/usr/bin/env python3
"""
Compare RunPod ``outputs/`` tree to local ``outputs/`` (by relative path + byte size)
and ``scp`` only files that are missing locally or have the wrong size.

Usage (from repo root, after ``ssh`` to the pod works)::

    python scripts/sync_runpod_missing_outputs.py --dry-run
    python scripts/sync_runpod_missing_outputs.py
    python scripts/sync_runpod_missing_outputs.py --parallel 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _ssh_manifest(
    user: str,
    host: str,
    port: int,
    key: Path,
    remote_repo: str,
) -> list[tuple[int, str]]:
    """Return list of (size_bytes, relative_posix_path) for every file under outputs/."""
    cmd = [
        "ssh",
        "-i",
        str(key),
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=720",
        "-p",
        str(port),
        f"{user}@{host}",
        f"cd {remote_repo} && find outputs -type f -printf '%s\\t%p\\n' 2>/dev/null",
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        sz_s, rel = line.split("\t", 1)
        rel = rel.replace("\\", "/").strip()
        if rel.startswith("./"):
            rel = rel[2:]
        rows.append((int(sz_s), rel))
    return rows


def _local_sizes(repo_root: Path) -> dict[str, int]:
    out_dir = repo_root / "outputs"
    sizes: dict[str, int] = {}
    if not out_dir.is_dir():
        return sizes
    skip_names = {
        "_remote_manifest.tsv",
        "_local_manifest.tsv",
        "_sync_missing_plan.tsv",
    }
    for p in out_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name in skip_names or p.name.endswith(".partial"):
            continue
        try:
            rel = p.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        sizes[rel] = p.stat().st_size
    return sizes


def _scp_one(
    user: str,
    host: str,
    port: int,
    key: Path,
    remote_repo_posix: str,
    rel_posix: str,
    dest: Path,
) -> None:
    """Download to a temp file under %TEMP% to avoid Windows AV locks on ``*.partial`` in-repo."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_path = f"{remote_repo_posix.rstrip('/')}/{rel_posix}"
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="btp_scp_", suffix=".tmp")
    os.close(tmp_fd)
    try:
        cmd = [
            "scp",
            "-i",
            str(key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=720",
            "-P",
            str(port),
            f"{user}@{host}:{remote_path}",
            tmp_path,
        ]
        subprocess.run(cmd, check=True)
        if not os.path.isfile(tmp_path):
            raise RuntimeError(f"scp produced no file: {rel_posix}")
        try:
            os.replace(tmp_path, dest)
        except OSError:
            if dest.exists():
                dest.unlink()
            shutil.move(tmp_path, str(dest))
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--ssh-user", default="root")
    p.add_argument("--ssh-host", default="203.57.40.227")
    p.add_argument("--ssh-port", type=int, default=10148)
    p.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    p.add_argument("--remote-repo", default="/workspace/BTP_2_Calm")
    p.add_argument("--dry-run", action="store_true", help="Only print what would be fetched.")
    p.add_argument("--parallel", type=int, default=1, help="Concurrent scp processes (1–3 recommended).")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Fetch at most this many files (0 = no limit). Useful for testing.",
    )
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    key = args.ssh_key.expanduser().resolve()
    if not key.is_file():
        print(f"Missing SSH key: {key}", file=sys.stderr)
        return 2

    print("Fetching remote file list...", flush=True)
    remote_rows = _ssh_manifest(args.ssh_user, args.ssh_host, args.ssh_port, key, args.remote_repo)
    local_sizes = _local_sizes(repo_root)

    need: list[tuple[int, str]] = []
    for sz, rel in remote_rows:
        if rel not in local_sizes or local_sizes[rel] != sz:
            need.append((sz, rel))

    need.sort(key=lambda x: -x[0])
    if not args.dry_run and args.limit and args.limit > 0:
        need = need[: args.limit]

    total_b = sum(s for s, _ in need)
    print(f"Remote files: {len(remote_rows)}  |  Local files under outputs/: {len(local_sizes)}", flush=True)
    print(f"To fetch (missing or size mismatch): {len(need)} files, {total_b / 1e9:.3f} GB", flush=True)

    plan_path = repo_root / "outputs" / "_sync_missing_plan.tsv"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as f:
        for sz, rel in sorted(need, key=lambda x: -x[0]):
            f.write(f"{sz}\t{rel}\n")
    print(f"Wrote plan: {plan_path}", flush=True)

    if args.dry_run or not need:
        return 0

    parallel = max(1, min(args.parallel, 8))
    failures: list[str] = []

    def work(item: tuple[int, str]) -> tuple[str, bool]:
        sz, rel = item
        dest = repo_root / rel.replace("/", os.sep)
        for attempt in range(1, args.max_retries + 1):
            try:
                _scp_one(
                    args.ssh_user,
                    args.ssh_host,
                    args.ssh_port,
                    key,
                    args.remote_repo,
                    rel,
                    dest,
                )
                got = dest.stat().st_size
                if got != sz:
                    raise RuntimeError(f"size {got} != expected {sz}")
                return (rel, True)
            except Exception as e:
                if attempt == args.max_retries:
                    return (rel, False)
                print(f"  retry {attempt}/{args.max_retries} {rel}: {e}", flush=True)
        return (rel, False)

    done = 0
    if parallel == 1:
        for item in sorted(need, key=lambda x: -x[0]):
            rel_ok, ok = work(item)
            done += 1
            if ok:
                print(f"[{done}/{len(need)}] ok {rel_ok}", flush=True)
            else:
                print(f"[{done}/{len(need)}] FAIL {rel_ok}", file=sys.stderr, flush=True)
                failures.append(rel_ok)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(work, item): item for item in need}
            for fut in as_completed(futs):
                done += 1
                rel_ok, ok = fut.result()
                if ok:
                    print(f"[{done}/{len(need)}] ok {rel_ok}", flush=True)
                else:
                    print(f"[{done}/{len(need)}] FAIL {rel_ok}", file=sys.stderr, flush=True)
                    failures.append(rel_ok)

    if failures:
        print(f"\nFailed ({len(failures)}):", file=sys.stderr)
        for r in failures[:50]:
            print(f"  {r}", file=sys.stderr)
        return 1
    print("All required files synced.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
