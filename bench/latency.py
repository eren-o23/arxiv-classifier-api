"""Latency, batch-vs-singles, cold start and peak RSS over real HTTP.

    make bench                                    # spawn a server, measure everything
    uv run python bench/latency.py --threads 1    # same, with NUM_THREADS=1
    uv run python bench/latency.py --url http://host:8000 -n 200   # a box you already run

Stdlib only. Sequential HTTP does not need a dependency, and this stays runnable
inside the M4 container.

Prints a markdown table. It does not write docs/numbers.md — the interpretation
there is a human job (SPEC.md §7).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from itertools import cycle, islice
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden.json"
BATCH = 32


def post(url: str, path: str, payload: dict, timeout: float = 60) -> tuple[float, dict]:
    """POST JSON. Returns (wall-clock ms, decoded body)."""
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return (time.perf_counter() - started) * 1000, body


def pct(values: list[float], q: float) -> float:
    """Nearest rank — no interpolation argument to have, correct at the edges."""
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def start(port: int, threads: int | None, body: dict) -> tuple[subprocess.Popen, float]:
    """Spawn a server and time process start -> first successful /predict.

    `sys.executable`, so the server runs in whatever environment bench does, and
    proc.pid is the server itself rather than a `uv run` wrapper whose RSS is
    meaningless. Plain uvicorn, never --reload: the reloader parent would break
    both this measurement and the RSS reading.
    """
    env = {"PYTHONUNBUFFERED": "1"}
    if threads:
        env["NUM_THREADS"] = str(threads)

    started = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.api:app", "--port", str(port),
         "--log-level", "warning"],
        cwd=REPO,
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    while True:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode} before serving")
        try:
            post(url, "/predict", body, timeout=30)
            return proc, (time.perf_counter() - started) * 1000
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # uvicorn refuses connections until the lifespan finishes, so this is
            # the expected path until the model is loaded and warmed.
            time.sleep(0.05)


def rss_mb(pid: int) -> float | None:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                         capture_output=True, text=True, check=False)
    return int(out.stdout.strip()) / 1024 if out.stdout.strip() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=1000, help="warm sequential requests (default 1000)")
    ap.add_argument("--url", help="measure a running server; skips cold start and RSS")
    ap.add_argument("--port", type=int, default=8001, help="port for the spawned server")
    ap.add_argument("--threads", type=int, help="NUM_THREADS for the spawned server")
    args = ap.parse_args()

    # Real papers already in the repo (tests/golden.json), not a new fixture.
    papers = json.loads(GOLDEN.read_text())["papers"]
    one = {"title": papers[0]["title"], "abstract": papers[0]["abstract"]}
    # Cycled to 32. Both sides of the batch comparison use this same list.
    batch = [{"title": p["title"], "abstract": p["abstract"]}
             for p in islice(cycle(papers), BATCH)]

    proc = None
    if args.url:
        url, cold_ms = args.url.rstrip("/"), None
    else:
        url = f"http://127.0.0.1:{args.port}"
        proc, cold_ms = start(args.port, args.threads, one)

    try:
        # The lifespan warms on a two-word input; settle on a real-length one.
        for _ in range(10):
            post(url, "/predict", one)

        wall, model = [], []
        for _ in range(args.n):
            ms, resp = post(url, "/predict", one)
            wall.append(ms)
            model.append(resp["latency_ms"])

        batch_wall, resp = post(url, "/predict/batch", {"items": batch}, timeout=300)
        batch_model = resp["latency_ms"]
        singles_wall = singles_model = 0.0
        for item in batch:
            ms, resp = post(url, "/predict", item)
            singles_wall += ms
            singles_model += resp["latency_ms"]

        # Read after the batch pass: padding 32 items to 384 tokens is the peak.
        peak = rss_mb(proc.pid) if proc else None
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=30)

    # In --url mode this script did not start the server, so it cannot know
    # the thread config: say so rather than label the run with a local value.
    threads = "set by the server" if args.url else (args.threads or "default")
    print(f"\n## {args.n} warm sequential requests, NUM_THREADS={threads}, {url}\n")
    print("| percentile | wall clock | model only |")
    print("|---|---|---|")
    for name, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        print(f"| {name} | {pct(wall, q):.1f} ms | {pct(model, q):.1f} ms |")

    print("\n| metric | wall clock | model only |")
    print("|---|---|---|")
    print(f"| batch of {BATCH}, one call | {batch_wall:.1f} ms | {batch_model:.1f} ms |")
    print(f"| {BATCH} singles | {singles_wall:.1f} ms | {singles_model:.1f} ms |")
    print(f"| batch speedup | {singles_wall / batch_wall:.1f}x | "
          f"{singles_model / batch_model:.1f}x |")

    print("\n| metric | value |")
    print("|---|---|")
    print(f"| cold start | {f'{cold_ms / 1000:.1f} s' if cold_ms else 'n/a (--url)'} |")
    print(f"| peak RSS | {f'{peak:.0f} MB' if peak else 'n/a (--url)'} |")
    if cold_ms:
        print("\nCold start is process spawn to first 200 from /predict: interpreter start,"
              "\nthe 265MB artifact load, and the lifespan warmup.")


if __name__ == "__main__":
    main()
