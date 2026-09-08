#!/usr/bin/env python3
"""The M4 gate, as something runnable: does the container serve a correct
prediction?

Stdlib only and no project imports — it talks to a running container over HTTP
the way anything else would, so it works just as well against the M5 VM
(`URL=https://... make smoke`). Inputs come from tests/golden.json rather than a
new fixture; the recorded confidences are the same ones test_golden_labels
asserts, so a container that disagrees with them has a different artifact or
different preprocessing, which is exactly what this is looking for.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("URL") or f"http://localhost:{os.environ.get('PORT', '8000')}"
GOLDEN = json.loads((pathlib.Path(__file__).resolve().parents[1] / "tests/golden.json").read_text())

failures: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" != {want!r}"))
    if not ok:
        failures.append(name)


def get(path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        URL + path, data=data, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def wait_for_health(timeout_s: float = 180) -> float:
    """Poll until ready. Returns seconds waited, which is NOT cold start — it
    counts from whenever this script started, not from container start."""
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        try:
            status, body = get("/health")
            if status == 200 and body.get("model_loaded"):
                return time.monotonic() - started
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass  # uvicorn refuses connections until the lifespan finishes
        time.sleep(0.5)
    sys.exit(f"never became healthy within {timeout_s}s — is `make docker-run` up?")


print(f"→ {URL}")
waited = wait_for_health()
print(f"ok   healthy after {waited:.1f}s (connection-refused until then, by design)")

status, health = get("/health")
check("health status", status, 200)
check("health model_version", health["model_version"], GOLDEN["model_version"])

status, meta = get("/metadata")
check("metadata status", status, 200)
check("metadata label count", len(meta["labels"]), 10)

paper = next(p for p in GOLDEN["papers"] if p["assert"] == "top1")
status, pred = get("/predict", {"title": paper["title"], "abstract": paper["abstract"]})
check("predict status", status, 200)
check(f"predict label ({paper['arxiv_id']})", pred["label"], paper["expected"])
check(
    "predict confidence within 0.01 of recorded",
    abs(pred["confidence"] - paper["confidence"]) < 0.01,
    True,
)
check("scores sum to 1", round(sum(pred["scores"].values()), 4), 1.0)
check("scores keys are the card's labels", sorted(pred["scores"]), sorted(meta["labels"]))
print(f"     latency_ms {pred['latency_ms']}, confidence {pred['confidence']:.3f}")

two = GOLDEN["papers"][:2]
status, batch = get(
    "/predict/batch",
    {"items": [{"title": p["title"], "abstract": p["abstract"]} for p in two]},
)
check("batch status", status, 200)
check("batch labels", [r["label"] for r in batch["results"]], [p["predicted"] for p in two])

status, bad = get("/predict", {"title": "", "abstract": "x"})
check("empty title is 400", status, 400)

print()
if failures:
    sys.exit(f"{len(failures)} failed: {', '.join(failures)}")
print("smoke passed")
