# Measured numbers — M3

All figures from `bench/latency.py` and `bench/load.sh`, on the machine below.
Nothing here is estimated. Re-run with `make bench` and `make load`.

## Test box

| | |
|---|---|
| Machine | Apple M2, 8 cores (4 performance + 4 efficiency), 8GB |
| OS / Python / torch | macOS 14.5 / 3.11.13 / 2.14.0 |
| Server | one uvicorn worker, no `--reload`, over loopback |
| Model | `erenrosman/arxiv-classifier-v1` @ `8eb5e473` (DistilBERT, 66M params) |

**This is a laptop, not the deployment target.** M5's VM is 2 vCPU / 4GB, so
every figure here will move. M6 re-runs the same two scripts against the
deployed box — `bench/latency.py --url` and `URL=... bench/load.sh` — and the
two sets sit side by side. The core count is the interesting difference, and
[the thread result](#pinning-threads-to-1-does-not-help-here-and-the-spec-said-it-would)
below is the reason to check it again there rather than assume.

`torch.get_num_threads()` defaults to **4** on this box (the performance cores),
not 8. "default" below means 4.

---

## Single request

1000 warm sequential requests, one fixed paper (1,858 chars → 384 tokens),
concurrency 1. "Model only" is the `latency_ms` the service reports for the
forward pass alone; the gap to wall clock is HTTP, JSON and FastAPI.

| percentile | default: wall | default: model | threads=1: wall | threads=1: model |
|---|---|---|---|---|
| p50 | **58.5 ms** | 57.0 ms | 74.5 ms | 72.9 ms |
| p95 | **64.3 ms** | 62.6 ms | 80.7 ms | 79.1 ms |
| p99 | **72.3 ms** | 70.7 ms | 93.0 ms | 90.9 ms |

## Concurrency sweep

`hey`, 200 requests per level, same paper.

| concurrency | default: rps | default: p95 | threads=1: rps | threads=1: p95 |
|---|---|---|---|---|
| 1 | 17.2 req/s | 62 ms | 13.4 req/s | 82 ms |
| 2 | 19.8 req/s | 111 ms | 17.1 req/s | 197 ms |
| 4 | 22.7 req/s | 203 ms | 22.0 req/s | 270 ms |
| 8 | **24.3 req/s** | **375 ms** | 23.7 req/s | 431 ms |
| 16 | 23.2 req/s | 805 ms | 23.4 req/s | 912 ms |

Concurrency 8 re-run at N=300 with the run order reversed, to rule out one
config warming the box for the other: `NUM_THREADS=1` 24.6 req/s / p95 458 ms,
default 24.8 req/s / p95 376 ms. Same conclusion both ways round.

## Batch, cold start, memory

| metric | default | `NUM_THREADS=1` |
|---|---|---|
| batch of 32, one call | 1806 ms | 2762 ms |
| the same 32 as singles | 1729 ms | 2020 ms |
| batch speedup | **0.96x** | **0.73x** |
| cold start | 3.8 s | 2.9 s |
| peak RSS | 582 MB | 607 MB |
| image size | TODO — M4's row | |

Cold start is process spawn → first 200 from `/predict`: interpreter start, the
265MB artifact load, and the lifespan warmup. It is not a measure of the model
load alone.

---

## What the numbers say

**p95 is 64 ms and the distribution is tight** — p99/p50 is 1.24, so there is no
long tail to speak of at concurrency 1. **HTTP and FastAPI cost about 1.5 ms**,
2.6% of the request; the other 97% is the forward pass. There is nothing to win
in the web layer, which is the useful thing to know before optimising it.

**Throughput ceilings at ~24 req/s and it is compute, not the server.** Going
from concurrency 8 to 16 buys no throughput (24.3 → 23.2 req/s) and doubles p95
(375 → 805 ms). Past saturation the extra concurrency is pure queueing, and
Little's law fits: at concurrency 8, 8 ÷ 24.3 req/s = 329 ms mean against a
measured p95 of 375 ms; at 16, 16 ÷ 23.2 = 690 ms against 805 ms. **The knee is
at concurrency 4–8.** A load balancer in front of this should cap in-flight
requests around there rather than let a queue build that only adds latency.

### Pinning threads to 1 does not help here, and the spec said it would

SPEC §3 predicts `torch.set_num_threads(1)` improves p95 under concurrency,
because torch otherwise uses every core per request and oversubscribes. On this
box it does the opposite — p95 is worse at **every** concurrency level (375 → 431
ms at 8, confirmed 376 → 458 ms on the re-run) and throughput is identical
(~24.7 req/s either way).

The oversubscription is real: at concurrency 8, 8 requests × 4 threads is 32
threads over 8 cores. It just is not the binding constraint. Throughput is set
by total available FLOPs, and both configurations reach the same ceiling, so the
only thing pinning changes is single-request latency — 27% worse (58.5 → 74.5 ms
p50) because one request no longer gets four cores. That penalty is paid at
every level and never earned back.

**This is worth re-testing on the 2 vCPU VM at M6, not assuming.** With 4× fewer
cores the ratio of threads to cores changes, and the prediction may hold there.
Keeping `NUM_THREADS` an env var rather than a `set_num_threads(1)` at import
time is what makes that a config change instead of a code change.

### Batching does not pay on CPU, and padding is why

SPEC §4 expects one forward pass over 32 texts to be several times faster than
32 passes. Measured, it is **0.96x** — slightly slower.

The cause is padding. `predict_many` pads the batch to its longest member, and
for these 32 papers that is 384 tokens against a median of 323:

| | tokens |
|---|---|
| shortest / median / longest | 204 / 323 / 384 |
| sum of actual lengths | 10,004 |
| padded batch (32 × 384) | 12,288 |
| **overhead** | **1.23x** |

So the batch does 23% more token-work than the singles do. It is genuinely more
efficient per token — otherwise 1.23x the work would take 1.23x the time — but
the efficiency only just covers the padding, and nets out at nothing.

**The `NUM_THREADS=1` column is the proof.** With one thread there is no
parallelism for a wider matrix to exploit, so time should track token-work
exactly: 1 ÷ 1.23 = 0.81 predicted, 0.73 measured. Batching wins by using idle
compute, and on a CPU box a *single* 384-token request already saturates the
cores, so there is none to use.

Batching would still pay if requests were short and varied — padding waste falls
with sequence length — or on a GPU, where one request cannot saturate the
device. Neither is this service. `/predict/batch` stays because one round trip
for 32 papers beats 32 round trips over a real network, which is a latency win
for the caller even when it is not a throughput win for the server.

### Memory

**Peak RSS is 582 MB against a 265 MB artifact** — 2.2x, which is fp32 weights
plus torch's allocator and the activations for a 32×384 batch. Comfortable on
the 4GB VM, and it sets the floor for M4's container limit. The `NUM_THREADS=1`
run peaks slightly higher (607 MB), which is per-thread arena bookkeeping, not
anything meaningful.

**Cold start is 3.8 s.** That is the number that matters for a rolling restart
or an autoscaler, not the load time alone. Health checks must not go green
before it finishes — which is exactly what the lifespan warmup and the 503
branch are for.
