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
| p50 | **59.0 ms** | 57.3 ms | 74.9 ms | 73.1 ms |
| p95 | **66.7 ms** | 64.5 ms | 80.0 ms | 78.2 ms |
| p99 | **82.3 ms** | 79.7 ms | 89.5 ms | 87.7 ms |

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
| batch of 32, one call | 1846 ms | 2607 ms |
| the same 32 as singles | 1622 ms | 2068 ms |
| batch speedup | **0.88x** | **0.79x** |
| cold start | 3.6 s | 3.0 s |
| peak RSS | 849 MB | 859 MB |
| image size | 562 MB | see below — it is not a per-config number |

Cold start is process spawn → first 200 from `/predict`: interpreter start, the
265MB artifact load, and the lifespan warmup. It is not a measure of the model
load alone.

Peak RSS is `getrusage(RUSAGE_CHILDREN).ru_maxrss` — the OS's high-water mark
for the server process, read after it exits. An earlier revision sampled `ps`
at the end of the run instead and reported **582 MB**; that is a *current*
reading, not a peak, and it understated the true figure by 46%. If you are
sizing a container from this page, 849 MB is the number.

---

## Image — M4

| | |
|---|---|
| **shipped image, compressed** | **562 MB** — `docker image inspect -f '{{.Size}}'`, what `make build` prints and what a registry transfers |
| the running container's filesystem | 1340 MB — `du -sx /` inside it |
| what the image costs Docker's local store | 1.98 GB — what `docker images` prints |
| dependency wheels, CPU-only index | **194 MB** across 40 wheels |
| dependency wheels, default PyPI index | **2921 MB** across 58 wheels |
| baked artifact | 257 MB |

Built `--platform linux/amd64` on an arm64 laptop, because the lock is hashed for
x86_64 and the M5 VM is the arch that matters.

**Three numbers, and they are not the same measurement.** 562 MB is the sum of
the compressed layers — the pull, and the number the "under 1GB" figures quoted
for CPU-only torch refer to, so it is the like-for-like comparison and the one
the M4 gate is met on. 1340 MB is what the container's filesystem actually
occupies (`docker history` sums to 1418 MB; the gap is overlay accounting).
1.98 GB is what `docker images` prints, because Docker's local store keeps the
compressed blobs *and* the unpacked snapshot — 562 + 1340 is most of it.

**Size the M5 VM's disk from 1.98 GB**, not 562 MB and not 1340 MB. The store is
what fills up, and this laptop already proved that failure mode.

`docker images` does *not* print the compressed size, which is easy to assume
and wrong in the direction that flatters the number. `make build` prints the
562 MB via `docker image inspect`, and torch alone is 575 MB unpacked — an image
containing it cannot be 562 MB on disk, which is the check that catches this.

### CPU-only torch is a 15x difference in what you download

SPEC §8 predicts ~2.5GB for default torch against under 1GB for CPU-only wheels.
Measured at the dependency layer, where the difference actually lives, it is
**194 MB against 2921 MB** for the identical set of top-level requirements.

The default index resolves 58 packages; the CPU index resolves 40. The 18 extra
are almost all CUDA:

| | MB |
|---|---|
| `torch` (default) | 529 |
| `torch` (`+cpu`) | 132 |
| `nvidia-cudnn-cu13` | 528 |
| `nvidia-cublas` | 404 |
| `triton` | 237 |
| `nvidia-nccl-cu13` | 206 |
| ...11 more `nvidia-*` | 619 |

None of it can ever execute on a 2 vCPU VM with no GPU. `torch==2.14.0+cpu` is a
different wheel, not the same wheel with a flag — 132 MB against 529 MB before a
single CUDA dependency is counted.

**The default-torch image was not built to completion.** It exhausted the disk on
this machine partway through, which is a blunter version of the same point than
any figure here: it is not that the CUDA image is larger, it is that it does not
fit. The 2921 MB is measured from the resolved wheel sizes the index serves, not
from a built image, and is labelled that way rather than quietly compared against
the 562 MB.

### 200MB of the wheel is unreachable at runtime

torch ships its C++ gtest binaries (85 MB), its C++ headers (63 MB) and a copy of
`protoc` (52 MB) inside the wheel. A serving image can never execute any of it,
so the builder stage deletes it — the same argument SPEC §8 makes for keeping
`training/` out, one layer further down. That is 620 → 562 MB, and 1561 → 1340 MB
on disk, with the smoke gate re-run afterwards to prove nothing that matters went
with it.

`.pyc` files stay, and they are 213 MB. The venv is root-owned and the service
runs as `app`, so a stripped venv could not rewrite them and every module would
recompile on every start. That trades cold start, repeatedly, against space saved
once.

### Container timings are not comparable to the rest of this page

`/predict` inside the container answered in **897 ms** against 59 ms native. That
is Rosetta emulating x86_64 on an arm64 laptop, not a property of the image, and
it is recorded here only so nobody reads it as a regression. **M6 measures the
container on the VM, where the arch is native**, using the same
`bench/latency.py --url` path M3 already verified.

---

## What the numbers say

**p95 is 67 ms and the distribution is tight** — p99/p50 is 1.4, so there is no
long tail to speak of at concurrency 1. **HTTP and FastAPI cost about 1.7 ms**,
2.9% of the request; the other 97% is the forward pass. There is nothing to win
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
32 passes. Measured, it is **0.88x** — slightly slower.

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
exactly: 1 ÷ 1.23 = 0.81 predicted, 0.79 measured. Batching wins by using idle
compute, and on a CPU box a *single* 384-token request already saturates the
cores, so there is none to use.

Batching would still pay if requests were short and varied — padding waste falls
with sequence length — or on a GPU, where one request cannot saturate the
device. Neither is this service. `/predict/batch` stays because one round trip
for 32 papers beats 32 round trips over a real network, which is a latency win
for the caller even when it is not a throughput win for the server.

### Memory

**Peak RSS is 849 MB against a 265 MB artifact** — 3.2x, which is fp32 weights
plus torch's allocator and the activations for a 32×384 batch. Still comfortable
on the 4GB VM, but it is the floor for M4's container limit and it is 46% above
what a `ps` sample at the end of the run suggested. The `NUM_THREADS=1` run
peaks marginally higher (859 MB), which is per-thread arena bookkeeping, not
anything meaningful.

**Cold start is 3.6 s.** That is the number that matters for a rolling restart
or an autoscaler, not the load time alone. Health checks must not go green
before it finishes — which is exactly what the lifespan warmup and the 503
branch are for.
