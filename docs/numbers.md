# Measured numbers

All figures from `bench/latency.py` and `bench/load.sh`. Nothing here is
estimated. Re-run with `make bench` and `make load`.

**Two boxes.** M3 measured the laptop; M6 re-ran the identical scripts against
the deployed VM, which is the box that actually serves traffic and the one the
README quotes. Both sets are kept, because the difference between them is the
content — the same code on a quarter of the cores.

| | laptop (M3) | deployed VM (M6) |
|---|---|---|
| Machine | Apple M2, 8 cores (4P + 4E), 8GB | Hetzner CX23, 2 vCPU Xeon Skylake, 3.7GB |
| OS / Python / torch | macOS 14.5 / 3.11.13 / 2.14.0 | Ubuntu 24.04 / 3.11 / 2.14.0+cpu |
| Server | one uvicorn worker, no `--reload`, over loopback | the shipped container, one uvicorn worker |
| Reached over | loopback | the compose bridge, and separately through Caddy |
| Model | `erenrosman/arxiv-classifier-v1` @ `8eb5e473` (DistilBERT, 66M params) | same |

`torch.get_num_threads()` defaults to **4** on the laptop (the performance
cores), not 8, and to **2** on the VM. "default" means those.

---

# The laptop — M3

**This is not the deployment target**; it is the comparison. Skip to
[the deployed box](#the-deployed-box--m6) for the numbers that describe the
running service.

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

# The deployed box — M6

Hetzner CX23, 2 vCPU. The same two scripts, in `--url` mode, run *on the VM* so
the numbers describe the service rather than the trip to it.

Two paths were measured. **Direct** is the api container's address on the compose
bridge — plain HTTP, no Caddy, no TLS, which is the like-for-like comparison with
the laptop's loopback. **Through Caddy** is the public hostname, which is what a
user actually gets. Procedure in [deploy.md](deploy.md#benchmarking-the-deployed-box-m6).

**The sweeps were taken with `RATE_LIMIT_EVENTS` raised to 100000 and the 30/min
limit restored afterwards** — otherwise a concurrency-16 sweep measures Caddy's
rejection rate rather than the service. Restoring it was verified: 30 requests
through, then 429 with `retry-after`, `/health` unaffected.

## Single request — VM

500 warm sequential requests direct, 200 through Caddy; the same fixed paper the
laptop used (1,858 chars → 384 tokens), concurrency 1.

| percentile | direct: wall | direct: model | threads=1: wall | threads=1: model | via Caddy: wall |
|---|---|---|---|---|---|
| p50 | **444.5 ms** | 438.7 ms | 689.5 ms | 680.0 ms | 447.5 ms |
| p95 | **579.0 ms** | 572.1 ms | 760.4 ms | 752.9 ms | 564.9 ms |
| p99 | **664.9 ms** | 658.6 ms | 877.2 ms | 871.5 ms | 672.0 ms |

## Concurrency sweep — VM

`hey`, 200 requests per level.

| concurrency | default: rps | default: p95 | threads=1: rps | threads=1: p95 | via Caddy: rps | via Caddy: p95 |
|---|---|---|---|---|---|---|
| 1 | 2.1 req/s | 580 ms | 1.4 req/s | 754 ms | — | — |
| 2 | **2.6 req/s** | 961 ms | 2.7 req/s | 954 ms | — | — |
| 4 | 2.6 req/s | 1805 ms | 2.7 req/s | 1725 ms | 2.6 req/s | 1811 ms |
| 8 | 2.6 req/s | 3404 ms | 2.7 req/s | 3328 ms | 2.6 req/s | 3381 ms |
| 16 | 2.6 req/s | 6502 ms | 2.7 req/s | 6408 ms | — | — |

## Batch, cold start, memory — VM

| metric | default | `NUM_THREADS=1` |
|---|---|---|
| batch of 32, one call | 15102 ms | 25224 ms |
| the same 32 as singles | 12548 ms | 18914 ms |
| batch speedup | **0.83x** | **0.75x** |
| cold start | **8.9 s** | 8.6 s |
| peak memory | **762 MB** | not sampled |

Cold start is `docker compose start api` → first 200 from `/predict`, which is
the container equivalent of M3's process-spawn definition. The Dockerfile's
`HEALTHCHECK --start-period=60s` has 6.7x headroom over it, which is the thing
that number is for.

Peak memory is cgroup v2's `memory.peak`, read after a run including the batch
pass. **It is not the same measurement as the laptop's 849 MB**, which is
`getrusage` RSS for one process; this is the container's whole cgroup. They are
comparable in spirit, not interchangeable — and the point either way is that the
service fits in 3.7GB with room to spare.

## Laptop vs VM

| | laptop | VM | ratio |
|---|---|---|---|
| p50 | 59.0 ms | 444.5 ms | 7.5x slower |
| p95 | 66.7 ms | 579.0 ms | 8.7x slower |
| throughput ceiling | 24.3 req/s | 2.6 req/s | 9.3x lower |
| cold start | 3.6 s | 8.9 s | 2.5x slower |
| knee | concurrency 4–8 | **concurrency 2** | tracks core count |

**A 2x core difference produced a 9x throughput difference**, so roughly 4.5x of
it is per-core: an M2 performance core against a shared Xeon Skylake vCPU. Worth
stating plainly, because "2 vCPU is half of 4 cores" would have predicted the
wrong number by a factor of four, and sizing a box on core count alone is how
that mistake gets made.

**The knee moved to concurrency 2, and that is the core count.** Past it the
sweep is pure queueing and Little's law fits at every level: 2 ÷ 2.6 = 769 ms
mean against a measured p95 of 961 ms, 4 ÷ 2.6 = 1538 vs 1805, 8 ÷ 2.6 = 3077 vs
3404, 16 ÷ 2.6 = 6154 vs 6502. The gap is the mean-to-p95 spread and it stays
flat, which is what saturation looks like when nothing else is going wrong.

### Caddy and TLS cost nothing measurable

p50 through the public hostname is **447.5 ms against 444.5 ms direct** — 3 ms,
and that is *including a fresh TLS handshake per request*, since
`bench/latency.py` uses stdlib `urllib` with no keep-alive. Under load the
sweeps are indistinguishable: 2.6 req/s and p95 1811/3381 ms through Caddy
against 1805/3404 ms direct.

That is not a compliment to Caddy so much as a statement about where the time
goes. The forward pass is ~97% of the request on this box, so a reverse proxy,
TLS, and a rate limiter are all free at this latency scale. **The corollary is
the useful part: there is nothing to win by removing them**, which is worth
knowing before someone proposes terminating TLS elsewhere to save a hop.

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
it is recorded here only so nobody reads it as a regression.

**M6 settled it on the VM, where the arch is native**, using the same
`bench/latency.py --url` path: [444.5 ms p50](#single-request--vm) for the same
container image and the same input. The 897 ms was emulation, as claimed — the
native container is 2x slower than the laptop's *native* 59 ms because the box
has a quarter of the cores, not 15x slower because the image is bad.

---

## What the numbers say

Written against the laptop's figures; the VM equivalents are in
[Laptop vs VM](#laptop-vs-vm) above. The conclusions transfer, the magnitudes
do not.

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

**Re-tested on the 2 vCPU VM at M6, and the result holds.** This was the case
where SPEC §3 had the best chance of being right — 4x fewer cores, so a much
tighter thread-to-core ratio — and it still is not. Pinning costs **55% at p50**
(444.5 → 689.5 ms) and **31% at p95** (579.0 → 760.4 ms), while the throughput
ceiling is unchanged at 2.6–2.7 req/s either way.

If anything the VM makes the mechanism plainer. Pinned throughput is *marginally
higher* — 2.7 against 2.6 req/s, which is within noise but in the direction
oversubscription would predict — and it buys nothing, because throughput was
never the thing pinning could improve. Both configurations reach the same FLOPs
ceiling on both boxes; the only variable pinning moves is how many cores one
request gets, and that is a latency term.

**So the recommendation is the opposite of the spec's, on two boxes with a 4x
core difference: leave `NUM_THREADS` unset.** The service runs that way. Keeping
it an env var rather than a `set_num_threads(1)` at import time is what made this
a config change instead of a code change, and it is why the re-test cost a
`docker compose up -d` rather than a rebuild.

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

**The VM reproduced it, and sharpened the mechanism.** Measured there: **0.83x**
on default threads and **0.75x** pinned, against the laptop's 0.88x and 0.79x.
The prediction for a box with no spare parallelism to exploit is 1 ÷ 1.23 = 0.81,
and the ratios line up with how much parallelism each configuration actually has:

| | cores available to one request | measured |
|---|---|---|
| laptop, default | 4 | 0.88x |
| laptop, pinned | 1 | 0.79x |
| VM, default | 2 | 0.83x |
| VM, pinned | 1 | 0.75x |

**The batch ratio tracks available parallelism, and nothing else does.** Fewer
cores per request moves it toward the pure token-work ratio, which is the padding
argument stated as a prediction and then checked on a second machine rather than
asserted once. The two pinned figures sit a little below 0.81 — 0.79 and 0.75 —
so the model is close but not exact; there is a few percent of per-call overhead
the singles pay 32 times and the batch pays once, working the other way.

Batching would still pay if requests were short and varied — padding waste falls
with sequence length — or on a GPU, where one request cannot saturate the
device. Neither is this service. `/predict/batch` stays because one round trip
for 32 papers beats 32 round trips over a real network, which is a latency win
for the caller even when it is not a throughput win for the server.

### Memory

**Peak RSS is 849 MB against a 265 MB artifact** — 3.2x, which is fp32 weights
plus torch's allocator and the activations for a 32×384 batch. It is the floor
for M4's container limit and it is 46% above what a `ps` sample at the end of the
run suggested. "Comfortable on the 4GB VM" was a prediction at M3; measured
there it is [762 MB of cgroup peak](#batch-cold-start-memory--vm) against 3.7GB
of RAM, so the prediction held. The `NUM_THREADS=1` run
peaks marginally higher (859 MB), which is per-thread arena bookkeeping, not
anything meaningful.

**Cold start is 3.6 s.** That is the number that matters for a rolling restart
or an autoscaler, not the load time alone. Health checks must not go green
before it finishes — which is exactly what the lifespan warmup and the 503
branch are for.
