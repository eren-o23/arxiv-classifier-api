# arxiv-classifier-api

[![ci](https://github.com/eren-o23/arxiv-classifier-api/actions/workflows/ci.yml/badge.svg)](https://github.com/eren-o23/arxiv-classifier-api/actions/workflows/ci.yml)

Fine-tuned DistilBERT classifying arXiv papers by subject, served as a FastAPI service — containerised, load-tested, and deployed.

```bash
curl -X POST https://arxiv-classifier.duckdns.org/predict \
  -H 'content-type: application/json' \
  -d '{"title": "Attention Is All You Need",
       "abstract": "We propose the Transformer, a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on machine translation tasks show these models to be superior in quality while being more parallelizable."}'
```

```json
{
  "label": "cs.CL",
  "confidence": 0.918,
  "scores": {"cs.CL": 0.918, "cs.AI": 0.070, "cs.LG": 0.006, "...": "all 10 labels"},
  "top3": ["cs.CL", "cs.AI", "cs.LG"],
  "model_version": "v1.0.0",
  "latency_ms": 115.61
}
```

Live, with a real certificate. Rate limited to 30 requests/minute per IP on
`/predict`; `/health` and `/metadata` are not limited. Deploy details in
[docs/deploy.md](docs/deploy.md).

## Numbers

Measured on **the box actually serving this URL** — a Hetzner CX23, 2 vCPU /
3.7GB — not estimated and not a laptop. Same two scripts as always,
`bench/latency.py --url` and `bench/load.sh`, run on the VM. Full run and
interpretation in [docs/numbers.md](docs/numbers.md), which keeps the laptop's
figures alongside for comparison.

| | |
|---|---|
| p50 / p95 / p99 latency | 445 / 579 / 665 ms |
| Throughput ceiling | 2.6 req/s — [the knee is at concurrency 2, which is the core count](docs/numbers.md#laptop-vs-vm) |
| Batch-of-32 vs 32 singles | 0.83x — [batching does not pay on CPU](docs/numbers.md#batching-does-not-pay-on-cpu-and-padding-is-why) |
| `NUM_THREADS=1` vs default | 55% worse at p50 — [the spec predicted the opposite, on both boxes](docs/numbers.md#pinning-threads-to-1-does-not-help-here-and-the-spec-said-it-would) |
| Cold start | 8.9 s (container start → first 200 from `/predict`) |
| Peak memory | 762 MB (cgroup peak, against 3.7GB of RAM) |
| Image size | 562 MB compressed, 1.98 GB in Docker's store ([CPU-only torch pulls 15x fewer wheel bytes](docs/numbers.md#cpu-only-torch-is-a-15x-difference-in-what-you-download)) |
| Top-1 / top-3 accuracy | 0.775 / 0.988 ([details](docs/model_eval.md)) |

**The latency row is a full-length input** — 1,858 chars, 384 tokens, the
tokenizer's ceiling. The curl above is a short abstract and comes back in 116 ms,
because on CPU the forward pass scales with token count and nothing else here
matters much: Caddy, TLS and the rate limiter together cost
[about 3 ms](docs/numbers.md#caddy-and-tls-cost-nothing-measurable).

## Status

**Complete — M0 through M6.** Model evaluation in
[docs/model_eval.md](docs/model_eval.md); measured numbers in
[docs/numbers.md](docs/numbers.md); deploy runbook in
[docs/deploy.md](docs/deploy.md).

```bash
make model   # fetch the artifact at its pinned revision (models/ is gitignored)
make test    # 42 tests
make run     # serve on :8000
make bench   # latency, batch, cold start, RSS — spawns its own server
make load    # hey sweep at concurrency 1/2/4/8/16 (needs `brew install hey`)
make build   # container image, artifact pulled at the pinned revision
make smoke   # the M4 gate: health, predict, batch, validation over HTTP
make deploy  # on the VM: git pull + compose up (see docs/deploy.md)
```

- [x] M0 — model trained, artifact + card published ([`erenrosman/arxiv-classifier-v1`](https://huggingface.co/erenrosman/arxiv-classifier-v1))
- [x] M1 — tested package (`make test`, 18 tests, golden set of 20 real papers)
- [x] M2 — API (four endpoints, lifespan load + warmup, validation, JSON request log)
- [x] M3 — benchmarks ([docs/numbers.md](docs/numbers.md))
- [x] M4 — container (multi-stage, CPU-only torch, non-root, healthcheck — 562 MB)
- [x] M5 — deployed ([arxiv-classifier.duckdns.org](https://arxiv-classifier.duckdns.org/health) — compose + Caddy, TLS, rate limited)
- [x] M6 — load tested on the deployed box, CI running lint, tests, image build and container smoke
