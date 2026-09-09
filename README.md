# arxiv-classifier-api

Fine-tuned DistilBERT classifying arXiv papers by subject, served as a FastAPI service — containerised, load-tested, and deployed.

```bash
curl -X POST https://TODO/predict \
  -H 'content-type: application/json' \
  -d '{"title": "Adaptive Gradient Clipping for Stable Low-Precision Training",
       "abstract": "We show that per-layer gradient clipping thresholds ..."}'
```

```json
{
  "label": "cs.LG",
  "confidence": 0.612,
  "top3": ["cs.LG", "stat.ML", "cs.AI"],
  "model_version": "v1.0.0",
  "latency_ms": 74.5
}
```

## Numbers

Measured on an Apple M2 (8 cores, 8GB), not estimated — `make bench` and
`make load`. Full run and interpretation in [docs/numbers.md](docs/numbers.md).
The M5 VM is 2 vCPU / 4GB, so M6 re-measures against the deployed box.

| | |
|---|---|
| p50 / p95 / p99 latency | 59.0 / 66.7 / 82.3 ms |
| Throughput @ concurrency 8 | 24.3 req/s (p95 375 ms) |
| Batch-of-32 vs 32 singles | 0.88x — [batching does not pay on CPU](docs/numbers.md#batching-does-not-pay-on-cpu-and-padding-is-why) |
| Cold start | 3.6 s (spawn → first 200 from `/predict`) |
| Peak RSS | 849 MB |
| Image size | 562 MB compressed, 1.98 GB in Docker's store ([CPU-only torch pulls 15x fewer wheel bytes](docs/numbers.md#cpu-only-torch-is-a-15x-difference-in-what-you-download)) |
| Top-1 / top-3 accuracy | 0.775 / 0.988 ([details](docs/model_eval.md)) |

## Status

In progress. Model evaluation in [docs/model_eval.md](docs/model_eval.md).

```bash
make model   # fetch the artifact at its pinned revision (models/ is gitignored)
make test    # 42 tests
make run     # serve on :8000
make bench   # latency, batch, cold start, RSS — spawns its own server
make load    # hey sweep at concurrency 1/2/4/8/16 (needs `brew install hey`)
make build   # container image, artifact pulled at the pinned revision
make smoke   # the M4 gate: health, predict, batch, validation over HTTP
```

- [x] M0 — model trained, artifact + card published ([`erenrosman/arxiv-classifier-v1`](https://huggingface.co/erenrosman/arxiv-classifier-v1))
- [x] M1 — tested package (`make test`, 18 tests, golden set of 20 real papers)
- [x] M2 — API (four endpoints, lifespan load + warmup, validation, JSON request log)
- [x] M3 — benchmarks ([docs/numbers.md](docs/numbers.md))
- [x] M4 — container (multi-stage, CPU-only torch, non-root, healthcheck — 562 MB)
- [ ] M5 — deployed
- [ ] M6 — load test + CI
