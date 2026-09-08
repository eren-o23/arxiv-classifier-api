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

Measured on TODO, not estimated.

| | |
|---|---|
| p50 / p95 / p99 latency | TODO |
| Throughput @ concurrency 8 | TODO |
| Batch-of-32 vs 32 singles | TODO |
| Cold start | TODO |
| Image size | TODO |
| Top-1 / top-3 accuracy | 0.775 / 0.988 ([details](docs/model_eval.md)) |

## Status

In progress. Model evaluation in [docs/model_eval.md](docs/model_eval.md).

```bash
make model   # fetch the artifact at its pinned revision (models/ is gitignored)
make test    # 36 tests
make run     # serve on :8000
```

- [x] M0 — model trained, artifact + card published ([`erenrosman/arxiv-classifier-v1`](https://huggingface.co/erenrosman/arxiv-classifier-v1))
- [x] M1 — tested package (`make test`, 18 tests, golden set of 20 real papers)
- [x] M2 — API (four endpoints, lifespan load + warmup, validation, JSON request log)
- [ ] M3 — benchmarks
- [ ] M4 — container
- [ ] M5 — deployed
- [ ] M6 — load test + CI
