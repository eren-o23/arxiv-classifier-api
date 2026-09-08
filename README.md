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
| Top-1 / top-3 accuracy | TODO |

## Status

In progress. Build plan in [SPEC.md](SPEC.md), original framing in [BRIEF.md](BRIEF.md).

- [ ] M0 — model trained, artifact + card published
- [ ] M1 — tested package
- [ ] M2 — API
- [ ] M3 — benchmarks
- [ ] M4 — container
- [ ] M5 — deployed
- [ ] M6 — load test + CI
