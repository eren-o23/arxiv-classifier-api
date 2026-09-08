# Model Serving Playground — Build Spec

Concrete plan for [model-serving-playground.md](model-serving-playground.md). Decisions are made, not listed as options — that's the point of a spec.

---

## 0. What you're building

A text classifier, fine-tuned once, served as an HTTP API, containerised, running on a real VM with TLS, with measured latency numbers in the README.

```
train once (Colab)  →  model artifact in the repo
                              │
                       loaded at startup
                              ↓
   request → validate → tokenize → inference → response
                              │
                       structured request log
```

The model is a fixed input to this project. Everything interesting is downstream of it.

---

## 1. The Model — build it, then stop touching it

**Task:** classify an arXiv paper (title + abstract) into its primary subject category.
**Base:** `distilbert-base-uncased`.
**Data:** [`TimSchopf/arxiv_categories`](https://huggingface.co/datasets/TimSchopf/arxiv_categories) — 204k titles + abstracts with the arXiv taxonomy attached. Filter to ~10 CS categories, sample ~40k rows, split 80/10/10.
**Where:** Colab free tier, ~15 minutes on a T4.

Suggested label set — pick for volume, and keep the overlapping ones:

```
cs.LG  machine learning        cs.CR  cryptography & security
cs.CV  computer vision         cs.SE  software engineering
cs.CL  computation & language  cs.DS  data structures & algorithms
cs.AI  artificial intelligence cs.NI  networking
cs.RO  robotics                stat.ML statistics / ML
```

**Keep cs.LG, cs.AI and stat.ML in, deliberately.** They overlap heavily — a paper on a new optimiser could legitimately carry any of the three, and which one it has depends on which the author happened to pick. Your top-1 accuracy will land somewhere around 75–82% and no amount of training fixes that, because the ceiling is in the labels, not the model.

That is the most valuable thing this dataset gives you. Report **top-1 and top-3 accuracy** and a confusion matrix. When the matrix shows your errors are concentrated in exactly the cs.LG/cs.AI/stat.ML block, you have a specific, evidenced answer to "what were the limitations of your model" — which beats a high number on an easy dataset in every interview that matters.

`training/train.py` — one script, run once, committed for provenance but never part of the runtime image.

Save to `models/arxiv-v1/`:

```
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.txt
model_card.json      ← you write this one
```

`model_card.json` is the file that makes this look like engineering:

```json
{
  "name": "arxiv-category-classifier",
  "version": "v1.0.0",
  "base_model": "distilbert-base-uncased",
  "trained_at": "2026-09-14T10:22:00Z",
  "dataset": "TimSchopf/arxiv_categories (cs subset, 40k)",
  "labels": ["cs.LG", "cs.CV", "cs.CL", "cs.AI", "cs.RO",
             "cs.CR", "cs.SE", "cs.DS", "cs.NI", "stat.ML"],
  "input_format": "title + \"\n\n\" + abstract",
  "max_input_chars": 4000,
  "max_tokens": 384,
  "metrics": {"top1_accuracy": 0.79, "top3_accuracy": 0.94, "macro_f1": 0.76}
}
```

Everything downstream reads this file: the API's `/metadata`, the label mapping, the validation limits, the tests. One source of truth, no constants duplicated in code.

**Input format matters and must be identical at train and serve time.** Title and abstract joined by a blank line. Getting this subtly different between `train.py` and `preprocessing.py` is *the* classic ML serving bug — the model quietly gets worse and every test still passes. Put the joining logic in `preprocessing.py` and have `train.py` import it. One function, two callers, impossible to drift.

### Where the artifact lives — not in git

A fine-tuned DistilBERT `model.safetensors` is ~265MB. GitHub hard-rejects any file over 100MB, so committing it isn't an option, and Git LFS on the free tier gives you 1GB of bandwidth a month that a few Docker builds would burn through.

Push it to the **HuggingFace Hub** instead (`eren-o23/arxiv-classifier-v1`), and have the Dockerfile pull it at build time, pinned to a commit SHA:

```python
snapshot_download("eren-o23/arxiv-classifier-v1", revision="<sha>", local_dir="/app/models/arxiv-v1")
```

`models/` stays gitignored. This isn't a workaround — separating the model registry from the code repo is what production setups actually do, and pinning the revision means a rebuild six months from now produces the identical image. It also makes `MODEL_DIR` and the version-swap story real rather than theoretical: a new model is a new Hub revision, not a new commit.

**If fine-tuning starts eating days, stop.** Grab a pretrained text classifier off HuggingFace, write a `model_card.json` for it, move on. The serving work is the project.

---

## 2. Layout

```
model-serving-playground/
├── Makefile                    # run, test, bench, build, deploy
├── Dockerfile                  # multi-stage
├── docker-compose.yml          # api + caddy
├── Caddyfile
├── pyproject.toml
├── requirements.lock           # fully pinned, hashes included
├── .dockerignore               # excludes training/, data/, notebooks/
├── models/
│   └── arxiv-v1/               # gitignored — pulled from the HF Hub
├── training/
│   └── train.py                # run once on Colab, not in the image
├── src/serving/
│   ├── config.py               # env vars, one place
│   ├── model.py                # load + predict, no web framework imports
│   ├── preprocessing.py        # clean + tokenize
│   ├── schemas.py              # Pydantic request/response
│   ├── api.py                  # FastAPI app, routes, lifespan
│   └── logging.py              # structured JSON request log
├── bench/
│   ├── latency.py              # p50/p95/p99, warm
│   └── load.sh                 # hey, concurrency sweep
├── tests/
│   ├── golden.json             # fixed inputs → expected outputs
│   ├── test_model.py
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   └── test_api.py
└── docs/
    └── numbers.md              # measured results
```

`model.py` must not import FastAPI. If it does, you can't test inference without spinning up a web app, and you've welded two layers together for no reason.

---

## 3. The Inference Path

Four details that separate "works on my laptop" from "serves traffic". Each is a few lines and each is a thing you can talk about.

**Load once, at startup.** FastAPI `lifespan`, model into `app.state`. Loading per request is the single most common mistake and it's a 500ms-per-call mistake.

```python
@asynccontextmanager
async def lifespan(app):
    app.state.model = ModelBundle.load(settings.model_dir)
    app.state.model.predict("warmup")     # see below
    yield
```

**Warm it up.** The first inference after load is 5–10× slower than steady state — lazy kernel initialisation. Fire one dummy prediction during startup so no real user eats it. Then note that the health check shouldn't report ready until the warmup completes.

**Pin the thread count.** `torch.set_num_threads(1)` on a small CPU box. Torch defaults to using every core per request; under concurrency that oversubscribes and makes p95 *worse*. Measure it both ways for the README — it's a surprising result and a good story.

**`torch.inference_mode()` around the forward pass.** Skips autograd bookkeeping you never use.

---

## 4. API Surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | One text → label, confidence, all class scores |
| `POST` | `/predict/batch` | Up to 32 texts in one forward pass |
| `GET` | `/health` | Liveness + model loaded + version + uptime |
| `GET` | `/metadata` | The model card, essentially verbatim |

**`POST /predict`**

```jsonc
// request
{
  "title": "Adaptive Gradient Clipping for Stable Low-Precision Training",
  "abstract": "We show that per-layer gradient clipping thresholds ..."
}

// response
{
  "label": "cs.LG",
  "confidence": 0.612,
  "scores": {"cs.LG": 0.612, "stat.ML": 0.201, "cs.AI": 0.09, ...},
  "top3": ["cs.LG", "stat.ML", "cs.AI"],
  "model_version": "v1.0.0",
  "latency_ms": 74.5
}
```

Take `title` and `abstract` as separate fields rather than one blob — the caller has them separately, and it keeps the train/serve join in your code where it belongs.

Ten classes is few enough to return every score, and `top3` alongside is worth it here: given the label overlap, top-3 is the number that reflects whether the model is actually right. A confidence of 0.61 with stat.ML second is the model correctly telling you the label is ambiguous.

**`POST /predict/batch`** exists to demonstrate a real throughput win: one forward pass over 32 texts is several times faster than 32 passes. Measure it, put the number in the README. Cap the batch at 32 and reject larger with a 400.

**`GET /health`** returns `{"status": "ok", "model_version": "v1.0.0", "model_loaded": true, "uptime_s": 8412}`. Returns 503 while the model is still loading or warming up — a health check that goes green before the service can serve is worse than none.

**Request logging** — one JSON line per request via middleware: `{"ts", "request_id", "path", "status", "latency_ms", "label", "confidence", "input_chars"}`. Log the predicted label and the character count, never the abstract text itself. That's a privacy decision you should be able to state out loud, and it's the correct default.

---

## 5. Input Validation

Pydantic checks that `text` is a string. It does not check that the string is sane. This is the trust boundary; don't be clever here.

| Rule | Response |
|---|---|
| Missing / wrong type | 422 (Pydantic handles it) |
| Empty / whitespace-only title or abstract | 400, naming which field |
| Over `max_input_chars` (4000, title + abstract) | 400, with the limit in the message |
| Batch over 32 items | 400 |
| Any item in a batch invalid | 400, reject the whole batch, name the index |

Cap the **characters** before tokenizing, not just `truncation=True` at the tokenizer. Truncation still tokenizes the whole string first — hand it 10MB and you've got a memory spike with a perfectly valid-looking response at the end.

Limits come from `model_card.json`, so they can't drift out of sync with what the model was trained for.

---

## 6. Tests — deterministic only

`golden.json`: 20 hand-picked papers with expected labels. Pull real abstracts off arXiv. Include the awkward ones deliberately — a clear cs.CV paper, a clear cs.CR paper, and three or four that sit right on the cs.LG / stat.ML / cs.AI boundary.

For the ambiguous ones, assert the correct label is **in the top 3**, not that it's top-1. Asserting top-1 on a genuinely ambiguous input gives you a test that fails for reasons that aren't bugs — and a test you learn to ignore is worse than no test.

| Test | Catches |
|---|---|
| `test_model_loads` | Artifact present, labels match the card, output shape `(1, 10)`, probabilities sum to 1.0 |
| `test_golden_labels` | The model still predicts what it predicted. Silent artifact swap, broken preprocessing |
| `test_train_serve_parity` | `preprocessing.join()` is the same function `train.py` used. The bug that breaks accuracy silently |
| `test_scores_valid` | Every score in [0,1], sums to 1, keys exactly match the card's labels |
| `test_determinism` | Same input twice → identical output, bit for bit |
| `test_preprocessing` | Whitespace, unicode, the truncation boundary at exactly max chars |
| `test_validation` | Every row of the table above returns the right status code |
| `test_api_contract` | TestClient: response shapes match the declared Pydantic models |
| `test_health_before_load` | `/health` returns 503 when the model isn't ready |

**No latency assertions in the test suite.** They pass on your laptop and fail randomly on a shared CI runner, and you'll end up ignoring red builds. Latency belongs in `bench/`, run deliberately.

On `test_golden_labels`: if you assert exact confidence values, a torch minor-version bump will break it. Assert the *label*, plus confidence within ±0.01 of a recorded value, and leave a comment saying regeneration is expected on a torch upgrade.

---

## 7. Benchmarks — where the interview numbers come from

`bench/latency.py` — 1000 warm sequential requests, report p50/p95/p99 and cold-start time (process start → first successful `/predict`).

`bench/load.sh` — `hey` at concurrency 1, 2, 4, 8, 16. Report throughput and p95 at each level. Find where it falls over and say why.

Record in `docs/numbers.md`:

- Single-request p50 / p95 / p99
- Batch-of-32 throughput vs 32 singles
- `set_num_threads(1)` vs default, at concurrency 8
- Cold start
- Image size, CPU-only vs default torch
- Peak RSS

Six real numbers about a thing you deployed puts you ahead of most candidates for this kind of role.

---

## 8. Docker

Multi-stage. The build stage gets pip, compilers, and the `snapshot_download` of the pinned model revision; the runtime stage gets a venv, your source, and the downloaded artifact — nothing else.

The one line that matters most:

```dockerfile
RUN pip install --no-cache-dir -r requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Default torch pulls CUDA libraries you will never use on this box: ~2.5GB image. CPU-only wheels: under 1GB. Report both figures — it's a concrete, verifiable win.

Also:
- `requirements.lock` fully pinned (`uv pip compile` or `pip-tools`). "Pin exact versions" in your .md means the lock file, not `>=`.
- Non-root user.
- `HEALTHCHECK` hitting `/health`.
- `.dockerignore` excluding `training/`, `data/`, `notebooks/`, `.git`. Training code in a runtime image is dead weight and extra attack surface.
- `MODEL_DIR` as an env var, defaulting to the baked-in artifact. Together with the pinned Hub revision, this is what makes version swapping real.
- The Hub revision SHA lives in one place (an ARG in the Dockerfile), and `model_card.json` travels with the artifact — so the running service can always tell you exactly which weights it has.

---

## 9. Deploy

Cheapest VM that comfortably fits: 2 vCPU / 4GB. A DigitalOcean droplet at ~$18/mo or an EC2 `t3.small`. DistilBERT on CPU is fine at 2GB but you'll be fighting it.

`docker-compose.yml`: your API + Caddy. Caddy over nginx+certbot — TLS is three lines and certificates renew themselves:

```
predict.yourdomain.com {
    reverse_proxy api:8000
}
```

`restart: unless-stopped` on both. Compose is your process supervisor; you don't need systemd on top of it.

Deploy is `git pull && docker compose up -d --build`, wrapped in `make deploy`. Not fancy, and fancy isn't what this project is proving.

---

## 10. CI

GitHub Actions on push: lint (`ruff`), tests, build the image. Optionally push to GHCR.

Building the image in CI is the part worth having — it catches "works locally, breaks in the container", which is exactly the failure this project exists to teach you about.

---

## 11. Milestones

Each ends with something runnable. Don't move on until the check passes.

**M0 — Model exists.** `training/train.py` runs on Colab, artifact + `model_card.json` land in `models/arxiv-v1/`, confusion matrix saved.
*Done when:* a 5-line local script loads the artifact and correctly labels an abstract you paste in from arXiv.
*Timebox: one day. Over that, use a pretrained checkpoint and move on.*

**M1 — Package.** `src/` layout, `model.py` + `preprocessing.py`, golden tests passing. No web framework yet.
*Done when:* `make test` is green and `model.py` has no FastAPI import.

**M2 — API.** All four endpoints, lifespan loading, warmup, validation, request logging.
*Done when:* `/health` returns 503 during startup and 200 after, and every validation row returns the right code.

**M3 — Measure.** `bench/latency.py`, thread-count comparison, batch vs single.
*Done when:* `docs/numbers.md` has real figures.

**M4 — Container.** Multi-stage build, CPU torch, pinned lock file, non-root, healthcheck.
*Done when:* the image is under 1GB and `docker run` serves a correct prediction.

**M5 — Deployed.** VM, compose, Caddy, TLS.
*Done when:* someone else can `curl https://your-domain/predict` and get a label back.

**M6 — Load tested + CI.** `hey` sweep against the deployed box, Actions running tests and building the image.
*Done when:* the README opens with your numbers and a working curl example.

---

## 12. Where Phase 1 Shows Up

| Source | Where |
|---|---|
| Fluent Python | `slots` dataclass for the model bundle, `lru_cache` on config, context manager for the inference timer, `Protocol` for the predictor interface, comprehensions over batch results |
| FastAPI docs | Lifespan, dependency injection, Pydantic v2 models, custom exception handlers, `TestClient` |
| Missing Semester | Makefile as the interface, `.dockerignore`, SSH + key setup on the VM, `hey`/`curl` for load testing, GitHub Actions, reading container logs |
| DDIA | Lightly — mostly ch 1 (reliability/latency percentiles: p95 and p99, not averages). The chapters land properly in a later project |
| SQL | Not here. Don't force it |

---

## 13. Deliberately Not Building

- **A model registry / MLflow** — `model_card.json` plus an env var does the version-reporting job at this scale.
- **Micro-batching / request queuing** — real serving systems batch across concurrent requests. Add it only if the load test proves you need it, so the decision is measurement-driven.
- **Multi-label** — arXiv papers carry cross-listed categories as well as a primary one, so the data supports it for free. It's a real upgrade (sigmoid, per-label thresholds, a different response schema) and a natural v2 if you want one after this ships. Not now.
- **Kubernetes** — one container on one VM. Reaching for k8s here would be the loudest possible signal you've never run anything in production.
- **A frontend** — `curl` in the README is the demo.
- **GPU** — DistilBERT on CPU is tens of milliseconds. A GPU would be spend without a reason.
- **Auth / rate limiting** — add a Caddy rate limit if you put the URL somewhere public. Otherwise out of scope.
