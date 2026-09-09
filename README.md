# arxiv-classifier-api

[![CI](https://github.com/eren-o23/arxiv-classifier-api/actions/workflows/ci.yml/badge.svg)](https://github.com/eren-o23/arxiv-classifier-api/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--only-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model%20registry-FFD21E?logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Caddy](https://img.shields.io/badge/Caddy-TLS-1F88C0?logo=caddy&logoColor=white)

**A fine-tuned DistilBERT, served as an HTTP API on a real box, with numbers that came off that box.**

**Why it exists.** Training a model is the part with a tutorial. Everything after it — loading the
weights once instead of per request, validating input before it reaches the tokenizer, keeping
train-time and serve-time text formatting identical, getting the artifact into a container without
committing 265MB to git, and finding out what the thing actually costs under load — is the part
that decides whether a model is a notebook or a service. This is that part, built end to end and
measured rather than estimated.

**Start here.** [What it actually predicts](#what-it-actually-predicts) is ten lines and contains
the project's central finding. If you read one design section, make it
[the train/serve seam](#the-trainserve-seam-is-the-spine). If you only want to know whether the
numbers are any good, [they are here](#the-numbers) and two of them contradict the spec I wrote
before measuring.

## The short version

- **Live at [arxiv-classifier.duckdns.org](https://arxiv-classifier.duckdns.org/docs)**, real
  certificate, on a 2 vCPU box costing about $9.20/month.
- **Four endpoints** — `/predict`, `/predict/batch`, `/health`, `/metadata` — with the model loaded
  once at startup and warmed before the health check goes green.
- **`model_card.json` is the single source of truth.** Labels, character limit, token limit and
  version all come from the artifact, so no constant is duplicated in code and nothing can drift.
- **The artifact is not in git.** It lives on the Hugging Face Hub pinned to a commit SHA, which
  the Makefile holds in one place and passes to the Docker build. A rebuild in six months produces
  the identical image.
- **562 MB image**, because CPU-only torch pulls 194 MB of wheels where the default pulls 2921 MB
  of CUDA libraries that a box with no GPU can never execute.
- **Measured, not asserted.** p50 445 ms, ceiling 2.6 req/s, cold start 8.9 s — all taken on the
  deployed VM with the same two scripts used on the laptop, and
  [two of the results contradict the spec](#two-predictions-the-measurements-broke).
- **42 tests**, no services needed, plus a stdlib-only smoke script that runs unchanged against a
  local container, the deployed box, and CI.

**Stack:** Python 3.11 · FastAPI · PyTorch (CPU-only) · Transformers · Hugging Face Hub ·
Docker · Caddy · uv · GitHub Actions.

## What it actually predicts

Four papers from `tests/golden.json`, all 2026 arXiv submissions the model never saw, sent to the
live service. `arxiv=` is the primary category the *author* filed it under; `pred=` is the model:

```
2609.05416  arxiv=cs.CV   pred=cs.CV   ok     512ms   cs.CV 0.900  cs.AI 0.091  cs.RO 0.004
    WorldSculpt: Generating Compositional Worlds from Grounded Videos

2609.05337  arxiv=cs.LG   pred=cs.LG   ok     386ms   cs.LG 0.874  cs.AI 0.048  stat.ML 0.035
    Variational Continuation for Double Pendulum Periodic Orbits

2609.05221  arxiv=cs.CL   pred=cs.AI   MISS   420ms   cs.AI 0.689  cs.CL 0.298  cs.LG 0.008
    A Verifier-Guided Explainable Reasoning Framework with Gold-Anchored QLoRA

2609.05363  arxiv=cs.LG   pred=cs.CL   MISS   466ms   cs.CL 0.549  cs.AI 0.434  cs.LG 0.010
    Distill Globally, Adapt Locally: Reasoning Distillation and Test-Time Training
```

**The two misses are the interesting rows, and they are not the same kind of miss.**

Row three is the failure this task is made of. A verifier-guided reasoning framework with QLoRA is
a language paper *and* an AI paper; the author picked cs.CL, the model picked cs.AI, and the
correct answer is sitting second with 0.298 of the mass. Nothing is broken — the label was a
judgement call and the model made a different one.

Row four is a worse miss and worth showing rather than hiding. The model splits 0.549/0.434 between
cs.CL and cs.AI while arXiv's answer, cs.LG, is third at 0.010. It scrapes into top-3 and no
further. A README that only showed row three would be making the model's failures look tidier than
they are.

That distinction is why this service returns **all ten scores and a `top3`**, not just a label.
A confidence of 0.549 with a live second place is the model telling you the input is ambiguous,
and that is information a caller can act on.

## The numbers

Two sets, and the gap between them is the point. Full working, including the arithmetic behind
every claim, in [docs/numbers.md](docs/numbers.md).

### Serving, on the deployed box

Hetzner CX23, 2 vCPU / 3.7GB. Measured **on the VM**, direct to the container, with
`bench/latency.py --url` and `bench/load.sh` — the same scripts that produced the laptop's figures.

| | value | |
|---|---|---|
| p50 / p95 / p99 | **445 / 579 / 665 ms** | a full 384-token input |
| throughput ceiling | **2.6 req/s** | [knee at concurrency 2, which is the core count](docs/numbers.md#laptop-vs-vm) |
| cold start | 8.9 s | container start → first 200 from `/predict` |
| peak memory | 762 MB | cgroup peak, against 3.7GB of RAM |
| Caddy + TLS + rate limiting | ~3 ms | [including a handshake per request](docs/numbers.md#caddy-and-tls-cost-nothing-measurable) |
| image | 562 MB compressed | [1.98 GB in Docker's store — three sizes, not interchangeable](docs/numbers.md#image--m4) |

**445 ms is not fast, and the honest reason is that nothing was done to make it fast.** This is
fp32 DistilBERT at the tokenizer's full 384 tokens on two shared Xeon vCPUs. Int8 dynamic
quantization or ONNX Runtime would typically be worth 2–3x and neither is here — that was out of
scope, not overlooked. The same service answers the short abstract in
[Run it](#run-it) in 116 ms, because on CPU the forward pass tracks token count and little else.

For comparison the laptop that trained it — an M2, 8 cores — does p50 59 ms and 24.3 req/s. **A 2x
core difference produced a 9x throughput difference**, so roughly 4.5x of it is per-core. Sizing a
box on core count alone would have been wrong by a factor of four.

### Model quality

Held-out split, 2,782 papers. Per-class breakdown and the confusion matrix in
[docs/model_eval.md](docs/model_eval.md).

| metric | value | |
|---|---|---|
| top-1 accuracy | 0.775 | majority-class baseline is 0.315 |
| top-3 accuracy | **0.988** | the one that matters here |
| macro F1 | 0.724 | ~0.80 across the nine live classes |

**The top-1/top-3 gap is the whole result.** At 0.988 top-3 the model nearly always understands the
paper; it loses top-1 on which of several defensible labels the author happened to tick. That
ceiling is in the labels, not the weights, and no amount of further training moves it.

Two weaknesses, both kept in and reported rather than tuned away:

- **cs.AI recall is 0.355.** Its errors go to cs.LG (109), cs.CL (87) and cs.CV (83) — papers
  genuinely about learning, language and vision that their authors filed under cs.AI. It is a
  grab-bag, not a subject.
- **stat.ML is a dead class: 0 of 17, F1 0.000.** As a *primary* category it is 0.6% of the data,
  because stat.ML papers list cs.LG first, so the model correctly learned that guessing it never
  pays. Dropping it would lift macro-F1 to about 0.80 and make the evaluation less true; macro-F1
  is reported precisely so a dead class cannot hide behind top-1 accuracy.

## Run it

Against the live service:

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

That JSON is the service's actual answer to the request above, replayed rather than typed. It is
also load-bearing: shortening that abstract by one sentence flips it to `cs.CV` at 0.549, which is
the label overlap showing up in the README. Rate limit is 30 requests/minute per IP on `/predict*`;
`/health` and `/metadata` are unlimited. The bare domain redirects to the Swagger UI.

Locally:

```bash
make model   # fetch the artifact at its pinned revision (models/ is gitignored)
make test    # 42 tests, ~6s
make run     # serve on :8000
make bench   # latency, batch, cold start, RSS — spawns its own server
make load    # hey sweep at concurrency 1/2/4/8/16
make build   # container image, artifact pulled at the pinned revision
make smoke   # health, predict, batch, validation over HTTP — works against any URL
```

## How it works

```
  title + abstract
        │
        ▼
  check_paper()  ─────────────► 400   empty field, >4000 chars, batch >32
        │                             (422 is Pydantic's, and means something else)
        ▼
  preprocessing.join()                ← the same function train.py imported
        │
        ▼
  tokenize, 384 max  →  DistilBERT  →  softmax over 10 labels
        │
        ▼
  label · confidence · all 10 scores · top3 · latency_ms
        │
        └────────────────────────────► one JSON log line per request:
                                       label, confidence, input_chars — never the text
```

- **`model.py`** owns loading and inference and **imports no web framework** — a test enforces it.
  `ModelBundle` is a frozen slots dataclass holding model, tokenizer and card; `load()` checks the
  card's limits against `preprocessing.py` and its label order against `config.json`, because a
  card claiming to be the source of truth is worth nothing if nothing verifies it.
- **`api.py`** holds the four routes and the lifespan that loads and warms the model once. The
  predict routes are `def`, not `async def`, so torch's blocking forward pass runs in FastAPI's
  threadpool instead of stalling the event loop.
- **`schemas.py`** carries the Pydantic models and `check_paper()`, which is the validation table
  in code.
- **`config.py`** reads four environment variables through an `lru_cache`d frozen dataclass. No
  `pydantic-settings`; four `os.environ` reads do not justify a dependency.
- **`logging.py`** writes one JSON line per request. The route deposits label, confidence and
  input length on `request.state`; the middleware merges status and total latency. It is registered
  *last* so it sits outermost — swap it with the body-size middleware and every 413 goes unlogged.

In production Caddy terminates TLS, caps bodies at 1MB and rate-limits `/predict*`. The api
container uses `expose`, never `ports`, so 8000 is unreachable from outside and neither cap can be
walked around by hitting the IP.

### The train/serve seam is the spine

`training/train.py` imports `join()` from `src/serving/preprocessing.py`. One function, two
callers, and it is the single most load-bearing line in the repo.

If train-time and serve-time text formatting ever diverge — a different separator, a strip that
one side does and the other does not — the model quietly gets worse and **every test still
passes**. There is no exception, no error, no bad status code. Accuracy just drops and nothing
tells you. It is the classic ML serving bug precisely because it is invisible.

The same reasoning produced the other two rules the codebase will not bend on: `model_card.json`
is the only place labels, limits and version live, and `config.json`'s `id2label` must match the
card's `labels` index for index — checked at load, because a silent permutation there returns
confidently wrong answers at full confidence. Both are verified by deliberately breaking a copy of
the artifact and confirming the load fails, not by reading the code and agreeing with it.

## Three decisions worth defending

**400 and 422 mean different things, so validation is not Pydantic validators.** Pydantic checks
shape — a missing field or a number where a string belongs is a 422, and that is its job. But an
empty abstract, or 10,000 characters of one, is a well-formed request that this service will not
serve, and that is a 400 naming the field and the limit. Collapsing the two would tell a caller
"you sent malformed JSON" when they did not. `check_paper()` exists to keep the distinction, and
every row of it has a test.

**The character cap runs before the tokenizer, not as `truncation=True`.** Truncation still
tokenizes the whole string first, so a 10MB abstract becomes a memory spike with a perfectly
valid-looking response at the end. The same argument applies one layer up: the batch cap checks
`Content-Length` before Pydantic builds the list, because a 17MB batch grew RSS by hundreds of MB
before reaching its 400. Now it is a 413 in 0.18 ms.

**The model lives on the Hub, pinned to a SHA, not in git.** A 265MB `safetensors` is over
GitHub's hard limit and LFS's free bandwidth would not survive a few Docker builds — but the real
reason is that separating the model registry from the code repo is what production setups do. The
SHA lives in exactly one place, the Makefile, and is passed to the Docker build as an argument. A
new model is a new Hub revision, not a new commit, which makes `MODEL_DIR` and the version-swap
story real rather than theoretical.

## Two predictions the measurements broke

I wrote a spec before building this, and it made two performance predictions confidently. Both are
wrong, and finding that out is worth more than either would have been if true.

**`torch.set_num_threads(1)` was supposed to improve p95 under concurrency.** The reasoning is
sound: torch uses every core per request, so concurrent requests oversubscribe. Measured on the
laptop it made p95 *worse* at every level. The obvious objection was core count — 8 cores is a lot
— so it was re-tested on the 2 vCPU VM, where the thread-to-core ratio is 4x tighter and the
prediction had its best chance. **It lost again: 55% worse at p50, 31% at p95, with the throughput
ceiling unchanged.** Both configurations hit the same FLOPs ceiling, so pinning only ever costs
single-request latency and never earns it back. The service runs with `NUM_THREADS` unset.

**A batch of 32 was supposed to be several times faster than 32 single calls.** It is **0.83x** —
slightly slower. The cause is padding: `predict_many` pads to the longest member, so 32 papers
whose actual lengths sum to 10,004 tokens become a 32×384 = 12,288-token batch, 1.23x the work.
The prediction for a box with no spare parallelism is 1 ÷ 1.23 = 0.81, and the measured ratio
tracks cores-available-to-one-request across four configurations:

| | cores per request | measured |
|---|---|---|
| laptop, default | 4 | 0.88x |
| VM, default | 2 | 0.83x |
| laptop, pinned | 1 | 0.79x |
| VM, pinned | 1 | 0.75x |

`/predict/batch` stays anyway, because one round trip for 32 papers still beats 32 round trips over
a real network. It is a latency win for the caller even though it is not a throughput win for the
server — which is a different claim from the one the spec made, and the honest one.

## Trade-offs and known ceilings

- **int8 quantization is measured but not shipped.** `QUANTIZE=1` is 1.5–1.7x on the VM at no
  measurable accuracy cost (top-1 +0.0014 over the full 2,782-paper split), but it moves 18 of 20
  golden confidences outside their ±0.01 band, and this service sells its score distribution as
  actionable. [The full result and the three reasons it is still off](docs/numbers.md#int8-quantization--measured-not-shipped).
  ONNX Runtime is the other candidate and would also drop torch from the image.
- **One uvicorn worker on one box.** 2.6 req/s is the ceiling and there is no horizontal story.
  The load test says the knee is at concurrency 2, so a balancer in front should cap in-flight
  requests there rather than let a queue build that only adds latency.
- **The rate limiter is in-memory**, so any redeploy clears every client's window.
- **`/health`'s 503 branch never fires under a single uvicorn process** — uvicorn refuses
  connections until the lifespan finishes, so an external probe gets a refused connection instead.
  The branch is still correct for a proxied or multi-worker setup and is tested through
  `TestClient`; it just is not observable in this deployment.
- **The golden set is curated, so it is not an accuracy measure.** 14 of its 20 papers assert top-1
  because they were chosen to. The honest numbers are in `docs/model_eval.md`, on a proper
  held-out split.
- **213 MB of `.pyc` is deliberately kept in the image.** The venv is root-owned and the service
  runs as `app`, so stripping them would mean recompiling every module on every start — trading
  cold start repeatedly to save space once. The 200 MB of gtest binaries, C++ headers and `protoc`
  that torch ships *is* stripped, because none of it can execute in a serving image.
- **No auth.** Out of scope; the rate limit is what stands between the URL and abuse.

## Tests

```bash
make test    # 42 tests, no network, no services
make lint
```

CI runs both on every push, then builds the image, starts the container and points the smoke script
at it — because "it built" and "it serves a correct prediction" are different claims and only the
second one matters. `scripts/smoke.py` is stdlib-only with no project imports, which is what lets
the same script be the gate for a local container, the deployed box over TLS, and CI.

The tests that earned their place were the ones written after something broke: a load-time check
that the card's limits match `preprocessing.py`, because the card claimed to be the source of truth
and nothing verified it; a guard on `golden.json`'s recorded model revision, because a golden file
that silently re-records itself stops catching the regression it exists for; and a health-check
ordering test, because the gate shares a module-level app with the session fixture and must not
depend on running first.

**Docs:** [numbers](docs/numbers.md) · [model evaluation](docs/model_eval.md) ·
[deploy runbook](docs/deploy.md)
