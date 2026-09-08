# Project: Model Serving Playground

Deploy a real ML model as a production-style API.

## The Core (What Gets Wrapped)

Take a genuinely useful ML task — e.g., a fine-tuned sentiment/topic classifier, a semantic search over a small corpus, or a simple recommendation engine — something with an actual model artifact (not just a script calling an LLM API). This matters for AI/ML eng roles: it shows you can handle model loading, inference, versioning, and latency, not just prompt orchestration.

## How the Pipeline Maps On

### 1. Tested Package

`src/` layout package with clear separation:
- `model/` (loading, inference)
- `preprocessing/`
- `api/`

Unit tests for preprocessing and inference logic (use a small fixed test set with known expected outputs), plus a test that the model loads correctly and produces consistent shapes/types. This is where you demonstrate ML-specific testing (not just software testing) — e.g. checking inference latency stays under a threshold, checking output distribution sanity.

### 2. FastAPI

- `POST /predict` with Pydantic input/output schemas
- `GET /health` that also reports model version/load status
- Ideally a `/metadata` endpoint (model version, training date, expected input shape)
- Basic request logging — useful talking point for "productionizing ML" in interviews

### 3. Containerise

- Multi-stage Dockerfile
- Pin exact dependency versions (critical for ML reproducibility — numpy/torch version mismatches are a classic pain point)
- Keep the image lean by not including training code/data in the runtime image

### 4. Deploy to a Cloud VM

- EC2 or a GPU-optional droplet depending on model size
- Docker + nginx/Caddy for TLS
- Add a simple load test (e.g. `locust` or `hey`) to report throughput/latency numbers — concrete numbers are strong interview material

## Why This Fits an ML Eng Job Specifically

- It demonstrates the full MLOps loop: model → tested code → API → container → deployed, with monitoring/metrics — which is what interviewers actually probe for.
- You could optionally layer in model versioning (swap models without downtime) or a simple CI pipeline (GitHub Actions running tests + building the image) to go further, since those keep coming up in ML eng interviews.
