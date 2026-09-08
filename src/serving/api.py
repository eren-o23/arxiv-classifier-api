"""The FastAPI app: four endpoints, lifespan loading, warmup, request logging."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from .config import get_settings
from .logging import configure, log_requests
from .model import ModelBundle
from .preprocessing import join
from .schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    Metadata,
    PredictRequest,
    PredictResponse,
    check_paper,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure(settings.log_level)
    app.state.bundle = ModelBundle.load(settings.model_dir, num_threads=settings.num_threads)
    # The first forward pass is 5-10x slower than steady state (lazy kernel
    # init). Eat it here so no real request does, and stay un-ready until it is
    # done — a health check that goes green before the service can serve is
    # worse than no health check.
    app.state.bundle.predict("warmup", "warmup")
    app.state.started = time.monotonic()
    app.state.ready = True
    yield


app = FastAPI(title="arxiv-classifier-api", lifespan=lifespan)
app.middleware("http")(log_requests)


def _bundle(request: Request) -> ModelBundle:
    """Every route that needs the model goes through here, so 503-before-ready
    is one check rather than one per endpoint."""
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(503, "model is still loading")
    return request.app.state.bundle


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response):
    state = request.app.state
    if not getattr(state, "ready", False):
        # Measured: a single uvicorn process does not accept connections until
        # the lifespan finishes, so an external probe gets a refused connection
        # here, not this 503. The branch is still the right contract — it is what
        # a proxied or multi-worker setup returns, and what the tests assert —
        # but do not expect to see it by curling a local `make run`.
        # Hand-rolled rather than `raise HTTPException`: the response model is
        # the same either way, and a probe reading `model_loaded` deserves the
        # real shape, not an error envelope.
        response.status_code = 503
        return HealthResponse(status="loading", model_version="", model_loaded=False,
                              uptime_s=0.0)
    return HealthResponse(
        status="ok",
        model_version=state.bundle.version,
        model_loaded=True,
        uptime_s=round(time.monotonic() - state.started, 1),
    )


@app.get("/metadata", response_model=Metadata)
def metadata(request: Request):
    return _bundle(request).card


# `def`, not `async def`: torch inference blocks, and FastAPI runs sync routes in
# a threadpool. As `async def` one prediction would stall the whole event loop.
@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest):
    bundle = _bundle(request)
    check_paper(body.title, body.abstract)

    started = time.perf_counter()
    result = bundle.predict(body.title, body.abstract)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    request.state.log = {
        "label": result["label"],
        "confidence": round(result["confidence"], 4),
        "input_chars": len(join(body.title, body.abstract)),
    }
    return PredictResponse(**result, model_version=bundle.version, latency_ms=latency_ms)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: Request, body: BatchRequest):
    bundle = _bundle(request)
    cap = get_settings().max_batch_size
    if not body.items:
        raise HTTPException(400, "batch is empty")
    if len(body.items) > cap:
        raise HTTPException(400, f"batch has {len(body.items)} items, limit is {cap}")
    # Validate everything before predicting anything: a batch is accepted or
    # rejected whole, so a bad item at index 30 must not cost 30 forward passes.
    for i, item in enumerate(body.items):
        check_paper(item.title, item.abstract, index=i)

    started = time.perf_counter()
    results = bundle.predict_many([(i.title, i.abstract) for i in body.items])
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    request.state.log = {
        "batch_size": len(results),
        "labels": [r["label"] for r in results],
        "input_chars": sum(len(join(i.title, i.abstract)) for i in body.items),
    }
    return BatchResponse(
        results=[PredictResponse(**r, model_version=bundle.version, latency_ms=latency_ms)
                 for r in results],
        model_version=bundle.version,
        latency_ms=latency_ms,
    )
