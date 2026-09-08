"""One structured JSON line per request.

Logs the predicted label and the input's character count, never the abstract
itself. That is a deliberate privacy default: the text is the user's, the shape
of it is what operating the service actually needs.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Any

from starlette.requests import Request

logger = logging.getLogger("serving.request")


def configure(level: str = "INFO") -> None:
    """Plain lines to stdout — the container runtime is the log collector."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


async def log_requests(request: Request, call_next: Any):
    """Middleware. Routes add prediction detail via `request.state.log`.

    The middleware cannot read the response body without buffering it, so the
    route deposits what only it knows (label, confidence, input_chars) and this
    merges it with what only the middleware knows (status, total latency).
    """
    # Truncated: the header is client-supplied and otherwise unbounded, and it
    # is written to every log line and echoed back.
    request_id = (request.headers.get("x-request-id") or "")[:64] or uuid.uuid4().hex[:12]
    request.state.log = {}
    request.state.request_id = request_id

    started = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except BaseException:
        # Starlette's ServerErrorMiddleware sits *outside* this one, so an
        # unhandled exception propagates past here and turns into a 500 that
        # this middleware never sees. Without this branch the only requests that
        # go unlogged are the ones that failed.
        status = 500
        raise
    finally:
        logger.info(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "request_id": request_id,
            "path": request.url.path,
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            **request.state.log,
        }))

    response.headers["x-request-id"] = request_id
    return response
