# Multi-stage. The builder holds pip's cache, the Hub's blob cache and the
# download machinery; the runtime gets a venv, the artifact and src/, nothing
# else. SPEC.md §8.
#
# No `apt install` in either stage: torch's manylinux wheels bundle their own
# libgomp, and nothing here compiles.
#
# --platform lives in the Makefile, not in these FROMs, so the M5 VM builds
# natively by overriding one variable instead of editing this file.
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# The lock lands before the model and before src/, so editing a route does not
# reinstall torch. The extra index is what makes `torch==X.Y.Z+cpu` resolvable —
# it is not on PyPI, which is the point.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cpu

# torch ships its C++ test suite, its headers and a copy of protoc in the wheel:
# 200MB of a runtime image that can never execute any of it. Same argument SPEC
# §8 makes for keeping training/ out, one layer down. `torch_shm_manager` is the
# one real binary in bin/ and it survives. Done here, in the builder, so the
# runtime layer never carries it in the first place.
#
# .pyc files stay, deliberately — they are 213MB, but the venv is root-owned and
# the service runs as `app`, so a stripped venv could not rewrite them and would
# recompile every module on every start. That is cold start, repeatedly, to save
# space once.
ENV SITE=/opt/venv/lib/python3.11/site-packages
RUN rm -rf $SITE/torch/test $SITE/torch/include \
           $SITE/pip $SITE/setuptools $SITE/pkg_resources && \
    find $SITE/torch/bin -type f ! -name torch_shm_manager -delete

# The artifact is not in git (265MB), so it comes from the Hub, pinned. Both
# values are --build-args from the Makefile, which is the single place the SHA
# lives; a copy here is how the two drift.
ARG HUB_REPO
ARG REVISION
# Under /tmp so the blob cache — a second full copy of the artifact — is thrown
# away with this stage rather than baked into the image.
ENV HF_HOME=/tmp/hf
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('${HUB_REPO:?required build-arg, use make build}', \
    revision='${REVISION:?required build-arg, use make build}', \
    local_dir='/opt/model')"


FROM python:3.11-slim AS runtime

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/model /opt/model
COPY src/ /app/src

# MODEL_DIR: the repo-relative default resolves inside the venv once the code is
# not a source tree, so the container has to say where the artifact is.
# PYTHONUNBUFFERED: one JSON log line per request goes to stdout, and `docker
# logs` should see it as it happens rather than in 8KB bursts.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    MODEL_DIR=/opt/model \
    PYTHONUNBUFFERED=1

# Nothing here needs to write to disk or bind a privileged port.
RUN useradd --system --no-create-home app
USER app

EXPOSE 8000

# stdlib rather than an apt-get for curl. urlopen raises on the 503 branch, so
# "still loading" reads as unhealthy, which is the contract. Cold start is 3.6s
# on a laptop; the start-period is generous because the 2 vCPU VM is slower and
# a container killed mid-load looks like a crash loop.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
