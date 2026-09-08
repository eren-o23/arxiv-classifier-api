.PHONY: model test lint run bench load lock

# The one place the artifact is pinned. At M4 the Dockerfile takes this as a
# --build-arg rather than hardcoding its own copy.
HUB_REPO  := erenrosman/arxiv-classifier-v1
REVISION  := 8eb5e4732aa4e3e2903df4d2e857f3cbf4ca6518
MODEL_DIR ?= models/arxiv-v1

# Default torch pulls ~2.5GB of CUDA libraries this box will never use. This
# index is the CPU-only build, and both `lock` and the Dockerfile need it — a
# lock resolved here pins `torch==X.Y.Z+cpu`, which does not exist on PyPI.
TORCH_CPU  := https://download.pytorch.org/whl/cpu

# models/ is gitignored, so a fresh clone has no artifact. This fetches it.
model:
	uv run python -c "from huggingface_hub import snapshot_download; \
	snapshot_download('$(HUB_REPO)', revision='$(REVISION)', local_dir='$(MODEL_DIR)')"

test:
	MODEL_DIR=$(MODEL_DIR) uv run pytest -q

lint:
	uv run ruff check .

run:
	MODEL_DIR=$(MODEL_DIR) uv run uvicorn serving.api:app --reload

# M3. Spawns its own server, so no `make run` in another shell first.
# NUM_THREADS is a settings knob, so the thread comparison is two runs of this,
# not a code change: `make bench THREADS="--threads 1"`.
bench:
	MODEL_DIR=$(MODEL_DIR) uv run python bench/latency.py $(THREADS)

load:
	MODEL_DIR=$(MODEL_DIR) bench/load.sh

# requirements.lock is generated, committed, and never hand-edited. uv strips
# --extra-index-url out of the header it writes, so this target is the only
# record of how the file was produced.
lock:
	uv pip compile pyproject.toml --python-version 3.11 \
	  --python-platform x86_64-unknown-linux-gnu \
	  --extra-index-url $(TORCH_CPU) --index-strategy unsafe-best-match \
	  --generate-hashes -o requirements.lock
