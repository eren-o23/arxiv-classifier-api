.PHONY: model test lint run

# The one place the artifact is pinned. At M4 the Dockerfile takes this as a
# --build-arg rather than hardcoding its own copy.
HUB_REPO  := erenrosman/arxiv-classifier-v1
REVISION  := 8eb5e4732aa4e3e2903df4d2e857f3cbf4ca6518
MODEL_DIR ?= models/arxiv-v1

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
