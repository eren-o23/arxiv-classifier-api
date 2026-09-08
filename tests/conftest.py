import json
from pathlib import Path

import pytest

from serving.model import DEFAULT_MODEL_DIR, ModelBundle

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def bundle() -> ModelBundle:
    """Loaded once for the whole run — it is ~265MB off disk.

    Hard-fails rather than skipping when the artifact is absent. models/ is
    gitignored, so a fresh clone and CI both start without it, and a green run
    that never touched the model is worse than a red one.
    """
    if not (DEFAULT_MODEL_DIR / "model_card.json").exists():
        pytest.exit(f"{DEFAULT_MODEL_DIR} not found — run `make model`", returncode=1)
    return ModelBundle.load()


@pytest.fixture(scope="session")
def golden() -> dict:
    return json.loads((REPO / "tests" / "golden.json").read_text())
