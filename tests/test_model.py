"""Inference tests. Deterministic only — latency belongs in bench/ at M3."""

import ast
import re
from pathlib import Path

import pytest

from serving.preprocessing import MAX_INPUT_CHARS

REPO = Path(__file__).resolve().parents[1]

# Confidences in golden.json are recorded from a specific torch build. A minor
# version bump moving one by more than this is expected, not a bug: re-record
# with `python scripts/build_golden.py --refresh`.
CONFIDENCE_TOLERANCE = 0.01


def test_model_loads(bundle):
    assert len(bundle.labels) == 10
    assert bundle.version == "v1.0.0"
    # load() already asserts card labels == config id2label; this pins the
    # shape and the probability contract that the API's response depends on.
    result = bundle.predict("A title", "An abstract")
    assert set(result["scores"]) == set(bundle.labels)
    assert sum(result["scores"].values()) == pytest.approx(1.0)


def test_scores_valid(bundle, golden):
    for paper in golden["papers"]:
        result = bundle.predict(paper["title"], paper["abstract"])
        assert list(result["scores"]) == bundle.labels, "score keys must keep card order"
        assert all(0.0 <= s <= 1.0 for s in result["scores"].values())
        assert sum(result["scores"].values()) == pytest.approx(1.0)
        assert result["top3"] == sorted(
            result["scores"], key=result["scores"].__getitem__, reverse=True)[:3]
        assert result["label"] == result["top3"][0]
        assert result["confidence"] == result["scores"][result["label"]]


def test_golden_labels(bundle, golden):
    """The model still predicts what it predicted. Catches a silent artifact
    swap, a reordered label map, and broken preprocessing."""
    papers = golden["papers"]
    results = bundle.predict_many([(p["title"], p["abstract"]) for p in papers])

    for paper, result in zip(papers, results):
        where = f"{paper['arxiv_id']} ({paper['expected']}, {paper['assert']})"
        if paper["assert"] == "top1":
            assert result["label"] == paper["expected"], where
        else:
            assert paper["expected"] in result["top3"], where
        assert result["label"] == paper["predicted"], where
        assert result["confidence"] == pytest.approx(
            paper["confidence"], abs=CONFIDENCE_TOLERANCE), where


def test_batch_matches_singles(bundle, golden):
    """One forward pass over N must give what N passes of one give. Padding in a
    batch is the thing that quietly breaks this."""
    papers = golden["papers"][:4]
    batched = bundle.predict_many([(p["title"], p["abstract"]) for p in papers])
    for paper, result in zip(papers, batched):
        single = bundle.predict(paper["title"], paper["abstract"])
        assert result["label"] == single["label"]
        assert result["confidence"] == pytest.approx(single["confidence"], abs=1e-4)


def test_golden_matches_the_shipped_artifact(bundle, golden):
    """golden.json records which artifact its numbers came from. If that drifts
    from what is actually loaded, the confidences are measuring one model while
    the file claims another, and nothing else goes red."""
    assert golden["model_version"] == bundle.version
    revision = re.search(r"^REVISION\s*:=\s*(\S+)", (REPO / "Makefile").read_text(), re.MULTILINE)
    assert revision and golden["model_revision"] == revision.group(1), \
        "golden.json and the Makefile disagree about which revision is shipped"


def test_empty_batch(bundle):
    # M2 rejects an empty batch at the schema layer, but predict_many() should
    # not be where it blows up with an IndexError out of torch.
    assert bundle.predict_many([]) == []


def test_determinism(bundle):
    title, abstract = "Sparse Attention", "We propose a sparse attention mechanism."
    assert bundle.predict(title, abstract) == bundle.predict(title, abstract)


def test_oversized_input_is_truncated_not_crashed(bundle):
    # join() caps characters before the tokenizer sees them, so a huge abstract
    # is a truncated prediction, not a memory spike.
    result = bundle.predict("Title", "word " * 200_000)
    assert result["label"] in bundle.labels
    assert sum(result["scores"].values()) == pytest.approx(1.0)


def test_train_serve_parity():
    """train.py must still get its text formatting from serving.preprocessing.

    Static check on purpose: importing train.py drags in datasets, sklearn and
    matplotlib, none of which are runtime dependencies. If train-time and
    serve-time formatting ever diverge the model quietly degrades and every
    other test in this file still passes.
    """
    tree = ast.parse((REPO / "training" / "train.py").read_text())
    imports_join = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "serving.preprocessing"
        and any(alias.name == "join" for alias in node.names)
        for node in ast.walk(tree)
    )
    defines_join = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "join"
        for node in ast.walk(tree)
    )
    assert imports_join, "train.py no longer imports join() from serving.preprocessing"
    assert not defines_join, "train.py defines its own join() — it must not have a second copy"


def test_model_has_no_web_framework_import():
    """The M1 gate: inference must be testable without standing up an app."""
    tree = ast.parse((REPO / "src" / "serving" / "model.py").read_text())
    imported = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    assert not imported & {"fastapi", "starlette", "uvicorn", "pydantic"}


def test_golden_expectations_are_sane(golden):
    """Guards the fixture itself: stat.ML is never predicted (0/17 recall), so a
    golden expectation naming it would fail for reasons that aren't bugs."""
    papers = golden["papers"]
    assert len(papers) == 20
    assert not any(p["expected"] == "stat.ML" for p in papers)
    assert len({p["arxiv_id"] for p in papers}) == len(papers)
    assert sum(p["assert"] == "top3" for p in papers) >= 4, "keep the ambiguous cases"
    assert all(len(p["abstract"]) < MAX_INPUT_CHARS for p in papers), \
        "a truncated golden abstract makes the fixture depend on the cap"
