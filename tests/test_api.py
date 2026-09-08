"""Contract tests. Shapes and status codes only — latency belongs in bench/ at M3."""

from fastapi.testclient import TestClient

from serving.api import app

PAPER = {"title": "Sparse Attention", "abstract": "We propose a sparse attention mechanism."}


def test_health_before_load_is_503(monkeypatch):
    """The M2 gate: the state a real container is in for its first seconds.

    No `with` block, so the lifespan never runs. `ready` is forced off as well
    because the session-scoped client fixture starts this same module-level app,
    and this test must not depend on running before it.
    """
    monkeypatch.setattr(app.state, "ready", False, raising=False)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False

    # The other routes must refuse too — a 200 from /predict while /health says
    # 503 would mean the readiness flag guards only the probe.
    assert client.post("/predict", json=PAPER).status_code == 503
    assert client.get("/metadata").status_code == 503


def test_health_after_startup_is_200(client, bundle):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == bundle.version
    assert body["uptime_s"] >= 0


def test_metadata_is_the_card(client, bundle):
    assert client.get("/metadata").json() == bundle.card


def test_predict_contract(client, bundle):
    body = client.post("/predict", json=PAPER).json()
    assert set(body) == {"label", "confidence", "scores", "top3",
                         "model_version", "latency_ms"}
    assert list(body["scores"]) == bundle.labels
    assert abs(sum(body["scores"].values()) - 1.0) < 1e-6
    assert body["top3"][0] == body["label"]
    assert body["confidence"] == body["scores"][body["label"]]
    assert body["model_version"] == bundle.version
    assert body["latency_ms"] > 0


def test_predict_matches_the_bundle_directly(client, bundle, golden):
    """The API must not quietly change the prediction. Anything between the
    request and predict() — a stray strip, a re-tokenize — shows up here."""
    paper = golden["papers"][0]
    direct = bundle.predict(paper["title"], paper["abstract"])
    served = client.post("/predict", json={"title": paper["title"],
                                           "abstract": paper["abstract"]}).json()
    assert served["label"] == direct["label"]
    assert served["scores"] == direct["scores"]


def test_batch_matches_singles_through_the_api(client, golden):
    papers = [{"title": p["title"], "abstract": p["abstract"]} for p in golden["papers"][:4]]
    batched = client.post("/predict/batch", json={"items": papers}).json()
    assert len(batched["results"]) == 4
    for paper, result in zip(papers, batched["results"]):
        single = client.post("/predict", json=paper).json()
        assert result["label"] == single["label"]
        assert abs(result["confidence"] - single["confidence"]) < 1e-4


def test_request_id_is_echoed(client):
    """Correlating a log line with the request that made it is the whole point
    of the id, so it has to come back on the response."""
    r = client.post("/predict", json=PAPER, headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


def test_request_log_has_no_abstract_text(client, caplog):
    """The privacy default, asserted rather than trusted to review."""
    with caplog.at_level("INFO", logger="serving.request"):
        client.post("/predict", json=PAPER)
    line = next(r.message for r in caplog.records if "/predict" in r.message)
    assert PAPER["abstract"] not in line
    assert '"label"' in line and '"input_chars"' in line
