"""Every row of SPEC.md §5, plus the boundaries either side of each limit."""

from serving.preprocessing import MAX_INPUT_CHARS

PAPER = {"title": "Sparse Attention", "abstract": "We propose a sparse attention mechanism."}


def test_missing_field_is_422(client):
    assert client.post("/predict", json={"title": "only a title"}).status_code == 422


def test_wrong_type_is_422(client):
    assert client.post("/predict", json={"title": 7, "abstract": ["x"]}).status_code == 422


def test_empty_title_is_400_naming_the_field(client):
    r = client.post("/predict", json={**PAPER, "title": ""})
    assert r.status_code == 400
    assert "title" in r.json()["detail"]


def test_whitespace_only_abstract_is_400_naming_the_field(client):
    r = client.post("/predict", json={**PAPER, "abstract": "  \n\t "})
    assert r.status_code == 400
    assert "abstract" in r.json()["detail"]


def test_over_the_char_limit_is_400_with_the_limit_in_the_message(client):
    r = client.post("/predict", json={"title": "t", "abstract": "x" * MAX_INPUT_CHARS})
    assert r.status_code == 400
    assert str(MAX_INPUT_CHARS) in r.json()["detail"]


def test_exactly_at_the_char_limit_is_accepted(client):
    # A payload join() produces exactly MAX_INPUT_CHARS from: 1 title char, the
    # 2-char blank line, and the rest abstract. Anything that reaches the model
    # untruncated must be served.
    abstract = "x" * (MAX_INPUT_CHARS - 3)
    r = client.post("/predict", json={"title": "t", "abstract": abstract})
    assert r.status_code == 200, r.json()


def test_one_char_over_the_limit_is_rejected(client):
    # The other side of that boundary. Summing the two fields without the
    # separator passes this input and then lets join() truncate it — silently
    # doing the thing the 400 exists to prevent.
    abstract = "x" * (MAX_INPUT_CHARS - 2)
    r = client.post("/predict", json={"title": "t", "abstract": abstract})
    assert r.status_code == 400
    assert str(MAX_INPUT_CHARS + 1) in r.json()["detail"], "report the real joined length"


def test_batch_over_the_cap_is_400(client):
    r = client.post("/predict/batch", json={"items": [PAPER] * 33})
    assert r.status_code == 400
    assert "33" in r.json()["detail"]


def test_batch_at_the_cap_is_accepted(client):
    r = client.post("/predict/batch", json={"items": [PAPER] * 32})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 32


def test_empty_batch_is_400(client):
    assert client.post("/predict/batch", json={"items": []}).status_code == 400


def test_one_bad_item_rejects_the_whole_batch_and_names_the_index(client):
    r = client.post("/predict/batch",
                    json={"items": [PAPER, {"title": "", "abstract": "x"}, PAPER]})
    assert r.status_code == 400
    assert "index 1" in r.json()["detail"]
    assert "results" not in r.json(), "a rejected batch must not return partial predictions"
