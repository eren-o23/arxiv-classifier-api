"""Text preparation shared by training and serving.

`train.py` and the serving path MUST both go through `join()`. If they drift,
the model quietly gets worse and every test still passes.
"""

MAX_INPUT_CHARS = 4000
MAX_TOKENS = 384


def join(title: str, abstract: str) -> str:
    """Title and abstract into the single string the model is trained on.

    Caps characters before tokenizing: `truncation=True` still tokenizes the
    whole string first, so a 10MB abstract is a memory spike with a
    perfectly valid-looking response at the end.
    """
    return f"{title.strip()}\n\n{abstract.strip()}"[:MAX_INPUT_CHARS]


if __name__ == "__main__":
    assert join(" A ", " B ") == "A\n\nB"
    assert join("t", "x" * 9999) == ("t\n\n" + "x" * 9999)[:MAX_INPUT_CHARS]
    assert len(join("t", "x" * 9999)) == MAX_INPUT_CHARS
    print("ok")
