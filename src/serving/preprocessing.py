"""Text preparation shared by training and serving.

`train.py` and the serving path MUST both go through `join()`. If they drift,
the model quietly gets worse and every test still passes.
"""

MAX_INPUT_CHARS = 4000
MAX_TOKENS = 384
SEPARATOR = "\n\n"


def join(title: str, abstract: str) -> str:
    """Title and abstract into the single string the model is trained on.

    Caps characters before tokenizing: `truncation=True` still tokenizes the
    whole string first, so a 10MB abstract is a memory spike with a
    perfectly valid-looking response at the end.
    """
    return f"{title.strip()}{SEPARATOR}{abstract.strip()}"[:MAX_INPUT_CHARS]


def joined_length(title: str, abstract: str) -> int:
    """How long join()'s output would be if it did not truncate.

    The API rejects oversized input rather than truncating it, which means
    knowing the untruncated length — and len(join(...)) cannot tell you, it is
    capped. Doing that arithmetic at the call site is how the separator gets
    forgotten and a 4002-character input slips through as "4000".
    """
    return len(title.strip()) + len(SEPARATOR) + len(abstract.strip())

