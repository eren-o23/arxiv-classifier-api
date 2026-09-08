"""join() is the train/serve contract. These are the ways it can silently break.

No artifact needed — ModelBundle.load() is what checks join()'s limits against
the card, and test_model_loads exercises that.
"""

from serving.preprocessing import MAX_INPUT_CHARS, join, joined_length


def test_separator_is_exactly_a_blank_line():
    # The model was trained on this exact shape. Anything else degrades it
    # silently — no error, just worse predictions.
    assert join("Title", "Abstract") == "Title\n\nAbstract"


def test_ends_are_stripped():
    assert join("  Title \n", "\t Abstract  ") == "Title\n\nAbstract"


def test_interior_whitespace_is_left_alone():
    assert join("A  B", "C\nD") == "A  B\n\nC\nD"


def test_truncates_at_the_limit():
    long = join("t", "x" * 9999)
    assert len(long) == MAX_INPUT_CHARS
    assert long == ("t\n\n" + "x" * 9999)[:MAX_INPUT_CHARS]


def test_boundary_exactly_at_the_limit_is_untouched():
    abstract = "x" * (MAX_INPUT_CHARS - 3)  # 3 = len("t") + len("\n\n")
    assert len(join("t", abstract)) == MAX_INPUT_CHARS
    assert join("t", abstract).endswith("x")
    # One character more and it must be cut, not passed through.
    assert len(join("t", abstract + "y")) == MAX_INPUT_CHARS
    assert not join("t", abstract + "y").endswith("y")


def test_limit_counts_characters_not_bytes():
    # A 4-byte emoji is one character. Counting bytes would truncate a unicode
    # abstract far earlier than an ASCII one.
    assert len(join("t", "🧪" * 5000)) == MAX_INPUT_CHARS


def test_empty_inputs_do_not_crash():
    # Rejecting these is the API's job at M2; join() must not be where it blows up.
    assert join("", "") == "\n\n"


def test_joined_length_agrees_with_join_below_the_cap():
    """The API rejects on joined_length(); the model consumes join(). If the two
    ever disagree about the same input, oversized text gets quietly truncated
    instead of refused."""
    for title, abstract in [("t", "x" * (MAX_INPUT_CHARS - 3)),
                            ("Title", "Abstract"),
                            ("  padded  ", "\n spaced \t")]:
        assert joined_length(title, abstract) == len(join(title, abstract))


def test_joined_length_sees_past_the_truncation():
    # Above the cap is the one place they must differ: len(join(...)) is pinned
    # at the limit, so only joined_length() can tell you the input was too big.
    assert len(join("t", "x" * 9999)) == MAX_INPUT_CHARS
    assert joined_length("t", "x" * 9999) == 10002
