"""Load the artifact and run inference.

No web framework imports here, deliberately (SPEC.md §2): welding inference to
FastAPI means you cannot test a prediction without spinning up a web app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocessing import MAX_INPUT_CHARS, MAX_TOKENS, join

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "arxiv-v1"


@dataclass(slots=True, frozen=True)
class ModelBundle:
    """Weights, tokenizer and model card, loaded once and reused."""

    model: Any
    tokenizer: Any
    card: dict

    @classmethod
    def load(cls, model_dir: str | Path | None = None,
             num_threads: int | None = None) -> ModelBundle:
        """Load from disk. Raises rather than half-loading.

        `num_threads` is an explicit knob, not a module-level
        `torch.set_num_threads(1)`: pinning threads helps p95 under concurrency
        but M3 has to measure it both ways, and a global side effect at import
        time makes that impossible.
        """
        d = Path(model_dir or DEFAULT_MODEL_DIR)
        card_path = d / "model_card.json"
        if not card_path.exists():
            raise FileNotFoundError(f"no model_card.json in {d} — run `make model`")

        if num_threads:
            torch.set_num_threads(num_threads)

        card = json.loads(card_path.read_text())
        model = AutoModelForSequenceClassification.from_pretrained(d).eval()
        tokenizer = AutoTokenizer.from_pretrained(d)

        # A card/config mismatch means confidently wrong labels for every
        # request. Cheapest possible place to catch it is before serving one.
        from_config = [model.config.id2label[i] for i in range(model.config.num_labels)]
        if from_config != card["labels"]:
            raise ValueError(f"card labels {card['labels']} != config {from_config}")

        # The card calls itself the source of truth for the limits, but
        # preprocessing.py is where they are enforced and train.py generated the
        # card from those constants. Assert they still agree.
        if (card["max_input_chars"], card["max_tokens"]) != (MAX_INPUT_CHARS, MAX_TOKENS):
            raise ValueError(
                f"card limits ({card['max_input_chars']}, {card['max_tokens']}) != "
                f"preprocessing ({MAX_INPUT_CHARS}, {MAX_TOKENS})"
            )

        return cls(model, tokenizer, card)

    @property
    def labels(self) -> list[str]:
        return self.card["labels"]

    @property
    def version(self) -> str:
        return self.card["version"]

    def predict_many(self, papers: list[tuple[str, str]]) -> list[dict]:
        """(title, abstract) pairs → one dict each, in one forward pass."""
        encoded = self.tokenizer(
            [join(title, abstract) for title, abstract in papers],
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            probs = self.model(**encoded).logits.softmax(-1)
        return [self._result(row) for row in probs]

    def predict(self, title: str, abstract: str) -> dict:
        return self.predict_many([(title, abstract)])[0]

    def _result(self, probs) -> dict:
        scores = dict(zip(self.labels, probs.tolist()))
        top3 = sorted(scores, key=scores.__getitem__, reverse=True)[:3]
        return {
            "label": top3[0],
            "confidence": scores[top3[0]],
            "scores": scores,
            "top3": top3,
        }
