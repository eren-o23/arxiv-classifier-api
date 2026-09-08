"""M0 done-check: load the artifact and label a real arXiv abstract.

    uv run --python 3.11 --with torch --with transformers \
        scripts/check_model.py --title "..." --abstract "..."

Pass --abstract - to read the abstract from stdin.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from serving.preprocessing import MAX_TOKENS, join  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(REPO / "models" / "arxiv-v1"))
    ap.add_argument("--title", required=True)
    ap.add_argument("--abstract", required=True)
    args = ap.parse_args()

    d = Path(args.model_dir)
    card = json.loads((d / "model_card.json").read_text())
    model = AutoModelForSequenceClassification.from_pretrained(d)
    tokenizer = AutoTokenizer.from_pretrained(d)

    # A card/config mismatch means the service returns confidently wrong
    # labels. Cheapest possible place to catch it.
    from_config = [model.config.id2label[i] for i in range(model.config.num_labels)]
    assert from_config == card["labels"], f"card {card['labels']} != config {from_config}"

    abstract = sys.stdin.read() if args.abstract == "-" else args.abstract
    text = join(args.title, abstract)
    with torch.inference_mode():
        logits = model(**tokenizer(text, truncation=True, max_length=MAX_TOKENS,
                                   return_tensors="pt")).logits[0]
    probs = logits.softmax(-1)
    top3 = probs.topk(3)

    print(f"model    {card['name']} {card['version']}  ({len(text)} chars in)")
    for score, idx in zip(top3.values.tolist(), top3.indices.tolist()):
        print(f"  {card['labels'][idx]:<8} {score:.3f}")


if __name__ == "__main__":
    main()
