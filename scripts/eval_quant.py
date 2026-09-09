#!/usr/bin/env python3
"""int8 dynamic quantization: what does it cost in accuracy?

    uv run --with datasets python scripts/eval_quant.py

Evaluates fp32 and int8 over the *same held-out split train.py used*, through
`ModelBundle` — so what is measured is the served path, not a reimplementation
of it. `training/train.py` is frozen (it produced the shipped artifact), so the
four constants it needs are mirrored here rather than imported; the row count is
the check that the mirror is faithful.

Latency is deliberately not measured here. Dynamic quantization dispatches to a
backend chosen by architecture — fbgemm on x86, qnnpack on ARM — so a speed
figure from a laptop says nothing about the VM. `bench/latency.py` answers that,
on the box that matters.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from serving.model import ModelBundle, resolve_model_dir  # noqa: E402

# Mirrored from training/train.py, which is committed for provenance and must
# not be edited. If any of these drifts, EXPECTED_ROWS stops matching and this
# script fails loudly rather than quietly evaluating a different split.
DATASET = "TimSchopf/arxiv_categories"
SEED = 42
N_TEST = 4_000
EXPECTED_ROWS = 2782  # the figure recorded in docs/model_eval.md


def primary(categories: list[str]) -> str:
    """arXiv lists the primary category first; rows look like 'Archive->cs.CV'."""
    return categories[0].split("->")[-1]


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """Unweighted mean per-class F1, with 0.0 for a class that is never
    predicted and never correct — the same convention sklearn uses, spelled out
    because a dead class (stat.ML) makes the zero-division branch load-bearing.
    """
    total = 0.0
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        total += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return total / len(labels)


def evaluate(bundle: ModelBundle, rows: list[tuple[str, str, str]], batch: int) -> dict:
    labels = bundle.labels
    y_true = [r[2] for r in rows]
    y_pred: list[str] = []
    top3_hit = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        for (_, _, gold), res in zip(chunk, bundle.predict_many([(t, a) for t, a, _ in chunk])):
            y_pred.append(res["label"])
            top3_hit += gold in res["top3"]
        print(f"\r  {min(i + batch, len(rows))}/{len(rows)}", end="", flush=True)
    print("\r" + " " * 24 + "\r", end="")
    return {
        "top1": sum(t == p for t, p in zip(y_true, y_pred)) / len(rows),
        "top3": top3_hit / len(rows),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "pred": y_pred,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, help="evaluate only the first N rows (a smoke run)")
    args = ap.parse_args()

    from datasets import load_dataset

    card_labels = ModelBundle.load(resolve_model_dir()).labels
    allowed = set(card_labels)

    ds = load_dataset(DATASET, split="test")
    ds = ds.filter(lambda r: primary(r["categories"]) in allowed)
    ds = ds.shuffle(seed=SEED).select(range(min(N_TEST, len(ds))))

    if len(ds) != EXPECTED_ROWS:
        raise SystemExit(
            f"split reproduced {len(ds)} rows, expected {EXPECTED_ROWS} — this is not "
            "the split train.py evaluated, so the comparison against "
            "docs/model_eval.md would be meaningless."
        )
    print(f"test split reproduced exactly: {len(ds)} rows")

    rows = [(r["title"], r["abstract"], primary(r["categories"])) for r in ds]
    if args.limit:
        rows = rows[:args.limit]
        print(f"--limit {args.limit}: NOT comparable to docs/model_eval.md")

    results = {}
    for name, quant in (("fp32", False), ("int8", True)):
        print(f"evaluating {name} ...")
        bundle = ModelBundle.load(resolve_model_dir(), quantize=quant)
        results[name] = evaluate(bundle, rows, args.batch)

    print("\n| metric | fp32 | int8 | delta |")
    print("|---|---|---|---|")
    for key, label in (("top1", "top-1 accuracy"), ("top3", "top-3 accuracy"),
                       ("macro_f1", "macro F1")):
        a, b = results["fp32"][key], results["int8"][key]
        print(f"| {label} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")

    agree = sum(a == b for a, b in zip(results["fp32"]["pred"], results["int8"]["pred"]))
    print(f"| label agreement | | | {agree}/{len(rows)} ({agree / len(rows):.2%}) |")

    flips = Counter(
        (a, b) for a, b in zip(results["fp32"]["pred"], results["int8"]["pred"]) if a != b
    )
    if flips:
        print("\nWhere the two disagree (fp32 -> int8):")
        for (a, b), n in flips.most_common(8):
            print(f"  {a:8} -> {b:-8} {n}")


if __name__ == "__main__":
    main()
