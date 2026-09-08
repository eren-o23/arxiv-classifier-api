"""Build and refresh tests/golden.json.

    python scripts/build_golden.py --fetch cs.CV,cs.CR --per 6   # find candidates
    python scripts/build_golden.py --refresh                     # re-record confidences

--fetch prints candidates from recent arXiv submissions (well past the training
data) with what the model does on each; picking which ones land in golden.json
is a human job. --refresh only rewrites recorded confidences — `expected` and
`assert` are never machine-written, which is what stops the file from becoming
a snapshot of the model marking its own homework.
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from serving.model import ModelBundle  # noqa: E402

GOLDEN = REPO / "tests" / "golden.json"
ATOM = "{http://www.w3.org/2005/Atom}"
API = "https://export.arxiv.org/api/query?"


def fetch(category: str, count: int, since: str = "202601010000") -> list[dict]:
    """Recent papers whose *primary* category is `category`."""
    query = urllib.parse.urlencode({
        "search_query": f"cat:{category} AND submittedDate:[{since} TO 202612312359]",
        "start": 0,
        "max_results": count * 3,          # over-fetch, most rows are cross-lists
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    with urllib.request.urlopen(API + query, timeout=30) as r:
        feed = ET.fromstring(r.read())

    papers = []
    for entry in feed.iter(f"{ATOM}entry"):
        primary = entry.find("{http://arxiv.org/schemas/atom}primary_category")
        if primary is None or primary.get("term") != category:
            continue
        papers.append({
            "arxiv_id": entry.findtext(f"{ATOM}id", "").rsplit("/", 1)[-1],
            "title": clean(entry.findtext(f"{ATOM}title", "")),
            "abstract": clean(entry.findtext(f"{ATOM}summary", "")),
            "expected": category,
        })
        if len(papers) == count:
            break
    return papers


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", help="comma-separated categories to pull candidates for")
    ap.add_argument("--per", type=int, default=5, help="candidates per category")
    ap.add_argument("--refresh", action="store_true", help="re-record confidences in place")
    args = ap.parse_args()

    bundle = ModelBundle.load()

    if args.refresh:
        golden = json.loads(GOLDEN.read_text())
        papers = golden["papers"]
        for paper, result in zip(papers, bundle.predict_many(
                [(p["title"], p["abstract"]) for p in papers])):
            paper["confidence"] = round(result["confidence"], 4)
            paper["predicted"] = result["label"]
        golden["torch"] = __import__("torch").__version__
        GOLDEN.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n")
        print(f"refreshed {len(papers)} papers")
        return

    if not args.fetch:
        ap.error("pass --fetch or --refresh")

    for category in args.fetch.split(","):
        for paper in fetch(category.strip(), args.per):
            result = bundle.predict(paper["title"], paper["abstract"])
            mode = ("top1" if result["label"] == paper["expected"]
                    else "top3" if paper["expected"] in result["top3"] else "MISS")
            print(json.dumps({**paper, "assert": mode,
                              "confidence": round(result["confidence"], 4),
                              "predicted": result["label"],
                              "top3": result["top3"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
