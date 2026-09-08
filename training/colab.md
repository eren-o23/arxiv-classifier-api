# Training on Colab

Free-tier T4, ~15 minutes. Runtime → Change runtime type → **T4 GPU** first.

```python
!pip -q install -U transformers datasets accelerate scikit-learn
!git clone https://github.com/eren-o23/arxiv-classifier-api.git
%cd arxiv-classifier-api
```

HTTPS clone on purpose — the personal SSH key stays off Colab.

```python
from huggingface_hub import notebook_login
notebook_login()          # write token, for the push at the end
```

Look at the data before trusting the filter:

```python
!python training/train.py --inspect
```

`categories` is a list like `["Computer Science Archive->cs.CV", ...]`; the first
entry is arXiv's primary category and `train.py` takes its leaf.

```python
!python training/train.py
```

Prints top-1 / top-3 / macro-F1, writes `docs/`, then pushes to
`eren-o23/arxiv-classifier-v1` and **prints the Hub commit SHA — copy it**, M4's
Dockerfile pins to it.

Then pull `docs/model_eval.md` and `docs/confusion_matrix.png` back into the repo
(download from the Colab file browser) and commit them.

## Expected

top-1 around 0.75–0.82, top-3 above 0.90, with errors concentrated in the
cs.LG / cs.AI / stat.ML block. That block is the point, not a defect — see
[SPEC.md](../SPEC.md) §1. Errors spread evenly across the matrix instead means
the labels got shuffled somewhere in the map step.

If this starts eating days: grab a pretrained classifier, hand-write a
`model_card.json` for it, move on. The serving work is the project.
