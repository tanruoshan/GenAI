# Semantic Repair of Damage-Occluded Historical Newspaper Text

GenAI coursework project (6 ECTS).

## Research question

Given pretrained generative and masked-language-model GenAI methods, to what extent can damaged
word spans in historical newspaper text be reconstructed so that the semantic content and key
facts of the original sentence are preserved, and how does reconstruction quality change as the
size of the damaged region grows?

## Status

Data and damage are frozen (`corrupted_v2`). Both repair methods run end to end on the dev split.
BERT reranking settings are being tuned on dev; the final run on the test split comes after that.

## Notebooks

1. `notebooks/01_dataset_and_damage_pipeline.ipynb`: data pool, calibration and the damage pipeline.
2. `notebooks/02_bert_repair.ipynb`: fine-tuned BERT with character-aware reranking (repair and dev tuning).
3. `notebooks/03_llm_repair.ipynb`: few-shot LLM repair (with and without the article as context) and the contamination probe.
4. `notebooks/04_evaluation.ipynb`: reads the stored predictions and scores BERT and the LLM on the same rows.

Predictions are stored in `runs/preds/`, one file per method and version. A finished row is never
requested again, so a run can be stopped and resumed. The test split stays closed until
`runs/repair_config_v1.yaml` exists and `ALLOW_TEST = True` is set in the notebook.

## LLM setup

Copy `.env.example` to `.env` and fill in the key (git-ignored). The client in `src/blnrepair/llm.py`
speaks the OpenAI-compatible API and only calls the model ids listed in `configs/llm.yaml`. The prompt
files in `prompts/` are never edited once used; a change goes into a new `_v2` file.

## Data

This project uses [BLN600](https://orda.shef.ac.uk/) (Booth, Thomas & Gaizauskas, 2024), a
password-protected, CC BY-NC-ND licensed corpus. It is not included in this repository. To run
the notebooks, unzip `BLN600.zip` into `data/raw/BLN600/`; the folder is git-ignored.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

Select the `.venv` interpreter as the Jupyter kernel in VS Code to run the notebooks.
