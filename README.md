# How Much Damage Can GenAI Repair? Fact Recovery in Damaged Historical Newspaper Text

GenAI coursework project (6 ECTS), University of Regensburg. Team: Bea Dippold, Ruo Shan Tan, Simon Manzenberger.

## Research question

When the damaged words of a sentence are known and their garbled letters are visible, how much of the
original wording, and in particular its fact words such as names, places and numbers, can a fine-tuned
masked language model and a few-shot instruction-following LLM restore, and how does this change as the
damaged share of the sentence grows? Hypotheses H1, H2, H2b and H3 are in the report's Introduction.

## Status (2026-09-29)

Data and damage are frozen (`corrupted_v2`). Six repair methods are frozen and their test predictions are
stored in `runs/preds/` (750 rows each): BERT with dictionary candidates, BERT without dictionary, the dictionary
lookup, the few-shot LLM (Qwen3-30B-A3B-Instruct-2507 on GWDG SAIA), the LLM with the whole article, and a
fine-tuned BART. The LLM contamination probe covers 116 of the 150 test sentences. No test score has been
computed yet: notebook 4 is run once on the test split, and those are the reported numbers.

`CLAUDE.md` and `docs/` (the project dossier and report source notes) are tracked in this repository
as internal working notes, kept alongside the instructor-facing material. The report is in
`report/overleaf-bln600/`.

## Notebooks

0. `notebooks/00_dataset_and_calibration.ipynb`: data pool, evaluation sample and calibration on real OCR errors.
1. `notebooks/01_damage_pipeline.ipynb`: the damage pipeline and the frozen damaged data.
2. `notebooks/02_bert_repair.ipynb`: fine-tuned BERT with character-aware reranking (repair and dev tuning).
   `notebooks/02b_dictionary.ipynb`: the dictionary lookup, a non-GenAI baseline (closest spelling in a word list, no context), and BERT with dictionary candidates.
   `notebooks/02c_bart.ipynb`: BART fine-tuned as a denoiser (training record and dev check; training and inference run on a rented GPU, `scripts/pod/pod.sh`).
3. `notebooks/03_llm_repair.ipynb`: few-shot LLM repair (with and without the article as context) and the contamination probe.
4. `notebooks/04_evaluation.ipynb`: reads the stored predictions and scores all methods on the same rows, with bootstrap intervals and paired tests.

Predictions are stored in `runs/preds/`, one file per method and version. A finished row is never
requested again, so a run can be stopped and resumed. The test split stays closed for a method until
its config in `configs/` says `frozen: true` and `ALLOW_TEST = True` is set in the notebook (for BART,
`--allow-test` in `scripts/run_bart.py`).

## LLM setup

Copy `.env.example` to `.env` and fill in the key (git-ignored). The client in `src/blnrepair/llm.py`
speaks the OpenAI-compatible API and only calls the model ids listed in `configs/llm.yaml`. The prompt
files in `prompts/` are never edited once used; a change goes into a new file with a new version number.

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
