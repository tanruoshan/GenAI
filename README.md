# How Much Damage Can GenAI Repair? Fact Recovery in Synthetically Damaged Historical Newspaper Text

Bea Dippold, Ruo Shan Tan, Simon Manzenberger. Faculty of Informatics and Data Science, University of Regensburg.
GenAI course project.

## What this repository does

We damage one contiguous span of words around a fact (a name, place, number) in 150 BLN600 test sentences at five
nested levels (1 word, 10, 25, 50 and 75% of the sentence), using character noise calibrated on BLN600's own OCR
errors. All methods get the same input: the intact context, plus the garbled form of each damaged word. The
methods are:

| Method | Where |
|---|---|
| Fine-tuned `bert-base-cased` + letter-similarity reranking, with and without dictionary candidates | notebooks 02, 02b |
| Dictionary lookup (closest spelling in a word list, no model) | notebook 02b |
| Few-shot LLM, Qwen3-30B-A3B-Instruct-2507 via GWDG SAIA (sentence only / whole article) | notebook 03 |
| Fine-tuned `facebook/bart-base` denoiser (descriptive only) | notebook 02c |

Repairs are scored with span BERTScore (`roberta-large`, layer 17, baseline rescaled) and an exact Fact Recovery
Rate, with 95% bootstrap intervals over sentences and Holm-corrected McNemar and Wilcoxon tests (notebook 04).

Main result: BERT with dictionary candidates restores the anchor fact in 58% of sentences at one damaged word and
53% at 75%. The lookup alone gets 51%. The few-shot LLM falls from 49% to 25%, and BERTScore ranks fluent
wrong-fact repairs too high.

## Repository layout

```
configs/        all settings (seed 42, damage, models); "frozen: true" marks the settings used on test
prompts/        the two LLM prompt files used on test (few-shot v4, article v3)
src/blnrepair/  the package: data, damage, slot view, repair methods, evaluation
notebooks/      00 to 04, run in order (see below); outputs are stored in the notebooks
scripts/        model training (BERT, BART), BART inference, paper tables and figures, fact-check sample
runs/           stored predictions (runs/preds/), training logs, run configs and SHA-256 hashes
reports/        calibration, damage statistics, pool statistics, manual fact-rule check
tests/          unit tests (pytest)
report/         not tracked; step 3 below writes the paper's tables and figures to report/overleaf-bln600/
```

## 1. Setup

Python 3.10 or newer (our runs used Python 3.13). A CPU is enough for everything except model training.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

`requirements.txt` pins the versions of torch, transformers, bert-score, scipy and the OpenAI client. Use the
`.venv` interpreter as the Jupyter kernel.

## 2. Data

BLN600 (Booth, Thomas and Gaizauskas, 2024) is licensed CC BY-NC-ND 4.0 and is not included in this repository.

1. Download BLN600 (Version 2) from https://doi.org/10.15131/shef.data.25439023. The ZIP is password protected;
   the password (`BLN600`) is given in the dataset's own readme.
2. Unzip it so that `data/raw/BLN600/Ground Truth/`, `data/raw/BLN600/OCR Text/` and
   `data/raw/BLN600/metadata.json` exist (`data/` is git-ignored).

Note: the stored predictions in `runs/preds/` contain the damaged and repaired test and dev sentences (about 170
sentences of BLN600 text) so that the scores can be recomputed without re-running the models.

## 3. Reproduce the reported numbers (CPU only)

This path re-scores the stored predictions. No GPU or API key is needed.

1. **Rebuild the inputs.** Run `notebooks/00_dataset_and_calibration.ipynb`, then `notebooks/01_damage_pipeline.ipynb`.
   Notebook 0 writes the sentence sample (`data/processed/sentences.jsonl`) and the OCR-error calibration
   (`reports/calibration.json`). Notebook 1 writes the damaged data `data/processed/corrupted_v2.jsonl` (1,020 rows:
   170 sentences × 6 levels) and stops if its hash differs from `runs/corrupted_v2.sha256`
   (`5798336d…`). Given the same `sentences.jsonl` and `calibration.json` (their hashes are in
   `runs/config_snapshot_v2.yaml`), the rebuild is byte-identical on Windows and Linux.
2. **Score.** Run `notebooks/04_evaluation.ipynb` (`SPLIT = "test"`). It loads the frozen data (with a hash check)
   and the seven prediction files below, then computes every score, interval and test in the report. The first
   run downloads `roberta-large` (about 1.4 GB) for BERTScore.
3. **Tables and figures.** `python scripts/paper_results.py` writes `tables/results.tex`, `tables/format.tex` and
   `figures/results_levels.pdf` into `report/overleaf-bln600/` (created if missing) from the notebook-4 outputs.
   `python scripts/fig_nested_spans.py` draws Figure 1.

Prediction files scored in notebook 4 (`runs/preds/`, 750 test rows each; the probe covers 116 sentences):

| Method in the report | File |
|---|---|
| BERT + dictionary (λ = 64, 5 list words) | `bert_rerank_ftv1-l64-b5-n10-x5-test.jsonl` |
| BERT, no dictionary (λ = 8) | `bert_rerank_ftv1-l8-b5-n10-test.jsonl` |
| Dictionary lookup | `lexicon_v1-test.jsonl` |
| LLM few-shot | `llm_fewshot_v4-qwen3-30b-a3b-instruct-2507-exb46768-test.jsonl` |
| LLM + article | `llm_fewshot_article_v3-qwen3-30b-a3b-instruct-2507-exb46768-test.jsonl` |
| BART | `bart_ftv1-greedy-test.jsonl` |
| Contamination probe | `llm_probe_v1-qwen3-30b-a3b-instruct-2507-test.jsonl` |

Files ending in `-dev.jsonl` are the dev-split tuning runs (λ and list-size grids) described in the notebooks.

Where the other report numbers come from: damage by level from notebook 1 (`reports/corruption_stats.json`);
pool, calibration and split numbers from notebook 0 (`reports/pool_stats.csv`, `reports/calibration.json`); a
summary of every number each notebook contributes in `reports/report_numbers_notebook0.csv` and `…notebook1.csv`;
the manual fact-rule check (46 of 50) in `reports/fact_check/`.

## 4. Re-run the repair methods (optional)

A prediction file is append-only: rows that are already stored are skipped. To re-run a method from scratch, move
its file out of `runs/preds/` first. The test split is closed unless the method's config says `frozen: true` and
the notebook sets `ALLOW_TEST = True` (for BART: `--allow-test`).

- **Dictionary lookup:** notebook 02b, CPU, a few seconds. Deterministic.
- **BERT:** notebook 02 writes the training pool `data/processed/bert_train_pool.jsonl` (checked against
  `runs/bert_train_pool.sha256`). Fine-tune with `python scripts/train_bert.py` on a GPU; our run took about 15
  minutes on a Colab Tesla T4 (settings in `configs/bert_ft.yaml`, run record in `runs/bert_ft_v1_config.yaml`,
  best epoch 2). The model goes to `models/bert_ft_v1/`. Then run the repair cells of notebooks 02 (no dictionary)
  and 02b (with dictionary).
- **BART:** `python scripts/train_bart.py`, then `python scripts/run_bart.py --split test --allow-test`, on a GPU
  (we used a RunPod A40 through `scripts/pod/pod.sh`; run record in `runs/bart_ft_v1_config.yaml`, best epoch 8).
- **LLM:** copy `.env.example` to `.env` and add a GWDG SAIA API key, then run notebook 03. The client
  (`src/blnrepair/llm.py`) calls only the model ids in `configs/llm.yaml`, at temperature 0.

Fine-tuned weights are not included (`models/` is git-ignored). Retraining on a GPU may not give bit-identical
weights; our weights' hashes are in `runs/bert_ft_v1.sha256` and `runs/bart_ft_v1.sha256`. The hosted LLM may
also change over time. The stored predictions are the ones reported.

## Data license and citation

BLN600 is CC BY-NC-ND 4.0. If you use the data, cite Booth, Thomas and
Gaizauskas (2024), *BLN600: A Parallel Corpus of Machine/Human Transcribed Nineteenth Century Newspaper Texts*,
LREC-COLING 2024.
