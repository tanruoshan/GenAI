# Semantic Repair of Damage-Occluded Historical Newspaper Text

GenAI coursework project (6 ECTS).

## Research question

Given pretrained generative and masked-language-model GenAI methods, to what extent can damaged
word spans in historical newspaper text be reconstructed so that the semantic content and key
facts of the original sentence are preserved, and how does reconstruction quality change as the
size of the damaged region grows?

## Status

Data and damage are frozen (`corrupted_v2`). Both repair methods run end to end on the dev split.
BERT is tuned on dev (λ = 8, `configs/bert_repair.yaml`) and its test run is done: 750 rows in
`runs/preds/bert_rerank_ftv1-l8-b5-n10-test.jsonl`, committed in this repository.
The LLM track is handed over for model and prompt tuning on another machine, then its test run; see
"Running the LLM track on your own machine" below.

`CLAUDE.md` and `docs/` (the project dossier and report source notes) are tracked in this repository
as internal working notes, kept alongside the instructor-facing material.

## Notebooks

1. `notebooks/01_dataset_and_damage_pipeline.ipynb`: data pool, calibration and the damage pipeline.
2. `notebooks/02_bert_repair.ipynb`: fine-tuned BERT with character-aware reranking (repair and dev tuning).
   `notebooks/02b_lexicon_baseline.ipynb`: the dictionary lookup, a non-GenAI baseline (closest spelling in a word list, no context).
3. `notebooks/03_llm_repair.ipynb`: few-shot LLM repair (with and without the article as context) and the contamination probe.
4. `notebooks/04_evaluation.ipynb`: reads the stored predictions and scores BERT and the LLM on the same rows.

Predictions are stored in `runs/preds/`, one file per method and version. A finished row is never
requested again, so a run can be stopped and resumed. The test split stays closed for a method until
its config says `frozen: true` (`configs/bert_repair.yaml` for BERT, `configs/llm.yaml` for the LLM) and
`ALLOW_TEST = True` is set in the notebook.

## LLM setup

Copy `.env.example` to `.env` and fill in the key (git-ignored). The client in `src/blnrepair/llm.py`
speaks the OpenAI-compatible API and only calls the model ids listed in `configs/llm.yaml`. The prompt
files in `prompts/` are never edited once used; a change goes into a new file with a new version number.

## Running the LLM track on your own machine

The LLM track is finished on dev with Llama 3.1 8B on GWDG SAIA, but the dev results are weak: 21% (few-shot)
and 27% (few-shot + article) of the answers have the wrong number of words, above the 15% limit set in the
project plan. The rest is done on a machine with its own LLM. Follow the steps in this order; everything
before step 3 uses the dev split only.

1. **Use your own LLM.** In `.env` (copied from `.env.example`) set `SAIA_BASE_URL` to your server's
   OpenAI-compatible address and `SAIA_API_KEY` to its key (any value if the server needs none); the
   variable names stay. In `configs/llm.yaml` set `model` to your model id and add the id to
   `allowed_models`. Only open-weight models may be used, never a hosted third-party model (BLN600 is
   CC BY-NC-ND). If your server is not OpenAI-compatible, adapt `SaiaClient` in `src/blnrepair/llm.py`
   (`chat` and `list_models`) and run `python -m pytest tests/test_llm.py`. The model id is part of the
   version name of every stored prediction, so your runs never mix with the Llama runs.
2. **Tune the prompts on dev.** A prompt file is never edited once used. Copy the file you want to change to a
   new version (for example `prompts/repair_fewshot_v3.txt`), point `TEMPLATES` in notebook 3 at it, and
   change the prompt version in the version names in `variant_versions` (`src/blnrepair/llm.py`); without
   that the notebook reads the old answers. The three few-shot examples can change too (`fewshot_picks` in
   `configs/llm.yaml`; they must be dev sentences and are left out of the scoring). Run notebook 3 and then
   notebook 4 with `SPLIT = "dev"`, and look at the format failures and the scores. The aim is below 15%
   format failures. Never choose anything by looking at the test split.
3. **Lock the config and open the test split.** When model, prompts and examples are final, set
   `frozen: true` in `configs/llm.yaml` and commit it. Then in notebook 3 set `SPLIT = "test"` and
   `ALLOW_TEST = True`. Without `frozen: true` the notebook stops.
4. **Run the LLM on the test split.** Run notebook 3 top to bottom: 750 rows for each of the two variants
   (few-shot and few-shot + article) and the contamination probe on the 150 test sentences, about 1,650
   requests. Every answer is saved as it arrives, so the run can be stopped and continued.
5. **Share the evaluation data.** The BERT test predictions are already in this repository
   (`runs/preds/bert_rerank_ftv1-l8-b5-n10-test.jsonl`): notebook 4 scores a method only when all its rows
   are stored. Run notebook 4 with `SPLIT = "test"`, and send back the executed notebook 4 (tables and
   figure) and the LLM test predictions `runs/preds/llm_*-test.jsonl` (commit them), for the report.

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
