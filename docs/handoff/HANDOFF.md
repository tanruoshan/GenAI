# Handoff: current state (Simon's sessions)

Last updated: 2026-09-29, by session "GenAI Projekt Einstieg", for the next session ("GenAI Worker 1").

This file is the **current state**; it is rewritten at every handoff. The history is in `SESSION_LOG.md` (append only). Design decisions and numbers are in the deviations log at the end of `CLAUDE.md` (newest entries at the bottom). If this file and the code disagree, the code wins; say so.

## Protocol for every session

1. **Start:** read this file, the last 2 entries of `SESSION_LOG.md`, and the last two sections of the deviations log in `CLAUDE.md`. Run `git fetch` and check `git log --oneline -5 origin/writing` for teammates' changes. Tell Simon in 3 to 5 lines what you understood and what you will do next.
2. **During:** every decision or deviation goes into the deviations log in `CLAUDE.md` (D = decided by Simon, C = changed by Claude, N = new number), as the team has done so far.
3. **End (or before a handoff):** append an entry to `SESSION_LOG.md`, rewrite this file, commit both on the working branch, push.

## People and ways of working

- **Simon** (this repo's user, GitHub `simon-mzb`): talk to him in **German**; everything written into the project (code, notebooks, docs, commits, PRs) is **English**. He works fast with AI help: no day-by-day schedules. Messages he forwards to teammates must be fact-checked.
- **Shan** (`tanruoshan`, repo owner): LLM track (notebook 3, SAIA), running the LLM test run now; will clean up the code on her own branch **after** that run. Notebook 3, the prompts and `configs/llm.yaml` are hers.
- **Bea**: report writing (`report/overleaf-bln600/sections/*.tex`). Her sections contain `\todo{team: ...}` notes that define what the evaluation must deliver (see "Next steps").
- **Git:** Simon works on `simon/baselines`; pull requests go into `writing` (the active branch: code + report). **Self-merging own PRs is allowed** (Shan agreed). Before merging: `origin/writing` must not have moved, and the PR must be CLEAN. No Claude co-author trailer in commits. Notebooks are JSON and merge badly: one person per notebook at a time; announce before editing a teammate's notebook.
- **Never commit:** `data/`, `models/`, `.env`, and `AGENTS.md` (an untracked copy of `CLAUDE.md` that is not ours).
- **Out of bounds:** `/Users/simonm/dev/Bachelorarbeit` is read only (Simon's thesis; its RunPod scripts in `experiments/` may be read). `/Users/simonm/UniRGB_local/sus-ai-studie` is a different project: never touch it.

## Deadlines and course rules (GRIPS, course HCAI M07, Prof. Bernd Ludwig)

- Submission: **Thu 1 Oct 2026, 00:00** (Gruppe 4). Report 7 to 9 pages, ACL style, no extensive appendix (announcement "Project Report", 17 Jul 2026).
- Defence: Gruppe 4, **14 Oct 2026, 10:30**, based on the paper. Anything we want credit for must be in the paper.
- Graded (announcement "Online Week", 30 May): report, demonstration, defence; coding does not count. Explicitly required: **a constructive research hypothesis** (the report has none yet), evaluation "according to state-of-the-art methods", interpretation "in terms of impact for an answer to your hypothesis".

## Local setup (Simon's Mac)

- Repo: `/Users/simonm/UniRGB_local/GenAI`. Data unpacked (git-ignored): `data/raw/BLN600/`, `data/processed/` (`corrupted_v2.jsonl` and `bert_train_pool.jsonl` hashes match `runs/`). Model: `models/bert_ft_v1/` (hash matches `runs/bert_ft_v1.sha256`).
- Python: `.venv` (Python 3.11.5, torch 2.14.0 with MPS, transformers 5.17.0). Tests: `PYTHONUTF8=1 .venv/bin/python -m pytest -q` (137 passed, 3 skipped).
- **Notebooks:** execute with `PYTHONUTF8=1 .venv/bin/jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=genai-venv <nb>` and afterwards set the notebook's `metadata.kernelspec` back to `{"name": "python3", "display_name": ".venv", "language": "python"}`. Plain `python -m jupyter nbconvert` picks Anaconda's nbconvert and the wrong kernel (`No module named blnrepair`).
- BERT on MPS gives exactly the same words as on the CPU (checked on 40 dev rows), about 2x to 10x faster.

## Current state

- **Frozen and done (test predictions stored, never scored yet):** BERT without dictionary (`bert_rerank_ftv1-l8-b5-n10-test`), dictionary lookup (`lexicon_v1-test`), BERT with dictionary candidates (`bert_rerank_ftv1-l64-b5-n10-x5-test`). All four configs have `frozen: true` (`bert_repair`, `bert_repair_lex`, `lexicon`, `llm`).
- **Running (Shan):** the LLM test run (Qwen3-30B-A3B via SAIA: few-shot v4 and few-shot + article v3, plus the probe; about 1,650 requests over 2 days). Its files will be `runs/preds/llm_fewshot_v4-qwen3-30b-a3b-instruct-2507-exb46768-test.jsonl`, `llm_fewshot_article_v3-...-test.jsonl`, `llm_probe_v1-...-test.jsonl`.
- **Merged:** PR #1 (dictionary baseline + BERT with dictionary candidates) into `writing` at `c9a7487`. `simon/baselines` = `writing` + nothing new yet (except this handoff).
- **Nobody has seen test scores.** Notebook 4 has no gate of its own (it reads `SPLIT`); it scores a method on a split only when all its rows exist. Do not run it with `SPLIT = "test"` until the evaluation is final (see below).
- Dev reference numbers (all 100 dev rows; exact words / Fact Recovery Rate): BERT no dictionary 0.267 / 0.126; lookup 0.528 / 0.426; BERT + dictionary 0.619 / 0.560. Qwen few-shot on the 15 scored dev sentences (notebook 4): exact 0.559, FRR 0.464, 7/75 format failures.

## Next steps (in this order)

1. **Extend notebook 4 on `simon/baselines`, on dev first** (Simon's go; Shan is not editing notebook 4). Add:
   - the two new methods next to BERT and the LLM: "dictionary lookup" (`lexicon`, `lexicon_version(load_lexicon_config(), SPLIT)`) and "BERT + dictionary" (`bert_rerank`, `repair_version(load_repair_config(ROOT / "configs" / "bert_repair_lex.yaml"), SPLIT)`); keep "BERT (fine-tuned)" as the no-dictionary reference (for the report: an ablation line, to save space);
   - what Bea's `\todo`s in `report/overleaf-bln600/sections/04_experiments.tex` ask for: 95% bootstrap CIs over sentences (10,000 resamples) for every cell of the results table; per level, McNemar's test on anchor recovery and a Wilcoxon signed-rank test on span BERTScore (main pair: BERT + dictionary vs LLM few-shot); anchor recovery; Fact Recovery Rate split into visible vs dropped fact slots (slots carry `anchor`, `fact`, `dropped` flags from `build_slots`); span CER pooled per level and pooled repair gain; LLM scores over valid answers only; sentence-level BERTScore (already computed, column "BERTScore sentence");
   - one number for SQ3, e.g. the share of rows with a high span BERTScore but a wrong fact (fix the threshold on dev before test).
   - Functions over about 10 lines go to `src/blnrepair/` with tests; notebook cells stay thin. Keep the fixed method order and colours (`plots.level_lines`, `ORDER`). The scored rows stay as they are (5 few-shot example sentences left out on dev; all 150 sentences on test).
   - Then commit, PR into `writing`, self-merge.
2. **Report drafts for Simon to pass to Bea and Shan** (ask Simon where they should go before editing any `.tex` file; the sections belong to Bea): hypotheses H1 to H3 (+ H2b) for the Introduction (draft in `SESSION_LOG.md`, 2026-09-28); two short Method paragraphs (dictionary lookup as the "confusion only" side of "context beats confusion", Evershed and Fitch 2014, already cited; dictionary candidates for BERT); fixes still open in Method/Intro/Table 2: the LLM is Qwen3-30B-A3B-Instruct-2507, **5** solved examples (one per level), numbered slots, frozen (the text still says Llama and three examples).
3. **When Shan's LLM test run is complete:** run notebook 4 once with `SPLIT = "test"`, commit the executed notebook, and hand the tables and the figure to the team for the Results section.
4. **Optional, only if it fits in the paper and the team agrees:** fine-tune BART-base (the lecture's denoising model) on the training pool damaged with our own generator, on Simon's RunPod A40 (about $0.45 to $0.50 per hour; his thesis scripts show the RunPod REST workflow). Not started.

## Open decisions (ask Simon)

- Where the report drafts go (own `.tex` file on the branch, or a message to Bea).
- Whether BART fine-tuning happens at all.
- The choice `lex_n` 5 / lambda 64 was made by the agreed rule; `lex_n` 50 at lambda 32 has a slightly higher exact rate on dev (0.628 vs 0.619, noise level). Simon has not objected; changing it would be a new test run under a new version name (still clean, since nobody has seen test scores).
