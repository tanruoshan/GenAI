# Handoff: current state (Simon's sessions)

Last updated: 2026-09-29 (night), by session "GenAI Worker 1" (the project lead since 2026-09-29), for its own next day or the next session.

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

- **Frozen, test predictions stored, never scored (6 methods):** BERT without dictionary (`bert_rerank_ftv1-l8-b5-n10-test`), dictionary lookup (`lexicon_v1-test`), BERT + dictionary (`bert_rerank_ftv1-l64-b5-n10-x5-test`), **BART** (`bart_ftv1-greedy-test`, 40 of 750 format failures, a count only). Configs `bert_repair`, `bert_repair_lex`, `lexicon`, `llm`, `bart_repair` all have `frozen: true`.
- **Running (Shan):** the LLM test run (Qwen3-30B-A3B: few-shot v4, few-shot + article v3, probe). Files: `runs/preds/llm_fewshot_v4-qwen3-30b-a3b-instruct-2507-exb46768-test.jsonl`, `llm_fewshot_article_v3-...-test.jsonl`, `llm_probe_v1-...-test.jsonl`.
- **Merged into `writing`:** PR #1 (dictionary methods), PR #2 (notebook 4: five methods, bootstrap CIs, McNemar/Wilcoxon with Holm, SQ3 rate), PR #3 (BART, notebook 2c, notebook 4 with six methods), at `efb3b80`. **Not yet merged:** `simon/baselines` commits `56f6682` (BART frozen) and `bd84f7e` (BART test predictions, pod upload retries).
- **Notebook 4 is final on dev** (six methods). Nobody has seen test scores. Run it with `SPLIT = "test"` exactly once, after Shan's LLM files are complete.
- **BART (notebook 2c):** `facebook/bart-base` fine-tuned on the training pool damaged by our generator, best epoch 8 of 10, 22.5 min on a RunPod A40; weights in `models/bart_ft_v1/` (local only, sha256 in `runs/bart_ft_v1.sha256`). Everything BART runs on a pod (`scripts/pod/pod.sh`): **never train or run BART on the laptop (8 GB; it crashed twice)**. RunPod balance 24.49 $.
- Dev reference (15 scored sentences, span BERTScore 1w / 75): BART 0.83 / 0.67, BERT + dictionary 0.85 / 0.43, few-shot 0.86 / 0.17, lookup 0.73 / 0.40. No paired test significant on dev (expected with 15 sentences).

## Next steps (in this order)

1. **Report drafts (decided: option 1)** into a new file `report/drafts/simon_sections.tex` on `simon/baselines` (Bea's sections stay untouched; she takes over what she wants). Contents: hypotheses H1 to H3 + H2b for the Introduction (draft in `SESSION_LOG.md`, 2026-09-28); Method paragraphs for the dictionary lookup (Evershed and Fitch 2014), BERT + dictionary, and BART (trained on our own noise: optimistic, say so); the statistics paragraph (bootstrap over sentences and why, McNemar exact, Wilcoxon, Holm, only the two pre-stated pairs tested; references Koehn 2004, Dror et al. 2018, Dietterich 1998: add to `references.bib` if missing); the SQ3 measure (1 - AUC, anchor, per method); Limitations additions (single training run and seed; BART on in-distribution noise); fixes in Bea's text: the LLM is Qwen3-30B-A3B-Instruct-2507, **5** solved examples (one per level), numbered slots. Space: the report now ends at about 6.85 pages of content, estimated 8.5 to 8.9 with all pending parts (limit 9, references not counted); compact figures across both columns save space.
2. **When Shan's LLM test run is complete:** check the three files (750 / 750 / 150 rows), run notebook 4 once with `SPLIT = "test"`, commit the executed notebook, PR into `writing`, hand tables and figure to the team for Results.
3. Then Results text, Interpretation (P1 to P5 in `05_interpretation.tex`), Abstract, with the team.

## Open decisions (ask Simon)

- The choice `lex_n` 5 / lambda 64 (by the agreed rule; `lex_n` 50 / lambda 32 is 0.628 vs 0.619 exact on dev, noise level). Simon has not objected.
- Which methods go into the main results table and which only into an ablation line (proposal: main = lookup, BERT + dictionary, few-shot, BART; ablation = BERT without dictionary, few-shot + article).
