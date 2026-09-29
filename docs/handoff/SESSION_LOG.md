# Session log (Simon's sessions)

Append only: one entry per session, newest at the bottom. Each entry: date, session name, what was done (with commits), what was decided, what was handed on. The current state is in `HANDOFF.md`; design decisions and numbers are also in the deviations log in `CLAUDE.md`.

## 2026-09-28 to 2026-09-29: "GenAI Projekt Einstieg" (Simon's first session)

**Onboarding and review**
- Read the repo, all branches, the report (`writing`), the configs, prompts and notebooks; checked GitHub access (private repo, push rights). Read Simon's bachelor thesis RunPod workflow (read only).
- Unpacked BLN600 and the processed data into the git-ignored `data/`, and the BERT weights into `models/bert_ft_v1/`; all hashes match `runs/`.
- Went through the GRIPS course (announcements, pages, all 11 lecture slide sets). Found: the 7-9 page limit (announcement "Project Report", 17 Jul), the grading criteria including "a constructive research hypothesis" (announcement "Online Week"), the defence date (Gruppe 4, 14 Oct, 10:30), and lecture links (BART as the lecture's denoiser, the diffusion slides' "hallucinate something similar", RAG and prompt parts in "LLM Engineering").
- Wrote a review dossier for Shan and Bea (sent by Simon via WhatsApp; Shan replied "yes and noted"). Main finding: BERT's candidates come from the context only; a plain dictionary lookup beat BERT on dev at every level and came close to Qwen. Draft hypotheses in the dossier:
  - H1 (severity): repair quality falls as the damaged share grows, and fact recovery falls faster than span BERTScore.
  - H2 (method): with identical input, the few-shot LLM restores more fact words than fine-tuned BERT, because it reads all garbled forms together.
  - H3 (metric): span BERTScore overstates repair quality: at every level a share of repairs gets a high BERTScore while the fact is wrong.
  - H2b (optional): GenAI repair beats a dictionary lookup on fact words, not only on common words.

**Built (branch `simon/baselines`, merged as PR #1 into `writing`, merge commit `c9a7487`)**
- `efc823b`: dictionary lookup baseline (`src/blnrepair/lexicon.py`, `configs/lexicon.yaml`, notebook 2b, 5 tests, dev predictions).
- `199fa29`: lookup frozen, test predictions stored (750 rows, not scored).
- `896d433`: BERT with dictionary candidates (`bert_repair.py` extension, off by default; the frozen v1 run reproduces exactly on CPU and MPS), dev grid `lex_n` 5/10/20/50 x lambda 1-64 (+128), chosen `lex_n` 5, lambda 64; test predictions stored (750 rows, not scored); notebook renamed `02b_dictionary.ipynb`.

**Decided by Simon**
- Dropped slots stay empty in the lookup; the lookup gets its own notebook (2b); several candidate counts tried; the same lambda rule as notebook 2; MPS allowed.
- Test discipline: all methods and metrics fixed on dev; notebook 4 runs on test once, and those are the reported numbers.
- The team allows self-merging own PRs.
- Report: no day-by-day plan; anything we want credit for must be in the paper (the defence is based on it); BART fine-tuning only if it fits in the paper.

**Handed on** to "GenAI Worker 1" on 2026-09-29: next step = extend notebook 4 on dev (see `HANDOFF.md`).

## 2026-09-29: "GenAI Worker 1" (took over as project lead)

**Built and merged (branch `simon/baselines` into `writing`)**
- PR #2 (`e68e837`, merge `cc39728`): notebook 4 scores five methods; `src/blnrepair/evaluation.py` (row scores with anchor recovery, visible and dropped fact slots, pooled CER and repair gain; bootstrap CIs over sentences; exact McNemar; Wilcoxon; Holm; the SQ3 rate) with tests; `scipy==1.17.1`.
- PR #3 (merge `efb3b80`): BART fine-tuned as a denoiser (`scripts/train_bart.py`, `scripts/run_bart.py`, `src/blnrepair/bart_data.py`, `configs/bart_ft.yaml`, `configs/bart_repair.yaml`, notebook `02c_bart.ipynb`, `plots.fit_curves`), RunPod helper `scripts/pod/pod.sh`, notebook 4 with six methods.
- Not yet merged: `56f6682` (BART frozen), `bd84f7e` (BART test predictions, 750 rows, 40 format failures; pod upload retries).

**Decided by Simon**
- Statistics: bootstrap CIs over sentences (reason must be in the Method text); SQ3 measure and the two tested pairs delegated to Claude (see the deviations log); every choice must be defensible in the oral defence.
- BART: do it; everything on RunPod (the local smoke run crashed the 8 GB laptop twice); weights downloaded, pods terminated after each run (no idle cost); no further tuning ("more effort would be in the wrong place"); frozen and run on test.
- Report drafts go into their own file `report/drafts/simon_sections.tex` (option 1).

**Numbers**
- Report length (main.pdf on `writing`, 28 Sep build, identical to a fresh compile): content ends at about 6.85 of 9 pages; pending parts estimated at about 1.95 pages minus 0.3 of placeholders; a BART paragraph plus one column about 0.35 to 0.4 pages.
- BART training: best epoch 8, validation exact 0.672, probe 0.879 (learns training sentences), validation loss flat from epoch 6. Pods: 0.20 $ + 0.27 $.

**Handed on** to the next session (or tomorrow): report drafts, then notebook 4 on test once Shan's run is complete (see `HANDOFF.md`).

## 2026-09-29 (late night): "GenAI Worker 2" (took over as project lead from "GenAI Worker 1")

**Done**
- Synced with `origin/writing` (Shan's report text, LLM test files, cleanup); tests 151 passed, 3 skipped. Checked all seven test files (six methods x 750 unique keys, probe 116) and that notebook 4 resolves to them.
- **Notebook 4 on the test split, run exactly once** (Simon's explicit go): PR #4, merge `5f1dad5`. This is the result record of the study.
- Report (branch `simon/results`, PR into `writing`): `scripts/paper_results.py` generates `tab:results`, `tab:format` and `fig:results` from the executed notebook; Results 4.3, Interpretation P1 to P5, Abstract and key outcomes written from test numbers; code-check fixes in the existing text; one citation corrected (Debaene et al. 2025) after checking the full text; text fits 9 pages with no slack.
- Read-only reviews: Codex `gpt-6-sol` (effort high) on results and interpretation, 15 findings, all but one applied (hypothesis wording kept: fixed before the test run); `gpt-6-luna` (effort medium) on the Debaene citation and on every number of the new text.

**Decided by Simon**
- Explicit go for the one test run of notebook 4; the report work delegated to this session; `lex_n` 5 / lambda 64 kept; results table filled by this session; workers read-only, only the lead changes the repo; literature is checked only against full texts (Shan's run before submission).

**Numbers**
- Test headline: see `HANDOFF.md`; every number is in the notebook 4 outputs and the last two deviations-log entries in `CLAUDE.md`.

**Handed on**: Shan's literature run, team read, submission (see `HANDOFF.md`).
