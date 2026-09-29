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
