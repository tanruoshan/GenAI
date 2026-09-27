# Project Dossier (Checkpoint): Repairing Damaged Historical Newspaper Text with GenAI

**For onboarding and day-to-day reference, read `docs/dossier_v2.md` instead: it is the short, current-state version.** This file stays as the complete record: the full design reasoning, and superseded numbers kept for the report's appendix material.

Plan of record for the report. Implementation lives in the project notebooks (start with `01_dataset_and_damage_pipeline.ipynb`) and the `blnrepair` package. If the two disagree on implementation details, the notebook wins; if they disagree on scope or design, this dossier wins.

Course: GenAI Master's coursework project (6 ECTS), team of 3. Report: max 9 pages in ACL style (excluding references), structured like the EMNLP example paper (Intro, Related Work, Method, Empirical Investigation, Interpretation).

**Current status (2026-09-26):** data and damage are frozen as `corrupted_v2.jsonl` (20-30% per-word damage intensity; sha256 `5798336d...`, replaces the original 40-60% intensity of `corrupted_v1.jsonl`, which is kept only as a local, untracked comparison file). BERT is fine-tuned and has one full run on the dev split. The LLM track dropped zero-shot after dev testing and has both remaining variants run and scored on dev. The test split (the final BERT vs LLM comparison) has not started: see §0 for the three open decisions that block it and the steps to run it.

## 0. Handoff notes for the team (read this first)
This section is the practical status: what is locked in, what is still open, and what to run next. Sections 1 to 14 below are the fuller design record for the report; where this section and a later one disagree on a number, this section is the more recent one.

**What is being compared, in plain terms**
Every comparison has the same shape: take a sentence with a damaged span (a run of words at one severity level: 0 (clean), 1 word, 10%, 25%, 50% or 75% of the sentence), give every method the exact same view of it (§6: context plus one "slot" per damaged word, showing its garbled letters), and score the repaired span against the original gold span. Three things are compared at every severity level:
1. **BERT + reranking.** A fine-tuned `bert-base-cased` model fills the damaged span left to right, guided by how close its guesses are to the garbled letters shown.
2. **LLM few-shot.** Llama 3.1 8B Instruct (via GWDG's SAIA API), given the same view plus 3 worked examples, asked to return one corrected word per slot.
3. **No repair.** The damaged text as it is, with nothing done to it. This is the floor: how bad the text would be if no method were used.

A fourth, secondary variant, **LLM few-shot + article**, gives the LLM the whole source article as extra context. It is compared only against the LLM's own few-shot run (does more context help the LLM), never against BERT, since BERT never sees that extra context and the comparison would not be fair.

Two headline scores are computed per severity level: **BERTScore** (does the repaired span mean the same as the original) and **Fact Recovery Rate** (were the names, numbers and dates recovered exactly). Secondary scores back these up: exact word match, whether the repair made the text better or worse (character error rate before versus after), how often the LLM returned the wrong number of words (format failure rate), and whether the LLM seems to have memorised the source text (contamination probe).

**Done so far**
- Data: `corrupted_v2.jsonl` frozen at 20-30% intensity (`runs/corrupted_v2.sha256`). It is the only version now: `corrupted_v1.jsonl` and its hash file are deleted.
- BERT: fine-tuned (`models/bert_ft_v1`, hash in `runs/bert_ft_v1.sha256`). The tuning grid on dev (lambda 1, 2, 4, 8, 16; beam 5 and 10 candidates fixed) is done: lambda 8 is chosen (8 and 16 are equal within noise; 8 keeps more weight on the context). Settings are in `configs/bert_repair.yaml`. The full dev run with them is stored (`runs/preds/bert_rerank_ftv1-l8-b5-n10-dev.jsonl`). The test run has not been done yet (project owner).
- LLM: prompts at v2 (a system message was added, needed to stop the model answering in Python code instead of JSON). Zero-shot was tried on dev, was weak, and was dropped from the study. Few-shot and few-shot + article both ran on the full dev split (17 of the 20 dev sentences; 3 are held out as the worked examples) and are scored.
- Evaluation: `notebooks/04_evaluation.ipynb` reads the stored predictions for BERT and the LLM and scores them side by side on the same rows. It works on dev; for test it only needs the test predictions to exist. On dev (not a result), BERT scores BERTScore 0.24 and Fact Recovery Rate 0.10, few-shot 0.28 and 0.22, few-shot + article 0.26 and 0.24; BERT is behind at low severity, and level with or ahead of the LLM at 50% and 75%, where the LLM's format failures all occur.
- Tests: the four older tests were pointed at v2 (commit `1842d6a`); the whole suite passes.

**Decisions**
1. **LLM model and prompts (open, handed to a teammate).** The pre-registered rule in §7.2a says switch if the dev format failure rate is above 15%. It came in at 21% (few-shot) and 27% (few-shot + article) with Llama 3.1 8B, so the rule is triggered. The teammate tunes the model and prompts on dev on a machine with an own LLM (steps below). Either the failure rate comes down, or it is reported as a finding.
2. **BERT settings (done).** Grid run on dev, lambda 8 chosen (see above).
3. **How the test split is opened (decided).** There is no separate `runs/repair_config_v1.yaml`. Each method has its own `frozen` flag in its config: `frozen: true` in `configs/bert_repair.yaml` for BERT and in `configs/llm.yaml` for the LLM. A notebook refuses `SPLIT = "test"` unless its config is frozen and `ALLOW_TEST = True` is set. Reasons: the tuned values stay where they are used, each stored prediction already carries its settings in its version name (model, lambda, beam, candidates, prompt version, few-shot examples, split), the BERT weights hash is in `runs/bert_ft_v1.sha256`, and the two methods can be opened at different times. Freezing means: set the flag and commit the config.

**Then, to get the final comparison**
- *Project owner, BERT:* set `frozen: true` in `configs/bert_repair.yaml`, then set `SPLIT = "test"` and `ALLOW_TEST = True` in notebook 02, section 7 (750 rows, about 1.7 hours on a laptop CPU; resumable).
- *Teammate, LLM* (the README has the same five steps):
  1. Change the API client to an own LLM: `.env` (`SAIA_BASE_URL`, `SAIA_API_KEY`), and `model` and `allowed_models` in `configs/llm.yaml`; open-weight models only.
  2. Tune the prompts on dev: new prompt file versions (a used file is never edited), update the prompt version in `variant_versions` in `src/blnrepair/llm.py`, optionally the few-shot picks; run notebooks 3 and 4 on dev; aim for under 15% format failures.
  3. Lock the config: `frozen: true` in `configs/llm.yaml`, commit; then `SPLIT = "test"` and `ALLOW_TEST = True` in notebook 3.
  4. Run the LLM on the test split: 750 rows for each of the two variants plus the contamination probe on the 150 test sentences (about 1,650 requests, resumable).
  5. Share the evaluation data: run notebook 4 with `SPLIT = "test"` (it needs the BERT test predictions first), and send back the executed notebook 4 and `runs/preds/llm_*-test.jsonl` for the report.
- *Then, writing:* the final numbers come from notebook 4 on the test split: BERTScore and Fact Recovery Rate per method per severity level, plus the secondary scores.

**Loose ends, not blocking**
- The stale 20-40% reports (`reports/cer_by_level_v2.json`, `reports/corruption_stats_v2.json`, `reports/review_v2.csv`) are deleted. `runs/config_snapshot_v2.yaml` is back on disk.
- This file is git-ignored, so a teammate does not get it by git; the README carries the same handoff steps.

## 1. Working title
"How Much Can Be Torn Away? Measuring the Limits of GenAI-Based Repair of Damaged Historical Newspaper Text" (placeholder).

## 2. Problem, research question, sub-questions
Historical newspapers suffer physical damage (tears, stains, tape, foreign objects). OCR on damaged regions produces garbled or missing words, which can shift or destroy the meaning of a passage, and the words most at risk are often the facts: names, places, dates, sums of money.

**RQ:** When the damaged region of a sentence is known and its garbled letters are visible, how well can GenAI models restore the original words so that meaning and key facts are preserved, and how does this change as the damaged share of the sentence grows?

- **SQ1 (severity):** Does repair quality decline smoothly with damage, or is there a point beyond which repair becomes unreliable? Does the absolute gap length matter beyond the damaged share?
- **SQ2 (method):** Given identical input, does a fine-tuned masked language model (BERT) or a zero/few-shot generative LLM repair better?
- **SQ3 (metric validity):** Does semantic similarity overstate repair quality? Can a repair read as "close in meaning" while getting names, numbers or dates wrong?

## 3. Motivation (for the Introduction)
Digitised newspaper archives are core sources for historical research, but OCR quality on damaged originals limits their use. GenAI models can fill in and correct text fluently, which creates a risk: plausible but invented content entering the historical record. Users need to know how far repair can be trusted as damage grows. This is a controlled robustness study of GenAI generation under increasing corruption, not a new system.

## 4. Data and splits
**BLN600** (Booth, Thomas & Gaizauskas, 2024): 600 excerpts from 19th-century British Library Newspapers (mostly London crime reporting, 1830s-1890s; 572 of 600 excerpts from three related publications: Lloyd's Weekly Newspaper (340), The Illustrated Police News (212) and Lloyd's Weekly London Newspaper (20)). Each excerpt has the source image, the original machine OCR, and a manually re-keyed gold transcription. License CC BY-NC-ND: fine for coursework analysis, do not redistribute a modified corpus (`data/processed/` is git-ignored).

- **Ground truth:** the gold transcription. The OCR field is used only to calibrate realistic character confusions.
- **Sentence pool:** gold text split with `pysbd`; sentences of 20-60 words with at least 2 fact tokens: 3,123 sentences from 592 excerpts.
- **Fact token:** a number or any word with a digit or a pound sign (`23`, `5s.`, `£100`), or a capitalised word that is not the first word of the sentence (names, places, days). Excluded: `I`, `Mr`, `Mrs`, `Dr`, `St`, `The`, `Miss`, `Sir`, `Rev`, `Messrs`, and a word right after a colon.
- **Length bands:** 20-29, 30-44, 45-60 words (three bands).
- **Evaluation sample (frozen):** 170 sentences from 157 excerpts, drawn with seed 42 and band quotas proportional to the pool: **150 test** (65 / 59 / 26 by band) and **20 dev** (9 / 8 / 3). At most 3 sentences per excerpt. No excerpt appears in both test and dev. Dev is used only to tune prompts and settings.
- **Calibration excerpts:** 29 excerpts outside the sample (one more dropped for coverage 0.35), used to measure real OCR confusions.
- **BERT training pool:** only excerpts outside the 157 sample excerpts (443 excerpts). No sentence from a test or dev excerpt may be used for training, because names repeat within an excerpt. The 29 calibration excerpts may be used for training; they shaped only the noise table, not the test data.
- The source images are not used in this project (see §13).

## 5. Damage simulation (frozen, `corrupted_v2.jsonl`; see §0 for the version history)
- **Anchor:** each sentence gets one anchor word, drawn among its fact tokens with a fixed seed. The anchor is always damaged and never dropped, so every damaged level contains at least one visibly damaged fact.
- **Span:** one contiguous block of damaged words around the anchor. Blocks are nested: every smaller block lies inside every larger one, so the same place gets worse across levels.
- **Levels (independent variable):** 0 (clean), 1w (one word), then 10%, 25%, 50%, 75% of the sentence's words (rounded half up). 170 sentences x 6 levels = 1,020 rows (900 test, 120 dev).
- **Per-word damage plan** (one plan per word, identical at every level that damages it): about 10% of damaged words are dropped completely (never the anchor); otherwise 20-30% of the word's letters and digits are altered (lowered from an initially-frozen 40-60%: that range sat above the third quartile of real OCR word-error severity measured in `reports/calibration.json`, so it was revised down to sit inside that real range; see §0 and the deviations log for the full reasoning); each altered character is deleted (about 15%, at most 2 per word) or substituted; substitutions come from the calibrated confusion table (85.5%), its lowercase entry written as a capital (3.1%), a look-alike map such as O/0, l/1/I, rn/m, cl/d (5.1%), or a random character of the same kind (6.3%). Cross-case confusions are allowed. Punctuation is never touched.
- **Resulting sentence-level CER against gold, current file (`corrupted_v2.jsonl`, 20-30% intensity, mean):** 1w 0.0098, 10% 0.0319, 25% 0.0793, 50% 0.1549, 75% 0.2307. Real BLN600 OCR is about 0.07 per excerpt, so this sits much closer to real severity than the original file, and 75% is now about 3 times heavier rather than 5. (Superseded numbers, 40-60% intensity, `corrupted_v1.jsonl`, median: 1w 0.018, 10% 0.047, 25% 0.118, 50% 0.236, 75% 0.351; kept only for comparison, e.g. as appendix material.)
- **Stored fields per row:** `gold_tokens`, `corrupted_text`, `span_start`, `span_end`, `k`, `damaged_idx`, `anchor_idx`, `fact_idx_in_span`, `ops_per_word` (gold word, damaged form `out`, `dropped` flag, edit ops), `frac`, `band`, `split`, `severity`.
- **Damaged words per level (test, median):** 20-29 band: 1 / 2 / 6 / 12 / 18; 30-44: 1 / 4 / 9 / 18 / 26; 45-60: 1 / 5 / 13 / 26 / 38 (levels 1w / 10 / 25 / 50 / 75).
- **Facts in the damaged span (test):** at least 1 at every level; mean 1.0 / 1.6 / 2.2 / 3.2 / 4.2. Share of damaged words that are facts: 100% / 48% / 26% / 19% / 17%.
- **Implementation note:** in `corrupted_text` a dropped word simply disappears. Any model input must be built from `gold_tokens` + `ops_per_word`, not from `corrupted_text`, or dropped words become invisible and slot counts break.

## 6. Shared repair input: the slot view
Both methods receive exactly the same information:

1. The undamaged context, unchanged.
2. The location of the damaged span.
3. One **slot** per damaged gold word, in order, showing that word's garbled form. A dropped word is an empty slot `⟨?⟩`. So the number of damaged gold words is known to both methods.

Example (test sentence `3200810696-001`, level 10, gold span "named Thomas Hurd, a"):
`A POWERFULLY-BUILT man, ⟨same⟩ ⟨Bbonias⟩ ⟨Huicl,⟩ ⟨e⟩ commission agent, who refused his address, ...`

Both methods return **one word per slot**. The same splice code writes the predictions back into the sentence, so context words are never rewritten and every score difference comes from the slots.

**Justification for the report:** OCR engines commonly flag low-confidence regions, and a reader of a damaged page still sees partial letters. Showing the garbled letters keeps the task an OCR-repair task and makes facts recoverable in principle (`1awsenc` -> `Lawrence`); hiding them would turn most fact repairs into pure guessing, which is not what archives face. Finding the damage without being told where it is stays future work (§13).

## 7. Repair methods (mapped to course lessons)
### 7.1 Fine-tuned BERT with character-aware reranking (BERT lesson)
- Model: `bert-base-cased`. Cased is required because fact tokens are partly defined by capitals, and an uncased model cannot output capitals at all.
- **Fine-tuning:** standard masked language model training on gold text from the BERT training pool (§4), masking contiguous spans around a fact token with the same size distribution as the evaluation levels. BERT never needs garbled text during training.
- **Inference (noisy-channel reranking):** fill slots left to right; later slots wait as `[MASK]`. For the current slot, try 1, 2 and 3 masks and collect candidate words from each (small beam over the pieces). Each candidate gets a score = BERT log-probability (mean per piece) + λ x character similarity to the slot's garbled form. The best candidate is written in and the next slot is filled. For a dropped slot `⟨?⟩` there is no garbled form, so only the BERT score counts. Trying 1-3 masks means BERT is never told the gold subword count.
- **Character similarity:** plain normalised edit distance (1 minus Levenshtein distance divided by the longer length). The confusion-weighted version is an ablation only (§7.3), because the confusion table also generated most of the damage.
- **λ, beam size and candidate count** are tuned on dev only. This mirrors the classic "context plus confusion" approach to OCR correction (Evershed & Fitch, 2014), in a basic form.
- Ownership: coded by one team member, training run by another, in parallel with the LLM track.

### 7.2 Generative LLM, zero/few-shot (GPT and LLM lessons)
- **Access:** GWDG Chat AI (SAIA, OpenAI-compatible API). **Model (O2, proposed, needs your confirmation): Llama 3.1 8B Instruct**, open-weight, GWDG-hosted. Candidate upgrade if instruction-following on the slot format is too weak after dev tuning: Qwen 3 30B A3B Instruct 2507 (also open-weight, GWDG-hosted). Reasons in §7.2a. Fallback for infrastructure only: open-weight model on the FAU/Bayern KI HPC.
- Prompt: the slot view, the number of slots, and an instruction to return a JSON list with exactly one corrected word per slot. Zero-shot, plus a few-shot variant with 2-3 examples from dev.
- Decoding: temperature 0 (or the lowest setting available). Record the exact model name and version.
- If the output list has the wrong length, record it as a format failure (report the rate), score those slots as wrong, and do not repair the output by hand.
- **Contamination probe:** give the LLM the first half of each clean test sentence and ask it to continue. Count near-verbatim continuations (for example character similarity to the gold second half of 0.9 or more). Report the rate; if it is high, flag LLM results as possibly inflated.
- **Update (2026-09-26):** zero-shot was tried on the full dev split and dropped from the study (weak results, and it was not needed to answer SQ2). The study now runs few-shot only, plus a secondary few-shot + article variant (whole source article as extra context, compared only against the LLM's own few-shot run, never against BERT). See §0 for current dev results and the open model-choice decision.

### 7.2a Why Llama 3.1 8B Instruct (O2, proposed)
Two reasons, not one:
1. **License and data handling.** BLN600 is CC BY-NC-ND, and the project rule is to send its text only where necessary and with the least exposure. GWDG states that for open-weight models it hosts itself, prompts and outputs are never stored. That guarantee is stated only for the open-weight models, not for the third-party-hosted ones (Claude, GPT), which forward the request to Anthropic or OpenAI under their own terms. So an open-weight, GWDG-hosted model is the safer and more defensible choice for this corpus, independent of capability.
2. **Precedent and fit.** Thomas, Gaizauskas & Lu (2024), the closest related work and the only prior study to run an LLM-family model on BLN600 itself, used a fine-tuned Llama 2 and found it beat a fine-tuned BART. Llama 3.1 8B Instruct continues that lineage as a newer generation, zero/few-shot rather than fine-tuned. It is also GWDG's own listed standard recommendation, and at 8B it is fast and cheap enough for the roughly 1,000 planned calls (§9), which matters given the limited SAIA quota.

Trade-off to state in the report: 8B is a small model, so a higher format-failure rate than a larger model is plausible; that is itself a result (§8), not a flaw to hide. If dev tuning shows format failures above about 15% or clearly weak repairs, switch to Qwen 3 30B A3B Instruct 2507 (still open-weight and GWDG-hosted, stronger instruction following, still economical because only 3.3B parameters are active per token) and record the switch and why.

**Update (2026-09-26): the switch rule has been triggered.** The full dev run came in at 21% format failures (few-shot) and 27% (few-shot + article), both above the 15% line. This is an open decision for the team (§0): switch to Qwen 3 30B A3B Instruct 2507 and rerun the dev prompts, or keep Llama 3.1 8B and report the high failure rate as a finding.

### 7.3 Optional ablations (cut first if short on time)
1. **Blank slots:** both methods with garbled letters hidden (dropped-word count still given). Measures how much the partial letters help each method.
2. **Confusion-weighted reranking:** BERT with edit distance weighted by the calibrated confusion table. Shows how much knowing the noise model helps. Report it as an upper bound, not as the main result: the same table produced 85.5% of the substitutions, so BERT would be using inside knowledge of the synthetic generator that the LLM does not get and real OCR would not match exactly.

### 7.4 Link to related work
The design mirrors the closest studies: a fine-tuned encoder or encoder-decoder versus a prompted generative LLM on historical OCR text. Thomas, Gaizauskas & Lu (2024) found an instruction-tuned generative LLM (Llama 2) beat a fine-tuned BART on post-OCR correction of BLN600 itself, while Debaene et al. (2025) found sequence-to-sequence models beat generative models on early modern Dutch. This project adds a controlled, fact-centred severity scale and a fact-level metric, under identical input for both methods, and moves from a fine-tuned Llama 2 to a zero/few-shot newer-generation open-weight model on the LLM side (§7.2a).

## 8. Evaluation
All metrics are computed per sentence and level, over the damaged span only unless stated.

- **Gap-level semantic similarity (primary):** BERTScore (Zhang et al., 2020) between the repaired span and the gold span. Full-sentence BERTScore is reported as a secondary number only: at low levels the unchanged context keeps full-sentence scores near the top of the range whatever the repair does.
- **Fact Recovery Rate (primary):** share of gold fact tokens in the span (`fact_idx_in_span`) restored exactly in their slot (punctuation around the word ignored). Defined at every level, because every span contains the anchor fact. At level 1w it is 0 or 1 per sentence.
- **Exact word recovery (secondary):** share of slots restored exactly.
- **Repair gain (secondary):** character error rate of the span before repair (damaged) and after repair, against gold. Shows whether a method improves the text or makes it worse.
- **Format failure rate (LLM):** share of outputs with the wrong number of slots.
- **Contamination rate (LLM):** share of near-verbatim continuations in the probe (§7.2).
- **Manual check (tertiary):** about 15-20 repairs across levels, to find fluent but wrong repairs for SQ3.
- **Analysis:** main figure: level (x) against gap-level BERTScore and Fact Recovery Rate (y), one line per method. Second figure or table: level x band, to separate damaged share from absolute gap length (SQ1). Levels are nested within the same sentence, so use paired comparisons across levels and between methods.

## 9. Experimental design summary
| Factor | Levels |
|---|---|
| Damage | one contiguous span around a fact anchor, character-level OCR noise, about 10% of words dropped (frozen v2, 20-30% per-word intensity; see §0) |
| Severity | 0, 1w, 10%, 25%, 50%, 75% of sentence words |
| Length band | 20-29, 30-44, 45-60 words |
| Input to models | slot view: span location, one slot per gold word, garbled letters visible, dropped words as empty slots |
| Methods | fine-tuned `bert-base-cased` + reranking by plain edit distance (1-3 masks per slot); Llama 3.1 8B Instruct few-shot via SAIA, zero-shot tried and dropped (§7.2, §0); secondary: LLM few-shot + whole-article context, compared only to the LLM's own few-shot run |
| Optional ablations | blank slots for both methods; confusion-weighted reranking for BERT |
| Evaluation data | 150 test sentences (20 dev for tuning only; test split not yet run, see §0) |
| Metrics | gap-level BERTScore, Fact Recovery Rate (primary); exact word recovery, repair gain, full-sentence BERTScore, format failures, contamination rate (secondary); manual check |

## 10. Fairness checklist and remaining asymmetries
Controlled:
- Same sentences, same slots, same garbled letters, same context for both methods.
- Both output one word per slot; identical splicing and scoring code.
- BERT is not told the gold subword count (1-3 masks tried per slot).
- The main reranking does not use the confusion table that generated the damage.
- No test or dev excerpt in BERT training; few-shot examples and all tuned settings from dev only.
- Deterministic decoding.

Remaining asymmetries (state in the report):
1. **How the letters are used.** The LLM reads garbled letters directly; BERT uses them only through the reranking score. This is a real property of the two approaches, not a setup error.
2. **Parallel prediction inside a slot.** BERT predicts the pieces of a multi-piece word in one pass, which weakens it on long rare names (Shen et al., 2020 note this for multi-token blanks). Words needing more than 3 pieces cannot be produced exactly.
3. **Scale and pretraining.** BERT-base has about 110M parameters and is fine-tuned in-domain; the LLM (Llama 3.1 8B Instruct) is larger and instruction-tuned but gets no in-domain training. SQ2 compares two practical approaches, not two architectures at equal size.
4. **Possible LLM data contamination.** BLN600 has been public since 2024, so the LLM may have seen the gold text (Sainz et al., 2023). Measured by the probe in §7.2, not removed. `bert-base-cased` predates the dataset; Llama 3.1's pretraining cutoff should be checked against the dataset's 2024 release and stated in the report.

## 11. Limitations for the report
- The damage location is given. Real use needs a detection step first; neither method is tested on that.
- Damage is synthetic, one contiguous span per sentence, and heavier than typical BLN600 OCR from level 10% up.
- Damage is centred on a fact token by design, so facts are over-represented in the damaged span (100% of damaged words at 1w, 17% at 75%). Results describe fact-centred damage, not random damage, and the fact mix changes across levels.
- Severity is the controlled variable, but what sits in the span (a stock phrase versus a rare name) strongly affects difficulty and is not controlled.
- Small sample (150 test sentences; only 26 in the 45-60 band): trends, not statistically powered effects.
- Fact Recovery Rate uses surface rules (numbers, capitals, money). It misses facts in lowercase words and counts exact matches only.
- English, 19th-century London crime reporting, mostly three related publications.
- The asymmetries in §10.

## 12. Team split and phases
1. **Data and damage (done, frozen).** Build the slot view from `gold_tokens` + `ops_per_word` (§5 note).
2. **Repair, in parallel:**
   - BERT: build the training pool, span-masking fine-tuning, reranking inference with 1-3 masks per slot and plain edit distance, tuning of λ, beam and candidate count on dev.
   - LLM: SAIA setup with Llama 3.1 8B Instruct (§7.2a), zero-shot and few-shot prompts with JSON slot output, logging of model version and format failures, contamination probe.
   - Shared: one slot-view builder, one splice function and one scoring module used by both tracks.
3. **Evaluation:** all metrics per level, band and method; figures; manual check; optional ablations (§7.3).
4. **Handoff:** the LLM tuning and test run are done by a teammate on an own machine (§0, README).
5. **Writing:** Method and Empirical Investigation first, then Introduction, Related Work, Interpretation and Limitations. Trim to 9 pages; list appendix material for the winter-term defence.

## 13. Future work (out of scope for this report)
- **Repair without a given location:** methods must first find the damage in raw OCR text, then repair it.
- **Transfer to real OCR errors:** evaluate on real OCR/gold word pairs (for example the calibration pairs) to check whether conclusions from synthetic damage hold.
- **Image-assisted repair:** give a vision-capable model the source scan with the damaged region blurred. BLN600 images have no word-level boxes, so words must first be located (e.g. with `pytesseract`, matched by reading order). Plain BERT cannot read images; vision-language BERT variants (VisualBERT, ViLBERT, LXMERT, VL-BERT) need separate region features and their own pretraining, so a vision-capable LLM is the practical tool. Precedent: "Reading the unreadable: creating a dataset of 19th century English newspapers using image-to-text language models" (Digital Scholarship in the Humanities; confirm authors before citing).
- **Other languages:** German historical newspapers.

## 14. Verified references
- Booth, C. W., Thomas, A., & Gaizauskas, R. (2024). BLN600: A Parallel Corpus of Machine/Human Transcribed Nineteenth Century Newspaper Texts. LREC-COLING 2024, pp. 2440-2446.
- Thomas, A., Gaizauskas, R., & Lu, H. (2024). Leveraging LLMs for Post-OCR Correction of Historical Newspapers. LT4HALA @ LREC-COLING 2024, pp. 116-121.
- Debaene, F., Maladry, A., Lefever, E., & Hoste, V. (2025). Evaluating Transformers for OCR Post-Correction in Early Modern Dutch Theatre. COLING 2025, pp. 10367-10374.
- Lyu, L., Koutraki, M., Krickl, M., & Fetahu, B. (2021). Neural OCR Post-Hoc Correction of Historical Corpora. TACL, 9, 479-493.
- Evershed, J., & Fitch, K. (2014). Correcting Noisy OCR: Context Beats Confusion. DATeCH 2014, pp. 45-51.
- Zhu, W., Hu, Z., & Xing, E. (2019). Text Infilling. arXiv:1901.00158.
- Shen, T., Quach, V., Barzilay, R., & Jaakkola, T. (2020). Blank Language Models. EMNLP 2020, pp. 5186-5198.
- Wettig, A., Gao, T., Zhong, Z., & Chen, D. (2023). Should You Mask 15% in Masked Language Modeling? EACL 2023.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR 2020.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
- Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR, 21(140), 1-67.
- Sainz, O., Campos, J., García-Ferrero, I., Etxaniz, J., Lopez de Lacalle, O., & Agirre, E. (2023). NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark. Findings of EMNLP 2023, pp. 10776-10787.
- Petrak, D., Tran, T. T., & Gurevych, I. (2025). Towards Automated Error Discovery: A Study in Conversational AI. arXiv:2509.10833. (The course's example paper; structural template only.)
