# Project Dossier: Repairing Damaged Historical Newspaper Text with GenAI

**One dossier (consolidated 2026-09-29; the claude.ai project copy `claude/project-dossier.md` is the same file).** Section numbers 1 to 11 and 13 keep the old `dossier.md` numbering, so references in the `CLAUDE.md` log (for example §5, §6, §7.1, §7.2a, §10) still point to the right place. This file replaces both `docs/dossier.md` (the long design record) and `docs/dossier_v2.md` (the short current-state version). `dossier_v2.md` is removed; its content is merged here. Superseded numbers are kept, labelled, where the report or the defence may need them.

Where things live: decisions and numbers with their reasons in the deviations log at the end of `CLAUDE.md` (newest at the bottom); every fact the report needs, by report section, in `docs/report_sources.md` (same file as `report/overleaf-bln600/notes/report_sources.md`); Simon's session state in `docs/handoff/` (his review of 28 Sep: `docs/handoff/review_2026-09-28_simon.md`). If this file and the code disagree on implementation, the code wins; on scope or design, this file wins.

Course: GenAI Master's coursework (6 ECTS), team of 3 (Bea Dippold, Ruo Shan Tan, Simon Manzenberger). Report: 7 to 9 pages, ACL style, no extensive appendix (GRIPS announcement "Project Report", 17 Jul 2026). Graded: report, demonstration, defence; explicitly "a constructive research hypothesis" and evaluation "according to state-of-the-art methods" (announcement "Online Week", 30 May). **Submission Thu 1 Oct 2026, 00:00. Defence Gruppe 4, 14 Oct 2026, 10:30**, based on the paper.

---

## 0. Status (2026-09-29, evening)

### What is compared
Every method gets the same input (§6, the slot view) and is scored by the same code on the same rows. Six methods plus the floor:

| # | Method (report name) | Stored as | Role | Test predictions |
|---|---|---|---|---|
| 1 | **BERT + dictionary** | `bert_rerank` `ftv1-l64-b5-n10-x5` | main MLM method (H2) | done, 750 rows |
| 2 | **LLM few-shot** (Qwen3-30B-A3B) | `llm_fewshot` `v4-...-exb46768` | main LLM method (H2, H2b) | done, 750 rows |
| 3 | LLM few-shot + article | `llm_fewshot_article` `v3-...-exb46768` | secondary, vs 2 only | done, 750 rows |
| 4 | **Dictionary lookup** (no AI) | `lexicon` `v1` | non-GenAI baseline (H2b) | done, 750 rows |
| 5 | BERT without dictionary | `bert_rerank` `ftv1-l8-b5-n10` | ablation line | done, 750 rows |
| 6 | **BART fine-tuned denoiser** | `bart` `ftv1-greedy` | added after the hypotheses, descriptive only | done, 750 rows (40 format failures, a count only) |
| - | No repair | (damaged text) | floor | n/a |
| - | Contamination probe (Qwen) | `llm_probe` `v1-...` | LLM only | **116 of 150, final** (stopped for API resources) |

### Done
- Data and damage frozen: `corrupted_v2.jsonl` (20-30% per-word intensity, sha256 `5798336d...`). v1 removed from git.
- BERT fine-tuned (`models/bert_ft_v1`); context-only reranking frozen (lambda 8, beam 5, top 10; `configs/bert_repair.yaml`).
- LLM frozen: Qwen3-30B-A3B-Instruct-2507 on GWDG SAIA, 5 few-shot examples (one per level), numbered-slot prompts v4 (few-shot) and v3 (article) (`configs/llm.yaml`).
- Dictionary lookup (notebook 2b, `configs/lexicon.yaml`) and BERT + dictionary candidates (notebook 2b, `configs/bert_repair_lex.yaml`, `lex_n` 5, lambda 64): both frozen, test run stored. Merged as PR #1.
- Notebook 4 extended to all methods: bootstrap intervals, paired tests, SQ3 number, anchor recovery, visible vs dropped facts, pooled CER, LLM valid-only. Scoring code in `src/blnrepair/evaluation.py`. Merged as PR #2. **Dev only so far.**
- BART-base fine-tuned on a RunPod A40 (22.5 min, $0.20), dev predictions stored, notebook 2c. Merged as PR #3. Frozen (`configs/bart_repair.yaml` `frozen: true`, `56f6682`) and run on test on a second A40 pod (`bd84f7e`, 750 rows, 40 format failures); BART in total $0.48. Merged into `writing` on 2026-09-29 (`8a6a988`).
- LLM test run (Shan, SAIA): few-shot 750/750, article 750/750, probe 116/150 (`d3da046`).
- Note on the A40: it was used for **BART only** (training, dev and test runs). All LLM calls go through SAIA (Shan's runs), none ran on the A40.
- Repo cleanup on `writing` (`6e9ecf7`): superseded LLM dev runs and prompts moved to the git-ignored `runs/_archive/`; LaTeX byproducts untracked. No code, config or notebook changed.

### Open, in order
1. **Notebook 4 on test, once** (`SPLIT = "test"`), **Simon** (his handoff, step 2; one person per notebook). All inputs are complete: six methods x 750 rows, probe 116 rows. Commit the executed notebook and hand the tables and figure to the team.
2. **Report, after the test run:** `tab:results` (BERTScore and FRR, columns BERT + dict., LLM, lookup, BART, none), Figure 2 (level x score with CIs), Results text in the order in `04_experiments.tex`, one verdict per hypothesis in the Interpretation, Abstract, the Introduction's key outcomes. About 1.25 pages are left (local build, 2026-09-29).
3. **References:** done. Lewis 2020, Koehn 2004, Dror 2018, Holm 1979 and Qwen3 are in `references.bib` (Shan, via Zotero); Dietterich 1998 dropped, Dror et al. (2018, §3.2) covers McNemar and Wilcoxon. `references.bib` also holds about 30 Zotero entries from other projects; uncited entries do not print.
4. **Before submission:** remove every `\todo`; check page count <= 9 without references; only test numbers in Results.

### Hypotheses (from Simon's review, 2026-09-28; in the Introduction since 2026-09-29)
- **H1 (severity):** repair quality falls as the damaged share grows, and fact recovery falls faster than span BERTScore.
- **H2 (method):** with identical input, the few-shot LLM restores more fact words than fine-tuned BERT, because it reads all garbled forms together. Tested as BERT + dictionary vs few-shot.
- **H2b (baseline):** GenAI repair beats a dictionary lookup on fact words, not only on common words. Tested as few-shot vs lookup.
- **H3 (metric):** span BERTScore overstates repair quality: at every level a share of wrong-fact repairs scores as high as right-fact repairs.

Honest framing for the report: the hypotheses were written **after** the dev results but **before** any test score was computed, and the paired tests were limited to H2 and H2b for that reason (§8). BART was added after the hypotheses and is compared descriptively only.

---

## 1. Working title
Report (team decision 2026-09-27): "How Much Damage Can GenAI Repair? Fact Recovery in Damaged Historical Newspaper Text". Earlier: "How Much Can Be Torn Away? ..." (dropped: it implied image-level repair).

## 2. Problem, research question, sub-questions
Historical newspapers suffer physical damage; OCR on damaged regions gives garbled or missing words, and the words most at risk are often facts: names, places, dates, sums of money.

**RQ (report wording, 2026-09-28):** When the damaged words of a sentence are known and their garbled letters are visible, how much of the original wording, and in particular its fact words such as names, places and numbers, can a fine-tuned masked language model and a few-shot instruction-following LLM restore, and how does this change as the damaged share of the sentence grows?

- **SQ1 (severity):** smooth decline or breakdown point? Does absolute gap length matter beyond the damaged share?
- **SQ2 (method):** given identical input, does a fine-tuned MLM or a few-shot LLM repair better? Since PR #1 also: does either beat a plain dictionary lookup?
- **SQ3 (metric validity):** does semantic similarity overstate repair quality compared with exact fact recovery?

Open point for the report: the RQ names two families; BART (a fine-tuned encoder-decoder) is a third. Either widen the RQ to "GenAI repair methods" or present BART as an extra reference line.

## 3. Motivation
Digitised newspaper archives are core sources; OCR on damaged originals limits their use. GenAI fills text fluently, which risks plausible but invented facts entering the record. This is a controlled robustness study, not a new system. Course links: BART is the lecture's denoising model ("reconstruct original text from corrupted input"); the diffusion slides' "hallucinate something similar" is the SQ3 point; the article variant is RAG-style context injection with perfect retrieval; the prompt has the lecture's five parts (task, context, examples, role, format).

## 4. Data and splits
**BLN600** (Booth, Thomas & Gaizauskas, 2024): 600 excerpts of 19th-century British newspapers (London crime reporting, 1834-1894), 572 from Lloyd's Weekly Newspaper (340), Illustrated Police News (212), Lloyd's Weekly London Newspaper (20). Image, OCR and gold per excerpt. CC BY-NC-ND: `data/` is git-ignored; never redistribute derived text.

- Sentence pool: `pysbd`, 20-60 words, at least 2 fact tokens: 3,123 sentences from 592 excerpts.
- Fact token: a number, any word with a digit or pound sign, or a capitalised non-initial word; stop list I, Mr, Mrs, Dr, St, The, Miss, Sir, Rev, Messrs; not after a colon. Surface rule, not NER.
- Evaluation sample (seed 42): 170 sentences from 157 excerpts: **150 test** (65/59/26 by band 20-29/30-44/45-60), **20 dev** (9/8/3). At most 3 per excerpt; no excerpt in both. Dev for tuning only.
- Scored dev set: **15 sentences**, 75 rows per method (the 5 few-shot example sentences are left out for every method).
- Calibration: 29 excerpts outside the sample (confusion table only).
- Training pool (BERT and BART): 443 excerpts outside the sample, 2,939 sentences (2,635 train / 304 validation), zero overlap with test/dev (tested). The dictionary word list is built from the same 443 excerpts (16,469 distinct words, 213,237 tokens).

## 5. Damage simulation (frozen, `corrupted_v2.jsonl`)
- One anchor per sentence among its fact tokens; always damaged, never dropped.
- One contiguous span around the anchor, nested across levels. Levels: 0 (clean), 1w, 10/25/50/75% of words. 170 x 6 = 1,020 rows; 850 with slots (750 test, 100 dev).
- Per damaged word: dropped with p 0.10 (never the anchor); else 20-30% of letters and digits altered; each altered character deleted (about 15%, at most 2 per word) or substituted. Substitution sources (realised): confusion table 85.7%, capitalised entry 3.6%, look-alike map 4.9%, random 5.8% (`reports/corruption_stats.json`; older figures 85.5/3.1/5.1/6.3 are superseded). Punctuation untouched.
- Why 20-30%: the first setting, 40-60%, sat above the third quartile of real per-word error among misrecognised words (median 0.25, IQR 0.17-0.40); 20-30% sits around the median.
- Sentence-level CER, mean: 1w 1.0%, 10% 3.2%, 25% 7.9%, 50% 15.5%, 75% 23.1% (real BLN600 OCR about 7% per excerpt). Superseded v1 (40-60%): 1.9 / 5.1 / 12.1 / 23.7 / 35.1% (appendix material only).
- Facts in span (test, mean) 1.0 / 1.6 / 2.2 / 3.2 / 4.2; share of damaged words that are facts 100 / 48 / 26 / 19 / 17%. About 1 in 10 fact slots is dropped.
- Build model input from `gold_tokens` + `ops_per_word`, never from `corrupted_text`.
- Calibration limitation: it measures which characters get confused in mild OCR, not how heavy real physical damage is.

## 6. Shared input: the slot view
Context unchanged; one slot per damaged gold word showing its garbled form; a dropped word is `⟨?⟩`. So the damaged-word count is known to all methods. Example (test `3200810696-001`, level 10, gold "named Thomas Hurd, a"): `A POWERFULLY-BUILT man, ⟨same⟩ ⟨Bbonias⟩ ⟨Huicl,⟩ ⟨e⟩ commission agent, ...`. Every method returns one word per slot; the same splice code writes it back inside the gold word's edge punctuation.

Presentation differs, information does not: the LLM prompts number the slots (`⟨1: Hnrd⟩`); BART uses `{ word }` brackets because its byte-level tokenizer breaks ⟨ ⟩ into byte pieces.

Justification (report, 2026-09-28): the detection/correction split of post-OCR work (Nguyen et al. 2021; Rigaud et al. 2019) and known-position restoration (Assael et al. 2019; Lazar et al. 2021). **Not** "OCR engines flag low-confidence regions": Evershed & Fitch (2014, §5.1) found OCR confidences unreliable.

## 7. Repair methods

### 7.1 BERT + reranking, context only (ablation line)
`bert-base-cased` (110M), fine-tuned on clean gold text only, inference-matched masking (option b). lr 5e-5, batch 16, early stop (patience 2, max 10), best epoch 2 of 4, validation loss per piece 5.61 (pretrained) to 4.14. Inference: slots left to right, 1-3 masks per slot, beam 5, top 10 per mask count; score = mean log-probability per piece + lambda x (1 - normalised Levenshtein). Lambda grid on dev (1/2/4/8/16): exact 0.164/0.212/0.239/0.267/0.263, FRR 0.042/0.092/0.116/0.126/0.128; **lambda 8**. Weakness: candidates come from context only, so a word BERT does not propose can never win (`Wrighb` gives Smith, White, andooll).

### 7.2 LLM few-shot (main LLM method; history in 7.2a)
Qwen3-30B-A3B-Instruct-2507 (MoE, 3.3B active parameters) on GWDG SAIA, open-weight only (GWDG does not store prompts for self-hosted open-weight models). temperature 0, max 1024 tokens. System message (needed: without it Llama answered in Python in 15 of 15 dev cases); 5 solved dev examples, one per level (`3206323884-017` 1w, `3206270194-012` 10, `3200807881-004` 25, `3206225730-001` 50, `3206237078-010` 75); numbered slots; JSON list of exactly k strings. Wrong length, invalid JSON or cut-off = format failure = scored as no repair; LLM also scored over valid answers only. Zero-shot tried on dev and dropped.

**7.2a History (keep for the defence).** Llama-3.1-8B-Instruct was the first choice (licence/data handling; Thomas et al. 2024 lineage). Its dev format failures were 21% (few-shot) and 27% (article), above the pre-registered 15% line, so the team switched to Qwen on 2026-09-28. Prompt versions: few-shot v2 (Llama), v3 (answer-shape example), **v4 (numbered slots, final)**; article v1, v2, **v3 (final)**.

### 7.3 Optional ablations (not run; appendix only)
Blank slots (letters hidden); confusion-weighted reranking (upper bound, since the table made 85.7% of substitutions).

### 7.4 LLM few-shot + article (secondary)
Whole excerpt as context with the target sentence as slot view between `<<<` `>>>`; the other sentences are gold, so this is an upper bound. Compared only with 7.2 (paired, descriptive).

### 7.5 Dictionary lookup (non-GenAI baseline, since 2026-09-28)
Each damaged word is replaced by the closest word-list entry (plain case-sensitive Levenshtein, ties by frequency then alphabet). No context, no model: the pure "confusion" side of "context beats confusion" (Evershed & Fitch 2014). A dropped slot stays empty (counts as wrong). Answers the examiner question "does GenAI beat a spell checker?". The list holds 95% of damaged dev words but only 77% of damaged fact words (48 of 62): rare names are where a lookup cannot help.

### 7.6 BERT + dictionary candidates (main MLM method, since 2026-09-29)
The `lex_n` word-list entries closest in spelling to each damaged form join BERT's candidates and get the same score. Same model, BERT still never sees damaged text in training. Off by default in code, so the old run reproduces exactly (40 of 40 rows checked, CPU and MPS). Grid on all 100 dev rows (`lex_n` 5/10/20/50 x lambda 1-64, +128): chosen **`lex_n` 5, lambda 64** by the agreed rule (best exact rate once FRR plateaus). Dev: exact 0.619, FRR 0.560 (vs 0.267 / 0.126 without dictionary; lookup alone 0.528 / 0.426). Note for the defence: `lex_n` 50 at lambda 32 has a slightly higher exact rate (0.628), noise level; and at lambda 64 the letters decide most slots, BERT's context mainly breaks ties and fills dropped words.

### 7.7 BART fine-tuned denoiser (added 2026-09-29, descriptive only)
`facebook/bart-base` (about 140M) reads the slot view and writes the sentence back with the gold word in each bracket; a wrong bracket count is a format failure. Trained on the training pool damaged by **our own generator** at all five levels (13,175 examples per epoch, new damage per epoch), same optimiser settings as BERT, greedy decoding with all settings passed explicitly (bart-base ships summarisation defaults, `no_repeat_ngram_size` 3, which would break copying). Best epoch 8 of 10 (validation loss 0.266, pretrained 2.507); validation exact 0.672. Overfitting visible: exact on training sentences with new damage 0.879 vs 0.672 on validation. **Must be defended:** unlike BERT and the LLM, BART is supervised on exactly the noise process of the test data, so its scores are optimistic. It links to Thomas et al. (2024), who compared a fine-tuned BART on BLN600.

### 7.8 Link to related work
Closest studies compare a fine-tuned encoder or encoder-decoder with a prompted generative LLM on historical OCR text: Thomas, Gaizauskas & Lu (2024) found an instruction-tuned Llama 2 beat a fine-tuned BART on BLN600 itself; Debaene et al. (2025) found fine-tuned seq2seq models beat generative ones on early modern Dutch. This project adds a controlled, fact-centred severity scale, a fact-level metric, identical input for all methods, a non-GenAI lookup baseline, and a few-shot newer open-weight LLM. Our BART line connects directly to Thomas et al.'s BART comparison.

## 8. Evaluation (as built in notebook 4, `src/blnrepair/evaluation.py`)
Computed over the damaged span unless stated; format failures scored as no repair.

- **Primary:** span BERTScore F1 (`roberta-large`, layer 17, baseline-rescaled, can be negative; `bert-score` 0.3.13); **Fact Recovery Rate** (gold fact tokens in the span restored exactly, punctuation ignored; mean over sentences).
- **Secondary:** anchor recovery (same word at every level, clean paired series for SQ1); FRR split into visible and dropped fact slots (pooled); exact-word rate; span CER before/after and repair gain, **pooled per level**; sentence-level BERTScore; LLM format failures per level and valid-only scores; contamination rate.
- **Intervals:** 95% bootstrap over sentences, 10,000 resamples, percentile, same resamples for all methods (Koehn 2004; Dror et al. 2018). Reason: slots of one sentence share context and are not independent.
- **Paired tests, only for the two stated comparisons:** BERT + dictionary vs few-shot (H2) and few-shot vs lookup (H2b). Per level: exact McNemar on anchor recovery, Wilcoxon signed-rank on span BERTScore (both cited via Dror et al. 2018, §3.2); Holm over 2 pairs x 5 levels per test family. Everything else descriptive.
- **SQ3 number (H3):** per method and level, the chance that a wrong-anchor repair gets at least as high a span BERTScore as a right-anchor repair (ties half) = 1 - AUC. 0 = BERTScore always ranks the right fact higher; 0.5 = BERTScore does not see the fact. No threshold to tune. Check: 0.00 at level 1w for every method on dev.
- **Contamination probe:** first half of each clean sentence (level 0 rows), model continues; normalised Levenshtein similarity (`rapidfuzz`) to the true second half, cut to its length; >= 0.9 = near-verbatim. The prompt does not name BLN600, so it is a simplified guided completion without the control condition of Golchin & Surdeanu (2024). Test: **116 of 150 sentences** (run stopped for API resources, final). With 0 hits the rule-of-three upper bound would be about 2.6% (vs 2.0% at 150).
- Manual check: 18 repairs (6 sentences at levels 10, 25, 50, seeded) for fluent-but-wrong examples.

## 9. Experimental design summary
| Factor | Levels |
|---|---|
| Severity | 1w, 10, 25, 50, 75% (level 0 only for the probe) |
| Length band | 20-29, 30-44, 45-60 words |
| Input | slot view, garbled letters visible, dropped words as empty slots |
| Methods | BERT + dictionary; LLM few-shot; dictionary lookup; no repair. Secondary: LLM + article (vs few-shot), BERT without dictionary (ablation), BART (descriptive) |
| Data | 150 test sentences, 750 rows per method; dev 20 (15 scored) for tuning |
| Metrics | span BERTScore, FRR (primary); anchor recovery, visible/dropped FRR, exact words, pooled CER and repair gain, sentence BERTScore, format failures, valid-only, SQ3 rate, contamination |
| Statistics | bootstrap 95% intervals; McNemar + Wilcoxon with Holm for H2 and H2b only |

## 10. Fairness and asymmetries
Controlled: same sentences, slots, letters and context; same splice and scoring; BERT not told the piece count; main reranking does not use the confusion table; no test/dev excerpt in any training data or word list; all settings chosen on dev; code gate (`frozen: true` + `ALLOW_TEST`) before any test run; deterministic decoding.

State in the report:
1. How letters are used: LLM and BART read them directly; BERT only through the reranking score; the lookup uses nothing else.
2. BERT predicts a word's pieces in parallel; words over 3 pieces are out of reach (0.9% of slots, 3.5% of fact slots).
3. Supervision and scale differ: BERT fine-tuned on clean in-domain text; BART fine-tuned on text damaged by our own generator (optimistic); LLM sees 5 examples only.
4. Possible LLM contamination: BLN600 public since 2024; the Qwen3 Technical Report (arXiv:2505.09388) states no pretraining cutoff, so the argument rests on the probe (116 of 150 test sentences).
5. Format failures count as no repair (LLM and BART).
6. Article variant uses gold context: upper bound.

## 11. Limitations
Damage location given, no detection. Synthetic OCR-like character noise, one contiguous span, intensity per word, not spatially correlated; calibrated on mild OCR, so it likely understates real damage (Belinkov & Bisk 2018 on synthetic vs natural noise). Fact-centred by design (100% of damaged words at 1w, 17% at 75%); span content not controlled. FRR is a surface rule with exact match (`M'Donald` vs `McDonald` counts as wrong). 150 test sentences (26 in the longest band), one noise draw per sentence: trends, not powered effects. English, London crime reporting, mostly two weeklies; one LLM. One training run and seed per fine-tuned model; hypotheses written after dev; probe on 116 of 150 test sentences. `runs/preds/` holds derived BLN600 text: acceptable only while the repo stays private.

## 12. Dev numbers (tuning context only, never report as results)
All from notebook 4 on the 15 scored dev sentences unless stated.

| Span BERTScore | 1w | 10 | 25 | 50 | 75 |
|---|---|---|---|---|---|
| BART | 0.83 | 0.77 | 0.73 | 0.69 | 0.67 |
| BERT + dictionary | 0.85 | 0.71 | 0.48 | 0.44 | 0.43 |
| LLM few-shot | 0.86 | 0.65 | 0.45 | 0.44 | 0.17 |
| Dictionary lookup | 0.73 | | 0.43 | | 0.40 |
| BERT, no dictionary | 0.59 | | 0.16 | | -0.03 |

(Empty cells: not recorded in a doc; read them from notebook 4 if needed.)

- Anchor recovery: BERT + dictionary 0.53-0.60, few-shot 0.40-0.60, lookup 0.40 at every level, BART 0.53 at every level. No method restores a dropped fact slot (10 on dev).
- Pooled repair gain positive for every method except BERT without dictionary (-0.16 to -0.35).
- No paired test significant after Holm (smallest Holm p 0.73): 15 sentences are far too few; the test needs its 150.
- SQ3 rate at 25%: 0.45 (few-shot) to 0.70 (BERT + dictionary); BART 0.00 / 0.16 / 0.21 / 0.29 / 0.36 by level.
- LLM valid-only: at 75% few-shot span BERTScore 0.45 on its 10 valid rows vs 0.17 with failures as no repair, so its drop at 75% is mostly format failures (5 of 15).
- Qwen format failures: few-shot v4 7/75 (25: 1, 50: 1, 75: 5); article v3 7/75 (50: 2, 75: 5). Probe on dev: 0 of 20 near-verbatim (median 0.25, max 0.31).
- Two Qwen exact-word numbers exist and differ by definition: 0.487 (pooled over slots, quick script) and 0.559 (mean over sentences, notebook 4). Use notebook 4's definition in the report.
- Superseded (Llama 3.1 8B, 17 dev sentences, kept as a record): span BERTScore few-shot 1w 0.73 ... 75% -0.10; FRR 0.22; format failures 21% / 27%.

**Pattern to watch on test (a reason to check, not a finding):** on dev, BART leads from 25% up, BERT + dictionary and few-shot are close, and the lookup is strong. If test agrees, the story shifts from "BERT vs LLM" to "letter evidence and supervision on the noise matter more than model size", with the BART caveat in §7.7.

## 13. Future work
Damage detection in raw OCR; real OCR/gold pairs; image-assisted repair with a vision-capable model (BLN600 images have no word boxes); whole-excerpt context for all methods (BERT's 512-token limit); German newspapers.

## 14. Verified references (already used)
- Booth, Thomas & Gaizauskas (2024). BLN600. LREC-COLING 2024, pp. 2440-2446.
- Thomas, Gaizauskas & Lu (2024). Leveraging LLMs for Post-OCR Correction of Historical Newspapers. LT4HALA 2024, pp. 116-121.
- Debaene, Maladry, Lefever & Hoste (2025). COLING 2025, pp. 10367-10374.
- Evershed & Fitch (2014). Correcting Noisy OCR: Context Beats Confusion. DATeCH 2014, pp. 45-51.
- Shen, Quach, Barzilay & Jaakkola (2020). Blank Language Models. EMNLP 2020, pp. 5186-5198.
- Zhang, Kishore, Wu, Weinberger & Artzi (2020). BERTScore. ICLR 2020.
- Devlin, Chang, Lee & Toutanova (2019). BERT. NAACL 2019.
- Sainz et al. (2023). NLP Evaluation in trouble. Findings of EMNLP 2023, pp. 10776-10787.
- Petrak, Tran & Gurevych (2025). arXiv:2509.10833 (course example paper, structure only).
- The full list used by the report is `report/overleaf-bln600/references.bib`.
## 15. What the report still needs (checked against `report/overleaf-bln600/sections/*.tex`, 2026-09-29 evening)
**Done on 2026-09-29 (Shan's session):** hypotheses H1, H2, H2b, H3 in the Introduction (and "five examples", lookup and BART named); Method (b) BERT + dictionary, (c) Qwen3-30B-A3B with 5 examples, numbered slots and the Llama switch rule, (d) dictionary lookup, (e) BART with its caveat, (f) no repair; `tab:settings` and `tab:design` updated; statistics paragraph and SQ3 number (1 - AUC); Bea's todos answered (level 0 = probe only; anchor and visible/dropped computed; probe measure and prompt; sentence BERTScore reported); probe 116 of 150; Related Work probe sentence fixed (our prompt does not name BLN600); Interpretation P2 and Limitations updated; appendix outline written.

**Still open:**
- **Results:** fill `tab:results` (now 5 method columns x BERTScore/FRR blocks) with CIs from notebook 4; Figure 2; BERT without dictionary and LLM + article as one sentence each; format failures and LLM valid-only as a footnote.
- **Interpretation:** P1, P3, P4, P5 first sentences from test numbers; one verdict per hypothesis; delete the unused variant in each paragraph.
- **Abstract** and the Introduction's key outcomes.
- **RQ scope (team):** the RQ names two families; the lookup and BART are presented as reference lines. If BART leads on test, reconsider the RQ wording.
- **References:** done (see §0, open item 3).
- **Length:** content ends at about 7.75 of 9 pages; about 1.25 pages left.

## 16. Appendix outline (for the defence)
Damage parameters, confusion table and the superseded 40-60% setting; BERT and BART training curves (BART over/underfitting record in `runs/bart_ft_v1_log.csv`); lambda and `lex_n` grids; full prompts, the v1 failure and system-message fix, the Llama run and switch; zero-shot dev run; subword ceiling; manual check; probe details; ablations not run.

