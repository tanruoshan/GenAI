# Report source book (consolidated)

One place for every fact the report needs, merged from three repo files and ordered by report section:
`docs/dossier.md` (design record, newest status, 2026-09-26), `docs/dossier_v2.md` (short version) and
`CLAUDE.md` (implementation and deviations log, up to 2026-09-27). Configs and run records were read
directly: `configs/*.yaml`, `runs/bert_ft_v1_config.yaml`, `runs/config_snapshot_v2.yaml`, `runs/*.sha256`,
`reports/corruption_stats.json`, `reports/calibration.json`, `prompts/*.txt`.

Source tags: [D] dossier.md, [D2] dossier_v2.md, [C] CLAUDE.md, [cfg] config or run file.
Rule used when they disagree: the newest dated entry wins, and config files beat prose.

---

## 0. Conflicts found between the three files (fix before writing)

| # | Topic | What the files say | What is true now | Action |
|---|---|---|---|---|
| 1 | CER by level | [D2] §3: "mean, of the damaged span". [D] §5 and [C]: "sentence-level CER against gold". | Sentence-level. The 75% level has 23.1% CER over the whole sentence, not the span. | Fixed in `03_method.tex` Table 1. Fix [D2] §3. |
| 2 | BERT settings | [D2]: tuning in progress (lambda 1, 2 done, 4 partway). | Grid lambda 1, 2, 4, 8, 16 done on dev; **lambda 8 chosen**, beam 5, top N 10 [D §0, C, cfg]. | [D2] is stale. |
| 3 | Test gate | [D2] and CLAUDE.md "Day 3": write `runs/repair_config_v1.yaml`. | No such file. Per-track `frozen: true` flag in `configs/bert_repair.yaml` (true) and `configs/llm.yaml` (false) [D §0, C]. | [D2] and CLAUDE.md header are stale. |
| 4 | BERT test run | [D2]: not started. | **Done** 2026-09-27 00:37, `runs/preds/bert_rerank_ftv1-l8-b5-n10-test.jsonl`, 750 rows, not yet scored [C]. | Do not look at it before the LLM test run exists. |
| 5 | LLM model and host | [D2]: Llama 3.1 8B on SAIA, switch decision open. | Handed to a teammate who tunes on their **own machine and own open-weight LLM**; model may change [D §0, README]. | Method text must wait for the final model id, host and prompt version. |
| 6 | corrupted_v1 | CLAUDE.md "Status": v1 kept locally. Project docs still list `corrupted_v1.jsonl`. | v1 deleted locally; only in git history [C, Cleanup]. | Remove the stale project copy or label it. |
| 7 | Dev BERT scores | [D §0]: BERTScore 0.24, FRR 0.10. [C]: 0.241, 0.096. | Same numbers, rounding only. | None. |
| 8 | Fine-tuning epochs | [D §7.1]: plain span masking, no epoch count. CLAUDE.md Section B: "3 epochs, 10% warmup". | Early stopping (max 10, patience 2), warmup 10% of **one** epoch, inference-matched masking (option b) [C, cfg]. | [D §7.1] needs the option (b) text. Use [C] in the report. |
| 9 | Subword limit | [D §10]: "words needing more than 3 pieces cannot be produced exactly". | Sub-slots at visible hyphens/apostrophes; only 0.9% of slots (3.5% of fact slots) still exceed 3 pieces [C, CHECKPOINT 1]. | Use the numbers. |
| 10 | Calibration as a severity measure | Dossier: silent. | Calibration under-measures real severity (mild OCR, unpaired words hide fragments). Damage models OCR output noise, not physical tears [C, 2026-09-27]. | Add to Limitations. |
| 11 | 20-30% refreeze reason | CLAUDE.md: "reason not recorded" (C entry). | [D §5] gives it: 40-60% sat above q3 of real per-word error; 20-30% sits around the median 0.25. | Use the dossier reason. |

---

## 1. Introduction

- **Problem.** Physical damage (tears, stains, tape) makes OCR output garbled or missing words. The words most at risk are often facts: names, places, dates, sums [D §2].
- **Why GenAI.** Fluent models can fill text, which risks plausible but invented facts entering the historical record [D §3]. This is a controlled robustness study, not a new system [D §3].
- **RQ.** When the damaged region is known and its garbled letters are visible, how well can GenAI models restore the original words so that meaning and key facts are preserved, and how does this change as the damaged share grows? [D §2]
- **SQ1** severity curve and breakpoint, share vs absolute gap length; **SQ2** fine-tuned BERT vs few-shot LLM on identical input; **SQ3** does semantic similarity overstate repair (wrong names, numbers, dates)? [D §2]
- **Working title (dossier).** "How Much Can Be Torn Away? Measuring the Limits of GenAI-Based Repair of Damaged Historical Newspaper Text" [D §1]. README uses "Semantic Repair of Damage-Occluded Historical Newspaper Text".
- **Title, team decision 2026-09-27.** Changed to "How Much Damage Can GenAI Repair? Fact Recovery in Damaged Historical Newspaper Text" (main.tex): the old title implied physical/image-level repair, but the study works on text only; image-level repair is now noted in the Introduction as future work.
- **RQ, team decision 2026-09-27.** "GenAI models" in the RQ replaced by naming the two tested families directly: a fine-tuned masked language model and a few-shot instruction-following LLM (sections/01_introduction.tex).
- **Key outcomes.** Wait for the test split.

## 2. Related work

- Thomas, Gaizauskas & Lu (2024): fine-tuned Llama 2 beat fine-tuned BART for post-OCR correction on BLN600 itself [D §7.4].
- Debaene et al. (2025): seq2seq beat generative models on early modern Dutch [D §7.4].
- Lyu et al. (2021): neural post-hoc OCR correction of historical corpora.
- Evershed & Fitch (2014): "context beats confusion"; our reranking is a basic form of context plus confusion [D §7.1].
- Shen et al. (2020): multi-token blanks are hard for parallel prediction [D §10].
- Zhang et al. (2020) BERTScore; Sainz et al. (2023) contamination per benchmark.
- **Our delta** [D §7.4]: controlled, fact-centred severity scale; fact-level metric; identical input for both methods; few-shot newer open-weight LLM instead of fine-tuned Llama 2.
- **Structure and draft, team decision 2026-09-27.** Related Work drafted in full prose (sections/02_related_work.tex), one paragraph per sub-question: RW-1 post-OCR correction (SQ1, gap), RW-2 restoring text at known gaps (SQ2 + task framing), RW-3 scoring meaning vs facts (SQ3), then a closing paragraph pointing to Method/Experiments and Table~1 (position table). Devil's-advocate points from the earlier audit are answered in-line: the "is this Pythia for newspapers" question is answered by the three-differences paragraph in RW-2 plus the table; the Intro/RW-1 wording mismatch is fixed (Intro P3 now says "post-OCR correction studies"); Hamdi et al. 2020 and 2023 are both kept, cited for different claims (2020 = graded synthetic degradation, RW-1; 2023 = in-depth NER/NEL analysis, Introduction). Belinkov & Bisk (2018) confirmed out of Related Work, earmarked for Limitations (synthetic-vs-natural-noise transfer threat) once that section is drafted.
- **"Is BERT GenAI?", team decision 2026-09-27.** Resolved in RW-2: BERT can be sampled to generate text \citep{wang2019bert}; filling a blank from context is generation conditioned on the rest of the sentence \citep{donahue2020enabling}. Wang & Cho (2019) cited only for the narrow "BERT can generate" claim, not for the paper's Markov-random-field derivation (Cho has since distanced from that part). Fallback for a skeptical reader: BERT is the fine-tuned, in-domain baseline the few-shot LLM must beat either way.
- Cross-reference labels added so Related Work's forward pointers resolve: `sec:method`, `sec:damage`, `sec:slot`, `sec:models` in sections/03_method.tex; `sec:metrics` in sections/04_experiments.tex. Method/Experiments prose itself is still `\todo` (not yet drafted).

## 3. Method

### 3.1 Data [D §4, cfg]
- BLN600: 600 excerpts, London crime reporting 1830s-1890s; 572 from Lloyd's Weekly Newspaper (340), Illustrated Police News (212), Lloyd's Weekly London Newspaper (20). Image, OCR and gold per excerpt. CC BY-NC-ND.
- Sentences split with `pysbd`; pool = 20-60 words with at least 2 fact tokens: 3,123 sentences from 592 excerpts.
- Fact token: number, any word with a digit or pound sign, or a capitalised non-initial word. Excluded: I, Mr, Mrs, Dr, St, The, Miss, Sir, Rev, Messrs, and a word right after a colon. Quirks: "Old" (Old Bailey) and "I'll" count as facts [C].
- Sample (seed 42): 170 sentences from 157 excerpts; test 150 (65/59/26 by band 20-29/30-44/45-60), dev 20 (9/8/3). At most 3 per excerpt; no excerpt in both splits.
- Calibration: 29 excerpts outside the sample (one dropped, coverage 0.35 < 0.60). 13,839 gold words: 11,430 equal, 1,148 paired, 1,261 unpaired; word error rate 0.0913; op mix: substitution 0.63, insertion 0.24, deletion 0.11 [cfg calibration.json]. Real erroneous-word error (words of 4+ chars): mean 0.30, median 0.25, q1 0.17, q3 0.40 [C]. Per-excerpt CER 0.016 to 0.428 [C].

### 3.2 Damage simulation [D §5, C, cfg config_snapshot_v2.yaml]
- One anchor per sentence among its fact tokens (seeded); always damaged, never dropped.
- One contiguous span around the anchor: start = clamp(anchor - k//2, 0, n - k). Spans are nested across levels.
- Levels: 0, 1w (k = 1), 10/25/50/75% with k = (pct x n + 50) // 100. 170 x 6 = 1,020 rows; 850 rows with slots (750 test, 100 dev) [C].
- Per damaged word: drop with p = 0.10 (anchor never); else alter 20-30% of letters and digits (uniform draw per word). Each altered character: deleted (~15%, max 2 per word) or substituted. Substitution sources: calibrated confusion table 85.7%, table entry capitalised 3.6%, look-alike map (O/0, l/1/I, rn/m, cl/d) 4.9%, random same-kind char 5.8% [cfg corruption_stats.json; D §5 gives 85.5/3.1/5.1/6.3 from an older run, use the json]. Punctuation never touched.
- Why 20-30%: 40-60% (v1) sat above q3 of real per-word error; 20-30% sits around the real median 0.25 [D §5, C].
- Result: 4,263 damaged words, 9.95% dropped; 87 digit words; 7 pure-punctuation tokens [cfg].
- Sentence-level CER mean (median): 1w 0.98% (0.88), 10% 3.19% (2.69), 25% 7.93% (7.29), 50% 15.49% (14.93), 75% 23.07% (22.64). Real BLN600 OCR about 7% per excerpt (Booth et al. 2024) [cfg, C].
- Damaged words per level, test median: 20-29 band 1/2/6/12/18; 30-44 1/4/9/18/26; 45-60 1/5/13/26/38 [D §5].
- Facts in span (test): mean 1.0/1.6/2.2/3.2/4.2; share of damaged words that are facts 100/48/26/19/17% [D §5].
- Fact slots: 692 (624 test), of which 68 dropped (62 test): about 1 in 10 facts has no visible letters [C].
- File: `corrupted_v2.jsonl`, sha256 `5798336d99ff...6b9eb9`.

### 3.3 Slot view [D §6, C 0a]
- Built from `gold_tokens` + `ops_per_word`, never from `corrupted_text`. Context unchanged; one slot per damaged gold word showing its garbled form; dropped word = `⟨?⟩`. Slot records hold no gold text (enforced by a test).
- Splice: prediction core framed by the **gold** word's edge punctuation (2 words where a deleted letter moved an inner apostrophe to the edge). Echoed punctuation is replaced by the frame. Pure-punctuation gold words are compared as whole tokens.
- Example: `A POWERFULLY-BUILT man, ⟨same⟩ ⟨Bbonias⟩ ⟨Huicl,⟩ ⟨e⟩ commission agent, ...` (gold "named Thomas Hurd, a").
- Justification: OCR engines flag low-confidence regions; showing garbled letters keeps it an OCR-repair task (`1awsenc` -> `Lawrence`).

### 3.4 BERT + reranking [D §7.1, C, cfg]
- `bert-base-cased` (cased needed: facts are partly defined by capitals). About 110M parameters.
- Training pool: 443 excerpts outside the sample (440 contribute), >= 1 fact token, no per-excerpt cap: 2,939 sentences, 96,792 words; train 2,635 sentences / 396 excerpts; val 304 / 44 (10% by excerpt). Zero doc_id and zero sentence overlap with test/dev (tested). Pool sha256 `d5313c92...`.
- Training input (option b, inference-matched): span words before the target are gold, the target is masked with its true piece count (loss only there), later span words are one `[MASK]` each. One example per span word: 32,131 examples in epoch 0 (28,981 train, 3,150 val). Example share by level 1w 1.9%, 10 5.5%, 25 15.6%, 50 31.0%, 75 46.1%. New span draw per epoch for train, fixed for val. Longest input 126 tokens (max length 192, no truncation). Clean gold text only, never garbled text.
- Fine-tuning: lr 5e-5, batch 16, max 10 epochs, early stopping patience 2, warmup 10% of one epoch (~180 steps) then linear decay, weight decay 0.01, grad clip 1.0, fp16 on GPU, seed 42. Colab Tesla T4, 4 epochs, 15.5 min, best epoch 2. Validation loss per target piece: pretrained 5.608; epochs 1-4: 4.297, **4.142**, 4.189, 4.330. Model sha256 `d483fd16...`.
- Inference: fill slots left to right, pending slots = one `[MASK]` each. For the current slot try 1, 2 and 3 masks; top b = 5 allowed pieces per position (word start first, then `##`; letters and digits only for visible slots); top N = 10 candidates per mask count. Slots with a visible inner hyphen or apostrophe are split into sub-slots, each filled with 1-3 masks.
- Score = mean log-probability per piece + lambda x (1 - Levenshtein / longer length), case-sensitive, against the garbled core (`rapidfuzz`). Dropped slot: BERT score only; punctuation-only candidates only there.
- Subword ceiling after sub-slots: 0.9% of all slots and 3.5% of fact slots need more than 3 pieces; 89% of slots need 1 piece [C].
- Lambda grid on dev (100 rows, beam 5, N 10), lambda 1/2/4/8/16: exact words 0.164/0.212/0.239/0.267/0.263; FRR 0.042/0.092/0.116/0.126/0.128; span CER after 0.756/0.671/0.604/0.572/0.570; repair gain -0.453/-0.368/-0.301/-0.269/-0.267. **Lambda 8 chosen** (plateau; best exact words; more weight on context) [C].
- Runtime (laptop CPU, test): 0.7 s/row at 1w up to 16.7 s at 75; 1.52 h for 750 rows [C].

### 3.5 LLM few-shot [D §7.2, C, cfg, prompts]
- Status: **model not final** (see conflict 5). Dev runs used `meta-llama-3.1-8b-instruct` on GWDG SAIA, allowed alternative `qwen3-30b-a3b-instruct-2507`. Open-weight only (licence and data handling: GWDG does not store prompts for self-hosted open-weight models) [D §7.2a].
- temperature 0, max_tokens 1024, timeout 120 s; retries only for 429/5xx/timeouts, never because of content [cfg].
- Prompt v2 (`prompts/repair_fewshot_v2.txt`): system message "You are a careful editor of 19th-century British newspaper text. You answer only in the format the user asks for, and you never write code." Added because without it SAIA's template pushed Llama 3.1 8B into Python/tool-call output (15/15 format failures with v1) [C]. User message: Victorian English, do not modernise, ⟨?⟩ = missing word, punctuation added back automatically, always a best guess, return a JSON list of exactly k strings.
- Few-shot picks (dev, excluded from scoring): `3206323884-017` at 1w (a name), `3200807881-004` at 25 (a dropped word, places), `3206237078-010` at 75 (dropped comma, sum `20l`) [cfg].
- Parsing: strip whitespace and code fences, take the first JSON array. Wrong length, no array, non-string item or `finish_reason == length` = format failure, scored as no repair. No retry, no manual fix [C].
- Secondary variant few-shot + article (`repair_fewshot_context_v1.txt`): whole excerpt (197-892 words, median 591) with the target sentence as slot view between `<<<` `>>>`. Other sentences are **gold**, so this is an upper bound for real use [C]. Compared only with the LLM's own few-shot run.
- Zero-shot tried on dev and dropped; consequence: the study cannot say how much the examples themselves help [C].
- Contamination probe: first half of each clean sentence, model continues; near-verbatim = rapidfuzz similarity >= 0.9 with the gold second half [C].

## 4. Empirical investigation

### 4.1 Design [D §9]
Severity (0, 1w, 10, 25, 50, 75) x length band (3) x method (BERT, LLM few-shot, no repair; secondary LLM + article). Test 150 sentences = 750 rows per method; dev 20 (17 scored for all methods, 3 are few-shot examples). Levels are nested in the same sentence: use paired comparisons.

### 4.2 Metrics as built (notebook 04) [C]
- BERTScore F1 of the span: `roberta-large`, default layer 17, no idf, `rescale_with_baseline=True`, `bert-score==0.3.13`. Rescaled, so values can be negative. Full-sentence BERTScore secondary.
- Fact Recovery Rate: share of gold fact tokens in the span restored exactly in their slot, punctuation ignored. Suggested split: visible vs dropped fact slots (68 of 692 dropped).
- Exact words; CER = Levenshtein / gold span length (chars, case-sensitive); repair gain = CER before - CER after; format failure rate; multiword count; contamination rate.
- Format failure scored as no repair (span = damaged text).
- Manual check: 18 repairs (6 sentences at levels 10, 25, 50, seeded).
- Note: repair gain per row is driven by short spans (one wrong word in a 1-word span gives CER > 1). A pooled CER is not computed yet.

### 4.3 Dev results (tuning context only, **never report as results**) [C]
- BERT (lambda 8) / few-shot / few-shot + article, 85 rows each: BERTScore span 0.241/0.282/0.261; FRR 0.096/0.217/0.237; exact 0.245/0.285/0.250; repair gain -0.284/-0.065/-0.118; format failures 0/18 (21%)/23 (27%).
- By level, BERTScore span BERT / few-shot: 1w 0.61/0.73, 10 0.36/0.46, 25 0.15/0.28, 50 0.12/0.05, 75 -0.03/-0.10. BERT behind on FRR at every level; crossing at 50 and 75 goes with the LLM's format failures (all at 50/75).
- LLM format failures: all wrong length, mostly one item short (12/18), 15/18 in rows with a dropped slot.
- Probe on 20 dev sentences: 0 near-verbatim (median 0.24, max 0.60).
- Typical LLM errors: short damaged words replaced with longer wrong ones (`Nev` -> `Newgate` for `New`), copies of damaged forms. Typical BERT errors: pieces glued into non-words (`andooll`), gold word never among candidates.

### 4.4 Test results
Pending: BERT stored (not scored), LLM not run. Filled from notebook 04 with `SPLIT = "test"`.

## 5. Interpretation hooks
- Answer RQ with the level where FRR or BERTScore falls near the no-repair floor.
- SQ2 link to asymmetries: letter access, parallel pieces, scale (110M vs 8B+), contamination.
- SQ3: compare BERTScore and FRR gaps; show 1 or 2 fluent-but-wrong repairs from the manual check.
- Practical message: flag repaired facts; do not trust names/numbers beyond level X.

## 6. Limitations [D §11, C]
- Damage location given; no detection step.
- Synthetic OCR-like character noise, not a model of physical damage; one contiguous span; intensity drawn per word, not spatially correlated.
- Calibration shows which characters get confused, not how heavy real damage is; it is a lower bound (mild OCR, unpaired words hide fragments).
- Fact-centred by design: facts are 100% of damaged words at 1w, 17% at 75%.
- Content of the span (stock phrase vs rare name) not controlled.
- 150 test sentences, 26 in the 45-60 band: trends, not powered effects.
- FRR is a surface rule with exact match; quirks like "Old", "I'll".
- English, London crime reporting, three related publications.
- Asymmetries: LLM reads letters directly; BERT pieces in parallel; scale and pretraining differ; LLM may have seen BLN600 (public since 2024). Check the final model's training cutoff.
- Article variant uses gold context: upper bound.
- `runs/preds/` holds derived BLN600 text; acceptable only while the repo stays private.

## 7. Appendix outline (for the winter-term defence)
Full damage parameters and confusion table; superseded v1 (40-60%) CER numbers (1w 1.86%, 10 5.05%, 25 12.10%, 50 23.70%, 75 35.13%); training-example construction and loss curve; lambda grid; prompts in full with the v1 failure and system-message fix; zero-shot dev run; subword ceiling; manual-check table; contamination probe details; optional ablations not run (blank slots, confusion-weighted reranking as upper bound).

## 8. Future work [D §13]
Damage detection in raw OCR; real OCR/gold pairs; image-assisted repair with a vision-capable model; whole-excerpt context for both methods (needs BERT redesign: 512-token limit, excerpts mean 490 words); German newspapers.

## 9. Reproducibility record
Seed 42; hashes: corrupted_v2 `5798336d...`, pool `d5313c92...`, model `d483fd16...`, sentences.jsonl `cdde36ee...`, calibration.json `44f0e48e...`. Local pins: torch 2.14.0 (CPU), transformers 5.17.0, bert-score 0.3.13, pysbd 0.3.4, rapidfuzz 3.14.6; Colab training torch 2.11.0+cu128, transformers 5.16.1. Repo: `tanruoshan/GenAI` (private), branch `dev/BERT_LLM`, commit `63ebe5d`.
