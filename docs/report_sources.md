# Report source book (consolidated)

One place for every fact the report needs, ordered by report section. **Updated 2026-09-29** for Simon's merged
work (PR #1 dictionary lookup and BERT + dictionary, PR #2 notebook 4 statistics, PR #3 BART) and the Qwen freeze.
Sources: `docs/dossier.md` (now the single dossier; `dossier_v2.md` merged into it and removed), `CLAUDE.md`
(implementation and deviations log, up to the BART entry of 2026-09-29), `docs/handoff/` (Simon's session state),
`message_to_shan.md` (Simon's review, 28 Sep), and the configs and run records: `configs/*.yaml`,
`runs/bert_ft_v1_config.yaml`, `runs/bart_ft_v1_config.yaml`, `runs/bart_ft_v1_log.csv`, `runs/config_snapshot_v2.yaml`,
`runs/*.sha256`, `reports/corruption_stats.json`, `reports/calibration.json`, `prompts/*.txt`.
This file and `report/overleaf-bln600/notes/report_sources.md` are the same file; edit both or copy.

Source tags: [D] dossier.md, [C] CLAUDE.md, [H] docs/handoff, [S] Simon's review, [cfg] config or run file.
([D2] in older rows = the former dossier_v2.md.) Rule when they disagree: the newest dated entry wins, and config
files beat prose.

---

## 0. Conflicts and stale points (fix before writing)

| # | Topic | What the files say | What is true now | Action |
|---|---|---|---|---|
| 1 | CER by level | [D2] §3: "mean, of the damaged span". [D] §5 and [C]: "sentence-level CER against gold". | Sentence-level. The 75% level has 23.1% CER over the whole sentence, not the span. | Fixed in `03_method.tex` Table 1. Fix [D2] §3. |
| 2 | BERT settings | [D2]: tuning in progress (lambda 1, 2 done, 4 partway). | Grid lambda 1, 2, 4, 8, 16 done on dev; **lambda 8 chosen**, beam 5, top N 10 [D §0, C, cfg]. | [D2] is stale. |
| 3 | Test gate | [D2] and CLAUDE.md "Day 3": write `runs/repair_config_v1.yaml`. | No such file. Per-track `frozen: true` flag: `bert_repair`, `bert_repair_lex`, `lexicon`, `llm` all true; **`bart_repair` still false** [cfg, 2026-09-29]. | Freeze `bart_repair.yaml` before BART's test run. |
| 4 | BERT test run | [D2]: not started. | **Done** 2026-09-27 00:37, `runs/preds/bert_rerank_ftv1-l8-b5-n10-test.jsonl`, 750 rows, not yet scored [C]. | Do not look at it before the LLM test run exists. |
| 5 | LLM model and host | [D2]: Llama 3.1 8B on SAIA, switch decision open. | **Resolved 2026-09-28:** `qwen3-30b-a3b-instruct-2507` on SAIA, 5 few-shot examples (one per level), numbered-slot prompts v4/v3, `configs/llm.yaml` `frozen: true` [C, cfg]. | Report still says Llama and three examples: Intro P5, 3.4(b), Table `tab:settings`, 5 P2. |
| 6 | corrupted_v1 | CLAUDE.md "Status": v1 kept locally. Project docs still list `corrupted_v1.jsonl`. | v1 deleted locally; only in git history [C, Cleanup]. | Remove the stale project copy or label it. |
| 7 | Dev BERT scores | [D §0]: BERTScore 0.24, FRR 0.10. [C]: 0.241, 0.096. | Same numbers, rounding only. | None. |
| 8 | Fine-tuning epochs | [D §7.1]: plain span masking, no epoch count. CLAUDE.md Section B: "3 epochs, 10% warmup". | Early stopping (max 10, patience 2), warmup 10% of **one** epoch, inference-matched masking (option b) [C, cfg]. | [D §7.1] needs the option (b) text. Use [C] in the report. |
| 9 | Subword limit | [D §10]: "words needing more than 3 pieces cannot be produced exactly". | Sub-slots at visible hyphens/apostrophes; only 0.9% of slots (3.5% of fact slots) still exceed 3 pieces [C, CHECKPOINT 1]. | Use the numbers. |
| 10 | Calibration as a severity measure | Dossier: silent. | Calibration under-measures real severity (mild OCR, unpaired words hide fragments). Damage models OCR output noise, not physical tears [C, 2026-09-27]. | Add to Limitations. |
| 11 | 20-30% refreeze reason | CLAUDE.md: "reason not recorded" (C entry). | [D §5] gives it: 40-60% sat above q3 of real per-word error; 20-30% sits around the median 0.25. | Use the dossier reason. |
| 12 | Methods compared | Report (Intro, 3.4, `tab:design`, `tab:results`): BERT, LLM few-shot, no repair (+ article). | Six methods: BERT + dictionary (main MLM), few-shot, article, dictionary lookup, BERT without dictionary (ablation), BART (descriptive) [C 2026-09-28/29]. | Add three Method paragraphs; decide which fit the results table (page limit). |
| 13 | Hypotheses | Report: RQ + SQ1-3, no hypothesis. | H1, H2, H2b, H3 drafted [S §2, H]; paired tests exist only for H2 and H2b [C]. | Add to Introduction (graded criterion). |
| 14 | Qwen dev exact-word rate | [C 2026-09-28]: 0.487. [H]: 0.559. | Different definitions: 0.487 pooled over slots (quick script), 0.559 mean over sentences (notebook 4). | Use notebook 4 definitions only. |
| 15 | Test runs | [H]: LLM test running. | Few-shot 750/750 and article 750/750 done; **probe 116/150**; **BART test not run** [runs/preds, 2026-09-29]. | Notebook 4 on test waits for both. |
| 16 | `CLAUDE.md` | Committed `writing` (HEAD `0e1aff0`, merge `c623c2a`) holds a half-resolved merge (`=======` / `>>>>>>>` without `<<<<<<<`), and the "Dossier docs updated" entry was lost. | Fixed in the working copy 2026-09-29, both entries kept. | Commit the fix. |
| 17 | "A40 LLM runs" | Team chat. | The A40 (RunPod) ran BART only (training + dev). All LLM calls are SAIA [C]. | Say "SAIA" for the LLM, "RunPod A40" for BART. |
| 18 | BERT uses letters | Report 5 P2: "BERT never sees [letters] as input, only through the reranking term". | Still true for the model input, but with dictionary candidates the letters also choose which words are candidates [C]. | Reword P2 for BERT + dictionary. |

---

## 1. Introduction

- **Problem.** Physical damage (tears, stains, tape) makes OCR output garbled or missing words. The words most at risk are often facts: names, places, dates, sums [D §2].
- **Why GenAI.** Fluent models can fill text, which risks plausible but invented facts entering the historical record [D §3]. This is a controlled robustness study, not a new system [D §3].
- **RQ.** When the damaged region is known and its garbled letters are visible, how well can GenAI models restore the original words so that meaning and key facts are preserved, and how does this change as the damaged share grows? [D §2]
- **SQ1** severity curve and breakpoint, share vs absolute gap length; **SQ2** fine-tuned BERT vs few-shot LLM on identical input; **SQ3** does semantic similarity overstate repair (wrong names, numbers, dates)? [D §2]
- **Working title (dossier).** "How Much Can Be Torn Away? Measuring the Limits of GenAI-Based Repair of Damaged Historical Newspaper Text" [D §1]. README uses "Semantic Repair of Damage-Occluded Historical Newspaper Text".
- **Title, team decision 2026-09-27.** Changed to "How Much Damage Can GenAI Repair? Fact Recovery in Damaged Historical Newspaper Text" (main.tex): the old title implied physical/image-level repair, but the study works on text only; image-level repair is now noted in the Introduction as future work.
- **RQ, team decision 2026-09-27.** "GenAI models" in the RQ replaced by naming the two tested families directly: a fine-tuned masked language model and a few-shot instruction-following LLM (sections/01_introduction.tex).
- **Hypotheses (not yet in the report) [S §2, H].** H1 (severity): repair quality falls as the damaged share grows, and fact recovery falls faster than span BERTScore. H2 (method): with identical input, the few-shot LLM restores more fact words than fine-tuned BERT, because it reads all garbled forms together. H2b: GenAI repair beats a dictionary lookup on fact words, not only on common words. H3 (metric): span BERTScore overstates repair quality: at every level some wrong-fact repairs score as high as right-fact repairs. Written after dev, before any test score; the Interpretation says per hypothesis: supported, partly, not.
- **RQ scope.** The RQ names two families (MLM, few-shot LLM). BART is a third (fine-tuned encoder-decoder): widen the RQ or present BART as an extra reference line [open].
- **Few-shot count.** Intro P5 says "three solved examples": now **five** (one per level) [cfg].
- **Key outcomes.** Wait for the test split.

## 2. Related work

- Thomas, Gaizauskas & Lu (2024): fine-tuned Llama 2 beat fine-tuned BART for post-OCR correction on BLN600 itself [D §7.4].
- Debaene et al. (2025): seq2seq beat generative models on early modern Dutch [D §7.4].
- Lyu et al. (2021): neural post-hoc OCR correction of historical corpora.
- Evershed & Fitch (2014): "context beats confusion"; our reranking is a basic form of context plus confusion [D §7.1].
- Shen et al. (2020): multi-token blanks are hard for parallel prediction [D §10].
- Dictionary lookup = the pure "confusion" side of Evershed & Fitch (2014); BERT + dictionary = context plus confusion with a larger candidate set [C 2026-09-28/29].
- BART (Lewis et al. 2020, **not yet in references.bib**): the course's denoising model; Thomas et al. (2024) compared a fine-tuned BART on BLN600, so our BART line links directly to them [S §6, C].
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
- **Model frozen 2026-09-28:** `qwen3-30b-a3b-instruct-2507` (MoE, 30B total, 3.3B active per token) on GWDG SAIA, `configs/llm.yaml` `frozen: true`. Switched from `meta-llama-3.1-8b-instruct` because its dev format failures (21% few-shot, 27% article) were above the pre-registered 15% line. Llama dev answers stay in `runs/preds/` as a record. Open-weight only (GWDG does not store prompts for self-hosted open-weight models) [D §7.2].
- temperature 0, max_tokens 1024, timeout 120 s; retries only for 429/5xx/timeouts, never because of content [cfg].
- System message: "You are a careful editor of 19th-century British newspaper text. You answer only in the format the user asks for, and you never write code." Needed because without it SAIA's template pushed Llama into Python/tool-call output (15/15 format failures with v1) [C]. User message: Victorian English, do not modernise, ⟨?⟩ = missing word, punctuation added back automatically, always a best guess, JSON list of exactly k strings.
- **Final prompts:** `repair_fewshot_v4.txt` (slots numbered `⟨1: Hnrd⟩`, item i answers slot i) and `repair_fewshot_context_v3.txt` (= v4 + the article block). Superseded: few-shot v2 (Llama), v3 (answer-shape example, 11/75 failures with Qwen); article v1, v2 [C 2026-09-28].
- **Few-shot picks: 5, one per level** (dev, left out of scoring for every method): `3206323884-017` 1w (a name), `3206270194-012` 10, `3200807881-004` 25 (dropped word, places), `3206225730-001` 50 (dropped word at that level), `3206237078-010` 75 (dropped comma, sum `20l`) [cfg]. Scored dev set: 15 sentences, 75 rows.
- Parsing: strip whitespace and code fences, take the first JSON array. Wrong length, no array, non-string item or `finish_reason == length` = format failure, scored as no repair. No retry, no manual fix. Also scored over valid answers only [C].
- Secondary variant few-shot + article: whole excerpt (dev 205-899 words, median 614; test 170-1,230, median 506) with the target sentence between `<<<` `>>>`. Other sentences are gold: upper bound. Compared only with the LLM's own few-shot run [C, notebook 3 output].
- Zero-shot tried on dev and dropped; consequence: the study cannot say how much the examples themselves help [C].
- Contamination probe: level 0 rows; first half of each clean sentence; prompt "Here is the beginning of a sentence from a 19th-century British newspaper. Continue it. Write only the rest of the sentence, nothing else." (does **not** name BLN600); normalised Levenshtein similarity (`rapidfuzz`) of the continuation, cut to the length of the true second half; >= 0.9 = near-verbatim [S §4, C].
- Test run status 2026-09-29: few-shot 750/750, article 750/750 (150 sentences x 5 levels, no duplicates); probe 116/150 [runs/preds].
- Still to check for the report: Qwen3's pretraining cutoff against BLN600's 2024 release (Yang et al. 2025, Qwen3 Technical Report, arXiv:2505.09388; not yet in references.bib).

### 3.6 Dictionary lookup, non-GenAI baseline [C 2026-09-28, cfg `lexicon.yaml`]
- Word list from the gold text of the 443 excerpts outside the sample (no dev or test excerpt): 16,469 distinct words, 213,237 tokens; tokens reduced to their core by the slot view's punctuation rule, case kept.
- Each damaged word -> closest entry by plain case-sensitive Levenshtein (the measure BERT's reranking uses); ties by frequency, then alphabet. Dropped slot = empty answer (counts as wrong; guessing the most frequent word was declined as flattering the baseline).
- Coverage (15 scored dev sentences): 95% of damaged words, 77% of damaged fact words (48 of 62) are in the list.
- Frozen, test run stored: `lexicon_v1-test.jsonl`, 750 rows. Runs in seconds, no GPU, no API. Notebook `02b_dictionary.ipynb`.
- Why in the report: answers "does GenAI beat a spell checker?"; the pure "confusion" side of Evershed & Fitch (2014).

### 3.7 BERT + dictionary candidates, main MLM method [C 2026-09-29, cfg `bert_repair_lex.yaml`]
- The `lex_n` word-list entries closest in spelling to each damaged form join BERT's own candidates (top 5 pieces per mask) and get the same score (mean log-probability per piece + lambda x letter similarity). Their BERT score comes from the pass whose mask count equals their piece count; more than 3 pieces or `[UNK]` = skipped. Same model `bert_ft_v1`; BERT still never sees damaged text in training.
- Why: context-only candidates miss words BERT does not propose (dev 1w: `Wrighb` -> Smith, White, andooll; `Suney` -> Saturday, Wednesday, Monday; `Hainos` -> Henry, van, John) [S §3].
- Off by default; the frozen v1 run reproduces exactly (40 of 40 dev rows, 0 of 320 slots different, CPU and MPS).
- Grid on all 100 dev rows (exact / FRR): `lex_n` 5 at lambda 8 0.416 / 0.275, 16 0.617 / 0.499, 32 0.619 / 0.537, **64 0.619 / 0.560**, 128 0.603 / 0.558; at lambda 1-4 dictionary words almost never win. Across `lex_n` 5/10/20/50 differences <= 0.03 (best exact 0.628 at `lex_n` 50, lambda 32). References: BERT without dictionary 0.267 / 0.126; lookup 0.528 / 0.426.
- **Chosen `lex_n` 5, lambda 64** (best exact once FRR plateaus; smallest `lex_n` since differences are noise). Defence note: chosen before the `lex_n` 10/50 rows at lambda 32/64 were read.
- Interpretation hook: at lambda 64 letters decide most slots; BERT's context breaks near ties and fills dropped words.
- Test: `bert_rerank_ftv1-l64-b5-n10-x5-test.jsonl`, 750 rows, 0.1 s/row (1w) to 1.3 s (75) on Apple MPS.
- BERT without dictionary (3.4) stays as an ablation line.

### 3.8 BART fine-tuned denoiser, descriptive only [C 2026-09-29, cfg `bart_ft.yaml`, `bart_repair.yaml`, runs/bart_ft_v1_*]
- `facebook/bart-base` (about 140M) reads the slot view and writes the sentence back with the gold word in each bracket; answers read from the brackets; wrong bracket count = format failure. Input brackets `{ word }` (BART's byte-level tokenizer splits ⟨ ⟩ into byte pieces); information identical.
- Training data: BERT's training pool (2,635 train / 304 val sentences), every sentence at all five levels per epoch (13,175 examples), new anchor and damage per epoch, **damaged by our own generator** (`corrupt.plan_word`, same confusion table). Validation: one fixed draw (1,520 examples).
- Settings copy BERT's: lr 5e-5, batch 16, max 10 epochs, patience 2, warmup 10% of one epoch, weight decay 0.01, clip 1.0, bf16. Greedy decoding with every setting explicit (bart-base ships `num_beams` 4, `no_repeat_ngram_size` 3, which would stop copying `} {` sequences).
- Run: RunPod A40, 22.5 min, $0.20 total, best epoch 8 of 10, validation loss 0.2663 (pretrained 2.507). Validation / training-probe exact per epoch: 1 0.396 / 0.446, 2 0.528 / 0.639, 4 0.602 / 0.791, 6 0.649 / 0.851, **8 0.672 / 0.879**, 10 0.670 / 0.896: BART memorises training sentences (overfitting signal); checkpoint rule (lowest val loss) fixed before the run. Val exact at epoch 8 by level: 1w 0.556, 10 0.710, 25 0.724, 50 0.695, 75 0.639.
- Dev check (100 rows): exact 1w 0.650, 10 0.674, 25 0.733, 50 0.744, 75 0.738; 2 format failures. Visible errors are fluent substitutions: `icyclc` -> `yacht` (gold `bicycle`), `£1,` -> `£6,` (gold `£2,`), `Southon` -> `Southwark` (gold `Southend`). Good SQ3 examples.
- **Caveats for Method and Limitations:** supervised on exactly the test noise process, so optimistic; added after the hypotheses, so descriptive only; **test run not done**, `bart_repair.yaml` `frozen: false`.
- Model sha256 `28de207d...5614` (`runs/bart_ft_v1.sha256`). Notebook `02c_bart.ipynb` (reads files only).

## 4. Empirical investigation

### 4.1 Design [D §9]
Severity (1w, 10, 25, 50, 75; level 0 for the probe only) x length band (3) x method. Main comparison: BERT + dictionary, LLM few-shot, dictionary lookup, no repair. Secondary: article vs few-shot; BERT without dictionary (ablation); BART (descriptive). Test 150 sentences = 750 rows per method; dev 15 scored sentences (5 are few-shot examples). Levels nested in the same sentence: paired comparisons.

### 4.2 Metrics as built (notebook 04, `src/blnrepair/evaluation.py`) [C 2026-09-29]
- BERTScore F1 of the span: `roberta-large`, layer 17, no idf, `rescale_with_baseline=True`, `bert-score==0.3.13`; can be negative. Sentence BERTScore also computed (answers Bea's SQ3 option (a) at no cost).
- Fact Recovery Rate: mean over sentences of the share of gold fact tokens in the span restored exactly, punctuation ignored. Split into visible and dropped fact slots (pooled over slots, since many sentences have no dropped fact). The count "68 of 692 fact slots dropped" (3.2) is questioned in Bea's 4.2 comment: recount from notebook 4 before using it.
- Anchor recovery: same word at every level, one per row, never dropped.
- Exact words; span CER pooled per level (all edits / all gold characters); repair gain = pooled CER before - after; format failures per level; LLM valid-only; multiword count; contamination rate.
- **Intervals:** 95% bootstrap over sentences, 10,000 resamples, percentile, same resampled sets for all methods and levels, seed from `configs/config.yaml`. Reason: slots of one sentence share context, so resampling slots makes intervals too narrow. Cite Koehn 2004; Dror et al. 2018 (both **not yet in references.bib**).
- **Paired tests, only H2 (BERT + dictionary vs few-shot) and H2b (few-shot vs lookup):** per level, exact McNemar on anchor recovery (Dietterich 1998), Wilcoxon signed-rank on span BERTScore (ties dropped, scipy default); Holm over 2 pairs x 5 levels = 10 tests per family. All other comparisons descriptive. Reason: test only what a hypothesis named before the test run.
- **SQ3 number (H3):** per method and level, P(wrong-anchor repair has span BERTScore >= right-anchor repair), ties half = 1 - AUC. 0 = BERTScore always ranks the right fact higher, 0.5 = blind to the fact. No threshold. Sanity: 0.00 at 1w on dev for every method.
- Manual check: 18 repairs (6 sentences at levels 10, 25, 50, seeded).
- Colours fixed: BERT + dictionary blue, few-shot orange, article aqua, lookup yellow, BERT without dictionary magenta, BART green.

### 4.3 Dev results (tuning context only, **never report as results**) [C 2026-09-29, notebook 4, 15 sentences]
- Span BERTScore 1w / 10 / 25 / 50 / 75: BART 0.83 / 0.77 / 0.73 / 0.69 / 0.67; BERT + dictionary 0.85 / 0.71 / 0.48 / 0.44 / 0.43; few-shot 0.86 / 0.65 / 0.45 / 0.44 / 0.17. At 1w / 25 / 75: lookup 0.73 / 0.43 / 0.40; BERT without dictionary 0.59 / 0.16 / -0.03.
- Anchor recovery: BERT + dictionary 0.53-0.60; few-shot 0.40-0.60; lookup 0.40 at every level; BART 0.53 at every level. FRR BART 0.53-0.58. No method restores a dropped fact slot (10 on dev).
- Pooled repair gain positive for all methods except BERT without dictionary (-0.16 to -0.35).
- Paired tests: none significant after Holm (smallest Holm p 0.73).
- SQ3 rate at 25%: few-shot 0.45 ... BERT + dictionary 0.70; BART by level 0.00 / 0.16 / 0.21 / 0.29 / 0.36.
- LLM valid-only at 75%: few-shot span BERTScore 0.45 on 10 valid rows vs 0.17 with failures as no repair (5 of 15 failed).
- Qwen format failures: few-shot 7/75 (25: 1, 50: 1, 75: 5); article 7/75 (50: 2, 75: 5). Probe dev: 0 of 20 near-verbatim (median 0.25, max 0.31).
- Superseded (Llama 3.1 8B, 17 dev sentences, 85 rows): span BERTScore BERT / few-shot / article 0.241 / 0.282 / 0.261; FRR 0.096 / 0.217 / 0.237; format failures 0 / 18 (21%) / 23 (27%); failures mostly one item short (12/18), 15/18 in rows with a dropped slot. Typical errors then: LLM `Nev` -> `Newgate` (gold `New`); BERT glued pieces (`andooll`).

### 4.4 Test results
Pending. Stored: BERT without dictionary, lookup, BERT + dictionary, few-shot, article (750 each). Missing: probe (116/150), BART (not run). Nobody has seen test scores. Filled from notebook 04 with `SPLIT = "test"`, run once.

## 5. Interpretation hooks
- RQ/H1: the level where anchor recovery or FRR falls near the no-repair floor; whether FRR falls faster than span BERTScore.
- H2: BERT + dictionary vs few-shot per level (McNemar, Wilcoxon, Holm); check any crossing against LLM valid-only scores (dev: the 75% drop is format failures).
- H2b: few-shot vs lookup on fact words, not only all words; the lookup cannot fill dropped slots or names outside its list.
- H3: the 1 - AUC rate per level; one or two fluent-but-wrong repairs (BART dev: `Southend` -> `Southwark`).
- BART, descriptive: if it leads, say why it may (supervised on the same noise) before saying it is better.
- Asymmetries: letter access (LLM, BART direct; BERT via reranking and dictionary candidates); parallel pieces; scale (110M / 140M / 30B with 3.3B active); contamination.
- Practical message: flag repaired facts; do not trust names and numbers beyond level X.

## 6. Limitations [D §11, C]
- Damage location given; no detection step.
- Synthetic OCR-like character noise, not physical damage; one contiguous span; intensity per word, not spatially correlated.
- Calibration shows which characters get confused, not how heavy real damage is; a lower bound (mild OCR, unpaired words hide fragments).
- Fact-centred by design: facts are 100% of damaged words at 1w, 17% at 75%. Span content not controlled.
- 150 test sentences, 26 in the 45-60 band, one noise draw per sentence: trends, not powered effects.
- FRR is a surface rule with exact match; quirks like "Old", "I'll".
- English, London crime reporting, mostly two weeklies (572 of 600 excerpts); one LLM.
- Asymmetries: see 5. BART is trained on our own damage generator: optimistic. Hypotheses written after dev.
- Article variant uses gold context: upper bound. Format failures count as no repair.
- `runs/preds/` holds derived BLN600 text; acceptable only while the repo stays private.

## 7. Appendix outline (for the winter-term defence)
Full damage parameters and confusion table; superseded v1 (40-60%) CER numbers (1w 1.86%, 10 5.05%, 25 12.10%, 50 23.70%, 75 35.13%); BERT training examples and loss curve; lambda grid and `lex_n` x lambda grid; BART training record and over/underfitting curves (`plots.fit_curves`); prompts in full with the v1 failure, the system-message fix, the Llama run and the switch rule; zero-shot dev run; subword ceiling; manual-check table; probe details; ablations not run (blank slots, confusion-weighted reranking).

## 8. Future work [D §13]
Damage detection in raw OCR; real OCR/gold pairs; image-assisted repair with a vision-capable model; whole-excerpt context for both methods (needs BERT redesign: 512-token limit, excerpts mean 490 words); German newspapers.

## 9. Reproducibility record
Seed 42; hashes: corrupted_v2 `5798336d...`, pool `d5313c92...`, BERT model `d483fd16...`, BART model `28de207d...`, sentences.jsonl `cdde36ee...`, calibration.json `44f0e48e...`. Local pins: torch 2.14.0, transformers 5.17.0, bert-score 0.3.13, pysbd 0.3.4, rapidfuzz 3.14.6, scipy 1.17.1; Colab BERT training torch 2.11.0+cu128, transformers 5.16.1; BART training on RunPod A40 (versions in `runs/bart_ft_v1_config.yaml`, code commit `298f782`). LLM: SAIA, `qwen3-30b-a3b-instruct-2507`, few-shot tag `exb46768`. Repo: `tanruoshan/GenAI` (private), branch `writing`, HEAD `0e1aff0` (2026-09-29).
