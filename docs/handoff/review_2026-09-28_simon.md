# GenAI project: review, findings and plan

Simon, 28 Sep 2026 (evening). For Shan and Bea. Based on the `writing` branch at commit `21f69eb` (Bea's writing included), the GRIPS course page, and the stored dev predictions. All numbers below are **dev only** (15 scored sentences): a reason to act, not results.

Hi Shan! Really impressive how clean the setup is: frozen data, the test gate, the paired levels, the prediction store. Bea's revision already covers several things I would have raised (valid-only LLM scores, pooled CER, anchor recovery, the stats and SQ3 to-dos). What is left is below, most important first.

## TL;DR

1. **One question first:** has anyone run notebook 4 on the test split, or looked at BERT's test scores? If not, please keep it that way until we've settled point 3.
2. **The report has no explicit hypothesis**, but the course grades "formulating a constructive research hypothesis". Cheap to fix, probably the most important point for the grade.
3. **An old-school baseline:** a simple dictionary lookup, with no AI and no context, beats BERT on dev and comes close to Qwen. I'd add it as a non-GenAI baseline, plus a "BERT + dictionary candidates" version.
4. **Your cleanup branch:** please push it (to its own branch) whenever it's ready, no need to wait for me. See section 5 for how we avoid getting in each other's way.
5. **My A40 (RunPod):** optional fine-tuning of **BART-base**, the denoising model from the lecture. Only if it makes it into the paper; otherwise we leave it out.

## 1. What the course asks for (from GRIPS)

- **Deadline:** Thursday 1 Oct 2026, 00:00 (so Wednesday night). Nothing is submitted yet for Gruppe 4.
- **Page limit is real:** announcement "Project Report" (17 Jul 2026, forum "Ankündigungen"): "Keep your text within 7-9 pages". Also from there: the EMNLP example paper, ACL style files optional, no extensive appendix, just say what would go into it at the defence. The README in `report/` matches this.
- **What is graded** (announcement "Online Week", 30 May): report, demonstration and defence. Coding itself does not count. The competences listed: a representative, high-quality dataset (and arguing for it), **a constructive research hypothesis**, a strategy to answer it, why it is relevant, a working pipeline, evaluation "according to state-of-the-art methods", interpreting the results "in terms of impact for an answer to your hypothesis", and practical impact.
- **Defence:** Gruppe 4 on **14.10.26, 10:30**, in Prof. Ludwig's office (PT 3.0.84C), per "Timetable for Project Presentation". The defence will be based on the paper, so whatever we want credit for has to be in the paper by Wednesday night.
- **Course material worth one or two sentences in the report:**
  - The lecture slides say "Encoders do not generate output sequences" (BERT under classification and NER). Our Related Work already answers this with Wang & Cho; expect the question at the defence.
  - BART is the lecture's example of a model that learns to "reconstruct original text from corrupted input". Denoising runs through the whole course (VAE, diffusion, BART), and the diffusion slides say iterative denoising "will not exactly reconstruct ... but hallucinate something similar". That is exactly our "fluent but wrong" point (SQ3) and a nice bridge to the course.
  - The "LLM Engineering" lecture covers RAG and the parts of a prompt (task, context, examples, role, format). Our article variant is RAG-style context injection with perfect retrieval, and our prompt has all five parts. One sentence each in the Method.

## 2. Hypotheses (missing so far)

The Introduction has an RQ and three sub-questions, but no hypothesis. A draft, one per sub-question, to adapt:

- **H1 (severity):** repair quality falls as the damaged share of the sentence grows, and fact recovery falls faster than span BERTScore.
- **H2 (method):** with identical input, the few-shot LLM restores more fact words than fine-tuned BERT, because it reads all garbled forms together.
- **H3 (metric):** span BERTScore overstates repair quality: at every level a share of repairs gets a high BERTScore while the fact is wrong.
- Optional, if we add the lookup baseline: **H2b:** GenAI repair beats a dictionary lookup on fact words, not only on common words.

The report's Interpretation section can then say for each: supported, partly, or not.

## 3. An old-school baseline, and a stronger BERT

**Why BERT underperforms.** BERT only proposes words from the context (the top 5 pieces per mask). The damaged letters are used only afterwards, to re-sort that list. If the right word is not in the list, BERT can never pick it, however close the letters are. From the stored dev predictions (level 1w):

| damaged | gold | BERT's top 3 candidates |
|---|---|---|
| Wrighb | Wright | Smith, White, andooll |
| Suney | Sunday | Saturday, Wednesday, Monday |
| Hainos | Haines | Henry, van, John |

**A dictionary lookup.** I built a word list from the 443 excerpts outside the sample (16k words, no dev or test excerpt in it) and took, for each damaged word, the entry with the closest spelling (edit distance, ties by frequency). No context, no model. Share of slots restored exactly, 15 scored dev sentences:

| level | 1w | 10 | 25 | 50 | 75 | all | facts only |
|---|---|---|---|---|---|---|---|
| BERT (lambda 8) | .13 | .42 | .28 | .22 | .17 | .22 | .10 |
| Qwen few-shot | .60 | .68 | .56 | .56 | .38 | .49 | .39 |
| dictionary lookup | .40 | .42 | .52 | .58 | .57 | .55 | .37 |

- The lookup ties or beats BERT at every level.
- It even beats Qwen at 50 and 75, but only because Qwen's format failures count as zero. On rows with a valid answer the two are about equal there (level 50: .63 vs .60, level 75: .61 vs .60). So Bea's "valid-only" scores matter a lot.
- On fact words, Qwen is only slightly ahead of the lookup (.39 vs .37).

**Why I like it as a baseline:**

- An examiner will very likely ask "does GenAI beat a spell checker?", and the course asks for evaluation "according to state-of-the-art methods". Better we answer it ourselves.
- It is old-school, cheap and still strong. My guess is that Ludwig likes that kind of thing: a simple, efficient method that holds its own next to a 30B model.
- It fits our framing: it is the pure "confusion" side of "context beats confusion" (Evershed & Fitch, already cited), while BERT and the LLM add context.
- It sharpens the RQ: where does GenAI actually add something, and where is a lookup enough? The dictionary covers 95% of all damaged dev words but only 77% of the fact words (48 of 62). Rare names are exactly where a lookup cannot help and GenAI could, plus the dropped words, which have no letters to look up.

**Proposal:**

- **a) Dictionary lookup as a non-GenAI baseline**, next to "no repair". No GPU, no API, runs in seconds.
- **b) "BERT + dictionary candidates":** the dictionary words close to the damaged form join BERT's candidate list and are scored with the same formula (BERT log-probability + lambda x letter similarity). BERT still never sees damaged text in training. The old BERT run stays under its own version name, so we can report both, which also shows how much the candidate step matters. Lambda chosen on dev only, then one test run (about 1.5 h on CPU, minutes on the A40).
- This only stays clean if nobody has looked at test results yet (TL;DR point 1).

## 4. The report after Bea's revision

**Still out of date** (the LLM track is frozen now):

- Intro (line 55), Method 3.4(b), the zero-shot sentence, and Table 2 still say "three solved examples" and "dev default: Llama-3.1-8B". It is now Qwen3-30B-A3B-Instruct-2507, **5** examples (one per level), numbered slots, frozen. Also "held out from all scoring": the 5 example sentences are dev sentences, so the scored dev set is 15 sentences.

**Bea's to-dos I can answer now:**

- *Level 0 only for the probe?* Yes. The probe uses the level 0 rows (`severity == "0"`), and notebook 4 scores only rows with slots (level 0 has none).
- *Anchor recovery and visible vs dropped fact slots from existing predictions?* Yes. Every slot already carries `anchor`, `fact` and `dropped` flags (`build_slots`), so both are a few lines in notebook 4, no new model runs.
- *Probe similarity measure and prompt:* normalised Levenshtein similarity (`rapidfuzz`, `Levenshtein.normalized_similarity`) between the continuation, cut to the length of the true second half, and that second half. The prompt does not name BLN600 ("Here is the beginning of a sentence from a 19th-century British newspaper. Continue it. Write only the rest of the sentence, nothing else.").
- *SQ3, option (a) sentence-level BERTScore:* already computed. Notebook 4 has a "BERTScore sentence" column, so (a) costs nothing extra.
- *Statistics:* bootstrap CIs, McNemar on anchor recovery and Wilcoxon on span BERTScore: I can add these to notebook 4.

## 5. How we work in the repo

- **Branches:** Shan's cleanup goes to its own branch, mine is `simon/baselines`. Both start from `writing` and come back via pull request into `writing`. `main` gets `writing` at the end.
- **Cleanup first, please:** push the cleanup branch as soon as the tests pass, and merge it into `writing`. My first steps are new files only (`src/blnrepair/lexicon.py` plus tests), so I can start in parallel. Before I touch `bert_repair.py` or notebook 4, I merge your cleanup in.
- **One request for the cleanup:** it must not change behaviour, meaning version names, scoring or which rows are read, or the stored predictions stop matching. A quick check: run the tests and notebook 4 on dev before and after, and the numbers must be identical.
- **What goes into my branch:**
  - `src/blnrepair/lexicon.py`: the dictionary lookup. The word list is built from the non-sample excerpts at run time, so no BLN600 text is committed. Plus tests.
  - BERT + dictionary candidates as a new option in `bert_repair.py`, with its own config (for example `configs/bert_repair_lex.yaml`). `configs/bert_repair.yaml` and `configs/llm.yaml` stay untouched and frozen.
  - New stored predictions in `runs/preds/` under new version names (dev first, then test).
  - The extra analysis in notebook 4: the new methods, CIs and tests, anchor recovery, visible vs dropped, pooled CER, valid-only LLM scores.
  - Short entries in the deviations log in `CLAUDE.md`, as you've been doing.
- **What I won't touch:** notebook 3, the prompts, `configs/llm.yaml`, your SAIA runs, and the report sections you two are writing.
- **Notebooks are JSON and hard to merge,** so one person per notebook at a time. I'll only edit notebook 4 after saying so here.
- **Pull requests:** small commits; when a piece is done I open a pull request into `writing` with a short description and the dev numbers. You look at the changes on GitHub and merge.
- **Never committed:** `data/`, `.env`, `models/`, and no change to the `frozen` flags.

## 6. The A40 and fine-tuning

- **Not needed for sections 3 and 4.** The lookup runs in seconds. BERT with the extra candidates should cost about as much as today; the A40 would just turn hours into minutes.
- **Optional: fine-tuned BART-base as a third GenAI method.**
  - Train BART-base (140M, about BERT's size) on "damaged slot view to clean words", using the BERT training pool (2,635 sentences), damaged with our own damage generator at the 5 levels (about 13k examples).
  - Why BART: it is the lecture's own denoising model, and Thomas et al. (2024), which we already cite, compare against a fine-tuned BART on BLN600. It adds "fine-tuned denoiser vs. prompted LLM" to the comparison.
  - Effort: on a dataset this small I expect training to take minutes to well under an hour on the A40 (estimate, to be checked with a short trial run), and inference on 750 test rows takes minutes.
  - Caveats to state: unlike BERT it sees damaged text in training, and it is trained with the same damage generator as the test data, so its numbers are optimistic.
  - It is only worth doing if it makes it into the paper, including its place in Method and Results within the 9 pages. Since the defence is based on the paper, we can't hand in a whole extra model later. If it doesn't fit, we leave it out, or mention it in one line as future work. Only if you both agree.
- **Cost:** about $0.45-0.50 per hour for an A40, paid from my RunPod balance, so no cost to you. The pod is deleted after each run.
- **Data:** BLN600 text would sit on a rented GPU we control (not a third-party model API). Please say if you see a licence problem.
- **Side note:** the course also offers NHR@FAU compute (page "Information for Bayern KI"), but setting up an account would take too long now.

## 7. What I need from you

1. The answer to TL;DR point 1 (has anyone seen test numbers?).
2. The fine-tuned BERT folder `models/bert_ft_v1` (via Drive?). I can start the lookup baseline without it.
3. OK on the branch setup in section 5, and the name of your cleanup branch.
4. Your and Bea's view on the hypotheses (section 2) and on the optional BART run (section 6).

Thanks, and great job so far!
