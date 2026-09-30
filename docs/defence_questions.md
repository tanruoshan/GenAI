# Defence preparation: likely questions and short answers

For the oral defence (Gruppe 4, 14 Oct 2026, 10:30, Prof. Ludwig). Written 2026-09-30 after a read-only review
(Codex `gpt-6-sol`) and a check of every number against the executed `notebooks/04_evaluation.ipynb` (test split).
Numbers are test numbers unless marked dev. Order: most dangerous first.

1. **Why should this predict anything about real damaged pages?**
   It does not claim to. The damage is synthetic OCR-like character noise in one known span; physical damage, real
   OCR on damaged pages and the detection step are untested (Limitations). The value is control: the same sentence
   and the same anchor at five levels, so every level comparison is paired.

2. **How realistic is the noise? You say it is calibrated.**
   The confusion table and the 20-30% per-word band come from real BLN600 OCR errors on 29 held-out excerpts
   (median per-word CER of misrecognised words 0.25, IQR 0.17-0.40). The drop rate (0.10) and the fallback
   substitutions (14.3% of substitutions) are our choice. Short words get more damage than 20-30% (at least one
   character: 45% on average for 1-3 characters). Sentence CER at 25% damage (7.9%) is close to real BLN600 OCR
   (about 7%). Real BLN600 OCR is mild, so heavy real damage may be worse than ours.

3. **Is BERT vs LLM a fair comparison?**
   It compares two practical set-ups under identical input and scoring, not two architectures under equal training:
   BERT is fine-tuned on 2,635 in-domain sentences, the LLM sees five solved examples. The paper says this (Method
   (c)). The LLM has about 270x the parameters of BERT (27x active per token).

4. **Is BERT + dictionary more than a spell checker?**
   Barely on facts: +0.01 to +0.07 Fact Recovery Rate over the lookup. Context helps on dropped words and near
   ties, but letters decide most slots at lambda 64. This is a finding, not a weakness: "context beats confusion"
   only holds when context proposes the right word, and for rare names it does not. No significance test was run
   for this pair (only the two pairs named by hypotheses were tested).

5. **Are the LLM's losses just formatting?**
   Partly. 106 of 750 answers fail the format (103 wrong word count, 3 no JSON array), 103 of them at 50% and 75%.
   Over valid answers only, the LLM's span BERTScore at 75% is 0.40 (as BERT + dictionary) but its Fact Recovery
   Rate stays lower (0.39 vs 0.48). Valid answers are a selected subset, so this is descriptive.

6. **Your SQ3 number (1 - AUC) compares repairs across different sentences. Is that a clean test?**
   No, it is a ranking diagnostic, not a controlled name-swap test: wrong-anchor and right-anchor repairs come from
   different sentences, so sentence difficulty is mixed in. It needs no threshold and has fixed reference points
   (0 = always ranks the right fact higher, 0.5 = cannot tell). Values 0.11 to 0.45 beyond one word. At one word it
   is 0 by construction (the span is the fact).

7. **BART has the highest BERTScore and also the highest exact-word rate (0.64-0.69 vs 0.57 for BERT + dict. from
   10% on). So is BERTScore not simply right that BART repairs better?**
   On common words, yes; BART restores more words overall. But it restores fewer facts (FRR 0.40-0.45 vs
   0.48-0.54). That is exactly the gap SQ3 is about: a meaning-level score rewards the many common words and
   hides the few wrong facts (*Alexander Fleming* for *Alexander Temple*). BART is descriptive only: trained on our
   own noise generator, added after the hypotheses.

8. **What are "facts"?**
   Fact tokens by a surface rule (numbers, words with digits or a pound sign, capitalised non-initial words minus a
   stop list), not a named-entity tagger. Quirks: "Old" in "Old Bailey" counts. Test: 624 fact slots at 75%, 62
   of them dropped. Scored by exact match, so spelling variants count as wrong.

9. **Why does the fact share fall with level, and does that distort FRR?**
   Spans grow around one anchor, so facts are 100% of damaged words at 1 word and 17% at 75%. That is why we also
   report anchor recovery: the same word at every level (BERT + dict. 0.58 to 0.53, lookup 0.51 at every level,
   LLM 0.49 to 0.25).

10. **One damage draw and one training seed: are the intervals meaningful?**
    The bootstrap (10,000 resamples over 150 sentences) covers sentence sampling only, conditional on this damage
    draw and these checkpoints. Seed and training variation are not measured (Limitations).

11. **Hypotheses after dev tuning: is that HARKing?**
    They were fixed after dev tuning but before any test score was computed; notebook 4 ran on test exactly once
    (commit history: PR #4). The wording was not changed after the test run; verdicts name the tested pairs.

12. **Did the LLM memorise BLN600?**
    The probe found 0 near-verbatim continuations in 116 of 150 test sentences (median similarity 0.26, max 0.51).
    That does not rule out other memorisation; the prompt does not name BLN600 and has no control condition.
    116, not 150, because the SAIA quota ran out.

13. **Why only one LLM, no zero-shot?**
    Llama-3.1-8B was dropped on dev because its format failures (21% / 27%) exceeded the 15% switch rule set in
    advance; zero-shot was tried on dev and dropped. So the paper cannot say how much the five examples help.

14. **Is BERT GenAI?**
    Masked infilling is generation conditioned on context (Donahue et al. 2020), and BERT can be sampled to
    generate text (Wang and Cho 2019, cited for that narrow claim only). Either way it is the in-domain baseline
    the LLM had to beat.

## Figures

`fig:results` (`scripts/paper_results.py`) and `fig:spans` (`scripts/fig_nested_spans.py`) use matplotlib with the
SciencePlots `science` style, Times fonts embedded as TrueType, Okabe-Ito colours (colour-blind safe) with one
marker per method, and viridis for the damage levels. Both regenerate from files in the repo.
