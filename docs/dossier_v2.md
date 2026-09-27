# BLN600 Damage Repair Study: Dossier (Revision 2, condensed)

This is the short, current-state version of the project plan: what the study does, where it stands, and what to run next. For the full design history, the original reasoning behind each decision, and superseded numbers, see `docs/dossier.md` (kept as the complete record). If the two ever disagree on a current number, this file is right; `docs/dossier.md` has the "why".

Course: GenAI Master's coursework project (6 ECTS), team of 3. Report: max 9 pages in ACL style (excluding references), structured like the EMNLP example paper (Introduction, Related Work, Method, Empirical Investigation, Interpretation).

## Status (2026-09-26) and what to do next

**Done:** data and damage frozen (`corrupted_v2.jsonl`, 20-30% per-word damage intensity); BERT fine-tuned and its reranking code built; LLM prompts settled on few-shot only (zero-shot tried and dropped); both methods have a full run on the 20-sentence dev split and are scored.

**In progress:** a small tuning search for BERT's reranking settings (lambda values 1, 2 done in full; 4 partway; beam size 3 not yet tried). Check `runs/preds/bert_rerank_ftv1-l*-b5-n10-dev.jsonl` for the latest.

**Three decisions before the test split (the real, final comparison) can run:**
1. **Model.** Llama 3.1 8B Instruct's dev format-failure rate (21% few-shot, 27% few-shot + article) is above the 15% line that was agreed would trigger a switch to Qwen 3 30B A3B Instruct. Decide: switch and rerun dev, or keep Llama and report the high failure rate as a finding.
2. **BERT settings.** Finish the tuning search above (or stop early and accept the current best) and pick one final lambda/beam/candidate-count setting.
3. **Freeze `runs/repair_config_v1.yaml`.** Once 1 and 2 are settled, write this file (model ids, BERT weights hash, lambda/beam/N, prompt version, few-shot example ids, seed). Both notebooks refuse to touch the test split until it exists, on purpose.

**Then, to get the final numbers:**
1. Set `ALLOW_TEST=True` and run notebook 02 (BERT) and notebook 03 (LLM) on the 150 test sentences (750 rows per method). SAIA's daily quota (1,000 requests/day) means the LLM run likely spans about 2 days; it resumes where it left off if interrupted.
2. Run the contamination probe on the 150 test sentences.
3. Run `notebooks/04_evaluation.ipynb` on the test split for the final BERT vs LLM comparison.

## 1. Research question

**RQ:** When the damaged region of a sentence is known and its garbled letters are visible, how well can GenAI models restore the original words so that meaning and key facts are preserved, and how does this change as the damaged share of the sentence grows?

- **SQ1 (severity):** Does repair quality decline smoothly with damage, or is there a point where it breaks down? Does the absolute gap length matter beyond the damaged share?
- **SQ2 (method):** Given identical input, does fine-tuned BERT or a few-shot LLM repair better?
- **SQ3 (metric validity):** Does semantic similarity overstate repair quality, missing wrong names, numbers or dates?

## 2. Data

**BLN600** (Booth, Thomas & Gaizauskas, 2024): 600 excerpts of 19th-century British newspapers (mostly London crime reporting), each with a gold (human-corrected) transcription and the original OCR. License CC BY-NC-ND: fine for coursework, never redistribute derived text (`data/` is git-ignored).

- **Sample (frozen):** 170 sentences from 157 excerpts: 150 test, 20 dev. At most 3 sentences per excerpt; no excerpt in both splits.
- **Calibration:** 29 separate excerpts, used only to measure realistic OCR letter-confusion rates.
- **BERT training pool:** 443 excerpts outside the 170-sentence sample (~2,900 sentences); zero overlap with test or dev, checked automatically.

## 3. Damage simulation (frozen, `corrupted_v2.jsonl`, sha256 `5798336d...`)

- One damaged **span** per sentence at 6 severity levels: 0 (clean), 1 word, 10%, 25%, 50%, 75% of the sentence's words.
- The span always includes an **anchor**: one key fact (a name, place, date or sum), always damaged, never fully dropped.
- Each damaged word: about 10% chance of being dropped entirely; otherwise 20-30% of its letters/digits are altered, using a realistic letter-confusion table built from the calibration excerpts.
- Resulting character error rate (mean, of the damaged span): 1 word 1.0%, 10% level 3.2%, 25% level 7.9%, 50% level 15.5%, 75% level 23.1%. Real BLN600 OCR error is about 7% per excerpt, so this scale is realistic at the low end and clearly heavier by the 50-75% levels, by design.

## 4. Shared input: the slot view

Both methods see exactly the same thing: the undamaged context unchanged, plus one **slot** per damaged word showing its garbled form (a dropped word is an empty slot `⟨?⟩`). Example:

`A POWERFULLY-BUILT man, ⟨same⟩ ⟨Bbonias⟩ ⟨Huicl,⟩ ⟨e⟩ commission agent, who refused his address, ...`

Both return one corrected word per slot. The same code splices predictions back in and scores them, so every score difference comes from how well the slots were filled, not from anything else.

## 5. Methods

**BERT + reranking.** Fine-tuned `bert-base-cased` (`models/bert_ft_v1`; trained on gold text only, never sees damaged text) fills slots left to right. For each slot it tries 1-3 mask tokens, scores candidates by BERT's confidence plus how closely they match the garbled letters shown (weighted by lambda), and keeps the best. Lambda, beam size and candidate count are tuned on dev only (in progress, see Status).

**LLM few-shot.** Llama 3.1 8B Instruct via GWDG's SAIA API (open-weight, hosted in Germany; chosen over commercial APIs so BLN600 text is never sent to a third party). Given the slot view plus 3 worked examples, asked to return one corrected word per slot as JSON. Zero-shot was tried and dropped (weak, not needed for SQ2). A secondary **few-shot + article** variant adds the whole source article as extra context, compared only against the LLM's own few-shot run (not against BERT, which never sees that context).

## 6. What is compared, and how it's scored

At every severity level, three things are scored against the gold span: **BERT + reranking**, **LLM few-shot**, and **no repair** (the damaged text as-is, a floor showing how bad things are with no intervention). The secondary **LLM few-shot + article** variant is scored too, but only compared to the LLM's own few-shot run.

Two headline metrics: **BERTScore** (does the repaired span mean the same thing) and **Fact Recovery Rate** (were names/numbers/dates recovered exactly). Secondary: exact word match, repair gain (error rate before vs. after), LLM format-failure rate, and the contamination probe (checks whether the LLM may have memorised BLN600's public text rather than genuinely repairing it).

**Dev results so far** (17 of 20 dev sentences, 3 held out as worked examples; mean BERTScore, few-shot / few-shot + article / no repair): 1w 0.73 / 0.70 / 0.37; 10% 0.46 / 0.51 / 0.01; 25% 0.28 / 0.26 / -0.16; 50% 0.05 / 0.04 / -0.27; 75% -0.10 / -0.21 / -0.34. Fact Recovery Rate: 1w 0.41 / 0.35; 10% 0.25 / 0.34; 25% 0.22 / 0.19; 50% 0.13 / 0.24; 75% 0.07 / 0.07. Contamination probe: 0 of 20 dev sentences near-verbatim. These are dev numbers only, for tuning; the test split is the real result.

## 7. Experimental design summary

| Factor | Levels |
|---|---|
| Severity | 0, 1 word, 10%, 25%, 50%, 75% of sentence words |
| Length band | 20-29, 30-44, 45-60 words |
| Input | slot view: garbled letters visible, dropped words as empty slots |
| Methods | fine-tuned BERT + reranking; Llama 3.1 8B few-shot (secondary: + article context) |
| Data | 150 test sentences (final result); 20 dev sentences (tuning only) |
| Metrics | BERTScore, Fact Recovery Rate (primary); exact word match, repair gain, format failures, contamination rate (secondary) |

## 8. Fairness and known asymmetries

Controlled: identical sentences, slots, garbled letters and context for both methods; identical scoring code; no test/dev excerpt ever used in BERT training or LLM few-shot examples; deterministic decoding.

Worth stating in the report: the LLM reads garbled letters directly while BERT only uses them through its scoring; BERT predicts multi-letter words in one parallel pass, which hurts it on long rare names; the two methods differ in scale and pretraining, so this compares two practical approaches, not two equal-sized architectures; the LLM may have seen BLN600's public gold text during pretraining (checked by the contamination probe).

## 9. Limitations

- Damage location is given; finding damage in raw OCR text is not tested here.
- Damage is synthetic and centres on a fact word by design, so facts are over-represented in small spans.
- Small test sample (150 sentences), so results are trends, not statistically powered effects.
- Fact Recovery Rate uses surface rules (capitals, numbers, money) and can miss facts written in lowercase.
- English, 19th-century London crime reporting, mostly three related publications.

## 10. Future work (out of scope for this report)

- Finding damage automatically in raw OCR text, rather than being told where it is.
- Testing on real OCR/gold word pairs, not just synthetic damage.
- Giving a vision-capable model the actual scanned image instead of text-only garbled letters.
- Whole-excerpt context at inference time: considered and set aside for now (512-token limit on BERT, would need retraining, and breaks the fairness rule that both methods see identical context).

## 11. References

- Booth, C. W., Thomas, A., & Gaizauskas, R. (2024). BLN600: A Parallel Corpus of Machine/Human Transcribed Nineteenth Century Newspaper Texts. LREC-COLING 2024, pp. 2440-2446.
- Thomas, A., Gaizauskas, R., & Lu, H. (2024). Leveraging LLMs for Post-OCR Correction of Historical Newspapers. LT4HALA @ LREC-COLING 2024, pp. 116-121.
- Debaene, F., Maladry, A., Lefever, E., & Hoste, V. (2025). Evaluating Transformers for OCR Post-Correction in Early Modern Dutch Theatre. COLING 2025, pp. 10367-10374.
- Lyu, L., Koutraki, M., Krickl, M., & Fetahu, B. (2021). Neural OCR Post-Hoc Correction of Historical Corpora. TACL, 9, 479-493.
- Evershed, J., & Fitch, K. (2014). Correcting Noisy OCR: Context Beats Confusion. DATeCH 2014, pp. 45-51.
- Shen, T., Quach, V., Barzilay, R., & Jaakkola, T. (2020). Blank Language Models. EMNLP 2020, pp. 5186-5198.
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR 2020.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
- Sainz, O., Campos, J., García-Ferrero, I., Etxaniz, J., Lopez de Lacalle, O., & Agirre, E. (2023). NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark. Findings of EMNLP 2023, pp. 10776-10787.
- Petrak, D., Tran, T. T., & Gurevych, I. (2025). Towards Automated Error Discovery: A Study in Conversational AI. arXiv:2509.10833. (The course's example paper; structural template only.)
