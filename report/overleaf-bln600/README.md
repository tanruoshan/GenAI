# BLN600 report: Overleaf framework

Upload the zip to Overleaf (New Project > Upload Project). Compiler: pdfLaTeX, main file `main.tex`.

## Course requirements (from the project brief)
- Format: ACL style (`acl.sty`, `acl_natbib.bst` from github.com/acl-org/acl-style-files). Any text processor is allowed, the PDF is what counts. Option `preprint` is used (names shown, page numbers on).
- Length: 7 to 9 pages of content, hard ceiling 9, references not counted. No extensive appendix. Instead, say in a few lines what the appendix would hold for the winter-term defence (`sections/07_appendix_outline.tex`).
- Content, following the EMNLP example (arXiv 2509.10833):
  1. Introduction: topic, why relevant for GenAI, objective, key outcomes.
  2. Related work: how far the objective is already solved, what we add.
  3. Method: what the solution looks like, how it derives from known work, what we add.
  4. Empirical investigation: data, experimental design, how the solution is used, results.
  5. Interpretation: results vs objective (fully or gradually reached), meaning and generality for the field.
- Writing rules (project instructions): dense concise paragraphs, simple English, few long dashes, basic theory only, tables and figures preferred over long text, every claim tied to a peer-reviewed source, no invented references, devil's advocate on every assertion.

## Page budget (9 pages)
| Section | Pages |
|---|---|
| Abstract + Introduction | 1.0 |
| Related work | 0.75 |
| Method (data, damage, slot input, methods) | 2.0 |
| Empirical investigation (design, metrics, results) | 2.5 |
| Interpretation and conclusion | 1.0 |
| Limitations | 0.3 |
| Appendix outline + slack | 0.45 |

## Where each fact lives
`notes/report_sources.md` merges `docs/dossier.md` (the single dossier; `dossier_v2.md` was merged into it on 2026-09-29), `CLAUDE.md` and the config/run files into one fact list ordered by report section, with a conflict table at the top. Every section stub points to its part of that file. Dev numbers are tuning context only and must not be reported as results.

## Blocked until results exist
- Table 3 and Figure 2 (test predictions for BERT and LLM, then `04_evaluation.ipynb` on test).
- Final LLM model id, host and prompt version (teammate). BERT settings are final: lambda 8, beam 5, top 10.
- Abstract, Interpretation, Introduction key outcomes.

## Before submitting
- [ ] Remove every `\todo` and the `\nocite` smoke-test line in `sections/02_related_work.tex`.
- [ ] Check each `references.bib` entry against the ACL Anthology / arXiv / publisher page. Delete unused entries.
- [ ] Insert real author names and affiliations.
- [ ] Page count <= 9 without references.
- [ ] Only test-split numbers in results.
- [ ] `inconsolata` is available on Overleaf. If compiling locally without it, comment out that line.
