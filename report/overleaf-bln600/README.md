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

## Status (2026-09-29)
- All six methods have frozen settings and stored test predictions (750 rows each); the LLM probe covers 116 of 150 test sentences (stopped for API resources, final).
- Blocked on one step: `notebooks/04_evaluation.ipynb` run once with `SPLIT = "test"` (Simon). Then fill Table `tab:results` (BERTScore and FRR, five columns), Figure 2, Results text, Interpretation P1 to P5 with one verdict per hypothesis, Abstract and the Introduction's key outcomes.
- Written on 2026-09-29: hypotheses H1, H2, H2b, H3 (Intro); Method (b) BERT + dictionary, (c) Qwen3 with 5 examples, (d) dictionary lookup, (e) BART; `tab:settings`, `tab:design`; statistics paragraph and SQ3 number (4.1, 4.2); probe details; Limitations and appendix outline.
- Length with these additions: content ends at about 7.75 of 9 pages (local pdfLaTeX build), so about 1.25 pages remain for Figure 2, Results text and the Interpretation fills. Deleting the unused variant in each Interpretation paragraph gives back some space.
- Six new citations (`lewis2020bart`, `koehn2004statistical`, `dror2018hitchhiker`, `dietterich1998approximate`, `holm1979simple`, `yang2025qwen3`) are checked and waiting in `notes/pending_refs.bib`. They show as (?) until Shan moves them into `references.bib`.

## Before submitting
- [ ] Remove every `\todo` and the `\nocite` smoke-test line in `sections/02_related_work.tex`.
- [ ] Move `notes/pending_refs.bib` into `references.bib` (no (?) citations left).
- [ ] Check each `references.bib` entry against the ACL Anthology / arXiv / publisher page. Delete unused entries.
- [ ] Insert real author names and affiliations.
- [ ] Page count <= 9 without references.
- [ ] Only test-split numbers in results.
- [ ] `inconsolata` is available on Overleaf. If compiling locally without it, comment out that line.
