# Live demo for the defence

One offline HTML page that walks through the project in six steps, one claim each. It is both the 5 to 10
minute summary and the place to dig deeper during the discussion: every step can be explored in place
(another sentence, level, measure or method), so there is only one mode.

| Key | Step | Claim and what can be explored |
|---|---|---|
| 1 | Damage | one fact and the words around it are damaged, from one word to 75%; other sentences, play the levels |
| 2 | Repairs | the same damaged sentence for every method, right and wrong words; then (next) their span BERTScore; ablations |
| 3 | Results | Fact Recovery Rate, anchor recovery, span BERTScore, exact words, CER with 95% intervals; H1 to H3, with the paired tests of H2 and H2b |
| 4 | BERTScore | span BERTScore of repairs with the right and the wrong anchor, per method and level; the SQ3 rate |
| 5 | Lookup | the dictionary baseline, live: type a damaged word and see the closest list words |
| 6 | Summary | answers to the hypotheses, the practical point and the limits |

Each step has a plain claim, one sentence on why we did it this way, the visual, and small controls. Right
arrow, Space or a presenter's clicker (PageDown) goes forward (in step 1 through the levels, in step 2 to the
BERTScore view), left arrow or PageUp back, 1 to 6 jump to a step. P plays the levels, R picks a random sentence,
up and down change the sentence (steps 1 and 2). The bar at the bottom does the same with the mouse. After
typing in the lookup box (step 5), press Escape or click outside it before the keys move again.

**Review notes.** The page is built as the production version for the defence (`let REVIEW=false` in
`template.html`): no notes, no notes button. For a team review build with the orange "To confirm" boxes, set
it to `true` and rebuild; N then hides and shows them.

**Look.** The page copies the submitted paper so it reads as the paper made interactive: the paper's own fonts
(TeX Gyre Termes, the OpenType version of the paper's Nimbus Roman, and Inconsolata, embedded from
`demo/fonts/` with their licences), black rules and booktabs tables, and figures drawn by matplotlib with the
SciencePlots style, Okabe-Ito colours and markers of the paper's Figure 2 (`demo/talk_figures.py`). Damage
levels use the viridis colours of Figure 1.

**Earlier versions.** Two git tags keep the earlier designs:

| Tag | Version |
|---|---|
| `demo-v1-shan` | Shan's original five-scene explorer (commit `ea3d44b`) |
| `demo-v2-two-modes` | the paper-style explorer plus a separate talk mode (commit `7ce820a`) |

To look at one without changing anything: `git fetch --tags`, then `git switch --detach demo-v1-shan`,
`python demo/build_demo.py --bertscore`, and `git switch demo` to come back. To make it the current version
again: `git checkout demo-v1-shan -- demo/`, `git rm -r demo/fonts demo/talk_figures.py`, commit.

## Build

The page is built on your own machine, because it embeds BLN600 text. It needs the setup and data steps of the
main README (the frozen damaged data in `data/processed/` and BLN600 in `data/raw/` for the word list). Run the commands from the repository
root inside its virtual environment, after `pip install -r requirements.txt` and `pip install -e .`; on a Mac
without an active environment, use `python3`.

```bash
python demo/build_demo.py --bertscore  # span BERTScore per repair and all figures; first run 5 to 10 min on CPU
python demo/build_demo.py              # reuses the cached BERTScores; without a cache, steps 3 and 4 have no figures
```

The page is written to `demo/out/bln600_demo.html`. Open it in any browser; it needs no server and no internet.
On a 1280 x 720 projector, use full screen (F11, or Cmd+Ctrl+F on a Mac) and zoom out once (Ctrl or Cmd and
minus) if a step does not fit; steps 2 to 5 may still need a short scroll.

## Where the numbers come from

- Intervals, paired tests and SQ3 rates are read from the stored outputs of `notebooks/04_evaluation.ipynb`, so
  they are the numbers in the paper.
- Per-sentence repairs are read from `runs/preds/` (the files listed in notebook 4) and scored with the same
  `row_scores` as notebook 4.
- The build loads the frozen data through `load_frozen` (hash check) and compares the per-sentence means with
  notebook 4. If any of them differs, it stops. `--data` uses another damaged-data file for testing; the page
  is then marked as unverified.
- Span BERTScore per repair uses the notebook's settings (`roberta-large`, baseline rescaled) and is cached in
  `demo/out/bertscore_span.json`.

## Working on the demo (for teammates and coding agents)

The project's decision log is not in this repository, so this section is the context you need.

**Rules that must not be broken**
- Every number on the page comes from the single test run of `notebooks/04_evaluation.ipynb` (read from its
  stored outputs) or is recomputed from `runs/preds/` with the notebook's own scoring. Never run notebook 4 on
  the test split again, and never type a number into the page by hand.
- The build checks 90 recomputed means against notebook 4 and stops if one differs. A build that does not
  end with "0 differ" must not be shown.
- Claims must stay within the paper: the hypothesis verdicts as in its Section 5, "best *main* method" (the
  LLM + article variant is an upper bound, not a main method), BART descriptive only (trained on our own
  damage), the LLM result tied to one model and one prompt, the damage synthetic and its location known.
- `demo/out/` holds BLN600 text. Never commit or upload it; this repository is public.

**Where things are**
- All on-page text (claims, "Why" lines, captions, hypothesis texts) is in `demo/template.html`, one function
  per step (`stepDamage` ... `stepSummary`) and the `HYPS` list. The orange review notes are `REVIEW_NOTES`.
- The example sentence of steps 1, 2 and 4 is `TALK` in `demo/build_demo.py`. It was chosen because BERT +
  dict. and BART restore the same number of words there and only BART gets the name wrong.
- Figures: `demo/talk_figures.py`. They are drawn at build time; `python demo/build_demo.py` reuses the
  BERTScore cache in `demo/out/`.

**After any change**
1. `python demo/build_demo.py` and check "0 differ".
2. `python -m pytest -q tests/test_demo_build.py`.
3. Open the page, click through all six steps (with and without the notes, N), and check that no "null",
   "undefined" or "NaN" appears and that step 3's hypotheses and measures still link both ways.

**State (13 Oct 2026)** The page has been checked against the paper's Tables 6 and 7 and notebook 4 by four
independent reviews; their fixes are in the git log of this branch. It is built as the production version
(`REVIEW=false`); the remaining team questions stay in `REVIEW_NOTES` for a review build.

## Licence

BLN600 is CC BY-NC-ND 4.0. `demo/out/` holds BLN600 text and is git-ignored: show the page, do not commit or
publish it.
