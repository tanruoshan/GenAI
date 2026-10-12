# Live demo for the defence

One offline HTML page that walks through the project in six steps, one claim each. It is both the 5 to 10
minute summary and the place to dig deeper during the discussion: every step can be explored in place
(another sentence, level, measure or method), so there is only one mode.

| Key | Step | Claim and what can be explored |
|---|---|---|
| 1 | Damage | one fact and the words around it are damaged, from one word to 75%; other sentences, play the levels |
| 2 | Repairs | the same damaged sentence for every method, right and wrong words; then (next) their span BERTScore; ablations |
| 3 | Results | Fact Recovery Rate, anchor recovery, span BERTScore, exact words, CER with 95% intervals; H1 to H3 with their tests |
| 4 | BERTScore | span BERTScore of repairs with the right and the wrong anchor, per method and level; the SQ3 rate |
| 5 | Lookup | the dictionary baseline, live: type a damaged word and see the closest list words |
| 6 | Summary | answers to the hypotheses, the practical point and the limits |

Each step has a plain claim, one sentence on why we did it this way, the visual, and small controls. Right
arrow, Space or a presenter's clicker (PageDown) goes forward (in step 1 through the levels, in step 2 to the
BERTScore view), left arrow or PageUp back, 1 to 6 jump to a step. P plays the levels, R picks a random sentence,
up and down change the sentence (steps 1 and 2). The bar at the bottom does the same with the mouse.

**Review notes.** Orange "To confirm" boxes mark points the team still has to agree on. N hides them; before
the defence set `REVIEW = false` in `template.html` and rebuild.

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
main README (the frozen damaged data in `data/processed/` and BLN600 in `data/raw/` for the word list).

```bash
python demo/build_demo.py --bertscore  # span BERTScore per repair and all figures; first run 5 to 10 min on CPU
python demo/build_demo.py              # without BERTScore: no figures (steps 3 and 4 stay empty)
```

The page is written to `demo/out/bln600_demo.html`. Open it in any browser; it needs no server and no internet.
On a 1280 x 720 projector, use full screen (F11, or Cmd+Ctrl+F on a Mac) and zoom out once if a step does not fit.

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

## Licence

BLN600 is CC BY-NC-ND 4.0. `demo/out/` holds BLN600 text and is git-ignored: show the page, do not commit or
publish it.
