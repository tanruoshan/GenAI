# Live demo for the defence

One offline HTML page with five scenes. It replaces most result slides: the slides keep motivation, research
questions and the method list, and the demo shows the damage, the repairs and the results.

| Key | Part | What it shows |
|---|---|---|
| 1 | Damage | one test sentence at clean, 1 word, 10, 25, 50 and 75%; damaged words, dropped words, anchor fact |
| 2 | Repairs | the same damaged input repaired by all six methods, each slot marked right or wrong |
| 3 | Results | anchor recovery, Fact Recovery Rate, span BERTScore, exact words and CER with 95% intervals; click H1 to H3 |
| 4 | BERTScore vs. facts | span BERTScore of repairs with the right and the wrong anchor (H3, SQ3), and the SQ3 rates |
| 5 | Lookup | the dictionary baseline, live: type a damaged word and see the closest list words |

The page copies the look of the paper on purpose, so the demo reads as the paper's figures made interactive:
Times, black rules and booktabs tables as on the ACL page, the Okabe-Ito colours and markers of the paper's
Figure 2, and the viridis level colours of its Figure 1. Every part ends with a caption in the paper's style.

Other keys: left and right arrows change the level, up and down the sentence, Space plays the levels, R picks a
random sentence. Escape leaves a text box.

## Build

The page is built on your own machine, because it embeds BLN600 text. It needs the setup and data steps of the
main README (the frozen damaged data in `data/processed/` and BLN600 in `data/raw/` for the word list).

```bash
python demo/build_demo.py              # a few seconds
python demo/build_demo.py --bertscore  # adds span BERTScore per repair (scene 4 dot plot); first run 5 to 10 min on CPU
```

The page is written to `demo/out/bln600_demo.html`. Open it in any browser; it needs no server and no internet.
On a 1280 x 720 projector, use full screen (F11) and zoom out once if a scene does not fit.

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
