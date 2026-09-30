"""Notebook 2b: the dictionary lookup, a non-GenAI baseline.

A word list is built from the gold text of the BLN600 excerpts outside the sample (no test or dev
excerpt). Each damaged word is replaced by the list entry with the smallest edit distance to its damaged
form; among equally close entries the most frequent one wins. There is no context and no model, so this
is the pure "confusion" side of "context beats confusion". A dropped slot has no letters to look up and
gets an empty answer (the word stays missing and counts as wrong).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from blnrepair.data import ROOT
from blnrepair.slots import split_punct


def load_lexicon_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "configs" / "lexicon.yaml").read_text(encoding="utf-8"))


def lexicon_version(cfg, split):
    """Version name of a stored lookup run, for example v1-dev."""
    return f"{cfg['version']}-{split}"


def build_lexicon(gold, exclude_docs):
    """Word cores and their counts over all excerpts not in exclude_docs. A core is a whitespace token
    without its edge punctuation (split_punct, the rule of the slot view); inner hyphens and apostrophes
    stay, and case is kept."""
    counts = Counter()
    for doc_id, paragraphs in gold.items():
        if doc_id in exclude_docs:
            continue
        for paragraph in paragraphs:
            counts.update(core for core in (split_punct(tok)[1] for tok in paragraph.split()) if core)
    return counts


class LexiconLookup:
    """predict(view) for the prediction store's runner: one word per slot, from the slot view only."""

    def __init__(self, counts):
        # most frequent first, so that among equally close entries the first one found is the most frequent
        self.words = sorted(counts, key=lambda w: (-counts[w], w))
        self.counts = counts

    def nearest(self, core):
        """(entry, edit distance, number of entries at that distance) for one damaged core."""
        dist = cdist([core], self.words, scorer=Levenshtein.distance, workers=-1)[0]
        best = int(dist.min())
        i = int(dist.argmin())  # first minimum = the most frequent of the closest entries
        return self.words[i], best, int((dist == best).sum())

    def closest(self, core, n):
        """The n entries closest to a damaged core: by edit distance, then by frequency (used by BERT
        with dictionary candidates)."""
        dist = cdist([core], self.words, scorer=Levenshtein.distance, workers=-1)[0]
        return [self.words[i] for i in np.argsort(dist, kind="stable")[:n]]

    def __call__(self, view):
        pred_words, log = [], []
        for slot in view["slots"]:
            if slot["dropped"] or not slot["core"]:
                pred_words.append("")
                log.append(None)
                continue
            word, dist, ties = self.nearest(slot["core"])
            pred_words.append(word)
            log.append([word, dist, ties, self.counts[word]])
        return {"raw_output": json.dumps(log, ensure_ascii=False), "pred_words": pred_words}
