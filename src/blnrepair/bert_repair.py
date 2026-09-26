"""Notebook 2, Part C: fill the slots with BERT, reranked by letter similarity to the damaged form.

Slots are filled left to right. A slot still waiting shows one [MASK] with its visible punctuation
(as waiting words looked in training). For the current slot BERT gets 1, 2 and 3 masks in one batched
forward pass; candidate words are built from the top pieces per mask, and each candidate gets
    score = mean log-probability per piece + lam * (1 - Levenshtein distance / longer length)
with the similarity taken against the slot's damaged core. A dropped slot has no letters, so only
the BERT score counts there, and only there a punctuation-only answer is allowed.

B1 sub-slots: a slot whose damaged core has inner punctuation (a hyphen, an apostrophe) is filled part
by part, keeping that punctuation; each part gets 1 to 3 masks. The model input is built from the
slot view only (build_slots), never from gold text.
"""
import itertools
import json
import math
import re

import torch
from rapidfuzz.distance import Levenshtein

MASK = "[MASK]"
INNER = re.compile(r"([^0-9A-Za-zÀ-ɏ]+)")  # the letter class of subwords.parts_of


def split_parts(slot):
    """(parts, separators) of a slot's damaged core: 'Pol1ce-coart' -> (['Pol1ce', 'coart'], ['-']).
    A dropped slot is one part with nothing visible."""
    if slot["dropped"]:
        return [""], []
    pieces = INNER.split(slot["core"])  # the core has no edge punctuation (split_punct), so no part is empty
    return pieces[0::2], pieces[1::2]


def char_sim(a, b):
    """1 minus the Levenshtein distance divided by the longer length (case-sensitive, plain)."""
    return Levenshtein.normalized_similarity(a, b)


def render(view, fills, current=None):
    """The sentence BERT sees. fills[i] = the chosen parts of slot i so far (a list). current =
    (slot index, number of masks): that slot shows its filled parts, then the masks for its next part,
    then one [MASK] per remaining part. Every other unfilled slot shows one [MASK]."""
    slots, words, s = view["slots"], [], 0
    for w in view["words"]:
        if w is not None:
            words.append(w)
            continue
        slot, done = slots[s], fills[s]
        parts, seps = split_parts(slot)
        if current and current[0] == s:
            shown = done + [" ".join([MASK] * current[1])] + [MASK] * (len(parts) - len(done) - 1)
            body = shown[0] + "".join(sep + p for sep, p in zip(seps, shown[1:]))
        elif len(done) == len(parts):
            body = done[0] + "".join(sep + p for sep, p in zip(seps, done[1:]))
        else:
            body = MASK
        words.append(body if slot["dropped"] else slot["lead"] + body + slot["trail"])
        s += 1
    return " ".join(words)


class BertReranker:
    """predict(view) for the prediction store's runner: one word per slot."""

    def __init__(self, model, tokenizer, lam=1.0, beam=5, top_n=10, max_masks=3, device="cpu"):
        self.model, self.tok, self.device = model.to(device).eval(), tokenizer, torch.device(device)
        self.lam, self.beam, self.top_n, self.max_masks = lam, beam, top_n, max_masks
        vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        special = torch.tensor([t.startswith("[") and t.endswith("]") for t in vocab])
        cont = torch.tensor([t.startswith("##") for t in vocab])
        word = torch.tensor([t.removeprefix("##").isalnum() for t in vocab])
        self.vocab = vocab
        # (first piece, later pieces) allowed: any piece for a dropped slot, letters and digits only
        # for a visible slot, so that punctuation never takes a place in the top `beam`
        self.allowed = {True: (~special & ~cont, ~special & cont), False: (~special & ~cont & word, cont & word)}

    @torch.no_grad()
    def mask_logprobs(self, texts):
        """For each text, log-probabilities at its first n masks, n = 1, 2, 3 for the three texts.
        The current part's masks are always the first masks: earlier slots and parts are filled."""
        enc = self.tok(texts, padding=True, return_tensors="pt").to(self.device)
        logps = torch.log_softmax(self.model(**enc).logits.float(), dim=-1)
        out = []
        for i, ids in enumerate(enc["input_ids"]):
            pos = (ids == self.tok.mask_token_id).nonzero().flatten()[:i + 1]
            out.append(logps[i, pos].cpu())
        return out

    def candidates(self, logps_by_m, allow_punct):
        """Candidate words with their BERT score (mean log-probability per piece). For each mask count,
        the top `beam` allowed pieces per position (first position: a word start; later: ## pieces),
        all their combinations, and the top_n of those by BERT score, kept per mask count so that
        multi-piece names are not crowded out before reranking. The same word from several mask
        counts keeps its best score. allow_punct: True only for a dropped slot."""
        start_ok, cont_ok = self.allowed[allow_punct]
        best = {}
        for lp in logps_by_m:
            tops = []
            for pos in range(lp.shape[0]):
                ok = start_ok if pos == 0 else cont_ok
                values, ids = lp[pos].masked_fill(~ok, -math.inf).topk(self.beam)
                tops.append(list(zip(ids.tolist(), values.tolist())))
            combos = [(self.vocab[c[0][0]] + "".join(self.vocab[i][2:] for i, _ in c[1:]), sum(v for _, v in c) / len(c))
                      for c in itertools.product(*tops)]
            for text, score in sorted(combos, key=lambda c: -c[1])[:self.top_n]:
                best[text] = max(score, best.get(text, -math.inf))
        return best

    def fill_part(self, view, fills, s, damaged):
        texts = [render(view, fills, (s, m)) for m in range(1, self.max_masks + 1)]
        dropped = view["slots"][s]["dropped"]
        cands = self.candidates(self.mask_logprobs(texts), allow_punct=dropped)
        scored = [(c, b, 0.0 if dropped else char_sim(c, damaged)) for c, b in cands.items()]
        scored.sort(key=lambda x: -(x[1] + self.lam * x[2]))
        assert scored, f"no candidate for slot {s} of {view['id']} {view['severity']}"
        return scored

    def __call__(self, view):
        fills, log = [[] for _ in view["slots"]], []
        for s, slot in enumerate(view["slots"]):
            parts, seps = split_parts(slot)
            for part in parts:
                scored = self.fill_part(view, fills, s, part)
                fills[s].append(scored[0][0])
                log.append([[c, round(b, 3), round(sim, 3)] for c, b, sim in scored[:3]])
        pred_words = [f[0] + "".join(sep + p for sep, p in zip(split_parts(slot)[1], f[1:]))
                      for f, slot in zip(fills, view["slots"])]
        return {"raw_output": json.dumps(log, ensure_ascii=False), "pred_words": pred_words}
