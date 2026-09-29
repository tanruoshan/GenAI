"""BART repair (notebook 2c): training examples and reading the answer.

BART is a denoising sequence-to-sequence model. It reads the sentence in the same slot view as the other
methods (context words as they are, each damaged word in brackets, a dropped word as a question mark in
brackets) and writes the same sentence back with the gold word inside each bracket. The answer is read
from the brackets. Only the bracket characters differ from the view BERT and the LLM get: "{ Hnrd }"
instead of "⟨Hnrd⟩", because BART's byte-level tokenizer splits ⟨ and ⟩ into three byte pieces each and
glues the word to them; "{", "}" and the spaces keep every slot a normal word. No BLN600 sentence contains
"{" or "}" (checked).

Unlike BERT and the LLM, BART is trained on damaged text: the sentences of the BERT training pool (443
excerpts outside the sample), damaged by the same generator and settings as the evaluation data
(corrupt.plan_word), with a new anchor and new damage in every epoch.
"""
import random
import re

from blnrepair.corrupt import level_sizes, plan_word, seed_int, select_span
from blnrepair.slots import build_slots

LEVELS_FOR_TRAINING = ["1w", "10", "25", "50", "75"]
SLOT = re.compile(r"\{(.*?)\}")


def damaged_record(row, level, draw, table, cfg):
    """A pool sentence damaged at one level, in the format of corrupted_v2.jsonl (the fields build_slots
    and splice need). draw ("0", "1", ... per epoch, or "val") enters every seed, so each draw gets its own
    anchor and its own damage; the anchor is never dropped, as in the evaluation data."""
    tokens = row["text"].split()
    key = f"{row['id']}|{draw}|{level}"
    anchor = random.Random(seed_int(cfg["seed"], "bart_anchor", key)).choice(row["fact_idx"])
    k = level_sizes(len(tokens))[level]
    start, end = select_span(len(tokens), anchor, k)
    plans = [plan_word(key, i, tokens[i], table, cfg, allow_drop=(i != anchor)) for i in range(start, end)]
    return {"id": row["id"], "severity": level, "split": row["usage"], "k": k, "gold_tokens": tokens,
            "anchor_idx": anchor, "span_start": start, "span_end": end, "ops_per_word": plans,
            "fact_idx_in_span": [i for i in row["fact_idx"] if start <= i < end]}


def bart_view(view):
    """The build_slots view as BART's input text (slots.slot_view with other brackets)."""
    slots = iter(view["slots"])
    parts = []
    for w in view["words"]:
        if w is None:
            s = next(slots)
            parts.append("{ ? }" if s["dropped"] else f"{{ {s['out']} }}")
        else:
            parts.append(w)
    return " ".join(parts)


def bart_input(record):
    return bart_view(build_slots(record))


def bart_target(record):
    """The gold sentence with every damaged word (with its punctuation) in brackets."""
    tokens = list(record["gold_tokens"])
    for p in record["ops_per_word"]:
        tokens[p["idx"]] = f"{{ {p['gold']} }}"
    return " ".join(tokens)


def epoch_examples(pool, table, cfg, draw):
    """Every sentence of the pool at every level (equal weight per level, as in the evaluation), with the
    damage of this draw. Validation sentences always use the fixed draw "val"."""
    out = []
    for row in pool:
        this_draw = "val" if row["usage"] == "val" else str(draw)
        for level in LEVELS_FOR_TRAINING:
            rec = damaged_record(row, level, this_draw, table, cfg)
            out.append({"id": row["id"], "usage": row["usage"], "level": level, "k": rec["k"],
                        "source": bart_input(rec), "target": bart_target(rec)})
    return out


def read_answer(text, k):
    """The words inside the brackets, in order. None when their number is not k (a format failure, as for
    the LLM)."""
    words = [w.strip() for w in SLOT.findall(text)]
    return words if len(words) == k else None


def load_bart_repair_config(path=None):
    import yaml
    from pathlib import Path
    from blnrepair.data import ROOT
    return yaml.safe_load(Path(path or ROOT / "configs" / "bart_repair.yaml").read_text(encoding="utf-8"))


def bart_version(cfg, split):
    """Version name of a stored BART run, for example ftv1-greedy-dev: model and decoding are in the name."""
    decode = "greedy" if cfg["num_beams"] == 1 else f"beam{cfg['num_beams']}"
    return f"{cfg['model_tag']}-{decode}-{split}"


class BartRepairer:
    """predict(view) for preds.run_method: one generation per row, deterministic (eval mode, no sampling)."""

    def __init__(self, model, tokenizer, device="cpu", num_beams=1, max_length=320):
        self.model, self.tokenizer, self.device = model.eval().to(device), tokenizer, device
        self.num_beams, self.max_length = num_beams, max_length

    def __call__(self, view):
        import json
        import torch
        source = bart_view(view)
        enc = self.tokenizer(source, return_tensors="pt")
        assert enc["input_ids"].shape[1] <= self.max_length, f"{view['id']}: input longer than {self.max_length} tokens"
        with torch.no_grad():
            out = self.model.generate(**enc.to(self.device), num_beams=self.num_beams, do_sample=False,
                                      max_length=self.max_length)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        words = read_answer(text, len(view["slots"]))
        error = None if words else f"expected {len(view['slots'])} bracketed words, got {len(SLOT.findall(text))}"
        return {"raw_output": json.dumps({"source": source, "output": text}, ensure_ascii=False),
                "pred_words": words, "error": error}
