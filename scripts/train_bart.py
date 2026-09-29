"""Fine-tune facebook/bart-base to repair damaged sentences (notebook 2c).

Smoke run (30 steps on 4 fixed batches, then save, reload and generate). Not on an 8 GB laptop: it crashed one.
    python scripts/train_bart.py --smoke

Full run (GPU; on a RunPod pod, see scripts/pod/pod.sh):
    python scripts/train_bart.py

Settings: configs/bart_ft.yaml; seed and damage settings: configs/config.yaml and reports/calibration.json
(the generator of the evaluation data). The pool must match runs/bert_train_pool.sha256. Every epoch
damages every training sentence again at all five levels (new anchor, new damage); the validation sentences
keep one fixed damage draw. The checkpoint with the lowest validation loss is kept (early stopping).

After every epoch three things are measured in eval mode, so that over- and underfitting can be read later:
- val: the validation sentences (never trained on), fixed damage;
- probe: as many training sentences, with a damage draw never used in training. Probe better than val means
  the model has learnt the training sentences themselves (overfitting); both poor means underfitting;
- per set: loss per target token, the share of slots repaired exactly (greedy decoding, compared as the
  evaluation does), per level, and the format failures.

Outputs: models/bart_ft_v1/ (best epoch), runs/bart_ft_v1_log.csv, runs/bart_ft_v1.sha256,
runs/bart_ft_v1_config.yaml (settings, hashes, best epoch, stop reason, times, versions, code commit) and
runs/bart_ft_v1_val_outputs.jsonl (the best epoch's answers on the validation set, for error analysis).
"""
import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from blnrepair.bart_data import LEVELS_FOR_TRAINING, epoch_examples, generation_settings, slots_right  # noqa: E402
from blnrepair.bert_train import load_pool_checked, sha256_file  # noqa: E402
from blnrepair.corrupt import load_table, seed_int, settings  # noqa: E402
from blnrepair.data import load_config  # noqa: E402


def device_and_dtype(cfg):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        return dev, (torch.bfloat16 if cfg["train"]["bf16_on_gpu"] and torch.cuda.is_bf16_supported() else None)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu"), None


def encode(batch, tokenizer, max_length, device):
    enc = tokenizer([e["source"] for e in batch], text_target=[e["target"] for e in batch], padding=True,
                    return_tensors="pt")
    assert enc["input_ids"].shape[1] <= max_length and enc["labels"].shape[1] <= max_length, "an example is too long"
    enc["labels"][enc["labels"] == tokenizer.pad_token_id] = -100
    return {k: v.to(device) for k, v in enc.items()}


def batches_of(examples, size, seed=None, draw=None):
    order = list(examples)
    if seed is not None:
        random.Random(seed_int(seed, "bart_shuffle", draw)).shuffle(order)
    return [order[i:i + size] for i in range(0, len(order), size)]


@torch.no_grad()
def evaluate(model, batches, tokenizer, t, device, dtype, generate=True, keep_outputs=False):
    """Loss per target token and, with generate, the exact-slot share overall and per level, the format
    failures (a failure counts all its slots as wrong) and optionally every answer."""
    model.eval()
    total, n = 0.0, 0
    right, slots, failures, outputs = defaultdict(int), defaultdict(int), 0, []
    for batch in batches:
        enc = encode(batch, tokenizer, t["max_length"], device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(**enc).logits
        total += torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), enc["labels"].flatten(), reduction="sum").item()
        n += int((enc["labels"] != -100).sum())
        if not generate:
            continue
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 **generation_settings(1, t["max_length"]))
        texts = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for e, text in zip(batch, texts):
            ok = slots_right(text, e["target"], e["k"])
            failures += ok is None
            right[e["level"]] += sum(ok) if ok else 0
            slots[e["level"]] += e["k"]
            if keep_outputs:
                outputs.append({"id": e["id"], "level": e["level"], "source": e["source"], "target": e["target"],
                                "output": text, "format_ok": ok is not None})
    model.train()
    result = {"loss": total / n}
    if generate:
        result.update(slot_exact=sum(right.values()) / sum(slots.values()), format_failures=failures,
                      **{f"exact_{lv}": right[lv] / slots[lv] for lv in LEVELS_FOR_TRAINING if slots[lv]})
    return result, outputs


def setup(cfg, seed):
    torch.manual_seed(seed)
    device, dtype = device_and_dtype(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"]).to(device)
    for key, value in generation_settings(1, cfg["train"]["max_length"]).items():
        setattr(model.generation_config, key, value)  # the saved model carries no summarisation defaults
    model.train()
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    print(f"device: {name}, bf16: {dtype is not None}")
    return tokenizer, model, device, dtype


def make_optimizer(model, t, steps_per_epoch, total_steps):
    """AdamW with weight decay except on biases and LayerNorm weights, warmup over a share of one epoch,
    then linear decay to 0 over total_steps (the settings of the BERT fine-tuning)."""
    no_decay = ("bias", "layer_norm.weight", "layernorm_embedding.weight")
    params = list(model.named_parameters())
    groups = [{"params": [p for n, p in params if not n.endswith(no_decay)], "weight_decay": t["weight_decay"]},
              {"params": [p for n, p in params if n.endswith(no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=t["learning_rate"])
    return opt, get_linear_schedule_with_warmup(opt, round(t["warmup_frac_of_epoch"] * steps_per_epoch), total_steps)


def train_step(model, batch, tokenizer, t, device, dtype, opt, sched):
    enc = encode(batch, tokenizer, t["max_length"], device)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        loss = model(**enc).loss
    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
    opt.step()
    sched.step()
    return loss.item(), grad_norm.item()


def load_inputs(cfg):
    config = load_config()
    calibration = json.loads((ROOT / "reports" / "calibration.json").read_text(encoding="utf-8"))
    pool, pool_sha = load_pool_checked(ROOT / cfg["pool"], ROOT / cfg["pool_sha256"])
    return config["seed"], settings(config, calibration), load_table(calibration), pool.to_dict("records"), pool_sha


def eval_sets(pool, table, damage, seed, batch_size):
    """val: every validation sentence with its fixed draw. probe: as many training sentences (seeded choice)
    with the draw "probe", which no training epoch uses."""
    val_rows = [r for r in pool if r["usage"] == "val"]
    train_rows = [r for r in pool if r["usage"] == "train"]
    probe_rows = random.Random(seed_int(seed, "bart_probe", "")).sample(train_rows, len(val_rows))
    val = epoch_examples(val_rows, table, damage, 0)
    probe = epoch_examples(probe_rows, table, damage, "probe")
    return batches_of(val, batch_size), batches_of(probe, batch_size)


def smoke_run(cfg):
    t, s = cfg["train"], cfg["smoke"]
    seed, damage, table, pool, _ = load_inputs(cfg)
    tokenizer, model, device, dtype = setup(cfg, seed)
    examples = epoch_examples(pool, table, damage, 0)
    train = [e for e in examples if e["usage"] == "train"][:s["n_train_examples"]]
    val = batches_of([e for e in examples if e["usage"] == "val"][:s["n_val_examples"]], s["eval_batch_size"])
    batches = batches_of(train, t["batch_size"])
    opt, sched = make_optimizer(model, t, s["steps"], s["steps"])
    rows = []
    for step in range(1, s["steps"] + 1):
        t0 = time.perf_counter()
        loss, _ = train_step(model, batches[(step - 1) % len(batches)], tokenizer, t, device, dtype, opt, sched)
        rows.append({"step": step, "train_loss": round(loss, 4), "seconds": round(time.perf_counter() - t0, 3)})
        print(f"step {step:2d}  loss {loss:.3f}")
    result, outputs = evaluate(model, val, tokenizer, t, device, dtype, keep_outputs=True)
    out_dir = ROOT / "models" / "bart_ft_smoke"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    reloaded = AutoModelForSeq2SeqLM.from_pretrained(out_dir).to(device)
    re_result, _ = evaluate(reloaded, val, tokenizer, t, device, dtype, generate=False)
    first, last = statistics.mean(r["train_loss"] for r in rows[:3]), statistics.mean(r["train_loss"] for r in rows[-3:])
    sec = statistics.median(r["seconds"] for r in rows[1:])
    steps_per_epoch = math.ceil(sum(e["usage"] == "train" for e in examples) / t["batch_size"])
    print(f"\ntrain loss {first:.3f} -> {last:.3f}; validation {result}; loss after reload {re_result['loss']:.4f}")
    print(f"example answer: {outputs[0]['output'][:200]}")
    print(f"{sec:.2f} s per step on {device.type}; one epoch = {steps_per_epoch} steps = about {sec * steps_per_epoch / 60:.1f} min")
    assert last < first, "the training loss did not go down"
    assert abs(re_result["loss"] - result["loss"]) < 1e-2, "the reloaded model gives a different loss"
    print("smoke run OK")


def log_row(epoch, step, kind, result, lr, seconds):
    return {"epoch": epoch, "step": step, "kind": kind, **{k: round(v, 5) if isinstance(v, float) else v for k, v in result.items()},
            "lr": lr, "seconds": seconds}


def full_run(cfg):
    t = cfg["train"]
    name = f"bart_ft_{cfg['version']}"
    out_dir, runs = ROOT / "models" / name, ROOT / "runs"
    seed, damage, table, pool, pool_sha = load_inputs(cfg)
    tokenizer, model, device, dtype = setup(cfg, seed)
    val, probe = eval_sets(pool, table, damage, seed, t["eval_batch_size"])
    n_train = sum(r["usage"] == "train" for r in pool) * len(LEVELS_FOR_TRAINING)
    steps_per_epoch = math.ceil(n_train / t["batch_size"])
    opt, sched = make_optimizer(model, t, steps_per_epoch, t["max_epochs"] * steps_per_epoch)
    started, t_start = datetime.now(timezone.utc).isoformat(timespec="seconds"), time.perf_counter()
    log = []
    for kind, batches in [("val", val), ("probe", probe)]:
        result, _ = evaluate(model, batches, tokenizer, t, device, dtype)
        log.append(log_row(0, 0, kind, result, 0.0, 0.0))
        print(f"pretrained model, {kind}: {result}")
    best, best_epoch, bad, step, stop = math.inf, None, 0, 0, None
    for epoch in range(1, t["max_epochs"] + 1):
        t0, running, norms = time.perf_counter(), [], []
        train = [e for e in epoch_examples(pool, table, damage, epoch - 1) if e["usage"] == "train"]
        for batch in batches_of(train, t["batch_size"], seed, epoch - 1):
            loss, norm = train_step(model, batch, tokenizer, t, device, dtype, opt, sched)
            running.append(loss)
            norms.append(norm)
            step += 1
            if step % t["log_every"] == 0:
                log.append(log_row(epoch, step, "train", {"loss": statistics.mean(running), "grad_norm": statistics.mean(norms)},
                                   sched.get_last_lr()[0], round(time.perf_counter() - t0, 1)))
                running, norms = [], []
        train_seconds = round(time.perf_counter() - t0, 1)
        result, outputs = evaluate(model, val, tokenizer, t, device, dtype, keep_outputs=True)
        probe_result, _ = evaluate(model, probe, tokenizer, t, device, dtype)
        seconds = round(time.perf_counter() - t0, 1)
        log.append(log_row(epoch, step, "val", result, sched.get_last_lr()[0], seconds))
        log.append(log_row(epoch, step, "probe", probe_result, sched.get_last_lr()[0], seconds))
        if result["loss"] < best:
            best, best_epoch, bad = result["loss"], epoch, 0
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            with (runs / f"{name}_val_outputs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                f.writelines(json.dumps({"epoch": epoch, **o}, ensure_ascii=False) + "\n" for o in outputs)
        else:
            bad += 1
        pd.DataFrame(log).to_csv(runs / f"{name}_log.csv", index=False, lineterminator="\n")
        print(f"epoch {epoch}: val loss {result['loss']:.4f}, exact {result['slot_exact']:.3f} "
              f"(probe loss {probe_result['loss']:.4f}, exact {probe_result['slot_exact']:.3f}); best loss {best:.4f} "
              f"at epoch {best_epoch}; train {train_seconds / 60:.1f} min, with evaluation {seconds / 60:.1f} min", flush=True)
        if bad >= t["patience"]:
            stop = f"early stop: no lower validation loss for {t['patience']} epochs"
            break
    stop = stop or f"max_epochs ({t['max_epochs']}) reached: validation loss may still fall (possible underfitting)"
    best_model = AutoModelForSeq2SeqLM.from_pretrained(out_dir).to(device)
    reloaded, _ = evaluate(best_model, val, tokenizer, t, device, dtype, generate=False)
    assert abs(reloaded["loss"] - best) < 1e-2, f"reloaded best model: loss {reloaded['loss']}, logged {best}"
    digest = sha256_file(out_dir / "model.safetensors")
    (runs / f"{name}.sha256").write_bytes(f"{digest}  {name}/model.safetensors\n".encode("utf-8"))
    commit = (ROOT / "POD_COMMIT").read_text().strip() if (ROOT / "POD_COMMIT").exists() else None
    snapshot = {**cfg, "seed": seed, "damage_settings": damage, "pool_hash": pool_sha, "model_sha256": digest,
                "code_commit": commit, "best_epoch": best_epoch, "best_val_loss": round(best, 5),
                "epochs_run": epoch, "steps": step, "steps_per_epoch": steps_per_epoch, "stop_reason": stop,
                "n_train_examples_per_epoch": n_train, "n_val_examples": sum(len(b) for b in val),
                "n_probe_examples": sum(len(b) for b in probe), "started": started,
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "minutes": round((time.perf_counter() - t_start) / 60, 1),
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else device.type, "bf16": dtype is not None,
                "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
                "versions": {p: version(p) for p in ("torch", "transformers")}
                | {"python": sys.version.split()[0], "cuda": torch.version.cuda}}
    (runs / f"{name}_config.yaml").write_bytes(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True).encode("utf-8"))
    print(f"{stop}. Best epoch {best_epoch}, validation loss {best:.4f}. Model sha256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "bart_ft.yaml"))
    parser.add_argument("--smoke", action="store_true", help="30-step check instead of the full run")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke_run(cfg) if args.smoke else full_run(cfg)
