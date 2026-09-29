"""Fine-tune facebook/bart-base to repair damaged sentences (notebook 2c).

Smoke run (any device; 30 steps on 4 fixed batches, then save, reload and generate):
    python scripts/train_bart.py --smoke

Full run (GPU; on a RunPod pod, see scripts/pod/README.md):
    python scripts/train_bart.py

Settings: configs/bart_ft.yaml; seed and damage settings: configs/config.yaml and reports/calibration.json
(the generator of the evaluation data). The pool must match runs/bert_train_pool.sha256. Every epoch
damages every training sentence again at all five levels (new anchor, new damage); the validation sentences
keep one fixed damage draw. The checkpoint with the lowest validation loss is kept (early stopping).
Outputs: models/bart_ft_v1/, runs/bart_ft_v1_log.csv, runs/bart_ft_v1.sha256, runs/bart_ft_v1_config.yaml.
"""
import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from blnrepair.bart_data import epoch_examples, read_answer  # noqa: E402
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
def validation(model, batches, tokenizer, t, device, dtype, generate=True):
    """Mean loss per target token, and (generate=True) the share of slots whose greedy answer equals the
    gold word, with a wrong number of brackets counted as all slots wrong."""
    model.eval()
    total, n, right, slots = 0.0, 0, 0, 0
    for batch in batches:
        enc = encode(batch, tokenizer, t["max_length"], device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(**enc).logits
        labels = enc["labels"]
        total += torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum").item()
        n += int((labels != -100).sum())
        if generate:
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], num_beams=1,
                                 do_sample=False, max_length=t["max_length"])
            for e, text in zip(batch, tokenizer.batch_decode(out, skip_special_tokens=True)):
                words, gold = read_answer(text, e["k"]), read_answer(e["target"], e["k"])
                right += sum(w == g for w, g in zip(words, gold)) if words else 0
                slots += e["k"]
    model.train()
    return total / n, (right / slots if slots else None)


def setup(cfg, seed):
    torch.manual_seed(seed)
    device, dtype = device_and_dtype(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"]).to(device)
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
    opt.step()
    sched.step()
    return loss.item()


def load_inputs(cfg):
    config = load_config()
    calibration = json.loads((ROOT / "reports" / "calibration.json").read_text(encoding="utf-8"))
    pool, pool_sha = load_pool_checked(ROOT / cfg["pool"], ROOT / cfg["pool_sha256"])
    return config["seed"], settings(config, calibration), load_table(calibration), pool.to_dict("records"), pool_sha


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
        loss = train_step(model, batches[(step - 1) % len(batches)], tokenizer, t, device, dtype, opt, sched)
        rows.append({"step": step, "train_loss": round(loss, 4), "seconds": round(time.perf_counter() - t0, 3)})
        print(f"step {step:2d}  loss {loss:.3f}")
    val_loss, val_exact = validation(model, val, tokenizer, t, device, dtype)
    out_dir = ROOT / "models" / "bart_ft_smoke"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    reloaded = AutoModelForSeq2SeqLM.from_pretrained(out_dir).to(device)
    re_loss, _ = validation(reloaded, val, tokenizer, t, device, dtype, generate=False)
    first, last = statistics.mean(r["train_loss"] for r in rows[:3]), statistics.mean(r["train_loss"] for r in rows[-3:])
    sec = statistics.median(r["seconds"] for r in rows[1:])
    steps_per_epoch = math.ceil(sum(e["usage"] == "train" for e in examples) / t["batch_size"])
    print(f"\ntrain loss {first:.3f} -> {last:.3f}; validation loss {val_loss:.4f}, after reload {re_loss:.4f}; "
          f"validation slot exact {val_exact:.3f}")
    print(f"{sec:.2f} s per step on {device.type}; one epoch = {steps_per_epoch} steps = about {sec * steps_per_epoch / 60:.0f} min")
    assert last < first, "the training loss did not go down"
    assert abs(re_loss - val_loss) < 1e-3, "the reloaded model gives a different loss"
    print("smoke run OK")


def full_run(cfg):
    t = cfg["train"]
    name = f"bart_ft_{cfg['version']}"
    out_dir, runs = ROOT / "models" / name, ROOT / "runs"
    seed, damage, table, pool, pool_sha = load_inputs(cfg)
    tokenizer, model, device, dtype = setup(cfg, seed)
    val = batches_of([e for e in epoch_examples(pool, table, damage, 0) if e["usage"] == "val"], t["eval_batch_size"])
    n_train = sum(r["usage"] == "train" for r in pool) * 5
    steps_per_epoch = math.ceil(n_train / t["batch_size"])
    opt, sched = make_optimizer(model, t, steps_per_epoch, t["max_epochs"] * steps_per_epoch)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    loss, exact = validation(model, val, tokenizer, t, device, dtype)
    log = [{"epoch": 0, "step": 0, "kind": "val", "loss": round(loss, 5), "slot_exact": exact, "lr": 0.0, "seconds": 0.0}]
    print(f"pretrained model: validation loss {loss:.4f}, slot exact {exact:.3f}")
    best, best_epoch, bad, step, stop = math.inf, None, 0, 0, None
    for epoch in range(1, t["max_epochs"] + 1):
        t0, running = time.perf_counter(), []
        train = [e for e in epoch_examples(pool, table, damage, epoch - 1) if e["usage"] == "train"]
        for batch in batches_of(train, t["batch_size"], seed, epoch - 1):
            running.append(train_step(model, batch, tokenizer, t, device, dtype, opt, sched))
            step += 1
            if step % t["log_every"] == 0:
                log.append({"epoch": epoch, "step": step, "kind": "train", "loss": round(statistics.mean(running), 5),
                            "slot_exact": None, "lr": sched.get_last_lr()[0], "seconds": round(time.perf_counter() - t0, 1)})
                running = []
        loss, exact = validation(model, val, tokenizer, t, device, dtype)
        seconds = round(time.perf_counter() - t0, 1)
        log.append({"epoch": epoch, "step": step, "kind": "val", "loss": round(loss, 5), "slot_exact": round(exact, 5),
                    "lr": sched.get_last_lr()[0], "seconds": seconds})
        if loss < best:
            best, best_epoch, bad = loss, epoch, 0
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        else:
            bad += 1
        pd.DataFrame(log).to_csv(runs / f"{name}_log.csv", index=False, lineterminator="\n")
        print(f"epoch {epoch}: validation loss {loss:.4f}, slot exact {exact:.3f} (best loss {best:.4f} at epoch {best_epoch}), "
              f"{seconds / 60:.1f} min")
        if bad >= t["patience"]:
            stop = f"early stop: no lower validation loss for {t['patience']} epochs"
            break
    stop = stop or f"max_epochs ({t['max_epochs']}) reached"
    best_model = AutoModelForSeq2SeqLM.from_pretrained(out_dir).to(device)
    reloaded, _ = validation(best_model, val, tokenizer, t, device, dtype, generate=False)
    assert abs(reloaded - best) < 1e-2, f"reloaded best model: loss {reloaded}, logged {best}"
    digest = sha256_file(out_dir / "model.safetensors")
    (runs / f"{name}.sha256").write_bytes(f"{digest}  {name}/model.safetensors\n".encode("utf-8"))
    snapshot = {**cfg, "seed": seed, "damage_settings": damage, "pool_hash": pool_sha, "model_sha256": digest,
                "best_epoch": best_epoch, "best_val_loss": round(best, 5), "epochs_run": epoch, "steps": step,
                "steps_per_epoch": steps_per_epoch, "stop_reason": stop, "started": started,
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else device.type, "bf16": dtype is not None,
                "versions": {p: version(p) for p in ("torch", "transformers")} | {"python": sys.version.split()[0]}}
    (runs / f"{name}_config.yaml").write_bytes(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True).encode("utf-8"))
    print(f"{stop}. Best epoch {best_epoch}, validation loss {best:.4f}. Model sha256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "bart_ft.yaml"))
    parser.add_argument("--smoke", action="store_true", help="30-step check instead of the full run")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke_run(cfg) if args.smoke else full_run(cfg)
