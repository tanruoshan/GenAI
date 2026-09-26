"""Fine-tune bert-base-cased as a masked language model on the BERT training pool (notebook 2, Part B).

Smoke run (local CPU is fine; 30 steps on 4 fixed batches, then save and reload):
    python scripts/train_bert.py --smoke

Full run on Google Colab (Runtime > Change runtime type > GPU), in a notebook cell:
    from google.colab import drive; drive.mount("/content/drive")
    !pip install -q pysbd rapidfuzz
    !python /content/drive/MyDrive/GenAI_proj/scripts/train_bert.py
If the session is cut off, run the last line again with --resume: it continues after the last
finished epoch.

Settings: configs/bert_ft.yaml; seed: configs/config.yaml. The pool must match
runs/bert_train_pool.sha256 (written by notebook 02). Full-run outputs: models/bert_ft_v1/ (the epoch
with the lowest validation loss), runs/bert_ft_v1_log.csv, runs/bert_ft_v1.sha256 and
runs/bert_ft_v1_config.yaml. Smoke-run outputs: models/bert_ft_smoke/ and runs/bert_ft_smoke_log.csv.
"""
import argparse
import math
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
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from blnrepair.bert_data import epoch_examples  # noqa: E402
from blnrepair.bert_train import (encode_batch, load_pool_checked, sha256_file, shuffled_batches,  # noqa: E402
                                  target_loss, validation_loss)
from blnrepair.data import load_config  # noqa: E402


def setup(cfg, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and cfg["train"]["fp16_on_gpu"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForMaskedLM.from_pretrained(cfg["model_name"]).to(device)
    model.train()
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}, mixed precision: {amp}")
    return tokenizer, model, device, amp


def make_optimizer(model, t, steps_per_epoch, total_steps):
    """AdamW with weight decay except on biases and LayerNorm weights (the usual BERT recipe);
    warmup over a share of one epoch, then linear decay to 0 over total_steps."""
    no_decay = ("bias", "LayerNorm.weight")
    params = list(model.named_parameters())
    groups = [{"params": [p for n, p in params if not n.endswith(no_decay)], "weight_decay": t["weight_decay"]},
              {"params": [p for n, p in params if n.endswith(no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=t["learning_rate"])
    sched = get_linear_schedule_with_warmup(opt, round(t["warmup_frac_of_epoch"] * steps_per_epoch), total_steps)
    return opt, sched


def train_step(model, batch, tokenizer, t, device, amp, opt, sched, scaler):
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        loss, _ = target_loss(model, encode_batch(batch, tokenizer, t["max_length"]), device)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
    scaler.step(opt)
    scaler.update()
    sched.step()
    return loss.item()


def examples_for(pool, seed, draw_epoch, tokenizer, usage):
    return [e for e in epoch_examples(pool, seed, draw_epoch, tokenizer.tokenize) if e["usage"] == usage]


def batches_of(examples, size):
    return [examples[i:i + size] for i in range(0, len(examples), size)]


def smoke_run(cfg, seed):
    """30 steps on 4 fixed training batches: the loss must go down. Then save, reload, and check
    the reloaded model gives the same validation loss. Prints the time per step."""
    t, s = cfg["train"], cfg["smoke"]
    pool, _ = load_pool_checked(ROOT / cfg["pool"], ROOT / cfg["pool_sha256"])
    tokenizer, model, device, amp = setup(cfg, seed)
    train = examples_for(pool, seed, 0, tokenizer, "train")[:s["n_train_examples"]]
    val = batches_of(examples_for(pool, seed, 0, tokenizer, "val")[:s["n_val_examples"]], t["eval_batch_size"])
    batches = shuffled_batches(train, t["batch_size"], seed, 0)
    opt, sched = make_optimizer(model, t, s["steps"], s["steps"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    val_before = validation_loss(model, val, tokenizer, t["max_length"], device, amp)
    rows = []
    for step in range(1, s["steps"] + 1):
        t0 = time.perf_counter()
        loss = train_step(model, batches[(step - 1) % len(batches)], tokenizer, t, device, amp, opt, sched, scaler)
        rows.append({"step": step, "train_loss": round(loss, 4), "seconds": round(time.perf_counter() - t0, 3)})
        print(f"step {step:2d}  loss {loss:.3f}")
    val_after = validation_loss(model, val, tokenizer, t["max_length"], device, amp)
    out_dir = ROOT / "models" / "bert_ft_smoke"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    reloaded = AutoModelForMaskedLM.from_pretrained(out_dir).to(device)
    val_reloaded = validation_loss(reloaded, val, tokenizer, t["max_length"], device, amp)
    pd.DataFrame(rows).to_csv(ROOT / "runs" / "bert_ft_smoke_log.csv", index=False, lineterminator="\n")

    first, last = statistics.mean(r["train_loss"] for r in rows[:3]), statistics.mean(r["train_loss"] for r in rows[-3:])
    sec = statistics.median(r["seconds"] for r in rows[1:])  # the first step includes warm-up costs
    steps_per_epoch = math.ceil(len(examples_for(pool, seed, 0, tokenizer, "train")) / t["batch_size"])
    print(f"\ntrain loss, mean of first 3 steps {first:.3f} -> last 3 steps {last:.3f} (must go down)")
    print(f"validation loss on {s['n_val_examples']} examples: before {val_before:.4f}, after {val_after:.4f}, "
          f"after reload {val_reloaded:.4f}")
    print(f"seconds per step (median, batch {t['batch_size']}, {device.type}): {sec:.2f}; "
          f"one epoch = {steps_per_epoch} steps = about {sec * steps_per_epoch / 60:.0f} min on this device")
    assert last < first, "the training loss did not go down"
    assert abs(val_reloaded - val_after) < 1e-4, "the reloaded model gives a different loss"
    print("smoke run OK")


def full_run(cfg, seed, resume):
    t = cfg["train"]
    name = f"bert_ft_{cfg['version']}"
    out_dir, state_path, runs = ROOT / "models" / name, ROOT / "models" / f"{name}_resume.pt", ROOT / "runs"
    pool, pool_sha = load_pool_checked(ROOT / cfg["pool"], ROOT / cfg["pool_sha256"])
    tokenizer, model, device, amp = setup(cfg, seed)
    val = batches_of(examples_for(pool, seed, 0, tokenizer, "val"), t["eval_batch_size"])  # fixed draw, any epoch
    steps_per_epoch = math.ceil(len(examples_for(pool, seed, 0, tokenizer, "train")) / t["batch_size"])
    opt, sched = make_optimizer(model, t, steps_per_epoch, t["max_epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    state = {"epochs_done": 0, "step": 0, "best": math.inf, "best_epoch": None, "bad": 0, "log": [], "stop": None,
             "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if resume and state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False)
        for obj, key in [(model, "model"), (opt, "opt"), (sched, "sched"), (scaler, "scaler")]:
            obj.load_state_dict(saved[key])
        state = saved["state"]
        print(f"resumed after epoch {state['epochs_done']}")
    else:
        pretrained = validation_loss(model, val, tokenizer, t["max_length"], device, amp)
        state["log"].append({"epoch": 0, "step": 0, "kind": "val", "loss": round(pretrained, 5), "lr": 0.0, "seconds": 0.0})
        print(f"validation loss of the pretrained model (reference): {pretrained:.4f}")

    while state["stop"] is None and state["epochs_done"] < t["max_epochs"]:
        epoch = state["epochs_done"] + 1  # the masking draw of this epoch is epoch - 1 (notebook 02 shows draw 0)
        t0, running = time.perf_counter(), []
        for batch in shuffled_batches(examples_for(pool, seed, epoch - 1, tokenizer, "train"), t["batch_size"], seed, epoch - 1):
            running.append(train_step(model, batch, tokenizer, t, device, amp, opt, sched, scaler))
            state["step"] += 1
            if state["step"] % t["log_every"] == 0:
                state["log"].append({"epoch": epoch, "step": state["step"], "kind": "train", "loss": round(statistics.mean(running), 5),
                                     "lr": sched.get_last_lr()[0], "seconds": round(time.perf_counter() - t0, 1)})
                running = []
        loss = validation_loss(model, val, tokenizer, t["max_length"], device, amp)
        seconds = round(time.perf_counter() - t0, 1)
        state["log"].append({"epoch": epoch, "step": state["step"], "kind": "val", "loss": round(loss, 5),
                             "lr": sched.get_last_lr()[0], "seconds": seconds})
        if loss < state["best"]:
            state.update(best=loss, best_epoch=epoch, bad=0)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        else:
            state["bad"] += 1
        if state["bad"] >= t["patience"]:
            state["stop"] = f"early stop: no lower validation loss for {t['patience']} epochs"
        state["epochs_done"] = epoch
        pd.DataFrame(state["log"]).to_csv(runs / f"{name}_log.csv", index=False, lineterminator="\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": state}, state_path)
        print(f"epoch {epoch}: validation loss {loss:.4f} (best {state['best']:.4f} at epoch {state['best_epoch']}), {seconds / 60:.1f} min")
    state["stop"] = state["stop"] or f"max_epochs ({t['max_epochs']}) reached"

    best_model = AutoModelForMaskedLM.from_pretrained(out_dir).to(device)
    reloaded = validation_loss(best_model, val, tokenizer, t["max_length"], device, amp)
    assert abs(reloaded - state["best"]) < 1e-3, f"reloaded best model: loss {reloaded}, logged {state['best']}"
    digest = sha256_file(out_dir / "model.safetensors")
    (runs / f"{name}.sha256").write_bytes(f"{digest}  {name}/model.safetensors\n".encode("utf-8"))
    snapshot = {**cfg, "seed": seed, "pool_hash": pool_sha, "model_sha256": digest, "best_epoch": state["best_epoch"],
                "best_val_loss": round(state["best"], 5), "pretrained_val_loss": state["log"][0]["loss"],
                "epochs_run": state["epochs_done"], "steps": state["step"], "steps_per_epoch": steps_per_epoch,
                "stop_reason": state["stop"], "started": state["started"],
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", "mixed_precision": amp,
                "versions": {p: version(p) for p in ("torch", "transformers", "accelerate")} | {"python": sys.version.split()[0]}}
    (runs / f"{name}_config.yaml").write_bytes(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True).encode("utf-8"))
    state_path.unlink()  # the resume file (about 1.3 GB) is no longer needed
    print(f"{state['stop']}. Best epoch {state['best_epoch']}, validation loss {state['best']:.4f}. Model sha256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "bert_ft.yaml"))
    parser.add_argument("--smoke", action="store_true", help="30-step check instead of the full run")
    parser.add_argument("--resume", action="store_true", help="continue a full run after the last finished epoch")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = load_config()["seed"]
    smoke_run(cfg, seed) if args.smoke else full_run(cfg, seed, args.resume)
