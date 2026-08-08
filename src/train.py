"""Train the LSTM seq2seq summarizer.

Usage:
    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --override epochs=1 train_size=2000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BucketBatchSampler, SummarizationDataset, collate_batch
from src.data.vocab import PAD_ID, Vocab
from src.models.seq2seq import ModelConfig, Seq2Seq
from src.utils.device import get_device, hardware_summary
from src.utils.seed import set_seed


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        cfg[key.strip()] = yaml.safe_load(raw)
    return cfg


class LabelSmoothedCrossEntropy(nn.Module):
    """Cross entropy with label smoothing, ignoring padded target positions.

    Smoothing matters more than usual here: with a 50k vocabulary and only ~80k
    training pairs, an unsmoothed objective drives the model to overconfident
    predictions on frequent tokens and worsens the repetition failure mode.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = PAD_ID) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        target = target.reshape(-1)

        keep = target.ne(self.ignore_index)
        logits, target = logits[keep], target[keep]
        if target.numel() == 0:
            return logits.sum() * 0.0

        logprobs = torch.log_softmax(logits, dim=-1)
        nll = -logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -logprobs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


@torch.no_grad()
def evaluate_loss(model: Seq2Seq, loader: DataLoader, criterion, device) -> dict:
    model.eval()
    total_loss, total_nll, total_tokens = 0.0, 0.0, 0
    nll_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss = criterion(logits, batch["tgt_out"])
        ntok = int(batch["tgt_mask"].sum())
        total_loss += float(loss) * ntok
        total_nll += float(nll_fn(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1)))
        total_tokens += ntok

    return {
        "loss": total_loss / max(total_tokens, 1),
        "nll": total_nll / max(total_tokens, 1),
        "ppl": math.exp(min(total_nll / max(total_tokens, 1), 20)),
        "tokens": total_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(cfg.get("seed", 1234))
    device = get_device(args.device)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"config: {json.dumps(cfg, indent=2)}")

    vocab = Vocab.load(cfg["vocab_file"])
    max_src, max_tgt = cfg["max_src_len"], cfg["max_tgt_len"]

    train_ds = SummarizationDataset(cfg["train_file"], vocab, max_src, max_tgt)
    val_ds = SummarizationDataset(cfg["val_file"], vocab, max_src, max_tgt)

    # Optional sub-subsampling, used by the smoke test and by the data-scale ablation.
    if cfg.get("train_size") and cfg["train_size"] < len(train_ds):
        n = cfg["train_size"]
        train_ds.src, train_ds.tgt = train_ds.src[:n], train_ds.tgt[:n]
        train_ds.records = train_ds.records[:n]
    if cfg.get("val_size") and cfg["val_size"] < len(val_ds):
        n = cfg["val_size"]
        val_ds.src, val_ds.tgt = val_ds.src[:n], val_ds.tgt[:n]
        val_ds.records = val_ds.records[:n]

    train_sampler = BucketBatchSampler(
        train_ds.src_lengths(), cfg["batch_size"], shuffle=True, seed=cfg.get("seed", 1234)
    )
    val_sampler = BucketBatchSampler(
        val_ds.src_lengths(), cfg["batch_size"], shuffle=False, seed=cfg.get("seed", 1234)
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate_batch)

    model_cfg = ModelConfig.from_dict({**cfg, "vocab_size": len(vocab)})
    model = Seq2Seq(model_cfg).to(device)
    params = model.num_parameters()
    print(f"parameters: {json.dumps(params)}")

    criterion = LabelSmoothedCrossEntropy(cfg.get("label_smoothing", 0.1))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.get("lr_decay", 0.5), patience=0
    )

    history: list[dict] = []
    best_val = float("inf")
    bad_epochs = 0
    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        running, seen_tokens = 0.0, 0
        epoch_start = time.time()

        # When stdout is a log file rather than a terminal, throttle the bar so the
        # log stays readable instead of filling with carriage returns.
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{cfg['epochs']}",
            unit="batch",
            mininterval=1.0 if sys.stdout.isatty() else 60.0,
        )
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["tgt_out"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 5.0))
            optimizer.step()

            ntok = int(batch["tgt_mask"].sum())
            running += loss.detach().item() * ntok
            seen_tokens += ntok
            if step % 50 == 0:
                pbar.set_postfix(
                    loss=f"{running / max(seen_tokens,1):.3f}",
                    gnorm=f"{grad_norm.detach().item():.1f}",
                )

        train_loss = running / max(seen_tokens, 1)
        val_metrics = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_nll": round(val_metrics["nll"], 4),
            "val_ppl": round(val_metrics["ppl"], 2),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(epoch_time, 1),
        }
        history.append(record)
        print(json.dumps(record))

        if val_metrics["loss"] < best_val - 1e-4:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "config": cfg,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                out_dir / "best.pt",
            )
            print(f"  saved new best (val_loss={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.get("patience", 3):
                print(f"early stopping after {bad_epochs} epochs without improvement")
                break

    total_time = time.time() - t_start
    summary = {
        "run_name": cfg.get("run_name", out_dir.name),
        "config": cfg,
        "model_config": asdict(model_cfg),
        "parameters": params,
        "hardware": hardware_summary(),
        "device": str(device),
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "total_train_seconds": round(total_time, 1),
        "total_train_hours": round(total_time / 3600, 3),
        "best_val_loss": round(best_val, 4),
        "history": history,
    }
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\ntraining finished in {total_time/60:.1f} min; best val loss {best_val:.4f}")
    print(f"artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
