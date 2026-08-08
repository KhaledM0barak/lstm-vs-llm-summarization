"""Generate summaries from a trained LSTM checkpoint.

Usage:
    python -m src.generate --checkpoint runs/base/best.pt \
        --test-file data/processed/test_llm.jsonl \
        --out runs/base/preds_test_llm.jsonl --decode beam
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BucketBatchSampler, SummarizationDataset, collate_batch
from src.data.tokenizer import detokenize
from src.data.vocab import Vocab
from src.models.seq2seq import ModelConfig, Seq2Seq
from src.utils.device import get_device, hardware_summary
from src.utils.seed import set_seed


def load_model(checkpoint: str, device) -> tuple[Seq2Seq, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = Seq2Seq(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-file", default="data/processed/test_llm.jsonl")
    ap.add_argument("--vocab-file", default="data/processed/vocab.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument(
        "--no-block-trigram",
        action="store_true",
        help="Disable repeated-trigram blocking (reported as a decoding ablation).",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    vocab = Vocab.load(args.vocab_file)
    model, ckpt = load_model(args.checkpoint, device)
    cfg = ckpt["config"]

    dataset = SummarizationDataset(
        args.test_file, vocab, cfg["max_src_len"], cfg["max_tgt_len"]
    )
    sampler = BucketBatchSampler(
        dataset.src_lengths(), args.batch_size, shuffle=False, seed=args.seed
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_batch)

    block_trigram = not args.no_block_trigram
    results: dict[int, dict] = {}
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"generate ({args.decode})",
            unit="batch",
            mininterval=1.0 if sys.stdout.isatty() else 30.0,
        ):
            src = batch["src"].to(device)
            src_len = batch["src_len"]
            src_mask = batch["src_mask"].to(device)

            if args.decode == "greedy":
                token_ids, _ = model.generate_greedy(
                    src, src_len, src_mask,
                    max_len=args.max_len, min_len=args.min_len,
                    block_trigram=block_trigram,
                )
            else:
                token_ids = model.generate_beam(
                    src, src_len, src_mask,
                    beam_size=args.beam_size, max_len=args.max_len,
                    min_len=args.min_len, length_penalty=args.length_penalty,
                    block_trigram=block_trigram,
                )

            for row, idx in enumerate(batch["index"].tolist()):
                rec = dataset.records[idx]
                tokens = vocab.decode(token_ids[row])
                results[idx] = {
                    "id": rec["id"],
                    "prediction": detokenize(tokens),
                    "reference": rec["summary"],
                    "src_len": rec["src_len"],
                    "pred_tokens": len(tokens),
                }

    elapsed = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # Restore original file order; the bucket sampler reorders for efficiency.
        for idx in range(len(dataset)):
            f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")

    meta = {
        "checkpoint": args.checkpoint,
        "test_file": args.test_file,
        "decode": args.decode,
        "beam_size": args.beam_size if args.decode == "beam" else None,
        "length_penalty": args.length_penalty,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "block_trigram": block_trigram,
        "n_examples": len(dataset),
        "device": str(device),
        "hardware": hardware_summary(),
        "total_seconds": round(elapsed, 1),
        "seconds_per_example": round(elapsed / max(len(dataset), 1), 4),
        "parameters": model.num_parameters(),
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
