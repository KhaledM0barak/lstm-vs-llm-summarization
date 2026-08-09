"""Build the shared source/target vocabulary from the training split only.

A shared vocabulary is standard for summarization: summary tokens are drawn from
the same distribution as article tokens, and sharing lets the encoder and decoder
embeddings be tied, which cuts parameters and helps a model this size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.data.dataset import read_jsonl
from src.data.tokenizer import tokenize
from src.data.vocab import Vocab
from src.utils.seed import set_seed


def _train_token_streams(records: list[dict]):
    for rec in tqdm(records, desc="counting tokens", unit="doc"):
        yield tokenize(rec["article"])
        yield tokenize(rec["summary"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--out", default="data/processed/vocab.json")
    ap.add_argument("--max-size", type=int, default=50_000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    records = read_jsonl(args.train_file)
    vocab = Vocab.build(
        _train_token_streams(records),
        max_size=args.max_size,
        min_freq=args.min_freq,
    )
    vocab.save(args.out)

    stats = vocab.coverage()
    stats["oov_token_rate_train"] = round(1.0 - stats["token_coverage"], 5)
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
