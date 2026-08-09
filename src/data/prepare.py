"""Download CNN/DailyMail 3.0.0 and write the frozen splits this project uses.

Dataset: CNN/DailyMail 3.0.0 (Hermann et al., 2015; See et al., 2017),
distributed on the Hugging Face Hub as `abisee/cnn_dailymail` under Apache-2.0.

Split policy
------------
We use the dataset's *official* train/validation/test splits, which are
article-disjoint by construction, so there is no leakage across splits. The only
thing this script adds is deterministic subsampling for tractability:

  * train      - subsampled to `--train-size` (compute budget)
  * validation - subsampled to `--val-size`
  * test       - kept in full, for the LSTM's headline number
  * test_llm   - a fixed random subset of test, drawn once with `--seed`, on which
                 *every* system (LSTM, ablations, and all four LLM settings) is
                 scored. This is the identical test set the comparison requires;
                 it exists because scoring an API model on all 11,490 test
                 articles would cost far more than the assignment's budget.

Subsampling happens before any model development and is seeded, so the splits are
byte-identical on every machine.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.data.tokenizer import normalize, tokenize
from src.utils.seed import set_seed

DATASET_NAME = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_LICENSE = "Apache-2.0"
DATASET_CITATION = (
    "Hermann et al. (2015), 'Teaching Machines to Read and Comprehend'; "
    "See et al. (2017), 'Get To The Point: Summarization with Pointer-Generator Networks'"
)


def _clean_summary(highlights: str) -> str:
    """CNN/DailyMail reference summaries are newline-separated highlight bullets.

    We join them into a single space-separated string. ROUGE is computed on this
    same flattened form for every system, so the choice is neutral across models.
    """
    parts = [p.strip() for p in highlights.split("\n") if p.strip()]
    joined = " . ".join(p.rstrip(" .") for p in parts)
    return normalize(joined + " .")


def _to_record(example: dict, idx: int) -> dict | None:
    article = normalize(example["article"])
    summary = _clean_summary(example["highlights"])
    if not article or not summary:
        return None

    src_tokens = tokenize(article)
    tgt_tokens = tokenize(summary)
    # Guard against degenerate pairs that would only add noise to training.
    if len(src_tokens) < 30 or len(tgt_tokens) < 5:
        return None

    return {
        "id": example.get("id", f"idx-{idx}"),
        "article": article,
        "summary": summary,
        "src_len": len(src_tokens),
        "tgt_len": len(tgt_tokens),
    }


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _process_split(split_data, limit: int | None, seed: int, desc: str) -> list[dict]:
    n = len(split_data)
    if limit is not None and limit < n:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(n), limit))
    else:
        indices = range(n)

    records = []
    for idx in tqdm(indices, desc=desc, unit="doc"):
        rec = _to_record(split_data[idx], idx)
        if rec is not None:
            records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--train-size", type=int, default=80_000)
    ap.add_argument("--val-size", type=int, default=3_000)
    ap.add_argument(
        "--llm-test-size",
        type=int,
        default=500,
        help="Size of the shared head-to-head test subset scored by every system.",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)

    print(f"Loading {DATASET_NAME} ({DATASET_CONFIG})  license={DATASET_LICENSE}")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, cache_dir=args.cache_dir)

    train = _process_split(ds["train"], args.train_size, args.seed, "train")
    val = _process_split(ds["validation"], args.val_size, args.seed + 1, "val")
    test = _process_split(ds["test"], None, args.seed + 2, "test")

    # Head-to-head subset, drawn once from the processed test split.
    rng = random.Random(args.seed + 3)
    llm_idx = sorted(rng.sample(range(len(test)), min(args.llm_test_size, len(test))))
    test_llm = [test[i] for i in llm_idx]

    _write_jsonl(train, out_dir / "train.jsonl")
    _write_jsonl(val, out_dir / "validation.jsonl")
    _write_jsonl(test, out_dir / "test.jsonl")
    _write_jsonl(test_llm, out_dir / "test_llm.jsonl")

    def stats(recs: list[dict]) -> dict:
        return {
            "n": len(recs),
            "mean_src_tokens": round(sum(r["src_len"] for r in recs) / len(recs), 1),
            "mean_tgt_tokens": round(sum(r["tgt_len"] for r in recs) / len(recs), 1),
        }

    meta = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "seed": args.seed,
        "official_split_sizes": {k: len(v) for k, v in ds.items()},
        "splits": {
            "train": stats(train),
            "validation": stats(val),
            "test": stats(test),
            "test_llm": stats(test_llm),
        },
        "test_llm_indices": llm_idx,
    }
    with (out_dir / "dataset_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps({k: v for k, v in meta.items() if k != "test_llm_indices"}, indent=2))


if __name__ == "__main__":
    main()
