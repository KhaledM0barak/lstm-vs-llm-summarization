"""Record the LLM's responses for the demo's fixed articles.

The live baseline needs Apple silicon, mlx-lm, and a 4.5 GB model download. The
demo always shows the same handful of articles, so a machine that only needs to
*display* the comparison can replay recorded responses instead -- which is what
lets a teammate record the video without the model.

Responses are produced by `src.demo.LLMSummarizer`, the same object a live run
uses, on the same 400-word window `src.demo` would show. So a replayed demo is
byte-identical to a live one; only the latency is a recording rather than a
measurement.

Run this on a machine with the live backend, then commit the cache:

    python scripts/build_llm_cache.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

from src.data.dataset import read_jsonl
from src.data.tokenizer import normalize
from src.demo import LLMSummarizer, cache_key, truncate_words

# Every article the walkthrough and the review packets show.
TEST_EXAMPLES = [3, 4, 112]
FILES = ["examples/demo_article_battery.txt"]


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/llm_cache.json")
    ap.add_argument("--test-file", default="data/processed/test_llm.jsonl")
    ap.add_argument("--llm-model", default="mlx-community/Llama-3.1-8B-Instruct-4bit")
    ap.add_argument("--max-src-words", type=int, default=400)
    args = ap.parse_args()

    records = read_jsonl(args.test_file)
    articles: list[tuple[str, str]] = []
    for i in TEST_EXAMPLES:
        articles.append((f"test example {i}", records[i]["article"]))
    for f in FILES:
        # `--file` normalizes before truncating; the cache key must match exactly.
        articles.append((f, normalize(Path(f).read_text(encoding="utf-8"))))

    print(f"Loading {args.llm_model} ...", flush=True)
    llm = LLMSummarizer(args.llm_model)

    # Rebuilt from scratch, not merged: an entry whose article has since been
    # edited would otherwise linger under a key nothing can reach.
    out_path = Path(args.out)
    entries: dict[str, dict] = {}

    for label, raw in articles:
        # Exactly what src.demo passes to the model.
        shown = truncate_words(raw, args.max_src_words)
        key = cache_key(shown)
        print(f"  {label:<40} {key[:12]} ...", end="", flush=True)
        prediction, latency = llm.summarize(shown)
        entries[key] = {
            "source": label,
            "prediction": prediction,
            "latency_s": round(latency, 3),
        }
        print(f" {latency:5.2f}s  {len(prediction.split()):3d} words")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.llm_model,
        "prompt_variant": "B (style-matched, zero-shot)",
        "sampling": "greedy",
        "max_src_words": args.max_src_words,
        "recorded_on": date.today().isoformat(),
        "recorded_at_commit": git_rev(),
        "entries": entries,
    }, indent=2) + "\n")
    print(f"\nWrote {out_path}  ({len(entries)} entries, {out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
