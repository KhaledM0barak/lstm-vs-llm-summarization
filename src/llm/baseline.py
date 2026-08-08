"""Run the Claude LLM baseline over the shared head-to-head test set.

Usage:
    python -m src.llm.baseline --setting A_zeroshot
    python -m src.llm.baseline --all
    python -m src.llm.baseline --all --dry-run     # no API calls; validates plumbing

Every system in this project is scored on `data/processed/test_llm.jsonl`.

Input parity
------------
By default the model receives the article truncated to the same `--max-src-words`
window the LSTM encoder sees. Giving the LLM the untruncated article while the
LSTM sees 400 tokens would confound "better model" with "more input", so the
matched-input condition is the headline number. `--full-article` runs the
unmatched condition as a separate, explicitly labelled setting; the report
discusses both.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.data.dataset import read_jsonl
from src.llm.prompts import VARIANTS, build_messages
from src.utils.seed import set_seed

MODEL = "claude-haiku-4-5"

# USD per million tokens, Claude Haiku 4.5 (input / output).
PRICE_PER_MTOK_IN = 1.00
PRICE_PER_MTOK_OUT = 5.00


def truncate_words(text: str, max_words: int | None) -> str:
    if max_words is None:
        return text
    words = text.split()
    return " ".join(words[:max_words])


def sample_exemplars(train_path: str, k: int, seed: int, max_src_words: int | None) -> list[dict]:
    """Draw k few-shot exemplars from the *training* split.

    Drawing them from train (never from validation or test) keeps the few-shot
    setting free of test-set leakage, which matters here because the report
    discusses contamination risk.
    """
    records = read_jsonl(train_path)
    rng = random.Random(seed)
    # Prefer mid-length articles: a 2000-word exemplar would dominate the prompt.
    pool = [r for r in records if 150 <= r["src_len"] <= 500]
    chosen = rng.sample(pool, k)
    return [
        {
            "article": truncate_words(r["article"], max_src_words),
            "summary": r["summary"],
        }
        for r in chosen
    ]


class UsageTracker:
    """Thread-safe accumulation of token usage, cost, and latency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.requests = 0
        self.errors = 0
        self.latencies: list[float] = []

    def add(self, usage, latency: float) -> None:
        with self._lock:
            self.requests += 1
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
            self.latencies.append(latency)

    def add_error(self) -> None:
        with self._lock:
            self.errors += 1

    def cost_usd(self) -> float:
        # Cache reads bill at ~0.1x and writes at ~1.25x; both are zero here
        # because the few-shot prefix is below Haiku 4.5's 4096-token cacheable
        # minimum, but the arithmetic is kept correct for other configurations.
        billed_in = self.input_tokens + 0.1 * self.cache_read_tokens + 1.25 * self.cache_write_tokens
        return (
            billed_in / 1e6 * PRICE_PER_MTOK_IN
            + self.output_tokens / 1e6 * PRICE_PER_MTOK_OUT
        )

    def summary(self) -> dict:
        lat = sorted(self.latencies)
        return {
            "requests": self.requests,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd(), 4),
            "cost_per_1k_summaries_usd": (
                round(self.cost_usd() / max(self.requests, 1) * 1000, 2)
            ),
            "latency_mean_s": round(sum(lat) / len(lat), 2) if lat else None,
            "latency_p50_s": round(lat[len(lat) // 2], 2) if lat else None,
            "latency_p95_s": round(lat[int(len(lat) * 0.95)], 2) if lat else None,
        }


def clean_output(text: str) -> str:
    """Strip preamble the model sometimes emits despite instructions.

    Kept deliberately minimal and symmetric across variants: this normalizes
    obvious wrappers only, so it cannot flatter one prompt over the other. The
    frequency of such preambles is itself reported in the error analysis.
    """
    text = text.strip()
    for prefix in ("Here are the highlights:", "Here is a summary:", "Summary:", "Highlights:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def run_setting(
    client,
    setting_name: str,
    variant_key: str,
    shot_count: int,
    records: list[dict],
    out_path: Path,
    args,
) -> dict:
    variant = VARIANTS[variant_key]
    max_src_words = None if args.full_article else args.max_src_words

    exemplars = (
        sample_exemplars(args.train_file, shot_count, args.seed, max_src_words)
        if shot_count
        else []
    )

    # Resume support: a rate-limit failure mid-sweep shouldn't cost a rerun.
    done_ids: set[str] = set()
    if out_path.exists() and not args.overwrite:
        for rec in read_jsonl(out_path):
            done_ids.add(rec["id"])
        if done_ids:
            print(f"  resuming: {len(done_ids)} already complete")

    todo = [r for r in records if r["id"] not in done_ids]
    tracker = UsageTracker()
    write_lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("a", encoding="utf-8")

    def one(rec: dict) -> None:
        article = truncate_words(rec["article"], max_src_words)
        messages = build_messages(variant, article, exemplars)
        t0 = time.time()
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=args.max_tokens,
                system=variant.system,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - logged and surfaced in the summary
            tracker.add_error()
            with write_lock:
                fh.write(
                    json.dumps({"id": rec["id"], "error": f"{type(exc).__name__}: {exc}"})
                    + "\n"
                )
                fh.flush()
            return

        latency = time.time() - t0
        tracker.add(resp.usage, latency)
        text = "".join(b.text for b in resp.content if b.type == "text")

        with write_lock:
            fh.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "prediction": clean_output(text),
                        "raw": text,
                        "reference": rec["summary"],
                        "stop_reason": resp.stop_reason,
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "latency_s": round(latency, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one, rec) for rec in todo]
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=setting_name,
                unit="doc",
                mininterval=1.0 if sys.stdout.isatty() else 30.0,
            ):
                pass
    fh.close()

    meta = {
        "setting": setting_name,
        "model": MODEL,
        "prompt_variant": variant.key,
        "prompt_variant_name": variant.name,
        "prompt_description": variant.description,
        "shots": shot_count,
        "input_condition": "full_article" if args.full_article else f"truncated_{args.max_src_words}_words",
        "max_tokens": args.max_tokens,
        "system_prompt": variant.system,
        "user_template": variant.user_template,
        "few_shot_exemplar_ids": [e["summary"][:60] for e in exemplars],
        "n_examples": len(records),
        "usage": tracker.summary(),
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


SETTINGS = {
    "A_zeroshot": ("A", 0),
    "A_fewshot": ("A", 4),
    "B_zeroshot": ("B", 0),
    "B_fewshot": ("B", 4),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setting", choices=sorted(SETTINGS))
    ap.add_argument("--all", action="store_true", help="Run all four settings.")
    ap.add_argument("--test-file", default="data/processed/test_llm.jsonl")
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--out-dir", default="runs/llm")
    ap.add_argument("--max-src-words", type=int, default=400)
    ap.add_argument(
        "--full-article",
        action="store_true",
        help="Send the untruncated article (unmatched-input condition).",
    )
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Cap examples (for smoke tests).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Validate prompts without calling the API.")
    args = ap.parse_args()

    if not args.setting and not args.all:
        ap.error("pass --setting NAME or --all")

    set_seed(args.seed)
    load_dotenv()

    records = read_jsonl(args.test_file)
    if args.limit:
        records = records[: args.limit]

    names = sorted(SETTINGS) if args.all else [args.setting]

    if args.dry_run:
        for name in names:
            vk, shots = SETTINGS[name]
            variant = VARIANTS[vk]
            max_src = None if args.full_article else args.max_src_words
            exemplars = (
                sample_exemplars(args.train_file, shots, args.seed, max_src) if shots else []
            )
            msgs = build_messages(variant, truncate_words(records[0]["article"], max_src), exemplars)
            approx = (len(variant.system) + sum(len(m["content"]) for m in msgs)) // 4
            print(f"\n=== {name} === ({len(msgs)} messages, ~{approx} input tokens)")
            print(f"system: {variant.system[:110]}...")
            print(f"final user turn: {msgs[-1]['content'][:220]}...")
        print("\ndry run complete - no API calls made")
        return

    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set.\n"
            "Create a .env file in the project root containing:\n"
            "    ANTHROPIC_API_KEY=sk-ant-...\n"
            "(.env is gitignored and never committed.)"
        )

    client = anthropic.Anthropic(max_retries=5)
    out_dir = Path(args.out_dir)
    all_meta = []

    for name in names:
        vk, shots = SETTINGS[name]
        suffix = "_fullarticle" if args.full_article else ""
        print(f"\n=== {name}{suffix} ===")
        meta = run_setting(
            client, name + suffix, vk, shots, records, out_dir / f"{name}{suffix}.jsonl", args
        )
        all_meta.append(meta)
        print(json.dumps(meta["usage"], indent=2))

    total_cost = sum(m["usage"]["cost_usd"] for m in all_meta)
    total_req = sum(m["usage"]["requests"] for m in all_meta)
    with (out_dir / "cost_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL,
                "price_per_mtok_input_usd": PRICE_PER_MTOK_IN,
                "price_per_mtok_output_usd": PRICE_PER_MTOK_OUT,
                "settings": all_meta,
                "total_requests": total_req,
                "total_cost_usd": round(total_cost, 4),
            },
            f,
            indent=2,
        )
    print(f"\nTOTAL: {total_req} requests, ${total_cost:.4f}")


if __name__ == "__main__":
    main()
