"""Run the LLM baseline over the shared head-to-head test set.

Usage:
    python -m src.llm.baseline --all                      # local MLX model (free)
    python -m src.llm.baseline --all --backend anthropic  # hosted API
    python -m src.llm.baseline --all --dry-run            # prompts only, no generation

Every system in this project is scored on `data/processed/test_llm.jsonl`.

Input parity
------------
By default the model receives the article truncated to the same `--max-src-words`
window the LSTM encoder sees. Giving the LLM the untruncated article while the
LSTM sees 400 tokens would confound "better model" with "more input", so the
matched-input condition is the headline number. `--full-article` runs the
unmatched condition as a separate, explicitly labelled setting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.data.dataset import read_jsonl
from src.llm.backends import GenResult, build_backend
from src.llm.prompts import VARIANTS, build_messages
from src.utils.seed import set_seed


def truncate_words(text: str, max_words: int | None) -> str:
    if max_words is None:
        return text
    return " ".join(text.split()[:max_words])


def sample_exemplars(train_path: str, k: int, seed: int, max_src_words: int | None) -> list[dict]:
    """Draw k few-shot exemplars from the *training* split.

    Drawing from train (never validation or test) keeps the few-shot setting free
    of test-set leakage, which matters because the report discusses contamination
    risk.
    """
    records = read_jsonl(train_path)
    rng = random.Random(seed)
    # Prefer mid-length articles: a 2000-word exemplar would dominate the prompt.
    pool = [r for r in records if 150 <= r["src_len"] <= 500]
    chosen = rng.sample(pool, k)
    return [
        {"article": truncate_words(r["article"], max_src_words), "summary": r["summary"]}
        for r in chosen
    ]


def completed_ids(path: Path) -> set[str]:
    """IDs already generated successfully in a previous run.

    Records carrying an `error` are deliberately excluded: a transient failure
    must be retried on resume, not treated as done. Treating it as done would
    silently drop that article from this setting, leaving systems scored on
    different subsets of the shared test set.
    """
    if not Path(path).exists():
        return set()
    return {r["id"] for r in read_jsonl(path) if "prediction" in r}


def prior_usage(path: Path, tracker: "UsageTracker") -> None:
    """Fold a previous run's usage into the tracker before resuming.

    Without this, an interrupted-then-resumed setting reports token counts, cost,
    and GPU-hours for only the final segment, understating the true totals that
    the report quotes.
    """
    if not Path(path).exists():
        return
    for rec in read_jsonl(path):
        if "prediction" not in rec:
            continue
        latency = rec.get("latency_s", 0.0)
        tracker.add(
            GenResult(
                text=rec.get("prediction", ""),
                input_tokens=rec.get("input_tokens", 0),
                output_tokens=rec.get("output_tokens", 0),
                latency_s=latency,
            )
        )
        # Wall-clock is measured from process start, so a resumed run would
        # otherwise report only this segment's GPU time.
        tracker.prior_seconds += latency


class UsageTracker:
    """Thread-safe accumulation of token usage, cost, and latency."""

    def __init__(self, backend) -> None:
        self._lock = threading.Lock()
        self.backend = backend
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0
        self.errors = 0
        self.latencies: list[float] = []
        self.wall_start = time.time()
        # Compute time carried over from a previous, interrupted run.
        self.prior_seconds = 0.0

    def add(self, res) -> None:
        with self._lock:
            if res.error:
                self.errors += 1
                return
            self.requests += 1
            self.input_tokens += res.input_tokens
            self.output_tokens += res.output_tokens
            self.latencies.append(res.latency_s)

    def summary(self) -> dict:
        lat = sorted(self.latencies)
        wall = (time.time() - self.wall_start) + self.prior_seconds
        cost = self.backend.cost_usd(self.input_tokens, self.output_tokens)
        out = {
            "requests": self.requests,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_clock_s": round(wall, 1),
            "latency_mean_s": round(sum(lat) / len(lat), 2) if lat else None,
            "latency_p50_s": round(lat[len(lat) // 2], 2) if lat else None,
            "latency_p95_s": round(lat[int(len(lat) * 0.95)], 2) if lat else None,
            "throughput_summaries_per_min": (
                round(self.requests / wall * 60, 1) if wall > 0 else None
            ),
        }
        if self.backend.kind == "api":
            out["cost_usd"] = round(cost, 4)
            out["cost_per_1k_summaries_usd"] = round(
                cost / max(self.requests, 1) * 1000, 2
            )
        else:
            # Local inference: the assignment asks for compute time instead of USD.
            out["cost_usd"] = 0.0
            out["gpu_hours"] = round(wall / 3600, 4)
            out["gpu_hours_per_1k_summaries"] = round(
                wall / max(self.requests, 1) * 1000 / 3600, 4
            )
        return out


def clean_output(text: str) -> str:
    """Strip preamble the model sometimes emits despite instructions.

    Deliberately minimal and applied identically to every setting, so it cannot
    flatter one prompt variant over another. How often such preambles occur is
    itself reported in the error analysis.
    """
    text = text.strip()
    for prefix in (
        "Here are the highlights:", "Here is a summary:", "Here's a summary:",
        "Summary:", "Highlights:",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def run_setting(
    backend,
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

    # Resume support: an interruption mid-sweep shouldn't cost a full rerun.
    if args.overwrite and out_path.exists():
        out_path.unlink()

    done_ids: set[str] = set() if args.overwrite else completed_ids(out_path)
    tracker = UsageTracker(backend)
    if done_ids:
        # Carry the earlier segment's tokens and latency forward, so the reported
        # totals describe the whole setting rather than just this run.
        prior_usage(out_path, tracker)
        print(f"  resuming: {len(done_ids)} already complete")

    todo = [r for r in records if r["id"] not in done_ids]
    write_lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("a", encoding="utf-8")
    progress = tqdm(
        total=len(todo), desc=setting_name, unit="doc",
        mininterval=1.0 if sys.stdout.isatty() else 30.0,
    )

    def record(rec: dict, res) -> None:
        tracker.add(res)
        payload = {"id": rec["id"]}
        if res.error:
            payload["error"] = res.error
        else:
            payload.update(
                {
                    "prediction": clean_output(res.text),
                    "raw": res.text,
                    "reference": rec["summary"],
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "latency_s": round(res.latency_s, 3),
                }
            )
        with write_lock:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh.flush()
        progress.update(1)

    if backend.supports_concurrency():
        def one(rec: dict) -> None:
            messages = build_messages(
                variant, truncate_words(rec["article"], max_src_words), exemplars
            )
            record(rec, backend.generate_one(variant.system, messages, args.max_tokens))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(one, r) for r in todo]):
                f.result()
    else:
        # Batched local generation: the GPU is the bottleneck, so parallelism
        # comes from the batch rather than from threads.
        bs = args.batch_size or getattr(backend, "batch_size", 8)
        for start in range(0, len(todo), bs):
            chunk = todo[start : start + bs]
            message_lists = [
                build_messages(
                    variant, truncate_words(r["article"], max_src_words), exemplars
                )
                for r in chunk
            ]
            results = backend.generate_batch(
                variant.system, message_lists, args.max_tokens
            )
            for rec, res in zip(chunk, results):
                record(rec, res)

    progress.close()
    fh.close()

    info = backend.info()
    meta = {
        "setting": setting_name,
        "backend": info.name,
        "backend_kind": info.kind,
        "model": info.model,
        "backend_details": info.details,
        "prompt_variant": variant.key,
        "prompt_variant_name": variant.name,
        "prompt_description": variant.description,
        "shots": shot_count,
        "input_condition": (
            "full_article" if args.full_article else f"truncated_{args.max_src_words}_words"
        ),
        "max_tokens": args.max_tokens,
        "system_prompt": variant.system,
        "user_template": variant.user_template,
        "few_shot_exemplar_summaries": [e["summary"] for e in exemplars],
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
    ap.add_argument("--backend", choices=["mlx", "anthropic"], default="mlx")
    ap.add_argument("--model", default=None, help="Override the backend's default model.")
    ap.add_argument("--test-file", default="data/processed/test_llm.jsonl")
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--out-dir", default="runs/llm")
    ap.add_argument("--max-src-words", type=int, default=400)
    ap.add_argument("--full-article", action="store_true",
                    help="Send the untruncated article (unmatched-input condition).")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8, help="API backend only.")
    ap.add_argument("--batch-size", type=int, default=None, help="Local backend only.")
    ap.add_argument("--limit", type=int, default=None, help="Cap examples (smoke tests).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the prompts without loading a model or calling an API.")
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
            msgs = build_messages(
                variant, truncate_words(records[0]["article"], max_src), exemplars
            )
            approx = (len(variant.system) + sum(len(m["content"]) for m in msgs)) // 4
            print(f"\n=== {name} === ({len(msgs)} messages, ~{approx} input tokens)")
            print(f"system: {variant.system[:110]}...")
            print(f"final user turn: {msgs[-1]['content'][:220]}...")
        print("\ndry run complete - nothing was generated")
        return

    backend = build_backend(args.backend, args.model, args.batch_size or 8)
    out_dir = Path(args.out_dir)
    all_meta = []

    for name in names:
        vk, shots = SETTINGS[name]
        suffix = "_fullarticle" if args.full_article else ""
        print(f"\n=== {name}{suffix} ({backend.info().model}) ===")
        meta = run_setting(
            backend, name + suffix, vk, shots, records,
            out_dir / f"{name}{suffix}.jsonl", args,
        )
        all_meta.append(meta)
        print(json.dumps(meta["usage"], indent=2))

    info = backend.info()
    summary_path = out_dir / "cost_summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    merged = {m["setting"]: m for m in previous.get("settings", [])}
    merged.update({m["setting"]: m for m in all_meta})
    settings = list(merged.values())

    total_cost = sum(s["usage"].get("cost_usd", 0.0) for s in settings)
    total_req = sum(s["usage"]["requests"] for s in settings)
    total_gpu_h = sum(s["usage"].get("gpu_hours", 0.0) for s in settings)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": info.name,
                "backend_kind": info.kind,
                "model": info.model,
                "backend_details": info.details,
                "price_per_mtok_input_usd": info.details.get("price_per_mtok_input_usd"),
                "price_per_mtok_output_usd": info.details.get("price_per_mtok_output_usd"),
                "settings": settings,
                "total_requests": total_req,
                "total_cost_usd": round(total_cost, 4),
                "total_gpu_hours": round(total_gpu_h, 4),
            },
            f,
            indent=2,
        )

    if info.kind == "api":
        print(f"\nTOTAL: {total_req} requests, ${total_cost:.4f}")
    else:
        print(f"\nTOTAL: {total_req} summaries, {total_gpu_h:.2f} GPU-hours, $0.00")


if __name__ == "__main__":
    main()
