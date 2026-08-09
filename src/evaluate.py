"""Score every system on the shared test set and produce the report tables.

Usage:
    python -m src.evaluate \
        --test-file data/processed/test_llm.jsonl \
        --system lstm_beam=runs/base/preds_test_llm.jsonl \
        --system llm_B_zeroshot=runs/llm/B_zeroshot.jsonl \
        --out-dir results/

Reports ROUGE-1/2/L with bootstrap confidence intervals, the same metrics
bucketed by input length and by reference abstractiveness, and diagnostic rates
(repetition, out-of-vocabulary, extractive copying) used in the error analysis.
A Lead-3 baseline is computed automatically from the source articles: on
CNN/DailyMail it is a famously strong baseline, and without it a summarization
score is hard to interpret.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.data.dataset import read_jsonl
from src.data.tokenizer import split_sentences, tokenize
from src.data.vocab import Vocab
from src.utils.seed import set_seed

ROUGE_TYPES = ["rouge1", "rouge2", "rougeLsum"]

# Function words are excluded from the copy/novelty diagnostics: they overlap
# between any two English texts and would wash out the signal.
STOPWORDS = set(
    """a an the and or but if while of to in on at by for with from as is are was were be been
    being it its this that these those he she they them his her their you your i we our not no
    has have had do does did will would can could should may might must than then there here
    what which who whom when where how all any both each few more most other some such only own
    same so too very s t don now""".split()
)


def sentences(text: str) -> list[str]:
    return split_sentences(text)


def rougeLsum_ready(text: str) -> str:
    """rougeLsum expects sentences separated by newlines."""
    return "\n".join(sentences(text))


def lead3(article: str) -> str:
    return " ".join(sentences(article)[:3])


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t.isalnum() and t not in STOPWORDS]


def diagnostics(prediction: str, article: str, vocab: Vocab | None) -> dict:
    """Per-example diagnostic rates used to characterize failure modes."""
    pred_tokens = tokenize(prediction)
    art_tokens = set(tokenize(article))

    tri = ngrams(pred_tokens, 3)
    dup_tri = 0.0
    if tri:
        counts = Counter(tri)
        dup_tri = 1.0 - len(counts) / len(tri)

    bigrams = ngrams(pred_tokens, 2)
    art_bigrams = set(ngrams(tokenize(article), 2))
    novel_bi = (
        sum(1 for b in bigrams if b not in art_bigrams) / len(bigrams) if bigrams else 0.0
    )

    pred_content = content_tokens(prediction)
    unsupported = (
        sum(1 for t in pred_content if t not in art_tokens) / len(pred_content)
        if pred_content
        else 0.0
    )

    oov = 0.0
    if vocab is not None and pred_tokens:
        oov = sum(1 for t in pred_tokens if t not in vocab.stoi) / len(pred_tokens)

    return {
        "length_tokens": len(pred_tokens),
        "dup_trigram_rate": dup_tri,
        "novel_bigram_rate": novel_bi,
        "unsupported_content_rate": unsupported,
        "oov_rate": oov,
        "empty": 1.0 if not pred_tokens else 0.0,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 1234) -> tuple[float, float]:
    """Percentile bootstrap CI over per-example scores."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 1234,
) -> dict:
    """Paired bootstrap significance test on the per-example difference a - b.

    Two systems scored on the same articles produce *paired* observations, and
    per-example scores are strongly correlated across systems (a hard article is
    hard for everyone). Comparing two independent confidence intervals throws
    that pairing away: non-overlapping intervals do imply a real difference, but
    overlapping intervals do **not** imply the absence of one — the usual case
    where an independent-CI reading is too conservative.

    This resamples article indices once per replicate and applies the same
    resample to both systems, so the correlation is preserved. The reported
    p-value is two-sided, for the null hypothesis that the mean difference is
    zero, computed by centring the bootstrap distribution on that null.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired test needs equal-length inputs, got {a.shape} and {b.shape}")

    n = len(a)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"), "p_value": float("nan")}

    diff = a - b
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    # One index draw per replicate, applied to both systems: this is what makes
    # the test paired.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided p-value: how often a null-centred bootstrap difference is at
    # least as extreme as the one observed.
    centred = boot - observed
    p = float((np.abs(centred) >= abs(observed)).mean())

    return {
        "n": n,
        "mean_diff": round(observed * 100, 3),
        "ci_low": round(float(lo) * 100, 3),
        "ci_high": round(float(hi) * 100, 3),
        "p_value": round(p, 4),
        "significant_at_05": bool(p < 0.05),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def compare_systems(
    scored: dict[str, dict],
    reference_system: str,
    metrics=("rouge1", "rouge2", "rougeLsum"),
    seed: int = 1234,
) -> dict:
    """Paired-bootstrap every system against one reference system.

    Only examples scored by *both* systems are used, so the pairing is genuine
    even if a system is missing predictions for some articles.
    """
    if reference_system not in scored:
        return {}

    ref_rows = {r["id"]: r for r in scored[reference_system]["per_example"]}
    out: dict[str, dict] = {}

    for name, s in scored.items():
        if name == reference_system:
            continue
        rows = {r["id"]: r for r in s["per_example"]}
        shared = sorted(set(ref_rows) & set(rows))
        if not shared:
            continue
        out[name] = {"vs": reference_system, "n_paired": len(shared)}
        for metric in metrics:
            a = np.array([rows[i][metric] for i in shared])
            b = np.array([ref_rows[i][metric] for i in shared])
            out[name][metric] = paired_bootstrap(a, b, seed=seed)
    return out


def score_system(
    name: str,
    preds: dict[str, str],
    references: dict[str, str],
    articles: dict[str, str],
    scorer: rouge_scorer.RougeScorer,
    vocab: Vocab | None,
) -> dict:
    ids = [i for i in references if i in preds]
    per_example = []

    for ex_id in tqdm(ids, desc=f"score {name}", unit="doc", leave=False):
        pred, ref = preds[ex_id], references[ex_id]
        scores = scorer.score(rougeLsum_ready(ref), rougeLsum_ready(pred))
        row = {"id": ex_id}
        for rt in ROUGE_TYPES:
            row[rt] = scores[rt].fmeasure
        row.update(diagnostics(pred, articles[ex_id], vocab))
        per_example.append(row)

    return {"name": name, "n": len(per_example), "per_example": per_example}


def aggregate(per_example: list[dict], seed: int = 1234) -> dict:
    out = {}
    for key in ROUGE_TYPES:
        vals = np.array([r[key] for r in per_example], dtype=float)
        lo, hi = bootstrap_ci(vals, seed=seed)
        out[key] = {
            "mean": round(float(vals.mean()) * 100, 2),
            "ci_low": round(lo * 100, 2),
            "ci_high": round(hi * 100, 2),
        }
    for key in (
        "length_tokens",
        "dup_trigram_rate",
        "novel_bigram_rate",
        "unsupported_content_rate",
        "oov_rate",
        "empty",
    ):
        vals = np.array([r[key] for r in per_example], dtype=float)
        out[key] = round(float(vals.mean()), 4)
    return out


def assign_buckets(records: list[dict], references: dict[str, str]) -> dict[str, dict[str, str]]:
    """Two bucketings for the consistency analysis the report requires.

    length      - source length terciles; tests whether the gap widens as the
                  recurrent encoder's bottleneck gets tighter.
    abstractive - terciles of the reference's novel-bigram rate against the
                  article; a proxy for how much genuine abstraction the example
                  demands rather than copying.
    """
    lengths = np.array([r["src_len"] for r in records], dtype=float)
    lo_len, hi_len = np.percentile(lengths, [33.3, 66.7])

    novelty = {}
    for rec in records:
        ref_bi = ngrams(tokenize(references[rec["id"]]), 2)
        art_bi = set(ngrams(tokenize(rec["article"]), 2))
        novelty[rec["id"]] = (
            sum(1 for b in ref_bi if b not in art_bi) / len(ref_bi) if ref_bi else 0.0
        )
    nov_vals = np.array(list(novelty.values()))
    lo_nov, hi_nov = np.percentile(nov_vals, [33.3, 66.7])

    buckets: dict[str, dict[str, str]] = {"length": {}, "abstractiveness": {}}
    for rec in records:
        n = rec["src_len"]
        buckets["length"][rec["id"]] = (
            f"short (<={int(lo_len)} tok)"
            if n <= lo_len
            else f"medium ({int(lo_len)}-{int(hi_len)} tok)"
            if n <= hi_len
            else f"long (>{int(hi_len)} tok)"
        )
        v = novelty[rec["id"]]
        buckets["abstractiveness"][rec["id"]] = (
            "extractive (low novelty)"
            if v <= lo_nov
            else "mixed"
            if v <= hi_nov
            else "abstractive (high novelty)"
        )
    return buckets


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(c[1] for c in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c[0], "")) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-file", default="data/processed/test_llm.jsonl")
    ap.add_argument("--vocab-file", default="data/processed/vocab.json")
    ap.add_argument(
        "--system",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Prediction file. Repeatable.",
    )
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-lead3", action="store_true")
    ap.add_argument(
        "--paired-reference",
        default="lstm_beam",
        help="System every other system is paired-bootstrap tested against.",
    )
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    records = read_jsonl(args.test_file)
    references = {r["id"]: r["summary"] for r in records}
    articles = {r["id"]: r["article"] for r in records}
    vocab = Vocab.load(args.vocab_file) if Path(args.vocab_file).exists() else None

    systems: dict[str, dict[str, str]] = {}
    for spec in args.system:
        if "=" not in spec:
            ap.error(f"--system expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        preds = {}
        for rec in read_jsonl(path):
            if "prediction" in rec:
                preds[rec["id"]] = rec["prediction"]
        systems[name] = preds
        print(f"loaded {name}: {len(preds)} predictions from {path}")

    if not args.no_lead3:
        systems["lead3_baseline"] = {r["id"]: lead3(r["article"]) for r in records}
        print(f"computed lead3_baseline: {len(records)} predictions")

    if not systems:
        ap.error("no systems given")

    # `use_stemmer=True` is the standard configuration for CNN/DailyMail ROUGE.
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    buckets = assign_buckets(records, references)

    scored = {
        name: score_system(name, preds, references, articles, scorer, vocab)
        for name, preds in systems.items()
    }

    results = {
        "test_file": args.test_file,
        "n_test": len(records),
        "rouge_config": {"types": ROUGE_TYPES, "use_stemmer": True, "bootstrap_n": 1000},
        "overall": {},
        "by_bucket": {},
        "coverage": {},
    }

    for name, s in scored.items():
        results["coverage"][name] = {"scored": s["n"], "expected": len(records)}
        results["overall"][name] = aggregate(s["per_example"], seed=args.seed)

    # Paired significance tests. Resolve the reference in decreasing order of
    # specificity, and fall back to the first system the user named rather than
    # silently skipping the tests when the default name is absent.
    ref = None
    if args.paired_reference in scored:
        ref = args.paired_reference
    else:
        for candidate in ("lstm_beam", "lstm", "base"):
            if candidate in scored:
                ref = candidate
                break
        if ref is None:
            user_named = [n for n in scored if n != "lead3_baseline"]
            if user_named:
                ref = user_named[0]
                print(
                    f"note: --paired-reference '{args.paired_reference}' not among the "
                    f"scored systems; pairing against '{ref}' instead"
                )

    if ref and len(scored) > 1:
        results["paired_vs"] = ref
        results["paired_tests"] = compare_systems(scored, ref, seed=args.seed)
    else:
        print("note: paired tests skipped (need at least two scored systems)")

    for bucket_kind, mapping in buckets.items():
        results["by_bucket"][bucket_kind] = {}
        bucket_names = sorted(set(mapping.values()))
        for bname in bucket_names:
            results["by_bucket"][bucket_kind][bname] = {}
            for name, s in scored.items():
                subset = [r for r in s["per_example"] if mapping.get(r["id"]) == bname]
                if subset:
                    results["by_bucket"][bucket_kind][bname][name] = aggregate(
                        subset, seed=args.seed
                    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (out_dir / "per_example_scores.json").open("w", encoding="utf-8") as f:
        json.dump({n: s["per_example"] for n, s in scored.items()}, f)

    # ---- markdown report tables ----
    lines = ["# Results\n", f"Test set: `{args.test_file}` ({len(records)} examples)\n"]
    lines.append("\n## Overall (ROUGE F1 x100, 95% bootstrap CI)\n")
    rows = []
    for name, agg in results["overall"].items():
        rows.append(
            {
                "system": name,
                "n": results["coverage"][name]["scored"],
                "r1": f"{agg['rouge1']['mean']} [{agg['rouge1']['ci_low']}, {agg['rouge1']['ci_high']}]",
                "r2": f"{agg['rouge2']['mean']} [{agg['rouge2']['ci_low']}, {agg['rouge2']['ci_high']}]",
                "rl": f"{agg['rougeLsum']['mean']} [{agg['rougeLsum']['ci_low']}, {agg['rougeLsum']['ci_high']}]",
                "len": agg["length_tokens"],
            }
        )
    rows.sort(key=lambda r: float(r["r1"].split()[0]), reverse=True)
    lines.append(
        markdown_table(
            rows,
            [("system", "System"), ("n", "N"), ("r1", "ROUGE-1"),
             ("r2", "ROUGE-2"), ("rl", "ROUGE-Lsum"), ("len", "Len")],
        )
    )

    lines.append("\n## Diagnostics (means)\n")
    drows = [
        {
            "system": name,
            "dup3": agg["dup_trigram_rate"],
            "novel2": agg["novel_bigram_rate"],
            "unsup": agg["unsupported_content_rate"],
            "oov": agg["oov_rate"],
            "empty": agg["empty"],
        }
        for name, agg in results["overall"].items()
    ]
    lines.append(
        markdown_table(
            drows,
            [("system", "System"), ("dup3", "Dup-trigram"), ("novel2", "Novel-bigram"),
             ("unsup", "Unsupported content"), ("oov", "OOV rate"), ("empty", "Empty")],
        )
    )

    if results.get("paired_tests"):
        ref = results["paired_vs"]
        lines.append(
            f"\n## Paired bootstrap vs. `{ref}`\n\n"
            "Per-example differences, resampling articles once per replicate and "
            "applying the same resample to both systems (10,000 replicates). A "
            "positive difference means the system beats "
            f"`{ref}`. Unlike comparing two independent CIs, this test accounts "
            "for the fact that a hard article is hard for every system.\n"
        )
        prows = []
        for name, t in results["paired_tests"].items():
            r1 = t["rouge1"]
            r2 = t["rouge2"]
            prows.append({
                "system": name,
                "n": t["n_paired"],
                "d1": f"{r1['mean_diff']:+.2f} [{r1['ci_low']:+.2f}, {r1['ci_high']:+.2f}]",
                "p1": f"{r1['p_value']:.4f}" + ("" if r1["significant_at_05"] else " (n.s.)"),
                "d2": f"{r2['mean_diff']:+.2f} [{r2['ci_low']:+.2f}, {r2['ci_high']:+.2f}]",
                "p2": f"{r2['p_value']:.4f}" + ("" if r2["significant_at_05"] else " (n.s.)"),
                "wl": f"{r1['wins']}/{r1['losses']}",
            })
        lines.append(markdown_table(
            prows,
            [("system", "System"), ("n", "N"), ("d1", "Δ ROUGE-1 [95% CI]"),
             ("p1", "p"), ("d2", "Δ ROUGE-2 [95% CI]"), ("p2", "p"),
             ("wl", "W/L")],
        ))

    for bucket_kind in results["by_bucket"]:
        lines.append(f"\n## ROUGE-1 by {bucket_kind}\n")
        bnames = sorted(results["by_bucket"][bucket_kind])
        brows = []
        for name in results["overall"]:
            row = {"system": name}
            for bname in bnames:
                cell = results["by_bucket"][bucket_kind][bname].get(name)
                row[bname] = cell["rouge1"]["mean"] if cell else "-"
            brows.append(row)
        lines.append(
            markdown_table(brows, [("system", "System")] + [(b, b) for b in bnames])
        )

    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out_dir/'results.json'} and {out_dir/'results.md'}")


if __name__ == "__main__":
    main()
