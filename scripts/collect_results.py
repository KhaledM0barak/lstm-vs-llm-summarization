"""Gather every measured number from the run artifacts into report tables.

Usage:
    python scripts/collect_results.py

Reads whatever exists under `runs/`, `results/`, and `data/processed/`, and
writes `reports/tables.md` (ready to paste into the report) plus
`reports/report_data.json`. Missing artifacts are reported as missing rather
than guessed, so a table in the report can only ever contain a number that some
committed script actually produced.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Display names, in the order they should appear in the report.
SYSTEM_ORDER = [
    ("lead3_baseline", "Lead-3 baseline"),
    ("lstm_beam", "LSTM + attention (beam 4)"),
    ("lstm_greedy", "LSTM + attention (greedy)"),
    ("lstm_beam_norepeat", "LSTM + attention (beam, no trigram block)"),
    ("no_attention", "— ablation: no attention"),
    ("unidirectional", "— ablation: unidirectional encoder"),
    ("short_context", "— ablation: 100-token encoder window"),
    ("llm_A_zeroshot", "LLM variant A (plain), zero-shot"),
    ("llm_A_fewshot", "LLM variant A (plain), few-shot k=4"),
    ("llm_B_zeroshot", "LLM variant B (style-matched), zero-shot"),
    ("llm_B_fewshot", "LLM variant B (style-matched), few-shot k=4"),
    ("llm_B_zeroshot_fullarticle", "LLM variant B, zero-shot, full article"),
]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fmt_rouge(agg: dict, key: str) -> str:
    m = agg[key]
    return f"{m['mean']:.2f} [{m['ci_low']:.2f}, {m['ci_high']:.2f}]"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    missing: list[str] = []
    data: dict = {}
    md: list[str] = ["# Generated result tables\n",
                     "Every number below is read from a run artifact. "
                     "Regenerate with `python scripts/collect_results.py`.\n"]

    # ---------------------------------------------------------------- dataset
    meta = load_json(DATA / "dataset_meta.json")
    vstats = load_json(DATA / "vocab.stats.json")
    if meta:
        data["dataset"] = {k: v for k, v in meta.items() if k != "test_llm_indices"}
        md.append("\n## Table 1 — Dataset\n")
        rows = [
            [name, f"{s['n']:,}", s["mean_src_tokens"], s["mean_tgt_tokens"]]
            for name, s in meta["splits"].items()
        ]
        md.append(table(["Split", "Documents", "Mean source tokens", "Mean summary tokens"], rows))
        md.append(f"\nDataset: `{meta['dataset']}` ({meta['config']}), license **{meta['license']}**.")
        md.append(f"Official split sizes: {meta['official_split_sizes']}.")
    else:
        missing.append("data/processed/dataset_meta.json (run src.data.prepare)")

    if vstats:
        data["vocab"] = vstats
        md.append(
            f"\nVocabulary: {vstats['vocab_size']:,} types from "
            f"{vstats['distinct_tokens_in_train']:,} distinct training tokens; "
            f"token coverage **{vstats['token_coverage']*100:.2f}%** "
            f"(**{vstats['oov_token_rate_train']*100:.2f}%** OOV)."
        )
    else:
        missing.append("data/processed/vocab.stats.json (run src.data.build_vocab)")

    # ---------------------------------------------------------------- training
    train_rows = []
    data["training"] = {}
    for run_dir in sorted(RUNS.glob("*/train_summary.json")):
        s = load_json(run_dir)
        if not s or s.get("run_name") == "smoke":
            continue
        name = s["run_name"]
        data["training"][name] = s
        best = min((h["val_loss"] for h in s["history"]), default=None)
        train_rows.append([
            name,
            f"{s['parameters']['total']:,}",
            len(s["history"]),
            f"{s['total_train_hours']:.2f}",
            f"{best:.4f}" if best is not None else "-",
            f"{min((h['val_ppl'] for h in s['history']), default=float('nan')):.1f}",
        ])
    if train_rows:
        md.append("\n## Table 2 — Training runs\n")
        md.append(table(
            ["Run", "Parameters", "Epochs run", "Train hours", "Best val loss", "Best val PPL"],
            train_rows,
        ))
        any_run = next(iter(data["training"].values()))
        hw = any_run["hardware"]
        md.append(
            f"\nHardware: {hw.get('cpu_model', hw.get('processor'))}, "
            f"{hw.get('ram_gb', '?')} GB RAM, GPU: {hw.get('gpu')}; "
            f"device `{any_run['device']}`, torch {hw.get('torch')}, Python {hw.get('python')}."
        )
    else:
        missing.append("runs/*/train_summary.json (run scripts/train_all.sh)")

    # ---------------------------------------------------------------- ROUGE
    # Tables are numbered off a running counter so a section that does not get
    # emitted (missing artifact) leaves no gap and no duplicate.
    table_no = 3
    res = load_json(RESULTS / "results.json")
    if res:
        data["results"] = res
        present = [(k, label) for k, label in SYSTEM_ORDER if k in res["overall"]]
        present += [(k, k) for k in res["overall"] if k not in dict(SYSTEM_ORDER)]

        md.append(f"\n## Table 3 — Main results (ROUGE F1 x100, 95% bootstrap CI, n={res['n_test']})\n")
        md.append(table(
            ["System", "ROUGE-1", "ROUGE-2", "ROUGE-Lsum", "Mean length"],
            [[label,
              fmt_rouge(res["overall"][k], "rouge1"),
              fmt_rouge(res["overall"][k], "rouge2"),
              fmt_rouge(res["overall"][k], "rougeLsum"),
              f"{res['overall'][k]['length_tokens']:.1f}"] for k, label in present],
        ))

        table_no += 1
        md.append(f"\n## Table {table_no} — Behavioral diagnostics (means)\n")
        md.append(table(
            ["System", "Dup-trigram", "Novel-bigram", "Unsupported content", "OOV rate", "Empty"],
            [[label,
              f"{res['overall'][k]['dup_trigram_rate']:.3f}",
              f"{res['overall'][k]['novel_bigram_rate']:.3f}",
              f"{res['overall'][k]['unsupported_content_rate']:.3f}",
              f"{res['overall'][k]['oov_rate']:.3f}",
              f"{res['overall'][k]['empty']:.3f}"] for k, label in present],
        ))

        # Numbered off a counter: there is one bucket table per bucket kind, so a
        # hard-coded number gives every one of them the same label.
        for bucket_kind, buckets in res["by_bucket"].items():
            bnames = sorted(buckets)
            table_no += 1
            md.append(f"\n## Table {table_no} — ROUGE-1 by {bucket_kind}\n")
            md.append(table(
                ["System"] + bnames,
                [[label] + [
                    f"{buckets[b][k]['rouge1']['mean']:.2f}" if k in buckets[b] else "-"
                    for b in bnames
                ] for k, label in present],
            ))

        # The headline comparison, computed rather than eyeballed.
        lstm = res["overall"].get("lstm_beam")
        llms = {k: v for k, v in res["overall"].items() if k.startswith("llm_")}
        if lstm and llms:
            best_llm_key = max(llms, key=lambda k: llms[k]["rouge1"]["mean"])
            best = llms[best_llm_key]
            gaps = {
                m: round(best[m]["mean"] - lstm[m]["mean"], 2)
                for m in ("rouge1", "rouge2", "rougeLsum")
            }
            overlap = not (
                best["rouge1"]["ci_low"] > lstm["rouge1"]["ci_high"]
                or lstm["rouge1"]["ci_low"] > best["rouge1"]["ci_high"]
            )
            data["headline_gap"] = {
                "lstm": "lstm_beam",
                "best_llm": best_llm_key,
                "absolute_gap": gaps,
                "ci_overlap_rouge1": overlap,
            }
            md.append("\n## Headline gap\n")
            md.append(
                f"Best LLM setting is `{best_llm_key}`. Absolute ROUGE gap over the LSTM: "
                f"R1 {gaps['rouge1']:+.2f}, R2 {gaps['rouge2']:+.2f}, "
                f"R-Lsum {gaps['rougeLsum']:+.2f}. "
                f"95% CIs on ROUGE-1 {'overlap' if overlap else 'do not overlap'}."
            )
    else:
        missing.append("results/results.json (run src.evaluate)")

    # ---------------------------------------------------------------- cost
    cost = load_json(RUNS / "llm" / "cost_summary.json")
    if cost:
        data["llm_cost"] = cost
        is_local = cost.get("backend_kind") == "local"
        unit_col = "GPU-h / 1k summaries" if is_local else "$ / 1k summaries"

        table_no += 1
        md.append(f"\n## Table {table_no} — LLM baseline compute and latency\n")
        rows = []
        for s in sorted(cost["settings"], key=lambda x: x["setting"]):
            u = s["usage"]
            unit = (
                f"{u.get('gpu_hours_per_1k_summaries', 0):.3f}"
                if is_local
                else f"${u.get('cost_per_1k_summaries_usd', 0):.2f}"
            )
            rows.append([
                s["setting"], s["shots"], f"{u['input_tokens']:,}", f"{u['output_tokens']:,}",
                f"{u.get('wall_clock_s', 0)/60:.1f}", unit,
                u.get("throughput_summaries_per_min", "-"),
                u["latency_p50_s"], u["errors"],
            ])
        md.append(table(
            ["Setting", "Shots", "Input tok", "Output tok", "Wall-clock (min)",
             unit_col, "Summaries/min", "p50 latency (s)", "Errors"],
            rows,
        ))

        if is_local:
            d = cost.get("backend_details", {})
            md.append(
                f"\nBackend: **local open-weights** — `{cost['model']}` "
                f"({d.get('quantization', '?')}, {d.get('sampling', '?')}) via "
                f"{d.get('framework', 'mlx-lm')} on the Apple silicon GPU. "
                f"Monetary cost **$0.00**; total compute "
                f"**{cost.get('total_gpu_hours', 0):.2f} GPU-hours** over "
                f"{cost['total_requests']:,} summaries."
            )
        else:
            md.append(
                f"\nBackend: **hosted API** — `{cost['model']}` at "
                f"${cost.get('price_per_mtok_input_usd', 0):.2f}/MTok input, "
                f"${cost.get('price_per_mtok_output_usd', 0):.2f}/MTok output. "
                f"**Total measured cost: ${cost['total_cost_usd']:.4f}** over "
                f"{cost['total_requests']:,} requests."
            )

        md.append("\n## Appendix A — Exact prompts\n")
        for s in cost["settings"]:
            # Tolerate entries written by an earlier version of the runner:
            # cost_summary.json merges settings across runs, so a stale entry
            # must degrade the appendix, not crash the whole collector.
            if not s.get("system_prompt"):
                md.append(f"\n### {s['setting']} — prompt not recorded in this run's metadata\n")
                continue
            md.append(
                f"\n### {s['setting']} (variant {s.get('prompt_variant', '?')} — "
                f"{s.get('prompt_variant_name', '?')}, {s.get('shots', '?')}-shot, "
                f"{s.get('input_condition', '?')})\n"
            )
            md.append("**System prompt:**\n```\n" + s["system_prompt"] + "\n```\n")
            md.append("**User template:**\n```\n" + s.get("user_template", "") + "\n```")
    else:
        missing.append("runs/llm/cost_summary.json (run src.llm.baseline --all)")

    # ---------------------------------------------------------------- inference cost
    gen_meta = load_json(RUNS / "base" / "preds_test_llm.meta.json")
    if gen_meta:
        data["lstm_inference"] = gen_meta
        md.append("\n## LSTM inference\n")
        md.append(
            f"Decoding: {gen_meta['decode']} (beam {gen_meta['beam_size']}), "
            f"{gen_meta['n_examples']} examples in {gen_meta['total_seconds']:.1f} s "
            f"= **{gen_meta['seconds_per_example']:.3f} s/summary** on {gen_meta['device']}. "
            f"Model size: {gen_meta['parameters']['total']:,} parameters "
            f"(~{gen_meta['parameters']['total']*4/1e6:.0f} MB in fp32)."
        )

    # ---------------------------------------------------------------- write
    REPORTS.mkdir(parents=True, exist_ok=True)
    if missing:
        md.append("\n## Not yet available\n")
        for m in missing:
            md.append(f"- {m}")

    (REPORTS / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORTS / "report_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("\n".join(md))
    print(f"\nwrote {REPORTS/'tables.md'} and {REPORTS/'report_data.json'}")
    if missing:
        print("\nSTILL MISSING:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
