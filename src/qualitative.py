"""Build the side-by-side qualitative comparison for the report.

Usage:
    python -m src.qualitative --scores results/per_example_scores.json \
        --system lstm_beam=runs/base/preds_test_llm.jsonl \
        --system llm_B_zeroshot=runs/llm/B_zeroshot.jsonl \
        --out results/qualitative.md

Selects examples by *behavior*, not by score: the assignment asks for cases that
illustrate distinct failure modes rather than a highlight reel. Each selector
below targets one phenomenon the error analysis discusses, and the emitted
markdown carries the measured diagnostics beside each output so the categories
can be checked against the data rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from src.data.dataset import read_jsonl


def pick(rows: list[dict], key, reverse: bool, taken: set[str], n: int = 1) -> list[str]:
    ordered = sorted(rows, key=key, reverse=reverse)
    out = []
    for row in ordered:
        if row["id"] in taken:
            continue
        out.append(row["id"])
        taken.add(row["id"])
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/per_example_scores.json")
    ap.add_argument("--test-file", default="data/processed/test_llm.jsonl")
    ap.add_argument("--system", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--lstm-system", default="lstm_beam", help="Name of the LSTM system.")
    ap.add_argument("--llm-system", default="llm_B_zeroshot", help="Name of the LLM system.")
    ap.add_argument("--out", default="results/qualitative.md")
    ap.add_argument("--article-chars", type=int, default=900)
    args = ap.parse_args()

    with open(args.scores, encoding="utf-8") as f:
        scores = json.load(f)

    records = {r["id"]: r for r in read_jsonl(args.test_file)}
    systems: dict[str, dict[str, str]] = {}
    for spec in args.system:
        name, path = spec.split("=", 1)
        systems[name] = {
            r["id"]: r["prediction"] for r in read_jsonl(path) if "prediction" in r
        }

    lstm_name, llm_name = args.lstm_system, args.llm_system
    if lstm_name not in scores or llm_name not in scores:
        raise SystemExit(
            f"scores file has {sorted(scores)}; "
            f"expected --lstm-system/--llm-system among them"
        )

    lstm_rows = {r["id"]: r for r in scores[lstm_name]}
    llm_rows = {r["id"]: r for r in scores[llm_name]}
    shared = [i for i in lstm_rows if i in llm_rows]
    rows = [
        {
            "id": i,
            "lstm": lstm_rows[i],
            "llm": llm_rows[i],
            "gap": llm_rows[i]["rouge1"] - lstm_rows[i]["rouge1"],
            "src_len": records[i]["src_len"],
        }
        for i in shared
    ]

    taken: set[str] = set()
    selectors = [
        ("LSTM's strongest case", lambda r: r["lstm"]["rouge1"], True, 1),
        ("LSTM's weakest case", lambda r: r["lstm"]["rouge1"], False, 1),
        ("Largest LLM advantage", lambda r: r["gap"], True, 1),
        ("LSTM beats the LLM", lambda r: r["gap"], False, 2),
        ("Most repetitive LSTM output", lambda r: r["lstm"]["dup_trigram_rate"], True, 1),
        ("Highest LSTM OOV rate", lambda r: r["lstm"]["oov_rate"], True, 1),
        ("Most unsupported LLM content (hallucination candidate)",
         lambda r: r["llm"]["unsupported_content_rate"], True, 1),
        ("Longest article", lambda r: r["src_len"], True, 1),
        ("Shortest article", lambda r: r["src_len"], False, 1),
        ("Most abstractive LSTM output", lambda r: r["lstm"]["novel_bigram_rate"], True, 1),
        ("Longest LLM output (format adherence)", lambda r: r["llm"]["length_tokens"], True, 1),
        ("Shortest LSTM output", lambda r: r["lstm"]["length_tokens"], False, 1),
    ]

    selected: list[tuple[str, str]] = []
    for label, key, reverse, n in selectors:
        for ex_id in pick(rows, key, reverse, taken, n):
            selected.append((label, ex_id))

    lines = [
        "# Qualitative comparison\n",
        f"{len(selected)} test examples selected by behavior (not by score) to illustrate "
        "distinct phenomena. Diagnostics shown per output: `dup3` = duplicate-trigram rate, "
        "`novel2` = novel-bigram rate vs. the article, `unsup` = content words absent from the "
        "article, `oov` = tokens outside the LSTM's 50k vocabulary, `R1` = ROUGE-1 F1.\n",
        "\n> The **Error category** field is a hypothesis to verify against the text, "
        "not an automatic label.\n",
    ]

    for n, (label, ex_id) in enumerate(selected, 1):
        rec = records[ex_id]
        lstm, llm = lstm_rows[ex_id], llm_rows[ex_id]
        article = rec["article"]
        shown = article[: args.article_chars]
        if len(article) > args.article_chars:
            shown += " [...]"

        lines.append(f"\n---\n\n## {n}. {label}\n")
        lines.append(f"**Example id:** `{ex_id}`, source length {rec['src_len']} tokens\n")
        lines.append("**Source (truncated for display):**\n")
        lines.append("> " + textwrap.fill(shown, 110).replace("\n", "\n> ") + "\n")
        lines.append("**Reference:**\n")
        lines.append("> " + textwrap.fill(rec["summary"], 110).replace("\n", "\n> ") + "\n")

        for sys_name, row in ((lstm_name, lstm), (llm_name, llm)):
            pred = systems.get(sys_name, {}).get(ex_id, "(missing)")
            lines.append(f"**{sys_name}:** ")
            lines.append(
                f"`R1={row['rouge1']*100:.1f}` "
                f"`dup3={row['dup_trigram_rate']:.2f}` "
                f"`novel2={row['novel_bigram_rate']:.2f}` "
                f"`unsup={row['unsupported_content_rate']:.2f}` "
                f"`oov={row['oov_rate']:.2f}` "
                f"`len={row['length_tokens']}`\n"
            )
            lines.append("> " + textwrap.fill(pred, 110).replace("\n", "\n> ") + "\n")

        for other in systems:
            if other in (lstm_name, llm_name):
                continue
            lines.append(f"**{other}:**\n")
            lines.append(
                "> "
                + textwrap.fill(systems[other].get(ex_id, "(missing)"), 110).replace("\n", "\n> ")
                + "\n"
            )

        lines.append("**Error category (LSTM):** _TODO, verify against the text_\n")
        lines.append("**Error category (LLM):** _TODO, verify against the text_\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(selected)} examples")


if __name__ == "__main__":
    main()
