"""Interactive side-by-side demo: LSTM vs. LLM on one article.

Usage:
    # A held-out test article, both systems, with reference and scores
    python -m src.demo --example 7

    # Your own text
    python -m src.demo --file article.txt
    python -m src.demo --text "Paste an article here..."

    # Interactive loop (best for a live demo)
    python -m src.demo --interactive

    # LSTM only -- starts instantly, no 4.5 GB model load
    python -m src.demo --example 7 --no-llm

    # Show what the ablated models produce for the same article
    python -m src.demo --example 7 --ablations

Both systems receive the *same* 400-word input window, matching the evaluation
protocol, so what you see on screen is the comparison the report describes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
import time
from pathlib import Path

import torch

from src.data.dataset import read_jsonl
from src.data.tokenizer import detokenize, normalize, tokenize
from src.data.vocab import Vocab
from src.generate import load_model
from src.utils.device import get_device
from src.utils.seed import set_seed

# ANSI styling. Disabled automatically when piped to a file.
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s

BOLD = lambda s: _c("1", s)
DIM = lambda s: _c("2", s)
CYAN = lambda s: _c("36", s)
GREEN = lambda s: _c("32", s)
YELLOW = lambda s: _c("33", s)
MAGENTA = lambda s: _c("35", s)
RED = lambda s: _c("31", s)

def terminal_width(default: int = 96, minimum: int = 60, maximum: int = 140) -> int:
    """Render width, adapted to the terminal.

    A fixed width wraps every line in a window narrower than it, which is exactly
    when the side-by-side comparison stops being readable. Clamped at both ends:
    below `minimum` the diagnostics no longer line up, and above `maximum` prose
    becomes hard to track across the line.

    `shutil.get_terminal_size` honours a `COLUMNS` override first, then queries
    the terminal, then falls back — so piped output and screenshots stay at a
    consistent `default` width, and the value can be forced for a recording.
    """
    cols = shutil.get_terminal_size(fallback=(default, 24)).columns
    return max(minimum, min(cols - 2, maximum))


WIDTH = terminal_width()


def rule(title: str = "", color=CYAN) -> None:
    if title:
        pad = WIDTH - len(title) - 4
        print(color("── " + title + " " + "─" * max(pad, 0)))
    else:
        print(color("─" * WIDTH))


def block(label: str, text: str, color=BOLD, meta: str = "") -> None:
    print()
    print(color(label) + (DIM("  " + meta) if meta else ""))
    for line in textwrap.wrap(text, WIDTH - 2) or ["(empty)"]:
        print("  " + line)


class LSTMSummarizer:
    def __init__(self, checkpoint: str, vocab_file: str, device) -> None:
        self.device = device
        self.vocab = Vocab.load(vocab_file)
        self.model, ckpt = load_model(checkpoint, device)
        self.cfg = ckpt["config"]
        self.params = self.model.num_parameters()["total"]
        self.name = Path(checkpoint).parent.name

    def summarize(self, article: str, beam: int = 4) -> tuple[str, float]:
        ids = self.vocab.encode(tokenize(article))[: self.cfg["max_src_len"]]
        if not ids:
            return "", 0.0
        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        src_len = torch.tensor([len(ids)], dtype=torch.long)
        mask = torch.ones(1, len(ids), dtype=torch.bool, device=self.device)

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate_beam(
                src, src_len, mask, beam_size=beam,
                max_len=self.cfg["max_tgt_len"], min_len=25,
                length_penalty=1.0, block_trigram=True,
            )
        return detokenize(self.vocab.decode(out[0])), time.time() - t0


class LLMSummarizer:
    def __init__(self, model: str, max_tokens: int = 200) -> None:
        from src.llm.backends import MLXBackend

        self.backend = MLXBackend(model=model, batch_size=1)
        self.max_tokens = max_tokens
        self.name = model

    def summarize(self, article: str) -> tuple[str, float]:
        from src.llm.baseline import clean_output
        from src.llm.prompts import VARIANTS, build_messages

        variant = VARIANTS["B"]  # style-matched: the report's best-performing prompt
        messages = build_messages(variant, article, [])
        t0 = time.time()
        res = self.backend.generate_one(variant.system, messages, self.max_tokens)
        return clean_output(res.text), time.time() - t0


def score(prediction: str, reference: str):
    """ROUGE for one pair, using the same configuration as the evaluation."""
    from rouge_score import rouge_scorer

    from src.evaluate import rougeLsum_ready

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    s = scorer.score(rougeLsum_ready(reference), rougeLsum_ready(prediction))
    return {k: v.fmeasure * 100 for k, v in s.items()}


def diagnostics(prediction: str, article: str, vocab: Vocab) -> str:
    from src.evaluate import diagnostics as diag

    d = diag(prediction, article, vocab)
    return (
        f"len={d['length_tokens']}  "
        f"repeat={d['dup_trigram_rate']:.2f}  "
        f"novel={d['novel_bigram_rate']:.2f}  "
        f"unsupported={d['unsupported_content_rate']:.2f}"
    )


def truncate_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def looks_degenerate(text: str) -> bool:
    """Detect the failure signature of a GPU out-of-memory condition.

    PyTorch's Metal backend does not raise when a command buffer runs out of
    memory -- it returns whatever was in the buffer, so generation silently
    produces empty or `the the the a the` output instead of erroring. That is
    indistinguishable from a broken model unless you check, and it is very easy
    to hit here because a locally running LLM can occupy most of the GPU. This
    guard catches it before anyone records a demo of it.
    """
    tokens = text.split()
    if len(tokens) < 5:
        return True
    return len(set(tokens)) / len(tokens) < 0.35


def run_one(article: str, reference: str | None, lstm, llm, vocab, args, ablations=None):
    shown = truncate_words(article, args.max_src_words)

    rule("SOURCE ARTICLE")
    preview = shown if args.full_source else shown[:1100] + (" [...]" if len(shown) > 1100 else "")
    for line in textwrap.wrap(preview, WIDTH - 2):
        print("  " + DIM(line))
    print()
    print(DIM(f"  ({len(article.split())} words total; both systems see the first "
              f"{args.max_src_words})"))

    if reference:
        block("REFERENCE (human-written highlights)", reference, GREEN)

    results = []

    pred, secs = lstm.summarize(shown, beam=args.beam)
    results.append(("LSTM + attention", pred, secs, MAGENTA))

    if llm is not None:
        pred_llm, secs_llm = llm.summarize(shown)
        results.append((f"LLM ({llm.name.split('/')[-1]})", pred_llm, secs_llm, YELLOW))

    rule("SUMMARIES")
    for label, pred, secs, color in results:
        meta = f"{secs:.2f}s   {diagnostics(pred, shown, vocab)}"
        if reference:
            s = score(pred, reference)
            meta += DIM(f"   R1={s['rouge1']:.1f} R2={s['rouge2']:.1f} RL={s['rougeLsum']:.1f}")
        block(label, pred, color, meta)

    if ablations:
        rule("ABLATIONS (same article)")
        for abl in ablations:
            pred_a, secs_a = abl.summarize(shown, beam=args.beam)
            meta = f"{secs_a:.2f}s   {diagnostics(pred_a, shown, vocab)}"
            if reference:
                s = score(pred_a, reference)
                meta += DIM(f"   R1={s['rouge1']:.1f}")
            block(f"— {abl.name}", pred_a, RED, meta)

    print()
    rule()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--example", type=int, help="Index into the held-out test set.")
    src.add_argument("--file", help="Path to a text file containing an article.")
    src.add_argument("--text", help="Article text passed directly.")
    src.add_argument("--interactive", action="store_true", help="Loop, reading pasted articles.")

    ap.add_argument("--checkpoint", default="runs/base/best.pt")
    ap.add_argument("--vocab-file", default="data/processed/vocab.json")
    ap.add_argument("--test-file", default="data/processed/test_llm.jsonl")
    ap.add_argument("--llm-model", default="mlx-community/Llama-3.1-8B-Instruct-4bit")
    ap.add_argument("--no-llm", action="store_true", help="Skip the LLM (instant startup).")
    ap.add_argument("--ablations", action="store_true", help="Also show the ablated models.")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-src-words", type=int, default=400)
    ap.add_argument("--full-source", action="store_true", help="Print the whole input window.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}.\n"
            "Train one first:  python -m src.train --config configs/base.yaml"
        )

    print()
    rule("LSTM seq2seq vs. LLM — abstractive summarization demo")
    lstm = LSTMSummarizer(args.checkpoint, args.vocab_file, device)

    # Self-check: decode a known article and verify the output is sane. If the
    # GPU is short on memory (e.g. a local LLM is resident) MPS returns garbage
    # rather than raising, so fall back to CPU -- the model is only 15M
    # parameters and CPU beam search is fast enough for an interactive demo.
    probe_records = read_jsonl(args.test_file) if Path(args.test_file).exists() else []
    if probe_records and str(device) != "cpu":
        probe, _ = lstm.summarize(truncate_words(probe_records[3]["article"], args.max_src_words),
                                  beam=args.beam)
        if looks_degenerate(probe):
            print(RED("  ! GPU produced degenerate output (likely out of memory —"))
            print(RED("    another process may be holding the GPU). Falling back to CPU."))
            device = torch.device("cpu")
            lstm = LSTMSummarizer(args.checkpoint, args.vocab_file, device)

    print(f"  LSTM      : {lstm.params:,} parameters, beam {args.beam}, device {device}")

    llm = None
    if not args.no_llm:
        print(f"  LLM       : loading {args.llm_model} ...")
        llm = LLMSummarizer(args.llm_model)
        print(f"  LLM       : loaded (4-bit, greedy)")
    else:
        print("  LLM       : skipped (--no-llm)")

    ablations = []
    if args.ablations:
        for name in ("no_attention", "unidirectional", "short_context"):
            ckpt = Path("runs") / name / "best.pt"
            if ckpt.exists():
                ablations.append(LSTMSummarizer(str(ckpt), args.vocab_file, device))
        print(f"  Ablations : {', '.join(a.name for a in ablations) or 'none found'}")

    records = probe_records

    if args.interactive:
        print()
        print(DIM("  Paste an article and press Enter twice. Type 'r N' for test example N, "
                  "or 'q' to quit."))
        while True:
            print()
            rule()
            try:
                first = input(BOLD("article> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return
            if first.lower() in {"q", "quit", "exit"}:
                print("bye")
                return
            if first.lower().startswith("r ") and records:
                idx = int(first.split()[1]) % len(records)
                rec = records[idx]
                print(DIM(f"  (test example {idx})"))
                run_one(rec["article"], rec["summary"], lstm, llm, lstm.vocab, args, ablations)
                continue
            lines = [first]
            while True:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line.strip():
                    break
                lines.append(line)
            text = normalize(" ".join(lines))
            if len(text.split()) < 30:
                print(RED("  Need at least ~30 words to summarize."))
                continue
            run_one(text, None, lstm, llm, lstm.vocab, args, ablations)
        return

    if args.file:
        article, reference = normalize(Path(args.file).read_text(encoding="utf-8")), None
    elif args.text:
        article, reference = normalize(args.text), None
    else:
        if not records:
            raise SystemExit(f"No test file at {args.test_file}; pass --file or --text.")
        idx = (args.example if args.example is not None else 0) % len(records)
        rec = records[idx]
        article, reference = rec["article"], rec["summary"]
        print(f"  Article   : held-out test example {idx} (id {rec['id'][:12]}...)")

    run_one(article, reference, lstm, llm, lstm.vocab, args, ablations)


if __name__ == "__main__":
    main()
