"""Edge cases in the surrounding pipeline: resume, accounting, prep, reporting.

These cover the modules the other test files don't reach, with an emphasis on
failure and restart paths -- the places that only execute when something has
already gone wrong, and are therefore the least likely to have been exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.prepare import _clean_summary, _to_record
from src.data.tokenizer import normalize
from src.demo import looks_degenerate
from src.llm.backends import GenResult
from src.utils.device import hardware_summary
from src.utils.seed import set_seed


# ------------------------------------------------------------ prepare/clean

def test_clean_summary_joins_highlight_bullets():
    out = _clean_summary("first highlight\nsecond highlight\nthird")
    assert "first highlight" in out and "second highlight" in out
    assert "\n" not in out


def test_clean_summary_ignores_blank_lines():
    assert _clean_summary("a\n\n\nb").count(".") >= 1


def test_to_record_rejects_degenerate_pairs():
    """Too-short articles and summaries would only add noise to training."""
    assert _to_record({"id": "x", "article": "short", "highlights": "tiny"}, 0) is None


def test_to_record_accepts_a_reasonable_pair():
    rec = _to_record(
        {"id": "x", "article": " ".join(["word"] * 100), "highlights": "a decent summary here"},
        0,
    )
    assert rec is not None
    assert rec["src_len"] >= 30 and rec["tgt_len"] >= 5


def test_to_record_reports_token_lengths_not_characters():
    rec = _to_record(
        {"id": "x", "article": " ".join(["word"] * 100), "highlights": "a decent summary here"},
        0,
    )
    assert rec["src_len"] == len(rec["article"].split())


# --------------------------------------------------------- degenerate guard

@pytest.mark.parametrize(
    "text,expected",
    [
        ("", True),
        ("the the the a the the", True),                        # MPS OOM signature
        ("the the the the the the the the", True),
        ("a mammoth fire broke out friday morning in a kentucky industrial park", False),
        ("short", True),                                        # too short to judge
    ],
)
def test_looks_degenerate(text, expected):
    """PyTorch's Metal backend returns garbage rather than raising on GPU OOM;
    this guard is what stops a broken demo from being recorded."""
    assert looks_degenerate(text) is expected


# --------------------------------------------------------------- provenance

def test_hardware_summary_has_what_the_report_needs():
    hw = hardware_summary()
    for key in ("platform", "python", "torch", "gpu"):
        assert key in hw and hw[key]


def test_set_seed_makes_numpy_and_torch_reproducible():
    import torch

    set_seed(99)
    a, ta = np.random.rand(5), torch.randn(5)
    set_seed(99)
    b, tb = np.random.rand(5), torch.randn(5)
    assert np.allclose(a, b)
    assert torch.allclose(ta, tb)


# ------------------------------------------------------- baseline: resuming

def write_jsonl(path: Path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_resume_retries_failed_examples(tmp_path):
    """An example that previously errored must be retried on resume, not treated
    as complete -- otherwise a transient failure permanently drops a test article
    and the systems end up scored on different subsets."""
    from src.llm.baseline import completed_ids

    out = tmp_path / "s.jsonl"
    write_jsonl(out, [
        {"id": "a", "prediction": "ok", "input_tokens": 5, "output_tokens": 2, "latency_s": 0.1},
        {"id": "b", "error": "RateLimitError: boom"},
    ])
    done = completed_ids(out)
    assert done == {"a"}, "an errored example was counted as complete"


def test_resume_recovers_prior_usage(tmp_path):
    """Usage totals must include work done before the interruption, or the
    reported cost and token counts describe only the final segment."""
    from src.llm.baseline import UsageTracker, prior_usage

    out = tmp_path / "s.jsonl"
    write_jsonl(out, [
        {"id": "a", "prediction": "x", "input_tokens": 100, "output_tokens": 20, "latency_s": 1.0},
        {"id": "b", "prediction": "y", "input_tokens": 200, "output_tokens": 30, "latency_s": 2.0},
        {"id": "c", "error": "boom"},
    ])

    class FakeLocal:
        kind = "local"
        def cost_usd(self, i, o):
            return 0.0

    t = UsageTracker(FakeLocal())
    prior_usage(out, t)
    t.add(GenResult("z", input_tokens=50, output_tokens=10, latency_s=0.5))

    s = t.summary()
    assert s["requests"] == 3, "prior completions were lost on resume"
    assert s["input_tokens"] == 350
    assert s["output_tokens"] == 60


def test_completed_ids_on_missing_file(tmp_path):
    from src.llm.baseline import completed_ids

    assert completed_ids(tmp_path / "nope.jsonl") == set()


# --------------------------------------------------- collect_results safety

def test_collect_results_reports_missing_rather_than_inventing(tmp_path, monkeypatch):
    """The collector must never emit a number whose artifact is absent."""
    import scripts.collect_results as cr

    monkeypatch.setattr(cr, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(cr, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cr, "DATA", tmp_path / "data")
    monkeypatch.setattr(cr, "REPORTS", tmp_path / "reports")

    cr.main()

    md = (tmp_path / "reports" / "tables.md").read_text()
    assert "Not yet available" in md
    # No fabricated ROUGE table when nothing has been scored.
    assert "Table 3" not in md


def test_collect_results_handles_local_and_api_backends(tmp_path, monkeypatch):
    import scripts.collect_results as cr

    runs = tmp_path / "runs" / "llm"
    runs.mkdir(parents=True)
    (runs / "cost_summary.json").write_text(json.dumps({
        "backend": "mlx", "backend_kind": "local", "model": "test-model",
        "backend_details": {"quantization": "4-bit", "sampling": "greedy", "framework": "mlx-lm"},
        "settings": [{
            "setting": "A_zeroshot", "shots": 0,
            "usage": {"input_tokens": 10, "output_tokens": 5, "wall_clock_s": 60.0,
                      "gpu_hours_per_1k_summaries": 0.5, "throughput_summaries_per_min": 20.0,
                      "latency_p50_s": 3.0, "errors": 0, "requests": 1},
        }],
        "total_requests": 1, "total_cost_usd": 0.0, "total_gpu_hours": 0.02,
    }))
    monkeypatch.setattr(cr, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(cr, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cr, "DATA", tmp_path / "data")
    monkeypatch.setattr(cr, "REPORTS", tmp_path / "reports")

    cr.main()
    md = (tmp_path / "reports" / "tables.md").read_text()
    assert "GPU-h / 1k summaries" in md
    assert "$0.00" in md or "0.00" in md


# ------------------------------------------------------------- report build

def test_build_pdf_flags_unfilled_placeholders(tmp_path, capsys):
    import scripts.build_pdf as bp

    src = tmp_path / "r.md"
    src.write_text("# Title\n\nSome text with [[FILL: something]] left in it.\n")
    try:
        bp.find_chrome()
    except SystemExit:
        pytest.skip("no Chrome available for PDF rendering")

    import sys
    argv = sys.argv
    sys.argv = ["build_pdf", "--input", str(src), "--output", str(tmp_path / "r.pdf")]
    try:
        bp.main()
    finally:
        sys.argv = argv

    out = capsys.readouterr().out
    assert "unfilled placeholders" in out
    assert "[[FILL: something]]" in out


# ------------------------------------------------------------ demo rendering

@pytest.mark.parametrize(
    "cols,expected",
    [
        (70, 68),     # narrow: track the window
        (96, 94),
        (120, 118),
        (200, 140),   # very wide: clamp, long lines are hard to read
        (40, 60),     # very narrow: clamp, diagnostics stop lining up
    ],
)
def test_terminal_width_tracks_and_clamps(cols, expected, monkeypatch):
    """A fixed render width wraps every line in a narrower window, which is
    exactly when the side-by-side comparison becomes unreadable."""
    from src.demo import terminal_width

    monkeypatch.setenv("COLUMNS", str(cols))
    assert terminal_width() == expected


def test_terminal_width_falls_back_when_there_is_no_terminal(monkeypatch):
    """Piped output and screenshots should stay at a stable width."""
    import shutil as _shutil

    from src.demo import terminal_width

    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setattr(_shutil, "get_terminal_size",
                        lambda fallback=(96, 24): __import__("os").terminal_size(fallback))
    assert terminal_width() == 94
