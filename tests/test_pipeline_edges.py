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


# ------------------------------------------------------------- LLM replay cache

def make_cache(tmp_path, article="a news article about a fire", prediction="a fire happened"):
    from src.demo import cache_key

    p = tmp_path / "cache.json"
    p.write_text(json.dumps({
        "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "recorded_on": "2026-08-09",
        "entries": {cache_key(article): {"source": "test", "prediction": prediction,
                                         "latency_s": 2.5}},
    }))
    return p


def test_cache_key_is_sensitive_to_the_exact_shown_text():
    """Keying on the shown window, not the article id, is what stops a cached
    response recorded at one truncation being replayed at another."""
    from src.demo import cache_key

    assert cache_key("abc") == cache_key("abc")
    assert cache_key("abc") != cache_key("abc ")


def test_cached_llm_replays_the_recorded_response(tmp_path):
    from src.demo import CachedLLM

    llm = CachedLLM(str(make_cache(tmp_path)))
    text, latency = llm.summarize("a news article about a fire")
    assert text == "a fire happened"
    assert latency == 2.5
    assert llm.replayed is True


def test_cached_llm_refuses_an_article_it_never_saw(tmp_path):
    """Returning the nearest entry, or an empty string, would put a response on a
    recorded demo that the model never produced for that article."""
    from src.demo import CachedLLM

    llm = CachedLLM(str(make_cache(tmp_path)))
    with pytest.raises(KeyError, match="not in the LLM cache"):
        llm.summarize("an article the cache does not contain")


class _Args:
    def __init__(self, **kw):
        self.no_llm = False
        self.replay_llm = False
        self.llm_cache = ""
        self.llm_model = "does-not-matter"
        self.__dict__.update(kw)


def test_load_llm_falls_back_to_the_cache_when_the_backend_fails(tmp_path, monkeypatch, capsys):
    """Most teammates have no Apple silicon and no 4.5 GB model. The demo must
    still run, and must say on screen that it is replaying."""
    import src.demo as demo

    def boom(*a, **k):
        raise ImportError("No module named 'mlx'")

    monkeypatch.setattr(demo, "LLMSummarizer", boom)
    llm, note = demo.load_llm(_Args(llm_cache=str(make_cache(tmp_path))))

    assert llm.replayed is True
    assert "replayed" in note
    assert "unavailable" in capsys.readouterr().out


def test_load_llm_reports_clearly_when_there_is_no_backend_and_no_cache(tmp_path, monkeypatch):
    import src.demo as demo

    def boom(*a, **k):
        raise ImportError("No module named 'mlx'")

    monkeypatch.setattr(demo, "LLMSummarizer", boom)
    with pytest.raises(SystemExit, match="--no-llm"):
        demo.load_llm(_Args(llm_cache=str(tmp_path / "absent.json")))


def test_replay_llm_never_loads_the_live_model(tmp_path, monkeypatch):
    """--replay-llm is what a recording machine uses; loading 4.5 GB anyway would
    defeat it."""
    import src.demo as demo

    def fail(*a, **k):
        raise AssertionError("the live backend must not be constructed under --replay-llm")

    monkeypatch.setattr(demo, "LLMSummarizer", fail)
    llm, _ = demo.load_llm(_Args(replay_llm=True, llm_cache=str(make_cache(tmp_path))))
    assert llm.replayed is True


def test_no_llm_short_circuits_before_touching_the_cache(monkeypatch):
    import src.demo as demo

    llm, note = demo.load_llm(_Args(no_llm=True, llm_cache="/nonexistent"))
    assert llm is None and "--no-llm" in note


def test_shipped_cache_covers_every_article_the_walkthrough_shows():
    """The walkthrough runs three demo commands. If any article is missing from
    the cache, a teammate's recording dies partway through."""
    from src.data.tokenizer import normalize
    from src.demo import cache_key, truncate_words

    cache_path = Path("examples/llm_cache.json")
    if not cache_path.exists():
        pytest.skip("cache not built")
    entries = json.loads(cache_path.read_text())["entries"]

    test_file = Path("data/processed/test_llm.jsonl")
    if test_file.exists():
        from src.data.dataset import read_jsonl

        records = read_jsonl(test_file)
        for i in (3, 4, 112):
            key = cache_key(truncate_words(records[i]["article"], 400))
            assert key in entries, f"test example {i} is missing from the LLM cache"

    battery = Path("examples/demo_article_battery.txt")
    if battery.exists():
        key = cache_key(truncate_words(normalize(battery.read_text(encoding="utf-8")), 400))
        assert key in entries, "the out-of-domain article is missing from the LLM cache"


# --------------------------------------------------- walkthrough display commands

def walkthrough_extractions():
    """The `run sed`/`run awk` lines in the walkthrough -- the ones that pull a
    table out of a results file for the video."""
    import re

    script = Path("scripts/walkthrough.sh")
    if not script.exists():
        return []
    return re.findall(r"^\s*run ((?:sed|awk|grep) .*)$", script.read_text(), re.M)


def test_every_walkthrough_extraction_actually_prints_a_table():
    """A range like /## Overall/,/^$/ ends at the blank line right after the
    heading, so it prints the heading and nothing else -- silently, exit 0. That
    emptied the Lead-3 segment, which is the credibility anchor of the whole
    demo, and neither `bash -n` nor a smoke run caught it."""
    import subprocess

    commands = walkthrough_extractions()
    assert commands, "no extraction commands found -- did the script change shape?"

    for cmd in commands:
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
        body = [ln for ln in out.splitlines() if ln.strip()]
        assert body, f"produced no output at all:\n  {cmd}"

        # A command pulling a section out of a results file must reach the data,
        # not stop at the heading. Other commands (the nn.* primitive count) just
        # have to print something.
        if "results/" in cmd:
            rows = [ln for ln in body
                    if ln.startswith("|") and not set(ln) <= set("|- ")]
            assert rows, (f"reached no data rows -- the range probably ends too "
                          f"early:\n  {cmd}\n  output was:\n{out}")


def test_the_walkthrough_shows_the_published_lead3_comparison():
    """The segment claims agreement with See et al. to ~0.3 ROUGE. If the number
    on screen ever stops being 40.04, the claim beside it is wrong."""
    import subprocess

    for cmd in walkthrough_extractions():
        if "lead3_fulltest" not in cmd:
            continue
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
        assert "40.04" in out and "17.5" in out and "36.34" in out, out
        return
    pytest.skip("lead3 segment not present")


# ---------------------------------------------------- demo fits its window

@pytest.mark.parametrize("cols", [80, 100, 105, 120, 140])
def test_demo_block_never_exceeds_the_window(cols, monkeypatch, capsys):
    """The LLM row is a long model name plus five diagnostics -- 128 characters.
    Folded by the terminal it breaks a ROUGE score across two lines and the
    comparison rows stop aligning, which is exactly what a recorded demo cannot
    afford. Long metadata must move to its own line instead."""
    import importlib

    monkeypatch.setenv("COLUMNS", str(cols))
    demo = importlib.reload(importlib.import_module("src.demo"))

    demo.block(
        "LLM (Llama-3.1-8B-Instruct-4bit · replayed)",
        "A mammoth fire broke out Friday morning in a Kentucky industrial park. "
        "The blaze began shortly before 7 a.m. at the General Electric Appliance Park.",
        meta="2.28s   len=61  repeat=0.00  novel=0.15  unsupported=0.03   "
             "R1=33.3 R2=18.4 RL=33.3",
    )
    out = capsys.readouterr().out
    for line in out.splitlines():
        assert len(line.rstrip()) <= cols, f"{len(line)} chars at COLUMNS={cols}: {line!r}"

    monkeypatch.delenv("COLUMNS", raising=False)
    importlib.reload(demo)


def test_walkthrough_banner_states_the_real_test_count():
    """The closing frame of the recorded video asserts a test count. It went
    stale at 162 while the suite grew to 177 -- a wrong number in the last thing
    a marker sees, and nothing pointed at it."""
    import re
    import subprocess
    import sys

    script = Path("scripts/walkthrough.sh")
    if not script.exists():
        pytest.skip("no walkthrough script")

    m = re.search(r"(\d+) tests · full report", script.read_text())
    assert m, "the banner no longer states a test count -- update this test"
    claimed = int(m.group(1))

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only"],
        capture_output=True, text=True,
    ).stdout
    actual = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert claimed == actual, f"banner says {claimed} tests, suite has {actual}"
