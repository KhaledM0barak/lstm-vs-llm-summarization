"""LLM baseline: prompt construction, input parity, leakage control, accounting.

No model is loaded here -- these cover the logic around generation, which is
where the experimental design lives. The backends themselves are exercised by
the integration test.
"""

from __future__ import annotations

import json

import pytest

from src.llm.backends import AnthropicBackend, GenResult
from src.llm.baseline import (
    SETTINGS,
    UsageTracker,
    clean_output,
    sample_exemplars,
    truncate_words,
)
from src.llm.prompts import VARIANTS, build_messages


# ------------------------------------------------------------------- prompts

def test_two_distinct_variants_exist():
    assert set(VARIANTS) == {"A", "B"}
    assert VARIANTS["A"].system != VARIANTS["B"].system
    assert VARIANTS["A"].user_template != VARIANTS["B"].user_template


def test_variant_b_specifies_reference_style():
    """Variant B exists to describe the CNN/DailyMail highlight style; if it
    stops doing that, the A-vs-B comparison loses its meaning."""
    sys_b = VARIANTS["B"].system.lower()
    assert "55 words" in sys_b or "3 to 4" in sys_b


def test_zero_shot_builds_a_single_user_turn():
    msgs = build_messages(VARIANTS["A"], "article text", [])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "article text" in msgs[0]["content"]


def test_few_shot_builds_alternating_turns():
    """Exemplars must be real conversation turns, not pasted into one message --
    otherwise the exemplar boundaries are ambiguous to the model."""
    ex = [{"article": f"art{i}", "summary": f"sum{i}"} for i in range(3)]
    msgs = build_messages(VARIANTS["B"], "target article", ex)

    assert len(msgs) == 7                      # 3 pairs + final user turn
    assert [m["role"] for m in msgs] == [
        "user", "assistant", "user", "assistant", "user", "assistant", "user"
    ]
    assert msgs[1]["content"] == "sum0"
    assert "target article" in msgs[-1]["content"]


def test_all_four_settings_are_two_variants_by_two_shot_counts():
    assert len(SETTINGS) == 4
    variants = {v for v, _ in SETTINGS.values()}
    shots = {s for _, s in SETTINGS.values()}
    assert variants == {"A", "B"}
    assert shots == {0, 4}, "assignment requires zero-shot and few-shot k=3-5"


def test_few_shot_k_is_in_the_required_range():
    for _, shots in SETTINGS.values():
        assert shots == 0 or 3 <= shots <= 5


# -------------------------------------------------------------- input parity

def test_truncate_words_enforces_the_window():
    text = " ".join(str(i) for i in range(1000))
    assert len(truncate_words(text, 400).split()) == 400


def test_truncate_words_none_keeps_everything():
    text = " ".join(str(i) for i in range(50))
    assert truncate_words(text, None) == text


def test_truncate_shorter_than_limit_is_unchanged():
    assert truncate_words("a b c", 400) == "a b c"


# ---------------------------------------------------------- leakage control

@pytest.fixture
def fake_train(tmp_path):
    p = tmp_path / "train.jsonl"
    with p.open("w") as f:
        for i in range(50):
            f.write(json.dumps({
                "id": f"train{i}",
                "article": " ".join(["word"] * 300),
                "summary": f"summary {i}",
                "src_len": 300,
                "tgt_len": 20,
            }) + "\n")
    return str(p)


def test_exemplars_come_from_the_given_file(fake_train):
    ex = sample_exemplars(fake_train, 4, seed=1, max_src_words=400)
    assert len(ex) == 4
    for e in ex:
        assert e["summary"].startswith("summary ")


def test_exemplar_sampling_is_deterministic(fake_train):
    a = sample_exemplars(fake_train, 4, seed=1, max_src_words=400)
    b = sample_exemplars(fake_train, 4, seed=1, max_src_words=400)
    assert a == b


def test_exemplar_sampling_varies_with_seed(fake_train):
    a = sample_exemplars(fake_train, 4, seed=1, max_src_words=400)
    b = sample_exemplars(fake_train, 4, seed=99, max_src_words=400)
    assert a != b


def test_exemplars_are_truncated_to_the_window(fake_train):
    ex = sample_exemplars(fake_train, 2, seed=1, max_src_words=50)
    for e in ex:
        assert len(e["article"].split()) <= 50


# --------------------------------------------------------------- output prep

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Here are the highlights: a b c", "a b c"),
        ("Summary: a b c", "a b c"),
        ("Here is a summary: a b c", "a b c"),
        ("  a b c  ", "a b c"),
        ("a b c", "a b c"),
        ("a\n\nb   c", "a b c"),
    ],
)
def test_clean_output_strips_preamble_and_normalizes(raw, expected):
    assert clean_output(raw) == expected


def test_clean_output_is_case_insensitive():
    assert clean_output("SUMMARY: a b") == "a b"


def test_clean_output_leaves_mid_text_alone():
    """It must only strip a leading label, never rewrite the body."""
    assert clean_output("the report summary: was late") == "the report summary: was late"


# ---------------------------------------------------------------- accounting

class FakeAPIBackend:
    kind = "api"

    def cost_usd(self, i, o):
        return i / 1e6 * 1.0 + o / 1e6 * 5.0


class FakeLocalBackend:
    kind = "local"

    def cost_usd(self, i, o):
        return 0.0


def test_usage_tracker_totals_tokens_and_cost():
    t = UsageTracker(FakeAPIBackend())
    t.add(GenResult("x", input_tokens=1_000_000, output_tokens=1_000_000, latency_s=1.0))
    s = t.summary()
    assert s["requests"] == 1
    assert s["input_tokens"] == 1_000_000
    assert s["cost_usd"] == pytest.approx(6.0)


def test_usage_tracker_reports_gpu_hours_for_local_backends():
    t = UsageTracker(FakeLocalBackend())
    t.add(GenResult("x", 100, 20, 1.0))
    s = t.summary()
    assert s["cost_usd"] == 0.0
    assert "gpu_hours" in s
    assert "cost_per_1k_summaries_usd" not in s


def test_usage_tracker_counts_errors_separately():
    t = UsageTracker(FakeLocalBackend())
    t.add(GenResult("", 0, 0, 0.1, error="boom"))
    t.add(GenResult("ok", 10, 5, 0.2))
    s = t.summary()
    assert s["errors"] == 1
    assert s["requests"] == 1, "a failed request must not count as a completed one"


def test_usage_tracker_latency_percentiles():
    t = UsageTracker(FakeLocalBackend())
    for i in range(100):
        t.add(GenResult("x", 1, 1, latency_s=float(i)))
    s = t.summary()
    assert s["latency_p50_s"] == pytest.approx(50.0, abs=1.0)
    assert s["latency_p95_s"] == pytest.approx(95.0, abs=1.0)


def test_anthropic_backend_prices_match_the_documented_model():
    assert AnthropicBackend.PRICE_PER_MTOK_IN == 1.00
    assert AnthropicBackend.PRICE_PER_MTOK_OUT == 5.00
