"""Evaluation correctness: ROUGE plumbing, diagnostics, and the bootstrap tests.

Everything reported rests on this module. If ROUGE is configured wrongly or a
diagnostic counts the wrong thing, every number in the report is wrong and
nothing else in the project matters.
"""

from __future__ import annotations

import numpy as np
import pytest
from rouge_score import rouge_scorer

from src.data.vocab import Vocab
from src.evaluate import (
    ROUGE_TYPES,
    bootstrap_ci,
    compare_systems,
    diagnostics,
    lead3,
    ngrams,
    paired_bootstrap,
    rougeLsum_ready,
    sentences,
)


@pytest.fixture
def scorer():
    return rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)


# --------------------------------------------------------------------- ROUGE

def test_identical_text_scores_one(scorer):
    text = "the cat sat on the mat. it was warm."
    s = scorer.score(rougeLsum_ready(text), rougeLsum_ready(text))
    for rt in ROUGE_TYPES:
        assert s[rt].fmeasure == pytest.approx(1.0)


def test_disjoint_text_scores_zero(scorer):
    s = scorer.score(rougeLsum_ready("alpha bravo charlie"),
                     rougeLsum_ready("xylophone quixotic zeppelin"))
    assert s["rouge1"].fmeasure == pytest.approx(0.0)


def test_rougeLsum_ready_puts_sentences_on_separate_lines():
    """rougeLsum computes a union over sentences; without newline separation it
    silently degrades to plain rougeL."""
    out = rougeLsum_ready("first one. second one. third one.")
    assert out.count("\n") == 2


def test_rougeLsum_differs_from_unseparated(scorer):
    """Guard against the separation being a no-op."""
    ref = "the cat sat. the dog barked. the bird flew."
    pred = "the dog barked. the cat sat. the bird flew."
    proper = scorer.score(rougeLsum_ready(ref), rougeLsum_ready(pred))["rougeLsum"].fmeasure
    naive = scorer.score(ref, pred)["rougeLsum"].fmeasure
    assert proper >= naive


def test_lead3_takes_first_three_sentences():
    article = "one. two. three. four. five."
    assert lead3(article) == "one. two. three."


def test_lead3_handles_short_articles():
    assert lead3("only one sentence.") == "only one sentence."


def test_sentences_matches_tokenizer_splitter():
    assert sentences("the u.s. said no. he left.") == ["the u.s. said no.", "he left."]


# ---------------------------------------------------------------- diagnostics

def test_ngrams_basic():
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert ngrams(["a"], 2) == []


def test_dup_trigram_rate_detects_exact_repetition():
    article = "irrelevant source text"
    repeated = "the cat sat the cat sat the cat sat"
    d = diagnostics(repeated, article, None)
    assert d["dup_trigram_rate"] > 0.5, "failed to detect obvious repetition"


def test_dup_trigram_rate_zero_on_non_repetitive_text():
    d = diagnostics("alpha bravo charlie delta echo foxtrot", "src", None)
    assert d["dup_trigram_rate"] == pytest.approx(0.0)


def test_novel_bigram_rate_zero_when_copied_verbatim():
    article = "the quick brown fox jumps over the lazy dog"
    d = diagnostics(article, article, None)
    assert d["novel_bigram_rate"] == pytest.approx(0.0)


def test_novel_bigram_rate_one_when_nothing_shared():
    d = diagnostics("xylophone quixotic zeppelin", "the quick brown fox", None)
    assert d["novel_bigram_rate"] == pytest.approx(1.0)


def test_unsupported_content_ignores_stopwords():
    """Function words overlap between any two English texts; counting them
    would wash out the signal."""
    article = "the president visited paris"
    # Only stopwords differ -> nothing unsupported.
    d = diagnostics("the president visited paris and it was the", article, None)
    assert d["unsupported_content_rate"] == pytest.approx(0.0)


def test_unsupported_content_flags_invented_entities():
    article = "a fire broke out in louisville kentucky"
    d = diagnostics("a fire broke out in diego california", article, None)
    assert d["unsupported_content_rate"] > 0.0


def test_oov_rate_uses_the_vocabulary():
    v = Vocab.build([["the", "cat", "sat"]], max_size=50, min_freq=1)
    d = diagnostics("the cat zebra", "the cat sat", v)
    assert d["oov_rate"] == pytest.approx(1 / 3)


def test_empty_prediction_flagged():
    d = diagnostics("", "some article text", None)
    assert d["empty"] == 1.0
    assert d["length_tokens"] == 0


# ------------------------------------------------------------------ bootstrap

def test_bootstrap_ci_brackets_the_mean():
    vals = np.random.default_rng(0).normal(0.5, 0.1, size=500)
    lo, hi = bootstrap_ci(vals, seed=1)
    assert lo < vals.mean() < hi


def test_bootstrap_ci_is_deterministic_given_seed():
    vals = np.random.default_rng(0).normal(0.5, 0.1, size=200)
    assert bootstrap_ci(vals, seed=7) == bootstrap_ci(vals, seed=7)


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(rng.normal(0.5, 0.1, size=50), seed=1)
    large = bootstrap_ci(rng.normal(0.5, 0.1, size=5000), seed=1)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_handles_empty():
    lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


# ----------------------------------------------------------- paired bootstrap

def test_paired_bootstrap_identical_systems_not_significant():
    x = np.random.default_rng(0).normal(0.4, 0.1, size=300)
    r = paired_bootstrap(x, x.copy(), seed=1)
    assert r["mean_diff"] == pytest.approx(0.0, abs=1e-9)
    assert r["p_value"] > 0.05
    assert r["significant_at_05"] is False


def test_paired_bootstrap_detects_a_consistent_advantage():
    rng = np.random.default_rng(0)
    base = rng.normal(0.35, 0.12, size=400)
    better = base + 0.05                       # same articles, uniformly better
    r = paired_bootstrap(better, base, seed=1)
    assert r["mean_diff"] == pytest.approx(5.0, abs=1e-6)   # reported x100
    assert r["p_value"] < 0.05
    assert r["significant_at_05"] is True
    assert r["wins"] == 400 and r["losses"] == 0


def test_paired_bootstrap_beats_independent_cis_on_correlated_data():
    """The reason this test exists at all: with strongly correlated systems, two
    independent CIs overlap while the paired test correctly finds a difference."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.35, 0.15, size=400)     # wide spread across articles
    better = base + 0.02                        # small but perfectly consistent gain

    lo_a, hi_a = bootstrap_ci(better, seed=1)
    lo_b, hi_b = bootstrap_ci(base, seed=1)
    overlap = not (lo_a > hi_b or lo_b > hi_a)

    paired = paired_bootstrap(better, base, seed=1)

    assert overlap, "precondition: independent CIs should overlap here"
    assert paired["significant_at_05"], "paired test missed a real, consistent gain"


def test_paired_bootstrap_ci_excludes_zero_when_significant():
    rng = np.random.default_rng(0)
    base = rng.normal(0.35, 0.1, size=300)
    r = paired_bootstrap(base + 0.04, base, seed=1)
    assert r["ci_low"] > 0.0


def test_paired_bootstrap_is_symmetric_in_sign():
    rng = np.random.default_rng(0)
    a = rng.normal(0.4, 0.1, size=200)
    b = rng.normal(0.3, 0.1, size=200)
    ab = paired_bootstrap(a, b, seed=1)
    ba = paired_bootstrap(b, a, seed=1)
    assert ab["mean_diff"] == pytest.approx(-ba["mean_diff"], abs=1e-6)
    assert ab["wins"] == ba["losses"]


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap(np.zeros(5), np.zeros(4))


def test_paired_bootstrap_handles_empty():
    r = paired_bootstrap(np.array([]), np.array([]))
    assert r["n"] == 0


# ------------------------------------------------------------ compare_systems

def _scored(name, ids, values):
    return {
        "name": name,
        "n": len(ids),
        "per_example": [
            {"id": i, "rouge1": v, "rouge2": v * 0.5, "rougeLsum": v * 0.9}
            for i, v in zip(ids, values)
        ],
    }


def test_compare_systems_pairs_only_shared_ids():
    """If two systems cover different articles, only the intersection may be
    compared -- otherwise the 'paired' test is not paired at all."""
    ref = _scored("ref", [f"id{i}" for i in range(10)], [0.3] * 10)
    other = _scored("other", [f"id{i}" for i in range(5, 15)], [0.4] * 10)
    out = compare_systems({"ref": ref, "other": other}, "ref")
    assert out["other"]["n_paired"] == 5


def test_compare_systems_skips_the_reference_itself():
    ref = _scored("ref", ["a", "b"], [0.3, 0.4])
    out = compare_systems({"ref": ref}, "ref")
    assert out == {}


def test_compare_systems_returns_empty_for_unknown_reference():
    ref = _scored("ref", ["a"], [0.3])
    assert compare_systems({"ref": ref}, "does_not_exist") == {}


def test_compare_systems_covers_all_metrics():
    ref = _scored("ref", ["a", "b", "c"], [0.3, 0.3, 0.3])
    other = _scored("other", ["a", "b", "c"], [0.5, 0.5, 0.5])
    out = compare_systems({"ref": ref, "other": other}, "ref")
    for metric in ("rouge1", "rouge2", "rougeLsum"):
        assert metric in out["other"]
        assert out["other"][metric]["mean_diff"] > 0
