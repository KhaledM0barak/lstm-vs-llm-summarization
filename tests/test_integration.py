"""End-to-end: synthetic corpus -> vocab -> train -> generate -> evaluate.

Runs the real entry points on a tiny synthetic dataset, so it exercises the same
code path as a full run in a few seconds. This is the test that would catch an
interface break between two components that are individually fine.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / "bin" / "python")


def run(args, **kw):
    """Run a module entry point exactly as a user would."""
    return subprocess.run(
        [PY, "-m", *args], cwd=REPO, capture_output=True, text=True, timeout=900, **kw
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A synthetic corpus with real structure: the summary is a compression of
    the article, so a model can actually learn something and ROUGE is meaningful.
    """
    d = tmp_path_factory.mktemp("data")
    rng = __import__("random").Random(0)
    subjects = ["the mayor", "a fire", "the company", "police", "the team"]
    verbs = ["announced", "reported", "confirmed", "denied", "investigated"]
    objects = ["a new policy", "the incident", "record profits", "the claims", "the results"]
    places = ["in london", "in berlin", "in toronto", "in madrid", "in tokyo"]

    def make(i):
        s, v, o, p = (rng.choice(x) for x in (subjects, verbs, objects, places))
        body = " ".join(
            f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(places)}."
            for _ in range(6)
        )
        article = f"{s} {v} {o} {p}. {body}"
        summary = f"{s} {v} {o} ."
        return {
            "id": f"doc{i}",
            "article": article,
            "summary": summary,
            "src_len": len(article.split()),
            "tgt_len": len(summary.split()),
        }

    for split, n in (("train", 240), ("validation", 32), ("test", 32)):
        with (d / f"{split}.jsonl").open("w") as f:
            for i in range(n):
                f.write(json.dumps(make(i)) + "\n")
    return d


@pytest.fixture(scope="module")
def vocab_file(corpus):
    out = corpus / "vocab.json"
    r = run(["src.data.build_vocab", "--train-file", str(corpus / "train.jsonl"),
             "--out", str(out), "--max-size", "500", "--min-freq", "1"])
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.exists()
    return out


@pytest.fixture(scope="module")
def trained(corpus, vocab_file, tmp_path_factory):
    """Train a small model through the real training entry point."""
    run_dir = tmp_path_factory.mktemp("run")
    cfg = tmp_path_factory.mktemp("cfg") / "test.yaml"
    cfg.write_text(f"""
run_name: itest
out_dir: {run_dir}
train_file: {corpus / 'train.jsonl'}
val_file: {corpus / 'validation.jsonl'}
vocab_file: {vocab_file}
seed: 1234
max_src_len: 64
max_tgt_len: 16
emb_dim: 32
hidden_size: 32
enc_layers: 1
dec_layers: 1
bidirectional: true
attention: bahdanau
attn_size: 32
dropout: 0.1
input_feeding: true
tie_embeddings: true
batch_size: 8
lr: 0.002
lr_decay: 0.5
epochs: 2
patience: 2
clip: 5.0
label_smoothing: 0.1
""")
    r = run(["src.train", "--config", str(cfg), "--device", "cpu"])
    assert r.returncode == 0, r.stderr[-3000:]
    return run_dir


def test_vocab_built_with_specials_first(vocab_file):
    blob = json.loads(vocab_file.read_text())
    assert blob["itos"][:4] == ["<pad>", "<unk>", "<bos>", "<eos>"]


def test_training_produces_checkpoint_and_summary(trained):
    assert (trained / "best.pt").exists()
    assert (trained / "train_summary.json").exists()


def test_training_summary_records_provenance(trained):
    s = json.loads((trained / "train_summary.json").read_text())
    for key in ("parameters", "hardware", "device", "total_train_seconds", "history", "config"):
        assert key in s, f"train_summary.json is missing {key}"
    assert s["parameters"]["total"] > 0
    assert len(s["history"]) >= 1


def test_model_actually_learns(trained):
    """Loss must fall. If it doesn't, something upstream is misaligned -- most
    likely the target shift -- even though nothing raised."""
    s = json.loads((trained / "train_summary.json").read_text())
    hist = s["history"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"], \
        f"training loss did not decrease: {[h['train_loss'] for h in hist]}"


def test_checkpoint_reloads_and_matches_config(trained):
    ck = torch.load(trained / "best.pt", map_location="cpu", weights_only=False)
    assert "model_state" in ck and "model_config" in ck
    assert ck["model_config"]["attention"] == "bahdanau"


@pytest.fixture(scope="module")
def predictions(trained, corpus, vocab_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("preds") / "preds.jsonl"
    r = run(["src.generate", "--checkpoint", str(trained / "best.pt"),
             "--test-file", str(corpus / "test.jsonl"),
             "--vocab-file", str(vocab_file),
             "--out", str(out), "--decode", "beam", "--beam-size", "2",
             "--min-len", "3", "--max-len", "16", "--device", "cpu"])
    assert r.returncode == 0, r.stderr[-3000:]
    return out


def test_generation_covers_every_test_example(predictions, corpus):
    preds = [json.loads(l) for l in predictions.open()]
    expected = [json.loads(l) for l in (corpus / "test.jsonl").open()]
    assert len(preds) == len(expected)
    assert [p["id"] for p in preds] == [e["id"] for e in expected], \
        "predictions are out of order relative to the test file"


def test_generation_produces_non_empty_text(predictions):
    preds = [json.loads(l) for l in predictions.open()]
    non_empty = [p for p in preds if p["prediction"].strip()]
    assert len(non_empty) >= 0.9 * len(preds), "most generations came back empty"


def test_generation_writes_metadata(predictions):
    meta = json.loads(predictions.with_suffix(".meta.json").read_text())
    for key in ("decode", "beam_size", "hardware", "total_seconds", "parameters"):
        assert key in meta


@pytest.fixture(scope="module")
def evaluated(predictions, corpus, vocab_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    r = run(["src.evaluate", "--test-file", str(corpus / "test.jsonl"),
             "--vocab-file", str(vocab_file),
             "--system", f"model={predictions}",
             "--out-dir", str(out)])
    assert r.returncode == 0, r.stderr[-3000:]
    return out


def test_evaluation_emits_all_artifacts(evaluated):
    for name in ("results.json", "results.md", "per_example_scores.json"):
        assert (evaluated / name).exists(), f"missing {name}"


def test_evaluation_scores_are_in_range(evaluated):
    res = json.loads((evaluated / "results.json").read_text())
    for system, agg in res["overall"].items():
        for rt in ("rouge1", "rouge2", "rougeLsum"):
            m = agg[rt]["mean"]
            assert 0.0 <= m <= 100.0, f"{system} {rt} out of range: {m}"
            assert agg[rt]["ci_low"] <= m <= agg[rt]["ci_high"], \
                f"{system} {rt} mean outside its own CI"


def test_evaluation_adds_lead3_automatically(evaluated):
    res = json.loads((evaluated / "results.json").read_text())
    assert "lead3_baseline" in res["overall"]


def test_evaluation_runs_paired_tests(evaluated):
    """The paired bootstrap must actually fire when >1 system is present."""
    res = json.loads((evaluated / "results.json").read_text())
    assert "paired_tests" in res
    assert res["paired_tests"], "paired tests were empty"
    for _, t in res["paired_tests"].items():
        assert "rouge1" in t and "p_value" in t["rouge1"]
        assert 0.0 <= t["rouge1"]["p_value"] <= 1.0


def test_every_system_scored_on_the_same_count(evaluated):
    """A like-for-like comparison requires equal N."""
    res = json.loads((evaluated / "results.json").read_text())
    counts = {n: c["scored"] for n, c in res["coverage"].items()}
    assert len(set(counts.values())) == 1, f"systems scored on different subsets: {counts}"


def test_results_markdown_has_the_report_tables(evaluated):
    md = (evaluated / "results.md").read_text()
    for heading in ("## Overall", "## Diagnostics", "## Paired bootstrap"):
        assert heading in md, f"results.md is missing '{heading}'"


def test_qualitative_report_builds(evaluated, predictions, corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("qual") / "qualitative.md"
    r = run(["src.qualitative",
             "--scores", str(evaluated / "per_example_scores.json"),
             "--test-file", str(corpus / "test.jsonl"),
             "--system", f"model={predictions}",
             "--system", f"lead3={predictions}",
             "--lstm-system", "model", "--llm-system", "lead3_baseline",
             "--out", str(out)])
    assert r.returncode == 0, r.stderr[-3000:]
    text = out.read_text()
    assert "# Qualitative comparison" in text
    assert text.count("## ") >= 10, "assignment requires at least 10 side-by-side examples"


def test_smoke_config_runs_end_to_end():
    """The command in the README that a grader is most likely to try first."""
    if not (REPO / "data" / "processed" / "vocab.json").exists():
        pytest.skip("real dataset not present")
    r = run(["src.train", "--config", "configs/smoke.yaml", "--device", "cpu"])
    assert r.returncode == 0, r.stderr[-3000:]
