"""Data pipeline: tokenization, vocabulary, batching, masking.

These target the places where a bug is silent, a wrong mask or a misaligned
target shifts every number in the report without raising anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import (
    SRC_LEN_MULTIPLE,
    TGT_LEN_MULTIPLE,
    BucketBatchSampler,
    collate_batch,
)
from src.data.tokenizer import detokenize, normalize, split_sentences, tokenize
from src.data.vocab import BOS_ID, EOS_ID, PAD_ID, UNK_ID, Vocab


# ----------------------------------------------------------------- tokenizer

def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize("  The   U.S.\n\nSAID  ") == "the u.s. said"


def test_tokenize_splits_punctuation_from_words():
    assert tokenize("hello, world!") == ["hello", ",", "world", "!"]


def test_tokenize_keeps_numbers_with_separators_together():
    toks = tokenize("it cost 3.5 million on 2015-04-01")
    assert "3.5" in toks
    assert "2015-04-01" in toks


@pytest.mark.parametrize(
    "text,expected",
    [
        # Assert the actual split, not just the count -- a count-only check
        # passes even when the boundary lands in the wrong place.
        ("the u.s. said no. he left.", ["the u.s. said no.", "he left."]),
        ("mr. smith met dr. jones. they talked.",
         ["mr. smith met dr. jones.", "they talked."]),
        # Sentence-final digits must terminate: "jan." merges as an
        # abbreviation, but "4." must not.
        ("it happened on jan. 4. everyone saw.",
         ["it happened on jan. 4.", "everyone saw."]),
        # Personal initials must not terminate.
        ("mr. j. smith left. he waved.", ["mr. j. smith left.", "he waved."]),
        # "no" is an ordinary word far more often than "No." the abbreviation.
        ("she said no. then she left.", ["she said no.", "then she left."]),
        ("it cost 3.5 million. that is a lot.",
         ["it cost 3.5 million.", "that is a lot."]),
        ("one. two. three.", ["one.", "two.", "three."]),
        ("no terminator here", ["no terminator here"]),
    ],
)
def test_split_sentences_boundaries(text, expected):
    """Without abbreviation protection this over-splits; with too aggressive
    protection it under-splits. Both shorten or corrupt Lead-3 and ROUGE-Lsum."""
    assert split_sentences(text) == expected


def test_split_sentences_handles_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_detokenize_reattaches_punctuation():
    assert detokenize(["hello", ",", "world", "."]) == "hello, world."


# -------------------------------------------------------------------- vocab

@pytest.fixture
def small_vocab():
    streams = [["the", "cat", "sat"], ["the", "dog", "sat"], ["the", "cat"]]
    return Vocab.build(streams, max_size=100, min_freq=1)


def test_vocab_reserves_special_ids(small_vocab):
    assert small_vocab.itos[PAD_ID] == "<pad>"
    assert small_vocab.itos[UNK_ID] == "<unk>"
    assert small_vocab.itos[BOS_ID] == "<bos>"
    assert small_vocab.itos[EOS_ID] == "<eos>"


def test_vocab_encode_decode_roundtrip(small_vocab):
    toks = ["the", "cat", "sat"]
    assert small_vocab.decode(small_vocab.encode(toks)) == toks


def test_vocab_maps_unknown_to_unk(small_vocab):
    assert small_vocab.encode(["zebra"]) == [UNK_ID]


def test_vocab_build_is_deterministic():
    """Ties must break alphabetically or the vocabulary differs across runs and
    every embedding index shifts."""
    streams = [["b", "a", "c"], ["a", "b", "c"]]
    v1 = Vocab.build(list(streams), max_size=50, min_freq=1)
    v2 = Vocab.build(list(streams), max_size=50, min_freq=1)
    assert v1.itos == v2.itos


def test_vocab_respects_min_freq():
    v = Vocab.build([["a", "a", "b"]], max_size=50, min_freq=2)
    assert "a" in v.stoi
    assert "b" not in v.stoi


def test_vocab_respects_max_size():
    streams = [[f"w{i}" for i in range(100)]]
    v = Vocab.build(streams, max_size=10, min_freq=1)
    assert len(v) == 10


def test_decode_stops_at_eos(small_vocab):
    ids = small_vocab.encode(["the", "cat"]) + [EOS_ID] + small_vocab.encode(["sat"])
    assert small_vocab.decode(ids) == ["the", "cat"]


def test_vocab_save_load_roundtrip(small_vocab, tmp_path):
    p = tmp_path / "v.json"
    small_vocab.save(p)
    loaded = Vocab.load(p)
    assert loaded.itos == small_vocab.itos
    assert loaded.stoi == small_vocab.stoi


# ------------------------------------------------------------------ collate

def make_examples(lengths, tgt_lengths):
    return [
        {
            "index": i,
            "src": np.arange(4, 4 + s, dtype=np.int32),
            "tgt": np.arange(4, 4 + t, dtype=np.int32),
        }
        for i, (s, t) in enumerate(zip(lengths, tgt_lengths))
    ]


def test_collate_pads_to_quantized_multiples():
    """Shape quantization is what keeps MPS from recompiling a kernel per step."""
    batch = collate_batch(make_examples([10, 70], [5, 5]))
    assert batch["src"].shape[1] % SRC_LEN_MULTIPLE == 0
    assert batch["tgt_in"].shape[1] % TGT_LEN_MULTIPLE == 0
    assert batch["src"].shape[1] >= 70


def test_collate_mask_marks_exactly_the_real_tokens():
    batch = collate_batch(make_examples([10, 25], [5, 5]))
    assert batch["src_mask"][0].sum().item() == 10
    assert batch["src_mask"][1].sum().item() == 25
    # And the mask must agree with the padding itself.
    assert torch.equal(batch["src_mask"], batch["src"].ne(PAD_ID))


def test_collate_target_shift_is_correct():
    """tgt_in must be <bos> w1..wn and tgt_out must be w1..wn <eos>; an
    off-by-one here trains the model to predict the wrong token."""
    batch = collate_batch(make_examples([10], [3]))
    tgt_in, tgt_out = batch["tgt_in"][0], batch["tgt_out"][0]

    assert tgt_in[0].item() == BOS_ID
    assert tgt_in[1:4].tolist() == [4, 5, 6]
    assert tgt_out[0:3].tolist() == [4, 5, 6]
    assert tgt_out[3].item() == EOS_ID
    # Everything after the real target is padding on both sides.
    assert (tgt_out[4:] == PAD_ID).all()


def test_collate_tgt_mask_excludes_padding():
    batch = collate_batch(make_examples([10], [3]))
    # 3 real tokens + 1 eos = 4 supervised positions.
    assert batch["tgt_mask"][0].sum().item() == 4


def test_collate_src_len_is_true_length_not_padded():
    batch = collate_batch(make_examples([10, 25], [5, 5]))
    assert batch["src_len"].tolist() == [10, 25]


# ---------------------------------------------------------- bucket sampler

def test_bucket_sampler_covers_every_example_exactly_once():
    lengths = np.random.default_rng(0).integers(10, 400, size=250)
    sampler = BucketBatchSampler(lengths, batch_size=16, shuffle=True, seed=1)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(250)), "sampler lost or duplicated examples"


def test_bucket_sampler_len_matches_actual_batches():
    lengths = np.random.default_rng(0).integers(10, 400, size=250)
    sampler = BucketBatchSampler(lengths, batch_size=16, shuffle=False, seed=1)
    assert len(sampler) == len(list(sampler))


def test_bucket_sampler_drop_last_keeps_batch_size_constant():
    lengths = np.random.default_rng(0).integers(10, 400, size=250)
    sampler = BucketBatchSampler(
        lengths, batch_size=16, shuffle=True, seed=1, drop_last=True
    )
    sizes = {len(b) for b in sampler}
    assert sizes == {16}, f"expected uniform batches, got sizes {sizes}"


def test_bucket_sampler_drops_under_two_percent():
    lengths = np.random.default_rng(0).integers(10, 400, size=80000)
    sampler = BucketBatchSampler(
        lengths, batch_size=64, shuffle=True, seed=1, drop_last=True
    )
    kept = sum(len(b) for b in sampler)
    assert (80000 - kept) / 80000 < 0.02


def test_bucket_sampler_reshuffles_each_epoch():
    """A different remainder must be dropped each epoch, or the same examples are
    permanently excluded from training."""
    lengths = np.random.default_rng(0).integers(10, 400, size=500)
    sampler = BucketBatchSampler(lengths, batch_size=16, shuffle=True, seed=1)
    sampler.set_epoch(0)
    first = [b[:] for b in sampler]
    sampler.set_epoch(1)
    second = [b[:] for b in sampler]
    assert first != second


def test_bucket_sampler_groups_similar_lengths():
    """The whole point of bucketing: within a batch, lengths should be close."""
    lengths = np.random.default_rng(0).integers(10, 400, size=1000)
    sampler = BucketBatchSampler(lengths, batch_size=16, shuffle=True, seed=1)
    spreads = [lengths[b].max() - lengths[b].min() for b in sampler]
    assert np.mean(spreads) < 40, "bucketing is not grouping similar lengths"


# ------------------------------------------------------- jsonl robustness

def test_read_jsonl_tolerates_a_truncated_final_line(tmp_path, capsys):
    """A process killed mid-write leaves a partial line. Refusing to parse the
    file would make an interrupted run unresumable, losing all completed work."""
    from src.data.dataset import read_jsonl

    f = tmp_path / "partial.jsonl"
    f.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "predicti')
    rows = read_jsonl(f)

    assert [r["id"] for r in rows] == ["a", "b"]
    assert "truncated" in capsys.readouterr().err


def test_read_jsonl_raises_on_corruption_mid_file(tmp_path):
    """A bad line followed by more content is corruption, not an interrupted
    append, and must not be silently skipped."""
    from src.data.dataset import read_jsonl

    f = tmp_path / "corrupt.jsonl"
    f.write_text('{"id": "a"}\n{"id": BROKEN\n{"id": "c"}\n')
    with pytest.raises(ValueError, match="corrupted rather than truncated"):
        read_jsonl(f)


def test_read_jsonl_handles_blank_lines(tmp_path):
    from src.data.dataset import read_jsonl

    f = tmp_path / "blanks.jsonl"
    f.write_text('{"id": "a"}\n\n\n{"id": "b"}\n')
    assert len(read_jsonl(f)) == 2
