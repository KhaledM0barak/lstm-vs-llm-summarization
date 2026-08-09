"""Model correctness: masking, shapes, loss equivalence, generation invariants.

The attention mask and the chunked loss are the two places where a silent bug
would change every reported number without raising anything, so both are
verified against explicit references rather than smoke-tested.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.vocab import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from src.models.attention import BahdanauAttention, LuongAttention, NoAttention
from src.models.seq2seq import ModelConfig, Seq2Seq, _block_repeat_trigrams
from src.utils.seed import set_seed

VOCAB = 200
torch.manual_seed(0)


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=VOCAB, emb_dim=32, hidden_size=32, attn_size=32,
        enc_layers=1, dec_layers=1, bidirectional=True,
        attention="bahdanau", dropout=0.0, input_feeding=True, tie_embeddings=True,
    )


@pytest.fixture
def model(cfg):
    set_seed(0)
    m = Seq2Seq(cfg)
    m.eval()
    return m


def make_batch(src_lens=(7, 4), tgt_lens=(5, 3), max_s=8, max_t=6):
    b = len(src_lens)
    src = torch.full((b, max_s), PAD_ID, dtype=torch.long)
    tgt_in = torch.full((b, max_t), PAD_ID, dtype=torch.long)
    tgt_out = torch.full((b, max_t), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_lens, tgt_lens)):
        src[i, :s] = torch.randint(4, VOCAB, (s,))
        tgt_in[i, 0] = BOS_ID
        toks = torch.randint(4, VOCAB, (t,))
        tgt_in[i, 1 : t + 1] = toks
        tgt_out[i, :t] = toks
        tgt_out[i, t] = EOS_ID
    return {
        "src": src,
        "src_len": torch.tensor(src_lens),
        "src_mask": src.ne(PAD_ID),
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "tgt_mask": tgt_out.ne(PAD_ID),
    }


# ----------------------------------------------------------------- attention

@pytest.mark.parametrize("cls", [BahdanauAttention, LuongAttention])
def test_attention_assigns_zero_weight_to_padding(cls):
    """The single most important invariant in the model. Without it the decoder
    puts probability mass on <pad> for every short article in a mixed batch."""
    torch.manual_seed(0)
    attn = cls(decoder_size=16, encoder_size=16, attn_size=16)
    query = torch.randn(2, 16)
    memory = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, False, False],
                         [True, True, False, False, False]])

    _, weights = attn(query, memory, mask)

    assert torch.allclose(weights[~mask], torch.zeros(1), atol=1e-6), \
        "attention put weight on padded positions"
    assert torch.allclose(weights.sum(-1), torch.ones(2), atol=1e-5), \
        "attention weights must still sum to 1 over real positions"


def test_attention_context_ignores_padded_content():
    """Changing what sits at a padded position must not change the context."""
    torch.manual_seed(0)
    attn = BahdanauAttention(16, 16, 16)
    query = torch.randn(1, 16)
    memory = torch.randn(1, 5, 16)
    mask = torch.tensor([[True, True, True, False, False]])

    c1, _ = attn(query, memory, mask)
    memory[:, 3:, :] = 999.0                      # garbage in the padded region
    c2, _ = attn(query, memory, mask)

    assert torch.allclose(c1, c2, atol=1e-5), "padding leaked into the context vector"


def test_luong_batched_matches_per_step():
    """The fast batched path must be numerically identical to the loop."""
    torch.manual_seed(0)
    attn = LuongAttention(16, 16, 16)
    queries = torch.randn(2, 4, 16)
    memory = torch.randn(2, 6, 16)
    mask = torch.tensor([[True] * 5 + [False], [True] * 3 + [False] * 3])

    ctx_b, w_b = attn.forward_batched(queries, memory, mask)
    for t in range(4):
        ctx_s, w_s = attn(queries[:, t], memory, mask)
        assert torch.allclose(ctx_b[:, t], ctx_s, atol=1e-5)
        assert torch.allclose(w_b[:, t], w_s, atol=1e-5)


def test_no_attention_ignores_padding_in_its_mean():
    attn = NoAttention(16, 16, 16)
    memory = torch.randn(1, 5, 16)
    mask = torch.tensor([[True, True, True, False, False]])
    c1, _ = attn(torch.randn(1, 16), memory, mask)
    memory[:, 3:, :] = 999.0
    c2, _ = attn(torch.randn(1, 16), memory, mask)
    assert torch.allclose(c1, c2, atol=1e-5)


def test_attention_batched_support_flags():
    """Additive attention cannot be batched over decoder steps; the decoder
    relies on this flag to pick the right path."""
    assert BahdanauAttention.supports_batched is False
    assert LuongAttention.supports_batched is True
    assert NoAttention.supports_batched is True


# --------------------------------------------------------------------- model

def test_forward_shapes(model, cfg):
    batch = make_batch()
    h = model(batch)
    assert h.shape == (2, 6, cfg.hidden_size)


def test_parameter_count_and_tying(model, cfg):
    p = model.num_parameters()
    assert p["embedding"] == VOCAB * cfg.emb_dim
    assert model.decoder.out_proj.weight.data_ptr() == model.embedding.weight.data_ptr(), \
        "output projection is not weight-tied to the embedding"


def test_padding_embedding_is_zero(model):
    assert torch.allclose(model.embedding.weight[PAD_ID], torch.zeros(1))


def test_encoder_output_width_matches_directions(cfg):
    set_seed(0)
    bi = Seq2Seq(cfg)
    assert bi.encoder.output_size == cfg.hidden_size * 2
    uni = Seq2Seq(ModelConfig(**{**cfg.__dict__, "bidirectional": False}))
    assert uni.encoder.output_size == cfg.hidden_size


def test_mismatched_layer_counts_rejected(cfg):
    with pytest.raises(ValueError, match="enc_layers must equal dec_layers"):
        Seq2Seq(ModelConfig(**{**cfg.__dict__, "enc_layers": 2, "dec_layers": 1}))


def test_tie_embeddings_requires_matching_dims(cfg):
    with pytest.raises(ValueError, match="tie_embeddings"):
        Seq2Seq(ModelConfig(**{**cfg.__dict__, "hidden_size": 64, "emb_dim": 32}))


def test_encoder_ignores_padded_source_content(model):
    """pack_padded_sequence must make the encoder blind to padding."""
    batch = make_batch(src_lens=(7, 4))
    m1, (h1, _) = model.encoder(batch["src"], batch["src_len"])
    src2 = batch["src"].clone()
    src2[1, 4:] = 99                                   # garbage after the real tokens
    m2, (h2, _) = model.encoder(src2, batch["src_len"])
    assert torch.allclose(h1, h2, atol=1e-5), "encoder state depends on padding"


# ---------------------------------------------------------------------- loss

def test_chunked_loss_matches_unchunked_reference(model):
    """The chunked projection was introduced to avoid a 1.28 GB tensor. It must
    be numerically identical to projecting everything at once."""
    batch = make_batch()
    h = model(batch)

    loss_sum, nll_sum, ntok = model.loss_from_states(
        h, batch["tgt_out"], label_smoothing=0.1, chunk=2
    )

    logits = model.decoder.project(h).reshape(-1, VOCAB)
    target = batch["tgt_out"].reshape(-1)
    ref_loss = F.cross_entropy(
        logits, target, ignore_index=PAD_ID, label_smoothing=0.1, reduction="sum"
    )
    ref_nll = F.cross_entropy(logits, target, ignore_index=PAD_ID, reduction="sum")

    assert torch.allclose(loss_sum, ref_loss, atol=1e-4)
    assert torch.allclose(nll_sum, ref_nll, atol=1e-4)
    assert ntok == int(batch["tgt_out"].ne(PAD_ID).sum())


def test_loss_chunk_size_does_not_change_result(model):
    batch = make_batch()
    h = model(batch)
    a, _, _ = model.loss_from_states(h, batch["tgt_out"], 0.1, chunk=1)
    b, _, _ = model.loss_from_states(h, batch["tgt_out"], 0.1, chunk=100)
    assert torch.allclose(a, b, atol=1e-4)


def test_loss_ignores_padded_targets(model):
    """Extending the batch with pure padding must not change the loss.

    Note the test has to *add* padding rather than overwrite existing padding
    with a token id -- overwriting turns those positions into real supervised
    targets, which would change the loss legitimately.
    """
    batch = make_batch()
    h = model(batch)
    a, _, ntok_a = model.loss_from_states(h, batch["tgt_out"], 0.1)

    pad_cols = 4
    tgt_padded = F.pad(batch["tgt_out"], (0, pad_cols), value=PAD_ID)
    h_padded = torch.cat([h, torch.randn(h.size(0), pad_cols, h.size(2))], dim=1)

    b, _, ntok_b = model.loss_from_states(h_padded, tgt_padded, 0.1)

    assert ntok_a == ntok_b, "padded positions were counted as target tokens"
    assert torch.allclose(a, b, atol=1e-5), "loss counted padded positions"


def test_gradients_reach_every_component(model):
    batch = make_batch()
    h = model(batch)
    loss_sum, _, ntok = model.loss_from_states(h, batch["tgt_out"], 0.1)
    (loss_sum / ntok).backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


# ---------------------------------------------------------------- generation

def test_greedy_never_emits_pad_or_unk(model):
    batch = make_batch()
    out, _ = model.generate_greedy(
        batch["src"], batch["src_len"], batch["src_mask"], max_len=12, min_len=2
    )
    for seq in out:
        assert PAD_ID not in seq and UNK_ID not in seq


def test_beam_never_emits_pad_or_unk(model):
    batch = make_batch()
    out = model.generate_beam(
        batch["src"], batch["src_len"], batch["src_mask"],
        beam_size=3, max_len=12, min_len=2,
    )
    for seq in out:
        assert PAD_ID not in seq and UNK_ID not in seq


def test_generation_respects_min_len(model):
    batch = make_batch()
    out = model.generate_beam(
        batch["src"], batch["src_len"], batch["src_mask"],
        beam_size=3, max_len=20, min_len=8,
    )
    for seq in out:
        assert len(seq) >= 8, f"min_len violated: produced {len(seq)} tokens"


def test_generation_respects_max_len(model):
    batch = make_batch()
    out, _ = model.generate_greedy(
        batch["src"], batch["src_len"], batch["src_mask"], max_len=6, min_len=1
    )
    for seq in out:
        assert len(seq) <= 6


def test_beam_is_deterministic(model):
    batch = make_batch()
    kw = dict(beam_size=3, max_len=12, min_len=2)
    a = model.generate_beam(batch["src"], batch["src_len"], batch["src_mask"], **kw)
    b = model.generate_beam(batch["src"], batch["src_len"], batch["src_mask"], **kw)
    assert a == b


def test_beam_returns_one_sequence_per_input(model):
    batch = make_batch(src_lens=(7, 4, 6), tgt_lens=(3, 3, 3), max_s=8, max_t=6)
    out = model.generate_beam(
        batch["src"], batch["src_len"], batch["src_mask"], beam_size=4, max_len=10, min_len=2
    )
    assert len(out) == 3


# ------------------------------------------------------- trigram blocking

def test_trigram_blocking_bans_the_completing_token():
    """Given ... a b c ... a b, the token c must be banned -- it would complete a
    repeated trigram."""
    logits = torch.zeros(1, 50)
    prefix = [[10, 11, 12, 20, 10, 11]]
    _block_repeat_trigrams(logits, prefix)
    assert logits[0, 12] == float("-inf"), "did not ban the repeat-completing token"
    assert torch.isfinite(logits[0, 13]), "banned an unrelated token"


def test_trigram_blocking_noop_on_short_prefix():
    logits = torch.zeros(1, 50)
    _block_repeat_trigrams(logits, [[7]])
    assert torch.isfinite(logits).all()


def test_trigram_blocking_is_per_row():
    logits = torch.zeros(2, 50)
    _block_repeat_trigrams(logits, [[10, 11, 12, 20, 10, 11], [1, 2, 3]])
    assert logits[0, 12] == float("-inf")
    assert torch.isfinite(logits[1]).all()
