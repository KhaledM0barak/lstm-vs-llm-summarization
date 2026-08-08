"""Attention mechanisms.

`BahdanauAttention` is the additive form (Bahdanau et al., 2015) and is the
default. `LuongAttention` (general/multiplicative, Luong et al., 2015) is included
so the report can ablate the scoring function itself, not just attention's
presence.

Both mask padded source positions before the softmax. Without that mask the
decoder can place probability mass on <pad>, which quietly corrupts the context
vector for every short article in a batch of long ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


class BahdanauAttention(nn.Module):
    """score(h_dec, h_enc) = v^T tanh(W_dec h_dec + W_enc h_enc)"""

    # Batching additive attention over decoder steps would materialize a
    # (B, T, S, attn_size) tensor -- ~330 GB at B=64, T=100, S=400 -- so this
    # variant is necessarily evaluated one step at a time.
    supports_batched = False

    def __init__(self, decoder_size: int, encoder_size: int, attn_size: int = 256) -> None:
        super().__init__()
        self.W_dec = nn.Linear(decoder_size, attn_size, bias=False)
        self.W_enc = nn.Linear(encoder_size, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)
        self.output_size = encoder_size

    def forward(
        self,
        query: torch.Tensor,      # (B, decoder_size)   decoder hidden at this step
        memory: torch.Tensor,     # (B, S, encoder_size) encoder outputs
        mask: torch.Tensor,       # (B, S) True at real positions
        memory_proj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # W_enc @ memory does not change across decoding steps, so the caller may
        # precompute it once per batch and pass it in.
        proj_mem = self.W_enc(memory) if memory_proj is None else memory_proj
        scores = self.v(torch.tanh(self.W_dec(query).unsqueeze(1) + proj_mem)).squeeze(-1)
        scores = scores.masked_fill(~mask, NEG_INF)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights

    def project_memory(self, memory: torch.Tensor) -> torch.Tensor:
        return self.W_enc(memory)


class LuongAttention(nn.Module):
    """score(h_dec, h_enc) = h_dec^T W h_enc  ('general' variant).

    Unlike the additive form, the multiplicative score is a plain matrix product,
    so it can be evaluated for every decoder step at once. That makes the whole
    decoder a handful of fused kernels instead of a Python loop, which is the
    difference between a run that finishes and one that does not on this
    hardware. See `forward_batched`.
    """

    supports_batched = True

    def __init__(self, decoder_size: int, encoder_size: int, attn_size: int = 256) -> None:
        super().__init__()
        self.W = nn.Linear(decoder_size, encoder_size, bias=False)
        self.output_size = encoder_size

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
        memory_proj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.bmm(memory, self.W(query).unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, NEG_INF)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights

    def forward_batched(
        self,
        queries: torch.Tensor,   # (B, T, H)
        memory: torch.Tensor,    # (B, S, E)
        mask: torch.Tensor,      # (B, S)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All decoder steps at once. Returns context (B,T,E), weights (B,T,S)."""
        scores = torch.bmm(self.W(queries), memory.transpose(1, 2))     # (B,T,S)
        scores = scores.masked_fill(~mask.unsqueeze(1), NEG_INF)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, memory)                            # (B,T,E)
        return context, weights

    def project_memory(self, memory: torch.Tensor) -> torch.Tensor:
        return None


class NoAttention(nn.Module):
    """Ablation: the decoder sees only the encoder's final state.

    Implemented as a fixed context (mean of unmasked encoder states) so the module
    interface is unchanged; this is the classic recurrent bottleneck the report
    compares against.
    """

    supports_batched = True

    def __init__(self, decoder_size: int, encoder_size: int, attn_size: int = 256) -> None:
        super().__init__()
        self.output_size = encoder_size

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
        memory_proj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = mask.unsqueeze(-1).to(memory.dtype)
        context = (memory * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        # Uniform weights over real positions, returned for interface parity.
        weights = m.squeeze(-1) / m.sum(dim=1).clamp(min=1.0)
        return context, weights

    def forward_batched(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, weights = self.forward(queries[:, 0], memory, mask)
        t = queries.size(1)
        return (
            context.unsqueeze(1).expand(-1, t, -1),
            weights.unsqueeze(1).expand(-1, t, -1),
        )

    def project_memory(self, memory: torch.Tensor) -> torch.Tensor:
        return None


ATTENTION_TYPES = {
    "bahdanau": BahdanauAttention,
    "luong": LuongAttention,
    "none": NoAttention,
}


def build_attention(kind: str, decoder_size: int, encoder_size: int, attn_size: int):
    if kind not in ATTENTION_TYPES:
        raise ValueError(f"unknown attention '{kind}', expected one of {sorted(ATTENTION_TYPES)}")
    return ATTENTION_TYPES[kind](decoder_size, encoder_size, attn_size)
