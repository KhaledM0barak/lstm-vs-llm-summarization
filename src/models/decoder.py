"""LSTM decoder with attention and optional input feeding."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import build_attention


class Decoder(nn.Module):
    """Embedding -> LSTM -> attention -> attentional vector -> output projection.

    Input feeding (Luong et al., 2015) concatenates the previous step's
    attentional vector onto the current input embedding, so the decoder knows what
    it already attended to. It forces a step-by-step loop; disabling it lets the
    LSTM run over the whole target in one call, which is markedly faster and is
    exposed as an ablation.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        encoder_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        attention: str = "bahdanau",
        attn_size: int = 256,
        dropout: float = 0.3,
        input_feeding: bool = True,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_feeding = input_feeding

        self.attention = build_attention(attention, hidden_size, encoder_size, attn_size)
        self.attn_out_size = self.attention.output_size

        lstm_input = embedding.embedding_dim + (hidden_size if input_feeding else 0)
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Luong's attentional vector: h_tilde = tanh(W_c [h_t ; c_t]).
        self.attn_combine = nn.Linear(hidden_size + self.attn_out_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(hidden_size, embedding.num_embeddings)
        if tie_embeddings:
            if hidden_size != embedding.embedding_dim:
                raise ValueError(
                    "tie_embeddings requires hidden_size == emb_dim "
                    f"(got {hidden_size} vs {embedding.embedding_dim})"
                )
            self.out_proj.weight = embedding.weight

    def init_feed(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def step(
        self,
        y_prev: torch.Tensor,                     # (B,) token ids
        state: tuple[torch.Tensor, torch.Tensor],
        memory: torch.Tensor,                     # (B, S, E)
        src_mask: torch.Tensor,                   # (B, S)
        feed: torch.Tensor | None,                # (B, H) previous attentional vector
        memory_proj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple, torch.Tensor, torch.Tensor]:
        """One decoding step. Returns (logits, new_state, new_feed, attn_weights)."""
        emb = self.dropout(self.embedding(y_prev)).unsqueeze(1)  # (B,1,D)
        if self.input_feeding:
            emb = torch.cat([emb, feed.unsqueeze(1)], dim=-1)

        out, state = self.lstm(emb, state)
        h_t = out.squeeze(1)                                     # (B,H)

        context, weights = self.attention(h_t, memory, src_mask, memory_proj)
        h_tilde = torch.tanh(self.attn_combine(torch.cat([h_t, context], dim=-1)))
        logits = self.out_proj(self.dropout(h_tilde))
        return logits, state, h_tilde, weights

    def forward(
        self,
        tgt_in: torch.Tensor,                     # (B, T)
        state: tuple[torch.Tensor, torch.Tensor],
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced decoding over the whole target.

        Returns the decoder's attentional vectors (B, T, H) and attention weights
        (B, T, S) -- deliberately *not* logits.

        Projecting to the 50k vocabulary here would materialize a (B, T, V)
        tensor: at batch 64 and 100 steps that is 1.28 GB in fp32 before the loss
        function has copied it even once, which on a 24 GB machine drives the
        whole run into swap. The projection is instead applied in time-chunks by
        `Seq2Seq.loss_from_states`, which keeps the GEMMs large enough to be
        efficient while capping peak memory.
        """
        bsz, tgt_len = tgt_in.shape
        memory_proj = self.attention.project_memory(memory)
        h_seq, weights_seq = [], []

        if self.input_feeding:
            feed = self.init_feed(bsz, memory.device, memory.dtype)
            for t in range(tgt_len):
                emb = self.dropout(self.embedding(tgt_in[:, t])).unsqueeze(1)
                emb = torch.cat([emb, feed.unsqueeze(1)], dim=-1)
                out, state = self.lstm(emb, state)
                h_t = out.squeeze(1)
                context, weights = self.attention(h_t, memory, src_mask, memory_proj)
                feed = torch.tanh(self.attn_combine(torch.cat([h_t, context], dim=-1)))
                h_seq.append(feed)
                weights_seq.append(weights)
        elif getattr(self.attention, "supports_batched", False):
            # Fast path. Without input feeding the recurrence does not depend on
            # the attention output, and a multiplicative score can be computed
            # for every step at once -- so the entire decoder is three fused
            # kernels (LSTM, bmm, linear) with no Python-level loop at all.
            emb = self.dropout(self.embedding(tgt_in))
            out, _ = self.lstm(emb, state)                       # (B,T,H)
            context, weights = self.attention.forward_batched(out, memory, src_mask)
            h_tilde = torch.tanh(self.attn_combine(torch.cat([out, context], dim=-1)))
            return h_tilde, weights

        else:
            # No input feeding, but an attention whose score cannot be batched
            # (additive): one fused LSTM call, then per-step attention.
            emb = self.dropout(self.embedding(tgt_in))
            out, _ = self.lstm(emb, state)                       # (B,T,H)
            for t in range(tgt_len):
                h_t = out[:, t]
                context, weights = self.attention(h_t, memory, src_mask, memory_proj)
                h_seq.append(torch.tanh(self.attn_combine(torch.cat([h_t, context], dim=-1))))
                weights_seq.append(weights)

        h_tilde = torch.stack(h_seq, dim=1)                      # (B,T,H)
        return h_tilde, torch.stack(weights_seq, dim=1)

    def project(self, h_tilde: torch.Tensor) -> torch.Tensor:
        """Map attentional vectors to vocabulary logits."""
        return self.out_proj(self.dropout(h_tilde))
