"""LSTM encoder (bidirectional by default)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """Embedding -> (Bi)LSTM.

    Returns per-step hidden states for attention, plus an initial decoder state
    built by projecting the encoder's final state. For the bidirectional case the
    forward and backward final states are concatenated before projection, so the
    decoder starts from a summary of the whole article rather than of one
    direction only.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Encoder output width seen by attention and by the decoder's context input.
        self.output_size = hidden_size * self.num_directions

        # Bridge: encoder final state -> decoder initial state.
        self.bridge_h = nn.Linear(self.output_size, hidden_size)
        self.bridge_c = nn.Linear(self.output_size, hidden_size)

    def forward(
        self, src: torch.Tensor, src_len: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        src      (B, S) padded token ids
        src_len  (B,)   true lengths

        returns
        memory   (B, S, output_size)
        state    (h, c), each (num_layers, B, hidden_size)
        """
        emb = self.dropout(self.embedding(src))

        # pack_padded_sequence needs lengths on CPU; MPS/CUDA tensors are rejected.
        packed = pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        memory, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1)
        )
        memory = self.dropout(memory)

        # h_n: (num_layers * num_directions, B, hidden). Fold directions into width.
        b = src.size(0)
        h_n = h_n.view(self.num_layers, self.num_directions, b, self.hidden_size)
        c_n = c_n.view(self.num_layers, self.num_directions, b, self.hidden_size)
        h_cat = torch.cat([h_n[:, i] for i in range(self.num_directions)], dim=-1)
        c_cat = torch.cat([c_n[:, i] for i in range(self.num_directions)], dim=-1)

        h_0 = torch.tanh(self.bridge_h(h_cat))
        c_0 = torch.tanh(self.bridge_c(c_cat))
        return memory, (h_0.contiguous(), c_0.contiguous())
