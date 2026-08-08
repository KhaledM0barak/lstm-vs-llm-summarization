"""Full encoder-decoder model plus greedy and beam-search generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.data.vocab import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from src.models.decoder import Decoder
from src.models.encoder import Encoder


@dataclass
class ModelConfig:
    vocab_size: int = 50_000
    emb_dim: int = 256
    hidden_size: int = 256
    enc_layers: int = 1
    dec_layers: int = 1
    bidirectional: bool = True
    attention: str = "bahdanau"
    attn_size: int = 256
    dropout: float = 0.3
    input_feeding: bool = True
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class Seq2Seq(nn.Module):
    """Embedding -> BiLSTM encoder -> attention -> LSTM decoder -> output projection."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # One embedding table shared by encoder, decoder, and (optionally) the
        # output projection.
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=PAD_ID)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg.emb_dim**-0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].fill_(0.0)

        self.encoder = Encoder(
            embedding=self.embedding,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.enc_layers,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout,
        )
        self.decoder = Decoder(
            embedding=self.embedding,
            encoder_size=self.encoder.output_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.dec_layers,
            attention=cfg.attention,
            attn_size=cfg.attn_size,
            dropout=cfg.dropout,
            input_feeding=cfg.input_feeding,
            tie_embeddings=cfg.tie_embeddings,
        )

        if cfg.enc_layers != cfg.dec_layers:
            raise ValueError(
                "enc_layers must equal dec_layers: the bridge maps the encoder's "
                "per-layer final state directly onto the decoder's initial state"
            )

    def forward(self, batch: dict) -> torch.Tensor:
        memory, state = self.encoder(batch["src"], batch["src_len"])
        logits, _ = self.decoder(batch["tgt_in"], state, memory, batch["src_mask"])
        return logits

    def num_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Tied weights are one tensor shared by two modules, so they are counted
        # once here; the embedding table dominates either way.
        return {
            "total": total,
            "trainable": trainable,
            "embedding": self.embedding.weight.numel(),
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
        }

    # ---------------------------------------------------------------- decoding

    @torch.no_grad()
    def generate_greedy(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int = 100,
        min_len: int = 10,
        block_trigram: bool = False,
        return_attention: bool = False,
    ) -> tuple[list[list[int]], torch.Tensor | None]:
        self.eval()
        memory, state = self.encoder(src, src_len)
        memory_proj = self.decoder.attention.project_memory(memory)

        bsz = src.size(0)
        device = src.device
        y = torch.full((bsz,), BOS_ID, dtype=torch.long, device=device)
        feed = self.decoder.init_feed(bsz, device, memory.dtype)

        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        outputs: list[list[int]] = [[] for _ in range(bsz)]
        attn_steps = []

        for t in range(max_len):
            logits, state, feed, weights = self.decoder.step(
                y, state, memory, src_mask, feed, memory_proj
            )
            # Never emit padding or unknown at inference: <unk> in a summary is a
            # pure error, and greedy decoding will otherwise produce it for rare
            # entities.
            logits[:, PAD_ID] = float("-inf")
            logits[:, UNK_ID] = float("-inf")
            if t < min_len:
                logits[:, EOS_ID] = float("-inf")
            if block_trigram:
                _block_repeat_trigrams(logits, outputs)

            y = logits.argmax(dim=-1)
            if return_attention:
                attn_steps.append(weights)

            for b in range(bsz):
                if not finished[b]:
                    tok = int(y[b])
                    if tok == EOS_ID:
                        finished[b] = True
                    else:
                        outputs[b].append(tok)
            if bool(finished.all()):
                break

        attn = torch.stack(attn_steps, dim=1) if (return_attention and attn_steps) else None
        return outputs, attn

    @torch.no_grad()
    def generate_beam(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        src_mask: torch.Tensor,
        beam_size: int = 4,
        max_len: int = 100,
        min_len: int = 10,
        length_penalty: float = 1.0,
        block_trigram: bool = True,
    ) -> list[list[int]]:
        """Beam search, run one source at a time for clarity.

        Uses the GNMT length penalty ((5+|Y|)/6)^alpha, which prevents beam search
        from collapsing onto very short summaries -- a real failure mode here,
        since <eos> is cheap once the decoder has emitted a fluent clause.
        """
        self.eval()
        results: list[list[int]] = []

        for i in range(src.size(0)):
            one_src = src[i : i + 1]
            one_len = src_len[i : i + 1]
            one_mask = src_mask[i : i + 1]

            memory, state = self.encoder(one_src, one_len)
            memory = memory.expand(beam_size, -1, -1).contiguous()
            mask_b = one_mask.expand(beam_size, -1).contiguous()
            memory_proj = self.decoder.attention.project_memory(memory)
            h, c = state
            state = (
                h.expand(-1, beam_size, -1).contiguous(),
                c.expand(-1, beam_size, -1).contiguous(),
            )

            device = src.device
            y = torch.full((beam_size,), BOS_ID, dtype=torch.long, device=device)
            feed = self.decoder.init_feed(beam_size, device, memory.dtype)

            # Only beam 0 is live at t=0; the rest are -inf so the first expansion
            # does not duplicate the same hypothesis beam_size times.
            scores = torch.full((beam_size,), float("-inf"), device=device)
            scores[0] = 0.0
            tokens: list[list[int]] = [[] for _ in range(beam_size)]
            completed: list[tuple[float, list[int]]] = []

            for t in range(max_len):
                logits, new_state, new_feed, _ = self.decoder.step(
                    y, state, memory, mask_b, feed, memory_proj
                )
                logits[:, PAD_ID] = float("-inf")
                logits[:, UNK_ID] = float("-inf")
                if t < min_len:
                    logits[:, EOS_ID] = float("-inf")
                if block_trigram:
                    _block_repeat_trigrams(logits, tokens)

                logprobs = torch.log_softmax(logits, dim=-1)
                cand = scores.unsqueeze(1) + logprobs           # (beam, V)
                flat = cand.view(-1)
                top_scores, top_idx = flat.topk(beam_size)
                beam_idx = torch.div(top_idx, logits.size(1), rounding_mode="floor")
                tok_idx = top_idx % logits.size(1)

                next_tokens, next_scores, next_beams = [], [], []
                for k in range(beam_size):
                    b_src = int(beam_idx[k])
                    tok = int(tok_idx[k])
                    seq = tokens[b_src] + [tok]
                    if tok == EOS_ID:
                        lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                        completed.append((float(top_scores[k]) / lp, tokens[b_src]))
                    else:
                        next_tokens.append(seq)
                        next_scores.append(float(top_scores[k]))
                        next_beams.append(b_src)

                if len(completed) >= beam_size or not next_tokens:
                    break

                # Re-pad the beam back to beam_size after removing finished ones.
                while len(next_tokens) < beam_size:
                    next_tokens.append(next_tokens[-1])
                    next_scores.append(float("-inf"))
                    next_beams.append(next_beams[-1])

                sel = torch.tensor(next_beams, dtype=torch.long, device=device)
                tokens = next_tokens
                scores = torch.tensor(next_scores, device=device)
                state = (new_state[0][:, sel].contiguous(), new_state[1][:, sel].contiguous())
                feed = new_feed[sel]
                y = torch.tensor([seq[-1] for seq in tokens], dtype=torch.long, device=device)

            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                results.append(completed[0][1])
            else:
                best = int(torch.argmax(scores))
                results.append(tokens[best])

        return results


def _block_repeat_trigrams(logits: torch.Tensor, prefixes: list[list[int]]) -> None:
    """Forbid tokens that would complete an already-seen trigram.

    Repetition is the canonical LSTM seq2seq failure mode. Blocking it in-place
    lets the report separate 'the model repeats' from 'the model is wrong',
    and the un-blocked variant is reported as an ablation.
    """
    for b, seq in enumerate(prefixes):
        if len(seq) < 2:
            continue
        seen = {(seq[i], seq[i + 1]): set() for i in range(len(seq) - 1)}
        for i in range(len(seq) - 2):
            seen[(seq[i], seq[i + 1])].add(seq[i + 2])
        banned = seen.get((seq[-2], seq[-1]))
        if banned:
            logits[b, list(banned)] = float("-inf")
