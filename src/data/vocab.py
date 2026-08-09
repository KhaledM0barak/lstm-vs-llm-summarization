"""Vocabulary construction.

The vocabulary is built from the *training split only*. Building it over the full
corpus would leak test-set lexical statistics into the model and would understate
the out-of-vocabulary rate that we report in the error analysis.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int]
    # Raw training counts, kept so we can report coverage and OOV rates.
    counts: dict[str, int]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        stoi = self.stoi
        return [stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: list[int], strip_specials: bool = True) -> list[str]:
        toks = []
        for i in ids:
            if i == EOS_ID:
                break
            if strip_specials and i in (PAD_ID, BOS_ID):
                continue
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else UNK)
        return toks

    @classmethod
    def build(
        cls,
        token_streams,
        max_size: int = 50_000,
        min_freq: int = 2,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for tokens in token_streams:
            counter.update(tokens)

        # Sort by frequency, then alphabetically, so ties break deterministically
        # and the vocabulary is byte-identical across runs.
        candidates = sorted(
            ((tok, c) for tok, c in counter.items() if c >= min_freq),
            key=lambda kv: (-kv[1], kv[0]),
        )
        kept = candidates[: max_size - len(SPECIALS)]

        itos = list(SPECIALS) + [tok for tok, _ in kept]
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(itos=itos, stoi=stoi, counts=dict(counter))

    def coverage(self) -> dict:
        """Token-level coverage of the training corpus by the kept vocabulary."""
        total = sum(self.counts.values())
        in_vocab = sum(c for tok, c in self.counts.items() if tok in self.stoi)
        return {
            "vocab_size": len(self.itos),
            "distinct_tokens_in_train": len(self.counts),
            "train_token_count": total,
            "token_coverage": in_vocab / total if total else 0.0,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"itos": self.itos, "counts": self.counts}, f)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with Path(path).open(encoding="utf-8") as f:
            blob = json.load(f)
        itos = blob["itos"]
        return cls(
            itos=itos,
            stoi={tok: i for i, tok in enumerate(itos)},
            counts=blob.get("counts", {}),
        )
