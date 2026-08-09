"""Torch dataset, padding/masking collate, and length-bucketed batching."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from src.data.tokenizer import tokenize
from src.data.vocab import BOS_ID, EOS_ID, PAD_ID, Vocab


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file, tolerating a truncated final line.

    The generation runners append one flushed line per completed example so a run
    can resume after an interruption. A process killed mid-write (or a full disk)
    leaves a partial final line, and refusing to parse the file would make the
    interrupted run unresumable -- losing hours of completed work for one bad
    trailing fragment.

    A malformed line *followed by more content* is different: that is genuine
    corruption rather than an interrupted append, and it is raised rather than
    silently skipped.
    """
    rows: list[dict] = []
    pending: tuple[int, str] | None = None

    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            if pending is not None:
                bad_line, msg = pending
                raise ValueError(
                    f"{path}: malformed JSON on line {bad_line} followed by further "
                    f"content on line {lineno}; the file is corrupted rather than "
                    f"truncated ({msg})"
                )
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                pending = (lineno, str(exc))

    if pending is not None:
        bad_line, _ = pending
        print(
            f"warning: {path} ends with a truncated line ({bad_line}); it was "
            f"skipped. This is expected if a run was interrupted mid-write.",
            file=sys.stderr,
        )
    return rows


class SummarizationDataset(Dataset):
    """Encoded (article, summary) pairs.

    Encoding happens once at construction: 80k articles of <=400 tokens fits
    comfortably in memory as int32 and keeps the training loop free of tokenizer
    overhead.
    """

    def __init__(
        self,
        path: str | Path,
        vocab: Vocab,
        max_src_len: int = 400,
        max_tgt_len: int = 100,
        show_progress: bool = True,
    ) -> None:
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.records = read_jsonl(path)

        self.src: list[np.ndarray] = []
        self.tgt: list[np.ndarray] = []
        iterator = (
            tqdm(
                self.records,
                desc=f"encode {Path(path).stem}",
                unit="doc",
                mininterval=1.0 if sys.stdout.isatty() else 30.0,
            )
            if show_progress
            else self.records
        )
        for rec in iterator:
            src_ids = vocab.encode(tokenize(rec["article"]))[:max_src_len]
            # Reserve one slot so the target always ends with <eos> after truncation.
            tgt_ids = vocab.encode(tokenize(rec["summary"]))[: max_tgt_len - 1]
            self.src.append(np.asarray(src_ids, dtype=np.int32))
            self.tgt.append(np.asarray(tgt_ids, dtype=np.int32))

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> dict:
        return {
            "index": idx,
            "src": self.src[idx],
            "tgt": self.tgt[idx],
        }

    def src_lengths(self) -> np.ndarray:
        return np.array([len(s) for s in self.src], dtype=np.int64)


# Padding lengths are rounded up to these multiples. The MPS backend compiles a
# Metal kernel per distinct tensor shape, and length-bucketed batches otherwise
# produce a near-unique (S, T) pair every step -- which makes shader compilation,
# not arithmetic, the dominant cost (observed: MTLCompilerService pinning two
# cores while training crawls). Quantizing collapses thousands of shapes into a
# few dozen that compile once and are then reused, at the cost of a little extra
# padding.
SRC_LEN_MULTIPLE = 64
TGT_LEN_MULTIPLE = 16


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def collate_batch(batch: list[dict]) -> dict:
    """Pad a batch and build the masks the encoder and attention need.

    Returns
    -------
    src        (B, S)  padded source ids
    src_len    (B,)    true source lengths, for pack_padded_sequence
    src_mask   (B, S)  True at real source positions; attention masks the rest
    tgt_in     (B, T)  decoder input,  <bos> w1 ... wT
    tgt_out    (B, T)  decoder target, w1 ... wT <eos>
    tgt_mask   (B, T)  True at real target positions; loss ignores the rest
    """
    bsz = len(batch)
    src_lens = [len(ex["src"]) for ex in batch]
    # +1 for the <bos>/<eos> that each side of the shifted pair carries.
    tgt_lens = [len(ex["tgt"]) + 1 for ex in batch]
    max_s = _round_up(max(src_lens), SRC_LEN_MULTIPLE)
    max_t = _round_up(max(tgt_lens), TGT_LEN_MULTIPLE)

    src = torch.full((bsz, max_s), PAD_ID, dtype=torch.long)
    tgt_in = torch.full((bsz, max_t), PAD_ID, dtype=torch.long)
    tgt_out = torch.full((bsz, max_t), PAD_ID, dtype=torch.long)

    for i, ex in enumerate(batch):
        s, t = ex["src"], ex["tgt"]
        src[i, : len(s)] = torch.from_numpy(s.astype(np.int64))
        t_long = torch.from_numpy(t.astype(np.int64))
        tgt_in[i, 0] = BOS_ID
        tgt_in[i, 1 : len(t) + 1] = t_long
        tgt_out[i, : len(t)] = t_long
        tgt_out[i, len(t)] = EOS_ID

    return {
        "index": torch.tensor([ex["index"] for ex in batch], dtype=torch.long),
        "src": src,
        "src_len": torch.tensor(src_lens, dtype=torch.long),
        "src_mask": src.ne(PAD_ID),
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "tgt_mask": tgt_out.ne(PAD_ID),
    }


class BucketBatchSampler(Sampler[list[int]]):
    """Group similar-length sources into batches to cut padding waste.

    With 400-token articles and unsorted batches, most of an encoder step is spent
    on padding. Bucketing shuffles within pools of similar length, so batches stay
    stochastic while padding stays low.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        pool_multiplier: int = 50,
        seed: int = 1234,
        drop_last: bool = False,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool_size = batch_size * pool_multiplier
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-shuffle differently each epoch while staying reproducible."""
        self.epoch = epoch

    def __iter__(self):
        n = len(self.lengths)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            order = rng.permutation(n)
        else:
            order = np.arange(n)

        batches: list[list[int]] = []
        for pool_start in range(0, n, self.pool_size):
            pool = order[pool_start : pool_start + self.pool_size]
            # Sort the pool by length, then slice into batches.
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            for b_start in range(0, len(pool), self.batch_size):
                batch = pool[b_start : b_start + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch + 10_000)
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return iter(batches)

    def __len__(self) -> int:
        # Batches are cut inside pools, so a short final pool yields its own
        # partial batch; count pool by pool rather than dividing n by batch_size.
        n = len(self.lengths)
        total = 0
        for pool_start in range(0, n, self.pool_size):
            pool_len = min(self.pool_size, n - pool_start)
            if self.drop_last:
                total += pool_len // self.batch_size
            else:
                total += (pool_len + self.batch_size - 1) // self.batch_size
        return total
