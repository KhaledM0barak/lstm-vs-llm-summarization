"""Deterministic seeding across python / numpy / torch.

Every entry point in this project calls `set_seed` before touching data or
parameters, so a fresh clone reproduces the reported numbers exactly.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 1234


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed every RNG this project draws from."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # MPS has no deterministic-algorithm registry of its own; these flags are
        # respected on CPU/CUDA and are harmless no-ops on Apple silicon.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive each worker's seed from the base seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
