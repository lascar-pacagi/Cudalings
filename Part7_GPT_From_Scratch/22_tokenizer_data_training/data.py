"""Tiny streaming dataloader -- the realistic file format for LLM training.

train.bin / val.bin are flat uint16 arrays of token ids. We memory-map
them so we don't load the whole file -- we only fault in the (B, T)
slices we need each step. This scales to multi-GB datasets unchanged.
"""

from pathlib import Path
import numpy as np
import torch


def open_shard(path: Path) -> np.ndarray:
    """Memory-map a flat uint16 token-id file."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    """Sample (B, T) input and target chunks. Targets are shifted by one.

    `data` is the memory-mapped uint16 array. We pick `batch_size` random
    starts, slice block_size+1 tokens from each, and split into x / y.
    """
    rng = np.random.default_rng()
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(data[i : i + block_size]).astype(np.int64) for i in starts])
    y = np.stack([np.asarray(data[i+1 : i + block_size + 1]).astype(np.int64) for i in starts])
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y
