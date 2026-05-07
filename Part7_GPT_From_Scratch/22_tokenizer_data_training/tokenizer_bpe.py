"""A from-scratch byte-pair encoding tokenizer.

BPE in one paragraph:
    Start with a vocabulary of all individual bytes (256 entries). Look at
    the frequency of every pair of adjacent tokens in the corpus. Replace
    the most-common pair with a single new token. Repeat until you've
    added vocab_size - 256 new merges.

That's it. Encoding is the inverse: greedily replace pairs in the order
they were added. The final vocabulary covers any byte string.

This implementation is ~100 lines and matches the spec well enough to
encode/decode arbitrary UTF-8. For production, use tiktoken (OpenAI) or
sentencepiece (Google) — both are C++ for speed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def _get_pair_counts(ids: list[int]) -> Counter:
    """Count how often each adjacent pair (a, b) appears in the sequence."""
    return Counter(zip(ids, ids[1:]))


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Walk through `ids` and replace every consecutive `pair` with `new_id`."""
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and (ids[i], ids[i+1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Minimal byte-level BPE tokenizer.

    Public API:
        tok = BPETokenizer.train(text, vocab_size=512)
        ids = tok.encode("hello world")
        text = tok.decode(ids)
        tok.save("bpe.json"); BPETokenizer.load("bpe.json")
    """

    def __init__(self, merges: dict[tuple[int, int], int], vocab: dict[int, bytes]):
        # `merges` keys are pairs of token ids; values are the resulting new id.
        # Order of insertion matters -- earlier merges are applied first.
        self.merges = merges
        self.vocab = vocab     # id -> bytes; lets us decode back to bytes
        self.vocab_size = len(vocab)

    # ----- training -----------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 512) -> "BPETokenizer":
        """Greedy BPE: each iteration finds the most-frequent adjacent pair
        and adds it as a new token. Stops when |vocab| == vocab_size.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one per byte)")

        # Start with raw bytes (UTF-8). Every byte is its own token.
        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        n_merges = vocab_size - 256
        for step in range(n_merges):
            counts = _get_pair_counts(ids)
            if not counts:
                break
            top_pair = counts.most_common(1)[0][0]
            new_id = 256 + step
            ids = _merge(ids, top_pair, new_id)
            merges[top_pair] = new_id
            vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
        return cls(merges, vocab)

    # ----- encoding / decoding -----------------------------------------
    def encode(self, text: str) -> list[int]:
        """UTF-8 bytes → list of token ids."""
        ids = list(text.encode("utf-8"))
        # Apply each merge in the same order it was learned.
        # (For really long inputs, you'd use a priority queue keyed by
        # merge index; this O(merges * len) is fine for tutorials.)
        while len(ids) >= 2:
            counts = _get_pair_counts(ids)
            # find the pair with the smallest merge index (i.e. learned first)
            best = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break    # no more known merges apply
            ids = _merge(ids, best, self.merges[best])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids → bytes → UTF-8 string."""
        b = b"".join(self.vocab[int(i)] for i in ids)
        # `errors="replace"` makes us robust to malformed sequences --
        # important during sampling where the model may output ill-formed bytes.
        return b.decode("utf-8", errors="replace")

    # ----- (de)serialization -------------------------------------------
    def save(self, path: str | Path) -> None:
        # Store merges as a list of triples [(a, b, new_id), ...] -- JSON
        # doesn't allow tuple keys, so we flatten.
        blob = {
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
            "vocab":  {str(k): list(v) for k, v in self.vocab.items()},
        }
        Path(path).write_text(json.dumps(blob))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        blob = json.loads(Path(path).read_text())
        merges = {(a, b): nid for a, b, nid in blob["merges"]}
        vocab = {int(k): bytes(v) for k, v in blob["vocab"].items()}
        return cls(merges, vocab)
