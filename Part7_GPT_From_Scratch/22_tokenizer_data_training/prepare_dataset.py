"""Download tinyshakespeare, train a BPE tokenizer, write tokens.bin.

Output layout (in this directory):
    input.txt        the raw corpus
    bpe.json         the learned tokenizer
    train.bin        90% of token ids, uint16
    val.bin          10% of token ids, uint16
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from tokenizer_bpe import BPETokenizer


HERE = Path(__file__).parent
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
VOCAB_SIZE = 1024     # small to fit Pascal -- ~1000 sub-word tokens for 1M chars


def main():
    raw = HERE / "input.txt"
    if not raw.exists():
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, raw)
    text = raw.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} chars")

    print(f"training BPE with vocab={VOCAB_SIZE} ...")
    tok = BPETokenizer.train(text, vocab_size=VOCAB_SIZE)
    tok.save(HERE / "bpe.json")

    print("encoding ...")
    ids = np.array(tok.encode(text), dtype=np.uint16)
    n_train = int(0.9 * len(ids))
    ids[:n_train].tofile(HERE / "train.bin")
    ids[n_train:].tofile(HERE / "val.bin")

    compress = len(text) / len(ids)
    print(f"done. {len(ids):,} tokens "
          f"(compression ratio: {compress:.2f}x vs chars)")


if __name__ == "__main__":
    main()
