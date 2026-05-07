"""Interactive sampling loop -- type a prompt, get a continuation.

Don't expect coherent answers from a 10M-param model trained on 1M tokens
of Shakespeare. The point is the *mechanism*: tokenize → forward →
sample → detokenize → print. That's what every chat UI is doing under
the hood; the only thing that scales it to ChatGPT is more data, more
parameters, and RLHF on top.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "21_nanogpt_pytorch"))
from model import GPT, GPTConfig                                  # noqa: E402

from tokenizer_bpe import BPETokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(HERE / "ckpt.pt", map_location=device, weights_only=False)
    cfg: GPTConfig = blob["cfg"]
    tok = BPETokenizer.load(blob["tok_path"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.train(False)

    print(f"chat ready -- vocab={tok.vocab_size}, ctx={cfg.block_size}.  Ctrl-D to quit.\n")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        ids = tok.encode(prompt)
        idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=50)
        new_ids = out[0, len(ids):].tolist()
        print(tok.decode(new_ids))
        print()


if __name__ == "__main__":
    main()
