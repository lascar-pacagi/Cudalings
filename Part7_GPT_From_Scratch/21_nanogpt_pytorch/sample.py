"""Load ckpt.pt and generate text from the trained model.

Usage:
    python sample.py                              # 500 chars, default seed
    python sample.py --prompt "ROMEO:"            # start with a prompt
    python sample.py --temperature 0.8 --top_k 50 # tame the sampler
"""

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(Path(__file__).parent / "ckpt.pt"))
    p.add_argument("--prompt", default="\n")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    stoi, itos = blob["stoi"], blob["itos"]

    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.train(False)

    # encode the prompt; if any char isn't in the vocab, fall back to "\n"
    seed_ids = [stoi.get(c, stoi["\n"]) for c in args.prompt]
    idx = torch.tensor(seed_ids, dtype=torch.long, device=device).unsqueeze(0)

    out = model.generate(idx, args.n,
                         temperature=args.temperature,
                         top_k=args.top_k)
    text = "".join(itos[int(i)] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
