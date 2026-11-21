import argparse
from collections import Counter

import torch

from bpe import BPE
from model import LowLLM


def load(ckpt_dir):
    tok = BPE.load(f"{ckpt_dir}/bpe.json")
    ck = torch.load(f"{ckpt_dir}/model.pt", map_location="cpu")
    model = LowLLM(len(tok.vocab)).eval()
    model.load_state_dict(ck["state_dict"])
    return tok, model


@torch.no_grad()
def generate(tok, model, prompt, max_new=120, top_k=40, temp=0.8, penalty=0.6, window=64):
    ids = tok.encode(prompt)
    for _ in range(max_new):
        logits = model(torch.tensor([ids[-384:]]))[0, -1].clone()

        if ids:
            for t, c in Counter(ids[-window:]).items():
                logits[t] -= penalty * c

        logits = logits / temp
        if top_k:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[-1]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1))
        ids.append(nxt)

        if nxt in (tok.stoi.get("\n", -1),) and len(ids) - len(prompt) > 400:
            break
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/content/drive/MyDrive/low-llm")
    ap.add_argument("--prompt", default="Москва — это ")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--penalty", type=float, default=0.6)
    ap.add_argument("--greedy", action="store_true", help="старое поведение для сравнения")
    args = ap.parse_args()

    tok, model = load(args.ckpt)

    if args.greedy:
        ids = tok.encode(args.prompt)
        for _ in range(args.n):
            with torch.no_grad():
                nxt = int(model(torch.tensor([ids[-384:]]))[0, -1].argmax())
            ids.append(nxt)
        print(tok.decode(ids))
    else:
        print(generate(tok, model, args.prompt, args.n, args.top_k, args.temp, args.penalty))


if __name__ == "__main__":
    main()
