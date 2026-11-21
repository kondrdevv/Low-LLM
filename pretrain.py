import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from bpe import BPE
from model import LowLLM


def build_dataset(texts, tok, seq_len=384):
    ids = []
    for t in texts:
        ids.extend(tok.encode(t))
        ids.append(tok.stoi.get("\n", 0))
    arr = np.array(ids, dtype=np.uint16)
    return arr


def make_batches(arr, seq_len, batch, device, start=0):
    n = len(arr) - seq_len - 1
    i = start
    while i + batch * seq_len < n:
        idx = np.random.randint(0, n, size=batch)
        x = np.stack([arr[j:j + seq_len] for j in idx]).astype(np.int64)
        y = np.stack([arr[j + 1:j + 1 + seq_len] for j in idx]).astype(np.int64)
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        i += batch * seq_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--out", default="/content/drive/MyDrive/low-llm")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = Path(args.data).read_text(encoding="utf-8", errors="ignore")
    print(f"corpus: {len(text) / 1e6:.1f}M chars, device: {device}")

    tok_path = out / "bpe.json"
    if tok_path.exists():
        tok = BPE.load(tok_path)
    else:
        tok = BPE.train(text[:8_000_000], vocab_size=args.vocab)
        tok.save(tok_path)
    print(f"vocab: {len(tok.vocab)}")

    arr = build_dataset([text], tok, args.seq)
    print(f"tokens: {len(arr) / 1e6:.2f}M")

    model = LowLLM(len(tok.vocab), seq_len=args.seq).to(device)
    print(f"params: {model.count_params():,}")

    ck_path = out / "model.pt"
    step0 = 0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        step0 = ck["step"]
        print(f"resumed from step {step0}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    WARM = 500

    for step in range(step0, args.steps):
        lr = args.lr * min(1.0, (step + 1) / WARM)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = np.random.randint(0, len(arr) - args.seq - 1, size=args.batch)
        x = torch.from_numpy(np.stack([arr[j:j + args.seq] for j in idx]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([arr[j + 1:j + args.seq + 1] for j in idx]).astype(np.int64)).to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, len(tok.vocab)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 50 == 0:
            print(f"step {step:6d} loss {loss.item():.3f} lr {lr:.2e}", flush=True)
        if step % 500 == 499:
            torch.save({"state_dict": model.state_dict(), "step": step + 1}, ck_path)
            print(f"  saved {ck_path}", flush=True)

    torch.save({"state_dict": model.state_dict(), "step": args.steps}, ck_path)
    tok.save(tok_path)
    print("done")


if __name__ == "__main__":
    main()
