import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import data


class Block(nn.Module):
    def __init__(self, dim, n_head, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class MiniLLM(nn.Module):
    def __init__(self, vocab=data.VOCAB, n_layer=4, n_head=4, dim=96, seq_len=192):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build_batch(pairs, max_len=192):
    x = torch.zeros(len(pairs), max_len, dtype=torch.long)
    y = torch.full((len(pairs), max_len), -100, dtype=torch.long)
    for i, (q, a) in enumerate(pairs):
        ids = data.encode_pair(q, a)
        tgt = ids[1:]
        start = ids.index(data.TEXT_START)
        lab = [-100] * (len(ids) - 1)
        lab[start:] = tgt[start:]
        x[i, :len(ids) - 1] = torch.tensor(ids[:-1])
        y[i, :len(ids) - 1] = torch.tensor(lab)
    return x, y


@torch.no_grad()
def greedy_answer(model, question, max_new=48):
    model.eval()
    ids = [data.BOS, data.ISTART] + data.encode(question) + [data.IEND, data.TEXT_START]
    for _ in range(max_new):
        x = torch.tensor([ids[-model.seq_len:]])
        logits = model(x)[0, -1]
        nxt = int(logits.argmax())
        ids.append(nxt)
        if nxt == data.EOS:
            break
    out = ids[ids.index(data.TEXT_START) + 1:]
    return render(out)


def render(ids):
    parts, buf = [], []
    in_num = False
    for i in ids:
        if i == data.NUM_START:
            parts.append(data.decode(buf)); buf = []
            parts.append("[")
            in_num = True
        elif i == data.NUM_END:
            parts.append("]"); in_num = False
        elif data.NUM_START + 1 <= i < data.NUM_START + 11:
            parts.append(str(i - data.NUM_START - 1))
        else:
            buf.append(i)
    parts.append(data.decode(buf))
    return "".join(parts).strip("\x00 \n")


def normalize(ans):
    return " ".join(ans.split()).strip(" .«»\"'")


@torch.no_grad()
def evaluate(model, pairs, limit=None):
    pairs = pairs[:limit] if limit else pairs
    good = 0
    for q, a in pairs:
        pred = greedy_answer(model, q)
        if normalize(pred) == normalize(a):
            good += 1
    return good / max(1, len(pairs))
