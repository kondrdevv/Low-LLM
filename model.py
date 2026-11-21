import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class LowLLM(nn.Module):
    def __init__(self, vocab, n_layer=6, n_head=6, dim=384, seq_len=384):
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


if __name__ == "__main__":
    m = LowLLM(vocab=8192)
    print(f"params: {m.count_params():,}")
