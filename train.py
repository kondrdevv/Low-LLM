import math
import time

import torch

import data
from model import MiniLLM, build_batch, greedy_answer

torch.manual_seed(0)
torch.set_num_threads(torch.get_num_threads())

CFG = dict(n_layer=2, n_head=4, dim=128, seq_len=160)
STEPS = 2000
BATCH = 64

train_pairs, eval_pairs, outside_pairs = data.build_corpus(10)
print(f"train pairs: {len(train_pairs)}, new eval: {len(eval_pairs)}, outside: {len(outside_pairs)}")

model = MiniLLM(**CFG)
print(f"parameters: {model.count_params():,}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
WARM = 100
t0 = time.time()

for step in range(STEPS):
    lr = 1e-3 * min(1.0, (step + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    for g in opt.param_groups:
        g["lr"] = lr
    idx = torch.randint(0, len(train_pairs), (BATCH,))
    x, y = build_batch([train_pairs[i] for i in idx], max_len=CFG["seq_len"])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, data.VOCAB), y.reshape(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0 or step == STEPS - 1:
        print(f"step {step:4d} loss {loss.item():.3f} lr {lr:.2e} {time.time() - t0:6.1f}s", flush=True)

torch.save({"state_dict": model.state_dict(), "config": CFG, "steps": STEPS}, "checkpoint.pt")
print(f"checkpoint saved, {time.time() - t0:.0f}s")

print("\n--- новые вопросы (не было в обучении) ---")
for q, a in eval_pairs[:10]:
    print(f"Q: {q}\n   надо: {a}\n   ИИ:   {greedy_answer(model, q)}")

print("\n--- слова вне учебника ---")
for q, a in outside_pairs[:6]:
    print(f"Q: {q}\n   надо: {a}\n   ИИ:   {greedy_answer(model, q)}")
