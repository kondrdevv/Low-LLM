import math
import sys
import time

import torch

import data
from model import MiniLLM, build_batch, greedy_answer

torch.manual_seed(0)
torch.set_num_threads(torch.get_num_threads())
device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(n_layer=4, n_head=6, dim=192, seq_len=224)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
BATCH = 32

train_pairs, eval_pairs, outside_pairs = data.build_corpus(0)
print(f"device: {device} | train {len(train_pairs)} | eval {len(eval_pairs)} | outside {len(outside_pairs)}")

model = MiniLLM(**CFG)
print(f"parameters: {model.count_params():,}")
model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
WARM = 100
t0 = time.time()
CKPT = "checkpoint_v2.pt"


def final(ans):
    return ans.split("Ответ:")[-1].strip(" .")


def quick_check():
    arith = [p for p in eval_pairs if "Шаг" in p[1]][:20]
    hit = sum(final(greedy_answer(model, q)) == final(a) for q, a in arith)
    print(f"    eval: арифметика {hit}/{len(arith)}")


for step in range(STEPS):
    lr = 1e-3 * min(1.0, (step + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    for g in opt.param_groups:
        g["lr"] = lr
    idx = torch.randint(0, len(train_pairs), (BATCH,))
    x, y = build_batch([train_pairs[i] for i in idx], max_len=CFG["seq_len"])
    logits = model(x.to(device))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, data.VOCAB), y.reshape(-1).to(device), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0 or step == STEPS - 1:
        print(f"step {step:4d} loss {loss.item():.3f} lr {lr:.2e} {time.time() - t0:6.1f}s", flush=True)
    if step % 400 == 399:
        torch.save({"state_dict": model.state_dict(), "config": CFG, "steps": step + 1}, CKPT)
        quick_check()

torch.save({"state_dict": model.state_dict(), "config": CFG, "steps": STEPS}, CKPT)
print(f"saved {CKPT}, {time.time() - t0:.0f}s")

print("\n--- новые вопросы ---")
for q, a in eval_pairs[:8]:
    print(f"Q: {q}\n   надо: {final(a)[:60]}\n   ИИ:   {greedy_answer(model, q)[:90]}")
