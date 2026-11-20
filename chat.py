import sys

import torch

from model import MiniLLM, greedy_answer

ck_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint.pt"
ck = torch.load(ck_path, weights_only=False)
model = MiniLLM(**ck["config"])
model.load_state_dict(ck["state_dict"])
model.eval()

print(f"checkpoint: {ck_path} | params: {model.count_params():,}")
print()

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        break
    print("AI :", greedy_answer(model, q))
    print()
