import sys

import torch

from model import MiniLLM, greedy_answer

ck = torch.load("checkpoint.pt", weights_only=False)
model = MiniLLM(**ck["config"])
model.load_state_dict(ck["state_dict"])
model.eval()

print(f"параметров: {model.count_params():,}")
print("Введи вопрос (пустая строка — выход).\n")

while True:
    try:
        q = input("Ты: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        break
    print("ИИ:", greedy_answer(model, q), "\n")
