import sys
from collections import Counter

import torch

from bpe import BPE
from model import LowLLM

CKPT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\kondr\low ai\Low-LLM-004"
tok = BPE.load(f"{CKPT}/bpe.json")
ck = torch.load(f"{CKPT}/model.pt", map_location="cpu")
model = LowLLM(len(tok.vocab)).eval()
model.load_state_dict(ck["state_dict"])

cfg = {"temp": 0.7, "top_k": 40, "penalty": 0.6, "n": 120, "greedy": False}

HELP = """команды:
  /set <параметр> <значение>   temp, top_k, penalty, n
  /greedy                      включить/выключить argmax
  /params                      показать настройки
  /reset                       вернуть значения по умолчанию
  /quit                        выход"""


@torch.no_grad()
def answer(prompt):
    ids = tok.encode(prompt)
    start = len(ids)
    for _ in range(cfg["n"]):
        logits = model(torch.tensor([ids[-384:]]))[0, -1].clone()
        if cfg["greedy"]:
            nxt = int(logits.argmax())
        else:
            for t, c in Counter(ids[-64:]).items():
                logits[t] -= cfg["penalty"] * c
            logits = logits / cfg["temp"]
            vals, _ = torch.topk(logits, cfg["top_k"])
            logits[logits < vals[-1]] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        ids.append(nxt)
        if tok.itos.get(nxt, "") == "\n":
            break
    return tok.decode(ids[start:]).strip()


def show():
    print(
        f"  temp={cfg['temp']}  top_k={cfg['top_k']}  penalty={cfg['penalty']}  "
        f"n={cfg['n']}  greedy={cfg['greedy']}"
    )


print(f"checkpoint: {CKPT} | step {ck.get('step', '?')} | params {model.count_params():,}")
show()
print(HELP)
print()

while True:
    try:
        q = input("Ты: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        break

    if q.startswith("/"):
        cmd, *rest = q.split()
        if cmd in ("/quit", "/q"):
            break
        elif cmd == "/params":
            show()
        elif cmd == "/help":
            print(HELP)
        elif cmd == "/greedy":
            cfg["greedy"] = not cfg["greedy"]
            show()
        elif cmd == "/reset":
            cfg = {"temp": 0.7, "top_k": 40, "penalty": 0.6, "n": 120, "greedy": False}
            show()
        elif cmd == "/set" and len(rest) == 2:
            key, val = rest
            if key == "greedy":
                cfg["greedy"] = val.lower() in ("1", "on", "true", "да")
            elif key in ("top_k", "n"):
                cfg[key] = int(val)
            elif key in ("temp", "penalty"):
                cfg[key] = float(val)
            else:
                print("  неизвестный параметр:", key)
            show()
        else:
            print("  " + HELP)
        print()
        continue

    full = q if "<вопрос>" in q else f"<вопрос>{q}\n<ответ>"
    print("ИИ:", answer(full))
    print()


