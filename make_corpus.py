import argparse
import json
import random
from pathlib import Path

CHITCHAT = [
    ("привет", "Привет!"), ("здравствуй", "Привет! Чем помочь?"),
    ("как дела?", "Хорошо, спасибо. А у тебя?"), ("кто ты?", "Я Low-LLM - маленькая языковая модель."),
    ("спасибо", "Пожалуйста."), ("пока", "Пока!"),
]

CAPITALS = [
    ("России", "Москва"), ("Франции", "Париж"), ("Германии", "Берлин"), ("Италии", "Рим"),
    ("Испании", "Мадрид"), ("Японии", "Токио"), ("Китая", "Пекин"), ("Египта", "Каир"),
]

PAIRS = [
    ("собака", "животное"), ("роза", "цветок"), ("дуб", "дерево"), ("яблоко", "фрукт"),
    ("морковь", "овощ"), ("молоко", "напиток"), ("волк", "хищник"),
]


def sum_steps(a, b):
    total = a + b
    width = max(len(str(a)), len(str(b)))
    da = [int(c) for c in str(a).zfill(width)][::-1]
    db = [int(c) for c in str(b).zfill(width)][::-1]
    carry = 0
    parts = []
    units = ["единиц", "десятков", "сотен", "тысяч"]
    for i in range(width):
        s = da[i] + db[i] + carry
        d, carry = s % 10, s // 10
        msg = f"Шаг {i + 1}: {units[i] if i < 4 else 'разряда'} {da[i]}+{db[i]}"
        if True:
            pass
        parts.append(msg + f"={s}, пишем {d}" + (f", перенос {carry}" if carry else ""))
    return total, parts


def arith_examples(rng, n):
    out = []
    for _ in range(n):
        a, b = rng.randint(1, 999), rng.randint(1, 999)
        total, steps = sum_steps(a, b)
        q = rng.choice([f"Сложи {a} и {b}.", f"Сколько будет {a}+{b}?", f"{a}+{b}=?"])
        out.append((q, " ".join(steps) + f". Ответ: {total}"))
    return out


def corpus_chat(rng, n):
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.35:
            out.append(rng.choice(CHITCHAT))
        elif kind < 0.6:
            c, a = rng.choice(CAPITALS)
            out.append((f"Какая столица {c}?", a))
        elif kind < 0.8:
            w, p = rng.choice(PAIRS)
            out.append((f"Что такое {w}?", f"Это {p}."))
        else:
            q, a = rng.choice(arith_examples(rng, 1))
            out.append((q, a))
    return out


def to_training_text(pairs):
    return "".join(f"<вопрос>{q}\n<ответ>{a}\n" for q, a in pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default="data/wiki.txt")
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--chat", type=int, default=40000)
    ap.add_argument("--mix", type=float, default=0.35)
    args = ap.parse_args()

    rng = random.Random(0)
    pairs = corpus_chat(rng, args.chat)
    chat_text = to_training_text(pairs)

    parts = []
    if Path(args.wiki).exists():
        wiki = Path(args.wiki).read_text(encoding="utf-8", errors="ignore")
        blocks = [b for b in wiki.split("\n\n") if len(b) > 200]
        rng.shuffle(blocks)
        need = int(len(chat_text) * (1 - args.mix) / args.mix)
        wiki_part = ""
        for b in blocks:
            wiki_part += b + "\n\n"
            if len(wiki_part) >= need:
                break
        parts.append(wiki_part)
        print(f"wiki part: {len(wiki_part) / 1e6:.1f}M chars")

    parts.append(chat_text)
    text = "".join(parts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"corpus: {len(text) / 1e6:.1f}M chars -> {args.out}")


if __name__ == "__main__":
    main()
