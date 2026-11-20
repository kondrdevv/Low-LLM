import random
import re

PAD = 0
BOS = 1
EOS = 2
ISTART = 3
IEND = 4
ANS = 5
TEXT_START = 6
NUM_START = 264
NUM_END = 265
VOCAB = 276

NUM_RE = re.compile(r"\b\d+\b")
NB = 201


def encode(text):
    return [b + 7 for b in text.encode("utf-8")]


def decode(ids):
    return bytes(max(0, i - 7) for i in ids).decode("utf-8", errors="replace")


def _num(n):
    return [NUM_START] + [NUM_START + 1 + int(c) for c in str(n)] + [NUM_END]


def _num_like(text):
    out = []
    last = 0
    for m in NUM_RE.finditer(text):
        out += encode(text[last:m.start()])
        out += _num(int(m.group()))
        last = m.end()
    out += encode(text[last:])
    return out


def encode_answer(text):
    ids = [ANS] + _num_like(text) + [EOS]
    return ids[:128]


BANK = [
    ("2+2", 4), ("3+5", 8), ("10-3", 7), ("6*7", 42), ("9+10", 19), ("15-8", 7),
    ("12*3", 36), ("25+17", 42), ("8*8", 64), ("100-46", 54), ("7*9", 63),
    ("45+45", 90), ("13*4", 52), ("60-23", 37), ("11*11", 121), ("144/12", 12),
    ("19+28", 47), ("31*3", 93), ("50-27", 23), ("16+29", 45), ("14*6", 84),
    ("75+25", 100), ("33-18", 15), ("21*2", 42), ("99-57", 42), ("6*9", 54),
    ("18+14", 32), ("72/8", 9), ("27*3", 81), ("85-49", 36), ("23+38", 61),
    ("7*7", 49), ("64/8", 8), ("56+39", 95), ("40-16", 24), ("17*5", 85),
    ("81/9", 9), ("36+48", 84), ("92-35", 57), ("12*12", 144), ("5*12", 60),
    ("70-42", 28), ("88+12", 100), ("15*4", 60), ("100-13", 87), ("29+31", 60),
    ("45/9", 5), ("22*4", 88), ("67-19", 48), ("53+29", 82), ("9*9", 81),
    ("13+47", 60), ("96/12", 8), ("51-27", 24), ("34*2", 68), ("76+18", 94),
    ("45-19", 26), ("8*6", 48), ("25*4", 100), ("43+37", 80), ("100-64", 36),
]

CAPITALS = [
    ("России", "Москва"), ("Франции", "Париж"), ("Германии", "Берлин"),
    ("Италии", "Рим"), ("Испании", "Мадрид"), ("Японии", "Токио"),
    ("Китая", "Пекин"), ("Египта", "Каир"), ("Канады", "Оттава"),
    ("Норвегии", "Осло"), ("Швеции", "Стокгольм"), ("Польши", "Варшава"),
    ("Греции", "Афины"), ("Турции", "Анкара"), ("Португалии", "Лиссабон"),
    ("Нидерландов", "Амстердам"), ("Бельгии", "Брюссель"), ("Австрии", "Вена"),
    ("Швейцарии", "Берн"), ("Финляндии", "Хельсинки"), ("Дании", "Копенгаген"),
    ("Чехии", "Прага"), ("Венгрии", "Будапешт"), ("Индии", "Нью-Дели"),
    ("Аргентины", "Буэнос-Айрес"), ("Мексики", "Мехико"), ("Кубы", "Гавана"),
    ("Ирландии", "Дублин"), ("Хорватии", "Загреб"), ("Румынии", "Бухарест"),
    ("Исландии", "Рейкьявик"), ("Монголии", "Улан-Батор"), ("Вьетнама", "Ханой"),
]

QUIZ_MISC = [
    ("Сколько дней в неделе?", "7"), ("Сколько месяцев в году?", "12"),
    ("Сколько часов в сутках?", "24"), ("Сколько минут в часе?", "60"),
    ("Сколько секунд в минуте?", "60"), ("Сколько сторон у треугольника?", "3"),
    ("Сколько сторон у квадрата?", "4"), ("Сколько градусов в прямом угле?", "90"),
    ("Сколько букв в русском алфавите?", "33"), ("Сколько пальцев на двух руках?", "10"),
    ("Сколько цветов у радуги?", "7"), ("Сколько струн у гитары?", "6"),
]

PAIRS = [
    ("столица", "город"), ("собака", "животное"), ("роза", "цветок"),
    ("дуб", "дерево"), ("яблоко", "фрукт"), ("морковь", "овощ"),
    ("молоко", "напиток"), ("волк", "хищник"), ("заяц", "травоядное"),
    ("квадрат", "фигура"), ("осень", "время года"), ("физика", "наука"),
    ("Москва", "город"), ("карп", "рыба"), ("орёл", "птица"),
]

QA_EXTRA = [
    ("Что больше: 5 или 9?", "9"), ("Что меньше: 12 или 7?", "7"),
    ("Какое число идёт после 19?", "20"), ("Какое число идёт перед 30?", "29"),
    ("Чётное ли число 8?", "да"), ("Чётное ли число 7?", "нет"),
    ("Сколько будет 0+0?", "0"), ("Сколько будет 1*1?", "1"),
]

LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
EN_RU = ("abcdefghijklmnopqrstuvwxyz", LETTERS)

CALC_TEMPLATES = [
    "Сколько будет {e}?", "Вычисли {e}.", "Посчитай: {e}", "Сколько равно {e}?",
    "Реши пример {e}.", "Сколько получится, если {e}?", "{e} = ?",
]

CAP_TEMPLATES = [
    "Какая столица {c}?", "Столица {c} — это?", "Назови столицу {c}.",
    "Какой город является столицей {c}?", "Столица страны {c}?",
]

MISC_TEMPLATES = ["{q}", "Ответь: {q}", "Вопрос: {q}", "{q} Скажи число."]

PAIR_TEMPLATES = [
    "Что такое {w}?", "К какой группе относится {w}?", "Объясни: {w}.",
    "{w} — это что?", "Назови группу для слова {w}.",
]

PAIR_ANSWER = [
    "{w} — это {p}.", "Это {p}.", "{w} относится к группе «{p}».",
    "Коротко: {p}.", "{p}.",
]

REVERSE_MI = [
    ("я люблю чай", "чай люблю я"), ("кот ест рыбу", "рыбу ест кот"),
    ("мы идём домой", "домой идём мы"), ("зима уже близко", "близко уже зима"),
    ("стол стоит тут", "тут стоит стол"), ("книга лежит рядом", "рядом лежит книга"),
    ("утро очень доброе", "доброе очень утро"), ("небо светит ярко", "ярко светит небо"),
    ("друг звонит часто", "часто звонит друг"), ("дождь идёт снова", "снова идёт дождь"),
]

REVERSE_CHARS = [
    ("азбука", "акубза"), ("луна", "анул"), ("палиндром", "морднилап"),
    ("кот", "ток"), ("дом", "мод"), ("рука", "акур"), ("город", "дорог"),
    ("слово", "оволс"), ("день", "ьнед"), ("весна", "ансев"),
]

SHIFT = [("а", "б"), ("к", "л"), ("я", "б"), ("ж", "з"), ("щ", "ъ"), ("ю", "я")]


def truth(v):
    return str(v)


SMALL_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def make_examples(rng, n, holdout=False, unseen_words=False):
    out = []
    for _ in range(n):
        kind = rng.choices(
            ["calc", "arith", "tens_sum", "cap", "misc", "pair", "extra", "revmi", "revch", "shift", "count"],
            weights=[16, 14, 8, 18, 10, 12, 6, 8, 5, 3, 8],
        )[0]
        if kind == "tens_sum":
            a, b = rng.randint(1, 9) * 10, rng.randint(1, 9)
            e = f"{a}+{b}"
            out.append((rng.choice(CALC_TEMPLATES).format(e=e), truth(a + b)))
        elif kind == "calc":
            e, a = rng.choice(BANK)
            q = rng.choice(CALC_TEMPLATES).format(e=e)
            out.append((q, truth(a)))
        elif kind == "arith":
            while True:
                a, b = rng.randint(1, 20), rng.randint(1, 20)
                op = rng.choice("+-*")
                if op != "-" or a >= b:
                    break
            if rng.random() < 0.5:
                a, b = b, a
            res = SMALL_OPS[op](a, b)
            e = f"{a}{op}{b}"
            q = rng.choice(CALC_TEMPLATES).format(e=e)
            out.append((q, truth(res)))
        elif kind == "count":
            s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 9)))
            out.append((f"Сколько букв в слове {s}?", str(len(s))))
        elif kind == "cap":
            c, a = rng.choice(CAPITALS)
            q = rng.choice(CAP_TEMPLATES).format(c=c)
            out.append((q, a))
        elif kind == "misc":
            q, a = rng.choice(QUIZ_MISC)
            q = rng.choice(MISC_TEMPLATES).format(q=q.rstrip("?"))
            out.append((q if q.endswith("?") else q + "?", a))
        elif kind == "pair":
            w, p = rng.choice(PAIRS)
            q = rng.choice(PAIR_TEMPLATES).format(w=w)
            a = rng.choice(PAIR_ANSWER).format(w=w, p=p)
            out.append((q, a))
        elif kind == "extra":
            q, a = rng.choice(QA_EXTRA)
            out.append((q, a))
        elif kind == "revmi":
            x, y = rng.choice(REVERSE_MI)
            q = f"Скажи наоборот: «{x}»."
            if rng.random() < 0.5:
                q = f"Переставь слова в обратном порядке: {x}."
            out.append((q, f"«{y}»."))
        elif kind == "revch":
            x, y = rng.choice(REVERSE_CHARS)
            q = f"Напиши слово {x} в обратном порядке."
            out.append((q, y))
        else:
            x, y = rng.choice(SHIFT)
            q = f"Какая буква идёт после буквы {x}?"
            out.append((q, y))
    return out


def make_upper(rng, n):
    out = []
    for _ in range(n):
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 9)))
        out.append((f"Напиши заглавными буквами: {s}.", s.upper()))
    return out


OUTSIDE_WORDS = [
    ("груша", "фрукт"), ("сосна", "дерево"), ("берёза", "дерево"), ("тигр", "хищник"),
    ("лиса", "хищник"), ("капуста", "овощ"), ("сок", "напиток"), ("тюльпан", "цветок"),
    ("круг", "фигура"), ("лето", "время года"), ("химия", "наука"), ("мёд", "продукт"),
    ("поезд", "транспорт"), ("щетка", "предмет"), ("апельсин", "фрукт"), ("окунь", "рыба"),
    ("воробей", "птица"), ("Казань", "город"), ("слива", "фрукт"), ("ёлка", "дерево"),
]

REVERSE_CHARS_OUTSIDE = [("радуга", "агудар"), ("кофе", "ефок"), ("море", "ером"), ("свет", "тевс")]


def make_outside(rng, n):
    out = []
    for _ in range(n):
        kind = rng.choices(["pair", "revch", "arith", "capital"], weights=[45, 15, 25, 15])[0]
        if kind == "pair":
            w, p = rng.choice(OUTSIDE_WORDS)
            q = rng.choice(PAIR_TEMPLATES).format(w=w)
            a = rng.choice(PAIR_ANSWER).format(w=w, p=p)
            out.append((q, a))
        elif kind == "revch":
            x, y = rng.choice(REVERSE_CHARS_OUTSIDE)
            out.append((f"Напиши слово {x} в обратном порядке.", y))
        elif kind == "arith":
            a, b = rng.randint(21, 60), rng.randint(21, 60)
            q = rng.choice(CALC_TEMPLATES).format(e=f"{a}+{b}")
            out.append((q, str(a + b)))
        else:
            q = rng.choice(CAP_TEMPLATES).format(c="Бразилии")
            out.append((q, "Бразилиа"))
    return out


def build_corpus(n_batches, seed=0):
    rng = random.Random(seed)
    train = []
    for _ in range(n_batches):
        train += make_examples(rng, 1000)
    train += make_upper(rng, 2400)
    train = list(dict.fromkeys(train))
    ev = list(dict.fromkeys(make_examples(rng, 400)))
    ev = [p for p in ev if p not in set(train)][:150]
    outside = make_outside(rng, 120)
    return train, ev, outside


def encode_pair(q, a):
    ids = [BOS, ISTART] + encode(q) + [IEND, TEXT_START] + encode_answer(a)
    return ids[:192]
