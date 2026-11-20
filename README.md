# Low-LLM - a tiny language model built from scratch

A very small transformer that trains and answers **fully offline** on an ordinary CPU.
No internet, no API, no pre-trained weights - just PyTorch and curiosity.

It is a **dumb little AI** by design: it learns from a tiny generated study book and gets
surprisingly good at a few narrow things, then fails in funny ways on everything else.
The point is to watch exactly where that line is - and then make it bigger.

## How it works

- `data.py` - builds the "study book": question -> answer pairs (step-by-step arithmetic,
  capitals, word categories, letter counting, word reversal).
- `model.py` - a decoder-only transformer. It learns to predict the next byte, nothing else.
  There are no rules, lookups or hardcoded answers anywhere in the code.
- `train.py` - trains from scratch (AdamW + warmup + cosine schedule).
- `chat.py` - an offline chat loop: type a question, get an answer.

## Requirements

- Python 3.10+
- PyTorch (CPU build is enough)

```powershell
pip install torch
```

## Run it

```powershell
cd mini_llm

# train from scratch (takes a while on CPU)
python train.py

# chat with the trained model
python chat.py
```

Pre-trained weights are included in `checkpoint.pt`, so `chat.py` works right away.
Re-running `train.py` overwrites the checkpoint.

## What it can and can't do

It does some things correctly:

```
You: привет
AI : Привет!

You: кто ты?
AI : Я маленькая языковая модель.

You: как дела?
AI : Хорошо, спасибо. А у тебя?

You: ывпзолр
AI : Не понял вопрос.

You: Сложи 2 и 2.
AI : Шаг 1: единиц 2+2=4, пишем 4. Ответ: 4

You: Какая столица Японии?
AI : Токио

You: Что такое собака?
AI : Слово «собака» — это животное.

You: Сколько букв в слове солнце?
AI : 6

You: Напиши слово луна в обратном порядке.
AI : анул

You: Скажи наоборот: «я люблю чай».
AI : «чай люблю я».
```

...and it fails at others, honestly:

```
You: сложи 12 и 25
AI : ... Шаг 2: десятков 1+2=4, пишем 4. Ответ: 47      (должно быть 37)

You: Сложи 12 и 39.
AI : ... Шаг 2: десятков 1+3+1=5, пишем 5. Ответ: 5     (обрывает цифры)

You: Сложи 232 и 101.
AI : Шаг 1: единиц 2+1=3, пишем 3. Шаг 2: десятков 3+]=3, пишем 3. Шаг 3: сотен
     3+1=4, ...

You: Напиши заглавными буквами: привет.
AI : РПРЕИТ
```

- **Arithmetic is hit or miss.** Single-digit and no-carry sums are right; a carry
  still corrupts the next step, and three-digit numbers fall apart.
- **It refuses on nonsense.** Random letter soup gets `Не понял вопрос.` instead of an
  invented answer - most of the time.
- **It can small-talk** a handful of phrases: greetings, `кто ты?`, `спасибо`.
- **It only works on its own material.** Words it never saw turn into plausible-looking nonsense.
- **It doesn't understand language.** It learned patterns and a handful of facts; reasoning and
  open-ended dialogue are way beyond it.
- **Character-level tasks are still shaky.** Reversal and letter counting work; uppercase
  conversion still scrambles letters.

So no, it is not an assistant. It's a sandbox for understanding how transformers learn
and where small models break.

## Files

| File | What it does |
|---|---|
| `data.py` | generates training pairs and encodes them into tokens |
| `model.py` | architecture, batching, generation, answer rendering |
| `train.py` | training loop and checkpoint saving |
| `chat.py` | interactive chat |
| `checkpoint.pt` | final 1.9M model (release 0.0.2) |

## Tune it yourself

Change `STEPS`, `BATCH` and `CFG` in `train.py` - the model size is defined by
`n_layer`, `dim`, `n_head` and `seq_len`.

## License

MIT - see [LICENSE](LICENSE).

Yes, this thing is legally protected. It still can't add 7 + 8, but at least the
lawyers are covered.
