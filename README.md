# Low-LLM

A language model built from scratch. No internet, no API, no pre-trained weights -
just PyTorch and curiosity.

It is a **dumb little AI** by design: it learns from its own material, gets surprisingly
good at a few narrow things, then fails in funny ways on everything else. The point is
to watch exactly where that line is - and then make it bigger.

## Versions

| | 0.0.1 | 0.0.3 | 0.0.4 |
|---|---|---|---|
| Parameters | 0.5M | 1.9M | **17M** |
| Data | Q&A only | Q&A only | 120M chars Wikipedia + Q&A |
| Tokenizer | bytes | bytes | BPE, 8192 |
| Trained on | CPU | CPU | Colab T4 |
| Spaces | - | - | yes |

## 0.0.4 in short

- **9x bigger model** - 6 layers x 384, ~17M parameters, BPE vocabulary of 8192.
- **Real Russian text** - 120M characters of Wikipedia mixed with the Q&A study book.
- **Spaces are back** - the tokenizer keeps a word-boundary marker, so answers read
  `Это животное.` instead of `Этоэтоживотное`.
- **`generate.py`** - top-k, temperature and repetition penalty instead of raw argmax.
- **`chat.py` can be tuned live** - `/set temp 0.5`, `/set top_k 20`, `/greedy`, `/params`, `/reset`.
- **Digit-wise addition** - the model writes out each place value and carries.

## Weights and vocabulary

**`model.pt` and `bpe.json` are not in this repository** - both are attached to the release,
because a model only works with the exact vocabulary it was trained with.

1. Open the [Releases](../../releases) page and download `model.pt` **and** `bpe.json`
   from the same version.
2. Put both into the project folder.

Without them the code runs, but `chat.py` has nothing to load.

## Requirements

- Python 3.10+
- PyTorch (CPU build is enough to chat)

```powershell
pip install torch
```

## Run

```powershell
# put model.pt and bpe.json next to the code, then:
python chat.py
```

If the files live elsewhere, pass the folder:

```powershell
python chat.py "C:\path\to\folder-with-model.pt"
```

Inside the chat:

```
/set temp 0.5        lower the temperature
/set top_k 20        choose from the 20 best tokens
/set penalty 1.0     stronger repetition penalty
/set n 200           answer length
/greedy              switch argmax on/off
/params              show settings
/reset               back to defaults (/quit to exit)
```

Defaults: `temp=0.7  top_k=40  penalty=0.6  n=120`.

## Generate without the chat

```powershell
python generate.py --prompt "Москва — это " --n 120
python generate.py --greedy --prompt "Москва — это "   # old argmax behaviour
```

## Train it yourself

Everything runs in Google Colab on a free T4 - see `COLAB.md` for the step-by-step.
Roughly 20-40 minutes of preprocessing and about 1.5 hours for 30 000 steps.

Locally (CPU) it works too, just many times slower.

## What it can and can't do

It does some things correctly:

```
Ты: Какая столица Франции?
ИИ: Париж

Ты: Какая столица Японии?
ИИ: Токио

Ты: Что такое роза?
ИИ: Это цветок.

Ты: 123 и 342.
ИИ: Шаг 1: единиц 3+2=5, пишем 5. Шаг 2: десятков 2+4=6, пишем 6.
     Шаг 3: сотен 1+3=4, пишем 4. Ответ: 465
```

...and it fails at others, honestly:

```
Ты: Сложи 12 и 39.
ИИ: ... Шаг 2: десятков 1+3=5, пишем 5. Ответ: 551        (перенос потерян)

Ты: Что такое квантовая физика?
ИИ: Это фрукт.                                             (выучен шаблон, не смысл)

Ты: ыапщжриогыщпилмрщшвакпро0зкауфыошпщыарвзпщшфрош
ИИ: Привет! Чем помочь?                                    (мусор = приветствие)
```

- **Facts it saw are solid.** Capitals 6/6, word categories 5/6 - everything from the study book.
- **A carry is found but not applied.** `12+39` loses the carried digit.
- **Anything unknown gets the most common template.** No understanding, just the format.
- **Random input is treated as a greeting.**

This is not an assistant. It's a sandbox for understanding how transformers learn
and where small models break.

## Files

| File | What it does |
|---|---|
| `bpe.py` | BPE tokenizer with word-boundary markers, train/save/load/encode/decode |
| `model.py` | the transformer itself - 6 layers x 384 |
| `make_corpus.py` | builds the mixed corpus: Wikipedia + generated Q&A |
| `pretrain.py` | training loop with fp16, checkpoints every 500 steps |
| `generate.py` | sampling: top-k, temperature, repetition penalty, `--greedy` |
| `chat.py` | interactive chat with live settings |
| `COLAB.md` | step-by-step Colab instructions |
| `bpe.json` | vocabulary of the trained model |
| `model.pt` | weights - **download from Releases** |

## License

MIT - see [LICENSE](LICENSE).

Yes, this thing is legally protected. It still can't add 12 + 39, but at least the
lawyers are covered.
