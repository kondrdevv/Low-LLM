# Low-LLM - a tiny language model built from scratch

A very small transformer that trains and answers **fully offline** on an ordinary CPU.
No internet, no API, no pre-trained weights — just PyTorch and curiosity.

It is a **dumb little AI** by design: it learns from a tiny generated study book and gets
surprisingly good at a few narrow things, then fails in funny ways on everything else.
The point is to watch exactly where that line is — and then make it bigger.

## How it works

- `data.py` — builds the "study book": question → answer pairs (arithmetic, capitals,
  word categories, letter counting, word reversal).
- `model.py` — a decoder-only transformer. It learns to predict the next byte, nothing else.
  There are no rules, lookups or hardcoded answers anywhere in the code.
- `train.py` — trains from scratch (AdamW + warmup + cosine schedule).
- `chat.py` — an offline chat loop: type a question, get an answer.

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
You: How many sides does a square have?
AI:  [4]

You: What is the capital of France?
AI:  Paris

You: Say this backwards: "i love tea".
AI:  "tea love i".
```

...and it fails at others, honestly:

```
You: How much is 7 + 8?
AI:  [7]

You: What is 1234 * 5678?
AI:  [5]

You: What is the capital of Mars?
AI:  Korengagord
```

- **It can't do real math.** Small sums get mangled — arithmetic doesn't fit in a model this size.
- **It only works on its own material.** Words it never saw turn into plausible-looking nonsense.
- **It doesn't understand language.** It learned patterns and a handful of facts; reasoning and
  open-ended dialogue are way beyond it.
- **Answers get cut off.** The model just runs into its length limit.

So no, it is not an assistant. It's a sandbox for understanding how transformers learn
and where small models break.

## Files

| File | What it does |
|---|---|
| `data.py` | generates training pairs and encodes them into tokens |
| `model.py` | architecture, batching, generation, answer rendering |
| `train.py` | training loop and checkpoint saving |
| `chat.py` | interactive chat |
| `checkpoint.pt` | trained weights |
| `checkpoint_v1.pt` | first version (before the corpus rework) |

## Roadmap

The goal is to keep upgrading it:

- [ ] bigger model — more layers and width
- [ ] better tokenization (BPE instead of raw bytes)
- [ ] reasoning-style tasks instead of pure recall
- [ ] step-by-step arithmetic so it can actually add
- [ ] proper evaluation: accuracy on held-out and out-of-distribution questions

## Tune it yourself

Change `STEPS`, `BATCH` and `CFG` in `train.py` — the model size is defined by
`n_layer`, `dim`, `n_head` and `seq_len`.

## License

MIT - see [LICENSE](LICENSE).

Yes, this thing is legally protected. It still can't add 7 + 8, but at least the
lawyers are covered.

