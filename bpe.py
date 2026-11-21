import json
import re
from collections import Counter
from pathlib import Path

SEP = "\u2581"

BASE = (
    [chr(c) for c in range(ord("а"), ord("я") + 1)]
    + ["ё"]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [str(d) for d in range(10)]
    + list(" .,!?-«»():;\"'/\n\r\t<>_=")
    + [SEP]
)

TOKEN_RE = re.compile(r"\n|[^\W\d_]+|\d+|[^\s\w]", re.UNICODE)


def split_words(text):
    words = []
    for tok in TOKEN_RE.findall(text):
        if tok == "\n" or not tok.strip():
            continue
        if tok[0].isalnum() or tok[0] == "_":
            words.append(SEP + tok)
        else:
            words.append(tok)
    return words


class BPE:
    def __init__(self, vocab, merges):
        self.vocab = list(vocab)
        self.merges = {tuple(k): v for k, v in merges}
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self._rank = {k: i for i, k in enumerate(self.merges)}

    @classmethod
    def train(cls, text, vocab_size=8192, max_words=200_000):
        counts = Counter(split_words(text[:2_000_000]))
        top = [w for w, _ in counts.most_common(max_words)]
        vocab = list(dict.fromkeys(BASE))
        merges = []
        word_syms = []
        for w in top:
            syms = list(w)
            for s in syms:
                if s not in vocab:
                    vocab.append(s)
            word_syms.append((syms, counts[w]))

        while len(vocab) < vocab_size:
            pairs = Counter()
            for syms, n in word_syms:
                for i in range(len(syms) - 1):
                    pairs[(syms[i], syms[i + 1])] += n
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            new = a + b
            if new in vocab:
                break
            vocab.append(new)
            merges.append(((a, b), new))
            updated = []
            for syms, n in word_syms:
                out, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                updated.append((out, n))
            word_syms = updated
        return cls(vocab, merges)

    def encode_word(self, word):
        if word in self.stoi:
            return [self.stoi[word]]
        syms = [s for s in word]
        syms = [s if s in self.stoi else "_" for s in syms]
        while len(syms) > 1:
            best_i, best_rank = -1, 1 << 30
            for i in range(len(syms) - 1):
                r = self._rank.get((syms[i], syms[i + 1]))
                if r is not None and r < best_rank:
                    best_i, best_rank = i, r
            if best_i < 0:
                break
            merged = self.merges[(syms[best_i], syms[best_i + 1])]
            syms = syms[:best_i] + [merged] + syms[best_i + 2:]
        return [self.stoi.get(s, self.stoi.get("_", 0)) for s in syms]

    def encode(self, text):
        out = []
        for tok in TOKEN_RE.findall(text):
            if tok == "\n":
                out.append(self.stoi.get("\n", 0))
            elif tok.strip() and (tok[0].isalnum() or tok[0] == "_"):
                out.extend(self.encode_word(SEP + tok))
            else:
                out.append(self.stoi.get(tok, 0))
        return out

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids).replace(SEP, " ")

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {"vocab": self.vocab, "merges": [[list(k), v] for k, v in self.merges.items()]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], [(tuple(k), v) for k, v in d["merges"]])
