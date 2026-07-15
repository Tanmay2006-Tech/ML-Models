import math
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Transformer encoder built from the actual pieces (attention, positional
# encoding, feed forward) instead of nn.Transformer, so it's clear what's
# happening under the hood.

POSITIVE = [
    "this movie was absolutely fantastic and moving",
    "I loved every minute of it, brilliant film",
    "a masterpiece, beautifully directed",
    "the performances were incredible and touching",
    "truly inspiring and well written",
    "an amazing experience from start to finish",
    "the cinematography was stunning",
    "highly recommend, a must watch",
    "great soundtrack and gripping story",
    "one of the best films of the decade",
    "a wonderfully crafted and heartfelt story",
    "the direction was flawless and captivating",
    "an unforgettable and powerful piece of cinema",
    "the writing was sharp and the acting superb",
    "a joyful and uplifting watch for the whole family",
    "the pacing was perfect and never dragged",
    "beautifully shot with a stellar cast",
    "a genuinely funny and clever comedy",
    "the emotional depth of this film blew me away",
    "an instant classic that I will watch again",
    "the chemistry between the leads was electric",
    "a bold and original take on the genre",
    "left the theater completely satisfied",
    "the score elevated every single scene",
    "a triumph of storytelling and visual design",
]

NEGATIVE = [
    "what a waste of time, terrible acting",
    "boring plot and awful dialogue",
    "I regret watching this, so bad",
    "worst film I have seen this year",
    "dull characters and a predictable story",
    "I fell asleep, it was that boring",
    "poorly executed with a weak script",
    "disappointing sequel, avoid it",
    "flat performances and no emotional depth",
    "a complete mess, badly edited",
    "the plot made no sense whatsoever",
    "wooden acting throughout the entire film",
    "a tedious and forgettable experience",
    "the jokes fell completely flat",
    "cheaply made with terrible special effects",
    "I wanted my money back after the first act",
    "the dialogue was cringeworthy and stiff",
    "an uninspired and lazy retread of better films",
    "way too long and desperately needed editing",
    "none of the characters were likable",
    "the ending made the whole film pointless",
    "a hollow, style-over-substance disaster",
    "painfully slow with no payoff",
    "the worst sequel in the franchise by far",
    "I couldn't wait for it to be over",
]

DATA = [(t, 1) for t in POSITIVE] + [(t, 0) for t in NEGATIVE]


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word in counter:
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=32):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)[:max_len]]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        return self.out(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128, num_layers=2, num_classes=2, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        mask = (ids != self.pad_idx).unsqueeze(1).unsqueeze(2)
        x = self.embed(ids) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        valid = (ids != self.pad_idx).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-6)
        return self.classifier(pooled)


vocab = build_vocab([t for t, _ in DATA])
X = torch.tensor([encode(t, vocab) for t, _ in DATA])
y = torch.tensor([label for _, label in DATA])

model = TransformerClassifier(vocab_size=len(vocab))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

losses = []
for epoch in range(30):
    opt.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

model.eval()
with torch.no_grad():
    acc = (model(X).argmax(dim=1) == y).float().mean().item()

print("training accuracy:", round(acc, 4))

plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title("transformer-nlp - training loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig("results.png", dpi=120)
