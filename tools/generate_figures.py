# -*- coding: utf-8 -*-
"""Genere toutes les figures + metriques du projet (parties I, II, III)."""
import os, sys, json, random, math, time, gzip, struct, urllib.request, zipfile, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import torch
from torch import nn
from torch.nn import functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "artifacts", "figures")
DATA = os.path.join(ROOT, "data")
os.makedirs(FIG, exist_ok=True)
RESULTS = {}

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
DEVICE = torch.device("cpu")
def savefig(name):
    plt.savefig(os.path.join(FIG, name), dpi=120, bbox_inches="tight"); plt.close()
    print("  figure ->", name)

# =====================================================================
# PARTIE I - MLP / Breast Cancer (donnees via scikit-learn)
# =====================================================================
print("\n=== PARTIE I : MLP / Breast Cancer ===")
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)

set_seed(42)
bc = load_breast_cancer()
X = bc.data.astype(np.float32)
# sklearn : target 0 = malin, 1 = benin  -> on encode malin=1 comme dans le notebook
y = (bc.target == 0).astype(np.int64)
Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=0.2, stratify=ytmp, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = sc.transform(Xtr).astype(np.float32), sc.transform(Xva).astype(np.float32), sc.transform(Xte).astype(np.float32)

from torch.utils.data import TensorDataset, DataLoader
def loader(Xa, ya, bs, sh): return DataLoader(TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)), batch_size=bs, shuffle=sh)
tl = loader(Xtr, ytr, 32, True); vl = loader(Xva, yva, 32, False); tel = loader(Xte, yte, 32, False)
IN, NC = Xtr.shape[1], 2

def build_mlp(hidden=(64, 32), drop=0.1):
    layers, prev = [], IN
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(drop)]; prev = h
    layers += [nn.Linear(prev, NC)]
    return nn.Sequential(*layers)

def init_gauss(m):
    if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)
def init_const(m):
    if isinstance(m, nn.Linear): nn.init.constant_(m.weight, 0.5); nn.init.zeros_(m.bias)
def init_xavier(m):
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

def train_mlp(model, epochs=60, lr=1e-3, wd=1e-4):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(); hist = {"val_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval(); vl_loss = 0; correct = total = 0
        with torch.no_grad():
            for xb, yb in vl:
                out = model(xb); vl_loss += crit(out, yb).item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item(); total += yb.size(0)
        hist["val_loss"].append(vl_loss / total); hist["val_acc"].append(correct / total)
    return hist

# Comparaison des 3 initialisations
inits = {"gaussienne": init_gauss, "constante": init_const, "xavier": init_xavier}
init_hist = {}
for name, fn in inits.items():
    set_seed(42); m = build_mlp(); m.apply(fn); init_hist[name] = train_mlp(m, epochs=40)
    print(f"  init {name:11s} val_acc finale={init_hist[name]['val_acc'][-1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, h in init_hist.items():
    ax[0].plot(h["val_loss"], label=name); ax[1].plot(h["val_acc"], label=name)
ax[0].set_title("Perte de validation selon l'initialisation"); ax[0].set_xlabel("epoque"); ax[0].set_ylabel("val_loss"); ax[0].legend()
ax[1].set_title("Accuracy de validation selon l'initialisation"); ax[1].set_xlabel("epoque"); ax[1].set_ylabel("val_acc"); ax[1].legend()
plt.tight_layout(); savefig("p1_init_comparison.png")

# Entrainement final + evaluation
set_seed(42); best = build_mlp(); best.apply(init_xavier); train_mlp(best, epochs=60)
best.eval()
with torch.no_grad():
    logits = torch.cat([best(xb) for xb, _ in tel])
y_pred = logits.argmax(1).numpy(); y_prob = F.softmax(logits, 1)[:, 1].numpy()
RESULTS["p1"] = {
    "accuracy": accuracy_score(yte, y_pred), "precision": precision_score(yte, y_pred),
    "recall": recall_score(yte, y_pred), "f1": f1_score(yte, y_pred),
    "roc_auc": roc_auc_score(yte, y_prob),
    "init_acc": {k: v["val_acc"][-1] for k, v in init_hist.items()},
    "n_params": sum(p.numel() for p in best.parameters()),
}
cm = confusion_matrix(yte, y_pred)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["benin", "malin"], yticklabels=["benin", "malin"])
plt.xlabel("Predit"); plt.ylabel("Reel"); plt.title("Matrice de confusion - test (MLP)")
plt.tight_layout(); savefig("p1_confusion.png")
print("  metriques test:", {k: round(v, 4) for k, v in RESULTS["p1"].items() if isinstance(v, float)})

# =====================================================================
# PARTIE II - CNN / Fashion-MNIST (miroir public)
# =====================================================================
print("\n=== PARTIE II : CNN / Fashion-MNIST ===")
fdir = os.path.join(DATA, "fashion_mnist"); os.makedirs(fdir, exist_ok=True)
FILES = {"train_img": "train-images-idx3-ubyte.gz", "train_lbl": "train-labels-idx1-ubyte.gz",
         "test_img": "t10k-images-idx3-ubyte.gz", "test_lbl": "t10k-labels-idx1-ubyte.gz"}
MIRRORS = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
           "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"]
def fetch(fname):
    dst = os.path.join(fdir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0: return dst
    for base in MIRRORS:
        try:
            req = urllib.request.Request(base + fname, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
            if os.path.getsize(dst) > 0: print("  telecharge:", fname); return dst
        except Exception as e:
            print(f"  echec {base}: {type(e).__name__}")
    raise RuntimeError("Impossible de telecharger " + fname)
def read_images(p):
    with gzip.open(fetch(p)) as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(n, 1, r, c).astype(np.float32) / 255.0
def read_labels(p):
    with gzip.open(fetch(p)) as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), np.uint8).astype(np.int64)

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
Xtr_f = read_images(FILES["train_img"]); ytr_f = read_labels(FILES["train_lbl"])
Xte_f = read_images(FILES["test_img"]); yte_f = read_labels(FILES["test_lbl"])

def subsample(X, y, n, seed=42):
    if n >= len(X): return X, y
    rng = np.random.default_rng(seed); idx = []
    per = n // len(np.unique(y))
    for cl in np.unique(y):
        ci = np.where(y == cl)[0]; idx.extend(rng.choice(ci, min(per, len(ci)), replace=False))
    idx = np.array(idx); rng.shuffle(idx); return X[idx], y[idx]
Xtr2, ytr2 = subsample(Xtr_f, ytr_f, 12000); Xte2, yte2 = subsample(Xte_f, yte_f, 2000)
nv = int(len(Xtr2) * 0.15)
Xva2, yva2 = Xtr2[:nv], ytr2[:nv]; Xtr2, ytr2 = Xtr2[nv:], ytr2[nv:]
ld_tr = loader(Xtr2, ytr2, 128, True); ld_va = loader(Xva2, yva2, 128, False); ld_te = loader(Xte2, yte2, 128, False)
print(f"  train={Xtr2.shape} val={Xva2.shape} test={Xte2.shape}")

# Echantillon d'images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, i in zip(axes.ravel(), range(10)):
    ax.imshow(Xtr2[i, 0], cmap="gray"); ax.set_title(CLASSES[ytr2[i]], fontsize=8); ax.axis("off")
plt.tight_layout(); savefig("p2_samples.png")

class LeNet(nn.Module):
    def __init__(self, c1=6, c2=16, padding=2, stride=1, pool="max", use_1x1=False, nc=10, bn=True):
        super().__init__()
        Pool = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, 5, padding=padding, stride=stride)]
            if bn: layers.append(nn.BatchNorm2d(cout))
            layers += [nn.ReLU(), Pool(2)]
            return layers
        L = block(1, c1)
        if use_1x1: L += [nn.Conv2d(c1, c1, 1), nn.ReLU()]
        L += block(c1, c2)
        self.features = nn.Sequential(*L)
        with torch.no_grad(): flat = self.features(torch.zeros(1, 1, 28, 28)).numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, 120), nn.ReLU(),
                                        nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, nc))
    def forward(self, x): return self.classifier(self.features(x))

def train_clf(model, tr, va, epochs=5, lr=1e-3):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    best_acc, best_state = 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        acc = clf_acc(model, va)
        if acc > best_acc: best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return best_acc
def clf_acc(model, ld):
    model.eval(); c = t = 0
    with torch.no_grad():
        for xb, yb in ld:
            c += (model(xb).argmax(1) == yb).sum().item(); t += yb.size(0)
    return c / t
def n_params(m): return sum(p.numel() for p in m.parameters())

set_seed(42); cnn = LeNet(); train_clf(cnn, ld_tr, ld_va, epochs=15)
RESULTS["p2"] = {"cnn_acc": clf_acc(cnn, ld_te), "cnn_params": n_params(cnn)}
print("  CNN test acc:", round(RESULTS["p2"]["cnn_acc"], 4))

# Etude architecturale (un facteur a la fois)
def run_variant(name, **kw):
    set_seed(42); m = LeNet(**kw); acc = train_clf(m, ld_tr, ld_va, epochs=6)
    return name, acc, n_params(m)
variants = [run_variant("base"), run_variant("padding=0", padding=0), run_variant("stride=2", stride=2),
            run_variant("avg-pooling", pool="avg"), run_variant("+filtres", c1=16, c2=32), run_variant("conv 1x1", use_1x1=True)]
RESULTS["p2"]["variants"] = [{"nom": n, "val_acc": round(a, 4), "params": p} for n, a, p in variants]
names = [v[0] for v in variants]; accs = [v[1] for v in variants]
plt.figure(figsize=(8, 4))
plt.barh(names, accs, color="steelblue")
plt.xlabel("accuracy de validation"); plt.title("Effet des choix architecturaux (CNN)")
plt.xlim(min(accs) - 0.03, max(accs) + 0.01)
for i, v in enumerate(accs): plt.text(v + 0.001, i, f"{v:.3f}", va="center")
plt.tight_layout(); savefig("p2_arch_experiments.png")

# Cartes de caracteristiques
acts = {}
h = cnn.features[0].register_forward_hook(lambda m, i, o: acts.__setitem__("c", o.detach()))
_ = cnn(torch.from_numpy(Xte2[0:1])); h.remove()
fmap = acts["c"][0]; nf = fmap.shape[0]
fig, axes = plt.subplots(1, nf + 1, figsize=(2 * (nf + 1), 2.2))
axes[0].imshow(Xte2[0, 0], cmap="gray"); axes[0].set_title(f"entree\n{CLASSES[yte2[0]]}", fontsize=8); axes[0].axis("off")
for k in range(nf):
    axes[k + 1].imshow(fmap[k], cmap="viridis"); axes[k + 1].set_title(f"filtre {k}", fontsize=8); axes[k + 1].axis("off")
plt.suptitle("Cartes de caracteristiques - 1ere convolution"); plt.tight_layout(); savefig("p2_feature_maps.png")

# Confusion CNN
cnn.eval()
with torch.no_grad():
    yp = torch.cat([cnn(xb).argmax(1) for xb, _ in ld_te]).numpy()
yt = np.concatenate([yb.numpy() for _, yb in ld_te])
cm = confusion_matrix(yt, yp)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.xlabel("Predit"); plt.ylabel("Reel"); plt.title("Matrice de confusion - CNN")
plt.tight_layout(); savefig("p2_confusion.png")

# MLP vs CNN
class MLPImg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x)
set_seed(42); mlpi = MLPImg(); train_clf(mlpi, ld_tr, ld_va, epochs=10)
RESULTS["p2"]["mlp_acc"] = clf_acc(mlpi, ld_te); RESULTS["p2"]["mlp_params"] = n_params(mlpi)
plt.figure(figsize=(5, 4))
bars = ["MLP dense", "CNN (LeNet)"]; vals = [RESULTS["p2"]["mlp_acc"], RESULTS["p2"]["cnn_acc"]]
plt.bar(bars, vals, color=["indianred", "steelblue"])
for i, v in enumerate(vals): plt.text(i, v + 0.002, f"{v:.3f}", ha="center")
plt.ylabel("accuracy test"); plt.ylim(min(vals) - 0.05, 1.0); plt.title("MLP vs CNN (meme dataset)")
plt.tight_layout(); savefig("p2_mlp_vs_cnn.png")
print("  MLP test acc:", round(RESULTS["p2"]["mlp_acc"], 4), "| CNN:", round(RESULTS["p2"]["cnn_acc"], 4))

# =====================================================================
# PARTIE III - RNN/LSTM/GRU + Seq2Seq / fra-eng (miroir public)
# =====================================================================
print("\n=== PARTIE III : RNN / Seq2Seq / fra-eng ===")
edir = os.path.join(DATA, "fra_eng"); os.makedirs(edir, exist_ok=True)
zp = os.path.join(edir, "fra-eng.zip")
if not glob.glob(os.path.join(edir, "**", "*.txt"), recursive=True):
    urllib.request.urlretrieve("https://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip", zp)
    with zipfile.ZipFile(zp) as z: z.extractall(edir)
txt = [p for p in glob.glob(os.path.join(edir, "**", "*.txt"), recursive=True) if "about" not in p.lower()][0]
print("  corpus:", txt)
tok_re = re.compile(r"[a-zà-ÿ0-9]+|[?.!,;:']", re.IGNORECASE)
def tok(s): return tok_re.findall(s.lower())
raw = []
with open(txt, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            te, tf = tok(parts[0]), tok(parts[1])
            if 1 <= len(te) <= 9 and 1 <= len(tf) <= 9: raw.append((te, tf))
random.Random(42).shuffle(raw); pairs = raw[:6000]
print("  paires:", len(pairs))

from collections import Counter
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
class Vocab:
    def __init__(self, lists, mf=2):
        c = Counter(t for l in lists for t in l)
        self.itos = [PAD, BOS, EOS, UNK] + [t for t, n in c.most_common() if n >= mf]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos, self.unk = 0, 1, 2, 3
    def __len__(self): return len(self.itos)
    def enc(self, ts): return [self.stoi.get(t, self.unk) for t in ts]
    def dec(self, ids):
        out = []
        for i in ids:
            t = self.itos[i]
            if t == EOS: break
            if t not in (PAD, BOS): out.append(t)
        return out
vs = Vocab([e for e, f in pairs]); vt = Vocab([f for e, f in pairs])
print(f"  vocab EN={len(vs)} FR={len(vt)}")
def pad_to(ids, L, p): return ids[:L] + [p] * max(0, L - len(ids))
ML = 9 + 2

# LM sur le francais
def lm_tensors(sents, voc):
    X, Y = [], []
    for ts in sents:
        seq = pad_to([voc.bos] + voc.enc(ts) + [voc.eos], ML, voc.pad)
        X.append(seq[:-1]); Y.append(seq[1:])
    return torch.tensor(X), torch.tensor(Y)
Xlm, Ylm = lm_tensors([f for e, f in pairs], vt)
nv = int(len(Xlm) * 0.1)
lm_tr = DataLoader(TensorDataset(Xlm[nv:], Ylm[nv:]), batch_size=128, shuffle=True)
lm_va = DataLoader(TensorDataset(Xlm[:nv], Ylm[:nv]), batch_size=128, shuffle=False)

class LM(nn.Module):
    def __init__(self, V, cell):
        super().__init__()
        self.emb = nn.Embedding(V, 96, padding_idx=0)
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell](96, 128, batch_first=True)
        self.fc = nn.Linear(128, V)
    def forward(self, x): o, _ = self.rnn(self.emb(x)); return self.fc(o)
def mce(logits, tgt, pad=0): return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=pad)
def ppl(model, ld):
    model.eval(); tl = tk = 0
    with torch.no_grad():
        for xb, yb in ld:
            n = (yb != 0).sum().item(); tl += mce(model(xb), yb).item() * n; tk += n
    return math.exp(tl / max(tk, 1))
def train_lm(model, epochs=5, clip=1.0, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); hist = []
    for ep in range(epochs):
        model.train()
        for xb, yb in lm_tr:
            opt.zero_grad(); mce(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
        hist.append(ppl(model, lm_va))
    return hist
lm_res = {}
for cell in ["rnn", "lstm", "gru"]:
    set_seed(42); m = LM(len(vt), cell); t0 = time.time(); h = train_lm(m, epochs=5)
    lm_res[cell] = {"ppl": h, "params": n_params(m), "time": (time.time() - t0) / 5}
    print(f"  {cell.upper():5s} PPL={h[-1]:.2f} params={n_params(m):,} t/ep={lm_res[cell]['time']:.2f}s")
RESULTS["p3_lm"] = {c: {"final_ppl": r["ppl"][-1], "params": r["params"], "time": r["time"]} for c, r in lm_res.items()}

plt.figure(figsize=(7, 4))
for c, r in lm_res.items(): plt.plot(range(1, 6), r["ppl"], marker="o", label=c.upper())
plt.xlabel("epoque"); plt.ylabel("perplexite (plus bas = mieux)"); plt.title("Perplexite de validation : RNN vs LSTM vs GRU")
plt.legend(); plt.tight_layout(); savefig("p3_perplexity.png")

# Gradient clipping : RNN tanh plus large + LR eleve pour provoquer l'explosion
class LMbig(nn.Module):
    def __init__(self, V, hidden=256):
        super().__init__(); self.emb = nn.Embedding(V, 96, padding_idx=0)
        self.rnn = nn.RNN(96, hidden, batch_first=True); self.fc = nn.Linear(hidden, V)
    def forward(self, x): o, _ = self.rnn(self.emb(x)); return self.fc(o)

def grad_norms(clip, lr=1.0, steps=120):
    set_seed(42); m = LMbig(len(vt)); opt = torch.optim.SGD(m.parameters(), lr=lr)
    norms = []; it = iter(lm_tr)
    for _ in range(steps):
        try: xb, yb = next(it)
        except StopIteration: it = iter(lm_tr); xb, yb = next(it)
        opt.zero_grad(); mce(m(xb), yb).backward()
        tot = math.sqrt(sum(p.grad.pow(2).sum().item() for p in m.parameters() if p.grad is not None))
        norms.append(tot)
        if clip is not None: nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
    return norms
theta = 1.0
norms_raw = grad_norms(None, lr=1.0)             # entrainement instable (LR eleve)
norms_clipped = [min(n, theta) for n in norms_raw]
plt.figure(figsize=(10, 4))
plt.plot(norms_raw, label="norme brute (sans clipping)", alpha=0.85)
plt.plot(norms_clipped, label=f"norme apres clipping (theta={theta})", alpha=0.95, lw=2)
plt.axhline(theta, ls="--", color="gray", lw=1)
plt.yscale("log"); plt.xlabel("iteration"); plt.ylabel("norme du gradient (log)")
plt.title("Effet du gradient clipping sur la stabilite de la BPTT"); plt.legend()
plt.tight_layout(); savefig("p3_grad_clip.png")
RESULTS["p3_lm"]["grad_max_noclip"] = max(norms_raw); RESULTS["p3_lm"]["grad_max_clip"] = max(norms_clipped)
print(f"  grad max brut={max(norms_raw):.1f} apres clipping={max(norms_clipped):.1f}")

# Seq2Seq
def s2s_tensors(prs):
    S, TI, TO = [], [], []
    for te, tf in prs:
        S.append(pad_to(vs.enc(te) + [vs.eos], ML, vs.pad))
        t = pad_to([vt.bos] + vt.enc(tf) + [vt.eos], ML, vt.pad)
        TI.append(t[:-1]); TO.append(t[1:])
    return torch.tensor(S), torch.tensor(TI), torch.tensor(TO)
S, TI, TO = s2s_tensors(pairs); nv = int(len(S) * 0.1)
s2s_tr = DataLoader(TensorDataset(S[nv:], TI[nv:], TO[nv:]), batch_size=128, shuffle=True)
s2s_va = DataLoader(TensorDataset(S[:nv], TI[:nv], TO[:nv]), batch_size=128, shuffle=False)
val_pairs = pairs[:nv]
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.e = nn.Embedding(len(vs), 96, padding_idx=0); s.g = nn.GRU(96, 128, batch_first=True)
    def forward(s, x): o, h = s.g(s.e(x)); return o, h
class Dec(nn.Module):
    def __init__(s):
        super().__init__(); s.e = nn.Embedding(len(vt), 96, padding_idx=0); s.g = nn.GRU(96, 128, batch_first=True); s.f = nn.Linear(128, len(vt))
    def forward(s, x, h): o, h = s.g(s.e(x), h); return s.f(o), h
class S2S(nn.Module):
    def __init__(s): super().__init__(); s.enc = Enc(); s.dec = Dec()
    def forward(s, src, ti): _, h = s.enc(src); o, _ = s.dec(ti, h); return o
set_seed(42); model = S2S()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
hist_tr, hist_va = [], []
N_EP_S2S = 12
for ep in range(N_EP_S2S):
    model.train(); run = tk = 0
    for src, ti, to in s2s_tr:
        opt.zero_grad(); loss = mce(model(src, ti), to); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        n = (to != 0).sum().item(); run += loss.item() * n; tk += n
    hist_tr.append(run / tk)
    model.eval(); vr = vt2 = 0
    with torch.no_grad():
        for src, ti, to in s2s_va:
            n = (to != 0).sum().item(); vr += mce(model(src, ti), to).item() * n; vt2 += n
    hist_va.append(vr / vt2)
print(f"  seq2seq val_loss final={hist_va[-1]:.3f} PPL={math.exp(hist_va[-1]):.2f}")
plt.figure(figsize=(7, 4))
ep_axis = range(1, N_EP_S2S + 1)
plt.plot(ep_axis, hist_tr, marker="o", label="train"); plt.plot(ep_axis, hist_va, marker="s", label="val")
plt.xlabel("epoque"); plt.ylabel("perte masquee"); plt.title("Apprentissage du Seq2Seq (EN->FR)"); plt.legend()
plt.tight_layout(); savefig("p3_seq2seq_loss.png")

def greedy(src_ids):
    model.eval()
    with torch.no_grad():
        _, st = model.enc(torch.tensor([src_ids])); t = torch.tensor([[vt.bos]]); out = []
        for _ in range(ML):
            lo, st = model.dec(t, st); nx = int(lo[0, -1].argmax())
            if nx == vt.eos: break
            out.append(nx); t = torch.tensor([[nx]])
    return vt.dec(out)
def beam(src_ids, k=3, alpha=0.7):
    model.eval()
    with torch.no_grad():
        _, st = model.enc(torch.tensor([src_ids])); beams = [(0.0, [vt.bos], st, False)]
        for _ in range(ML):
            cand = []
            for lp, tks, s, done in beams:
                if done: cand.append((lp, tks, s, True)); continue
                lo, s2 = model.dec(torch.tensor([[tks[-1]]]), s); logp = F.log_softmax(lo[0, -1], -1)
                tv, ti2 = logp.topk(k)
                for v, i in zip(tv.tolist(), ti2.tolist()):
                    cand.append((lp + v, tks + [i], s2, i == vt.eos))
            cand.sort(key=lambda c: c[0] / (len(c[1]) ** alpha), reverse=True); beams = cand[:k]
            if all(b[3] for b in beams): break
        best = max(beams, key=lambda c: c[0] / (len(c[1]) ** alpha))
    return vt.dec(best[1][1:])
def ngc(toks, n): return Counter(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))
def sbleu(cand, ref, mn=4):
    if not cand: return 0.0
    lp = 0
    for n in range(1, mn + 1):
        cg = ngc(cand, n); rg = ngc(ref, n)
        ov = sum(min(c, rg.get(g, 0)) for g, c in cg.items()); tot = max(sum(cg.values()), 1)
        lp += (1 / mn) * math.log((ov + 1e-9) / tot)
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / max(len(cand), 1))
    return bp * math.exp(lp)
def cbleu(fn, n=400):
    sc = []
    for te, tf in val_pairs[:n]:
        s = pad_to(vs.enc(te) + [vs.eos], ML, vs.pad); sc.append(sbleu(fn(s), tf))
    return float(np.mean(sc))
bg = cbleu(lambda s: greedy(s)); bb = cbleu(lambda s: beam(s))
RESULTS["p3_seq2seq"] = {"val_ppl": math.exp(hist_va[-1]), "bleu_greedy": bg, "bleu_beam": bb}
samples = []
for te, tf in val_pairs[:6]:
    s = pad_to(vs.enc(te) + [vs.eos], ML, vs.pad)
    samples.append({"en": " ".join(te), "ref": " ".join(tf),
                    "greedy": " ".join(greedy(s)), "beam": " ".join(beam(s))})
RESULTS["p3_seq2seq"]["samples"] = samples
print(f"  BLEU greedy={bg:.4f} beam={bb:.4f}")

with open(os.path.join(FIG, "results.json"), "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)
print("\nTERMINE. Figures + results.json dans", FIG)
