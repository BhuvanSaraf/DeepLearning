"""Shared helpers for CS601T Assignment 4: data loading, FCNN classifier, autoencoders, plotting."""
import os, json, glob, random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---------------- config ----------------
DATA_DIR = os.environ.get("A4_DATA", os.path.join(os.path.dirname(__file__), "data"))
# expects DATA_DIR/{train,val,test}/<class>/*.png|jpg
OUT = os.environ.get("A4_OUT", "results")
DIMS = [32, 64, 128, 256]
FCNN_ARCHS = {"A": [64], "B": [128, 64], "C": [256, 128, 64]}   # same set used in every task
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/models", exist_ok=True)


def seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------- data ----------------
def load_split(split):
    """Returns X (N,784) float32 in [0,1], y (N,) int, class names.
    If your dataset is stored differently (same as Assignment 3), replace only this function."""
    split_dir = os.path.join(DATA_DIR, split)
    classes = sorted([c for c in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, c))])
    X, y = [], []
    for ci, c in enumerate(classes):
        for f in sorted(glob.glob(os.path.join(split_dir, c, "*"))):
            X.append(np.asarray(Image.open(f).convert("L"), dtype=np.float32).reshape(-1) / 255.0)
            y.append(ci)
    return np.stack(X), np.array(y), classes


def load_all():
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    return tr[0], tr[1], va[0], va[1], te[0], te[1], tr[2]


def t(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def batches(n, bs, shuffle=True):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


# ---------------- FCNN classifier ----------------
def make_fcnn(d_in, hidden, n_cls):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, n_cls))
    return nn.Sequential(*layers).to(DEVICE)


def train_fcnn(Ztr, ytr, Zva, yva, hidden, n_cls, lr=1e-3, bs=64, max_ep=500, patience=20):
    """Adam + cross-entropy, early stopping on validation loss (best weights restored)."""
    seed(0)
    model = make_fcnn(Ztr.shape[1], hidden, n_cls)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt, Yt, Xv, Yv = t(Ztr), torch.tensor(ytr, device=DEVICE), t(Zva), torch.tensor(yva, device=DEVICE)
    best, best_state, bad = 1e9, None, 0
    count = 0;
    for ep in range(max_ep):
        count += 1
        model.train()
        for b in batches(len(Xt), bs):
            opt.zero_grad(); 
            lossf(model(Xt[b]), Yt[b]).backward();
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), Yv).item()
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    #print("Number of epochs: " , count)
    model.load_state_dict(best_state)
    return model


def predict(model, Z):
    model.eval()
    with torch.no_grad():
        return model(t(Z)).argmax(1).cpu().numpy()


def standardize(Ztr, *others):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    return [(Ztr - mu) / sd] + [(Z - mu) / sd for Z in others]


def classify_representation(Ztr, ytr, Zva, yva, Zte, yte, classes, tag, archs=None):
    """Train every FCNN arch on one representation; pick best by val acc; report test acc + confusion matrix."""
    Ztr, Zva, Zte = standardize(Ztr, Zva, Zte)
    archs = archs or FCNN_ARCHS
    res = {"val_acc": {}, "train_acc": {}}
    models = {}
    for name, hidden in archs.items():
        m = train_fcnn(Ztr, ytr, Zva, yva, hidden, len(classes))
        models[name] = m
        res["train_acc"][name] = float((predict(m, Ztr) == ytr).mean())
        res["val_acc"][name] = float((predict(m, Zva) == yva).mean())
        print(f"  [{tag}] arch {name} {hidden}: train {res['train_acc'][name]:.4f}  val {res['val_acc'][name]:.4f}")
    best = max(res["val_acc"], key=res["val_acc"].get)
    pred = predict(models[best], Zte)
    res["best_arch"] = best
    res["test_acc"] = float((pred == yte).mean())
    res["test_acc_all"] = {n: float((predict(m, Zte) == yte).mean()) for n, m in models.items()}
    cm = confusion_matrix(yte, pred)
    res["confusion_matrix"] = cm.tolist()
    plot_cm(cm, classes, f"{tag} (arch {best}) test acc={res['test_acc']:.4f}", f"{OUT}/cm_{tag}.png")
    print(f"  [{tag}] best arch {best} -> test acc {res['test_acc']:.4f}")
    return res


def plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="white" if cm[i][j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------------- autoencoders ----------------
class AE(nn.Module):
    """n_hidden=1: 784 -> k(linear) -> 784
       n_hidden=3: 784 -> 400(sigmoid) -> k(linear) -> 400(sigmoid) -> 784
       Output layer uses logistic sigmoid since pixels are in [0,1]."""
    def __init__(self, k, n_hidden=1):
        super().__init__()
        if n_hidden == 1:
            self.encoder = nn.Sequential(nn.Linear(784, k))
            self.decoder = nn.Sequential(nn.Linear(k, 784), nn.Sigmoid())
        else:
            self.encoder = nn.Sequential(nn.Linear(784, 400), nn.Sigmoid(), nn.Linear(400, k))
            self.decoder = nn.Sequential(nn.Linear(k, 400), nn.Sigmoid(), nn.Linear(400, 784), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def add_noise(x, p):
    """Gaussian noise: each pixel is perturbed with N(0,1) with probability p; otherwise unchanged."""
    noise = (torch.rand_like(x) < p).float()
    return x + noise * torch.randn_like(x)


def train_ae(Xtr, Xva, k, n_hidden=1, noise=0.0, lr=1e-3, bs=64, max_ep=500, patience=20):
    """Adam + MSE. For denoising (noise>0) input is corrupted, target is the clean image.
    Early stopping on clean validation reconstruction loss."""
    seed(0)
    model = AE(k, n_hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt, Xv = t(Xtr), t(Xva)
    best, best_state, bad = 1e9, None, 0
    for ep in range(max_ep):
        model.train()
        for b in batches(len(Xt), bs):
            xb = Xt[b]
            inp = add_noise(xb, noise) if noise > 0 else xb
            opt.zero_grad(); lossf(model(inp), xb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), Xv).item()
        if vl < best - 1e-6:
            best, bad, best_state = vl, 0, {kk: v.clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    print(f"  AE k={k} hidden={n_hidden} noise={noise}: stopped at epoch {ep + 1}, best val MSE {best:.5f}")
    return model


def recon_error(model, X):
    """Average reconstruction error = mean over images of sum-of-squared pixel errors."""
    model.eval()
    with torch.no_grad():
        Xt = t(X)
        return float(((model(Xt) - Xt) ** 2).sum(1).mean().item())


def encode(model, X):
    model.eval()
    with torch.no_grad():
        return model.encoder(t(X)).cpu().numpy()


def reconstruct(model, X):
    model.eval()
    with torch.no_grad():
        return model(t(X)).cpu().numpy()


def one_per_class(y):
    return [int(np.where(y == c)[0][0]) for c in np.unique(y)]


def plot_recon_grid(rows, row_labels, classes, title, path):
    """rows: list of (n_classes,784) arrays; first row is usually the originals."""
    nr, nc = len(rows), len(classes)
    fig, axes = plt.subplots(nr, nc, figsize=(1.4 * nc + 1, 1.4 * nr))
    for i, r in enumerate(rows):
        for j in range(nc):
            ax = axes[i, j]
            ax.imshow(r[j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(f"class {classes[j]}", fontsize=8)
            if j == 0: ax.set_ylabel(row_labels[i], fontsize=8)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_json(obj, name):
    with open(f"{OUT}/{name}", "w") as f:
        json.dump(obj, f, indent=2)


def load_json(name):
    with open(f"{OUT}/{name}") as f:
        return json.load(f)
