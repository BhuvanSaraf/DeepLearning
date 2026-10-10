"""Task 1: PCA (eigenvectors from training data only) -> FCNN classification for 32/64/128/256 dims."""
import numpy as np
from common import *

Xtr, ytr, Xva, yva, Xte, yte, classes = load_all()
mu = Xtr.mean(0)                                   # training mean used for all splits
C = np.cov(Xtr - mu, rowvar=False)
evals, evecs = np.linalg.eigh(C)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]

results = {}
for k in DIMS:
    W = evecs[:, :k]
    Ztr, Zva, Zte = (Xtr - mu) @ W, (Xva - mu) @ W, (Xte - mu) @ W
    print(f"PCA k={k}: explained variance {evals[:k].sum() / evals.sum():.4f}")
    r = classify_representation(Ztr, ytr, Zva, yva, Zte, yte, classes, f"task1_pca{k}")
    r["explained_variance"] = float(evals[:k].sum() / evals.sum())
    results[k] = r
best_k = max(results, key=lambda k: results[k]["test_acc"])
print(f"Best PCA dimension: {best_k} (test acc {results[best_k]['test_acc']:.4f})")
save_json({"per_dim": results, "best_dim": best_k}, "task1.json")
