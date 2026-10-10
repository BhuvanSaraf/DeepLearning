"""Task 2: 1-hidden and 3-hidden-layer autoencoders, bottleneck 32/64/128/256 (linear), Adam.
Saves models (used by Tasks 3, 4, 6), reconstruction errors and reconstruction figures."""
import os
from common import *

Xtr, ytr, Xva, yva, Xte, yte, classes = load_all()
os.makedirs(f"{OUT}/models", exist_ok=True)
splits = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
errors = {}
for nh in [1, 3]:
    models = {}
    for k in DIMS:
        m = train_ae(Xtr, Xva, k, n_hidden=nh)
        torch.save(m.state_dict(), f"{OUT}/models/ae{nh}_{k}.pt")
        models[k] = m
        errors[f"ae{nh}_{k}"] = {s: recon_error(m, X) for s, (X, _) in splits.items()}
        print(f"  recon error ae{nh}_{k}: {errors[f'ae{nh}_{k}']}")
    for s, (X, y) in splits.items():
        idx = one_per_class(y)
        rows = [X[idx]] + [reconstruct(models[k], X[idx]) for k in DIMS]
        plot_recon_grid(rows, ["original"] + [f"k={k}" for k in DIMS], classes,
                        f"{nh}-hidden-layer AE reconstructions ({s})", f"{OUT}/recon_ae{nh}_{s}.png")
save_json(errors, "task2.json")
