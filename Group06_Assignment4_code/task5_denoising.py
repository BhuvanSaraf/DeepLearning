"""Task 5: 1-hidden-layer denoising AEs (20% and 40% Gaussian noise), bottleneck = best dim from Task 3."""
from common import *

Xtr, ytr, Xva, yva, Xte, yte, classes = load_all()
t3 = load_json("task3.json")
k = int(t3["best_dim"]); t3_best_arch = t3["per_dim"][str(k)]["best_arch"]
print(f"Bottleneck k={k} (best from Task 3); Task-3 best FCNN arch for it: {t3_best_arch}")
splits = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
out = {"k": k, "task3_best_arch": t3_best_arch}
for p in [0.2, 0.4]:
    tag = f"dae{int(p * 100)}"
    m = train_ae(Xtr, Xva, k, n_hidden=1, noise=p)
    torch.save(m.state_dict(), f"{OUT}/models/{tag}_{k}.pt")
    r = {"recon_error": {s: recon_error(m, X) for s, (X, _) in splits.items()}}
    print(f"  recon error {tag}: {r['recon_error']}")
    for s, (X, y) in splits.items():
        idx = one_per_class(y)
        seed(); noisy = add_noise(t(X[idx]), p).cpu().numpy()
        rows = [X[idx], reconstruct(m, X[idx]), noisy, reconstruct(m, noisy)]
        plot_recon_grid(rows, ["original", "recon(clean)", f"noisy {int(p*100)}%", "recon(noisy)"], classes,
                        f"Denoising AE {int(p*100)}% noise, k={k} ({s})", f"{OUT}/recon_{tag}_{s}.png")
    c = classify_representation(encode(m, Xtr), ytr, encode(m, Xva), yva, encode(m, Xte), yte, classes, f"task5_{tag}")
    c["test_acc_task3_arch"] = c["test_acc_all"][t3_best_arch]
    r["classification"] = c
    out[tag] = r
save_json(out, "task5.json")
