"""Collects all task results into results/summary.md (tables to copy into the report)."""
from common import *

A3_BEST_TEST_ACC = None   # <-- put your best Assignment-3 test accuracy here (e.g. 0.9812)
L = []

def cls_table(title, res):
    archs = list(FCNN_ARCHS)
    L.append(f"\n### {title}\n")
    L.append("| dim | " + " | ".join(f"val acc {a}" for a in archs) + " | best arch | test acc (best) |")
    L.append("|---" * (len(archs) + 3) + "|")
    for k, r in res["per_dim"].items():
        L.append(f"| {k} | " + " | ".join(f"{r['val_acc'][a]:.4f}" for a in archs)
                 + f" | {r['best_arch']} | {r['test_acc']:.4f} |")
    b = res["best_dim"]
    L.append(f"\nBest dimension: **{b}** (test acc {res['per_dim'][str(b)]['test_acc']:.4f}); "
             f"confusion matrix: `cm_*{b}.png`")

L.append("# Assignment 4 results summary\n")
L.append("FCNN architectures (tanh hidden, Adam, early stopping on val loss): "
         + ", ".join(f"{a}={h}" for a, h in FCNN_ARCHS.items()))
t1, t3, t4 = load_json("task1.json"), load_json("task3.json"), load_json("task4.json")
cls_table("Task 1: PCA", t1)

L.append("\n### Task 2: average reconstruction error (sum of squared pixel errors per image)\n")
L.append("| model | train | val | test |\n|---|---|---|---|")
for m, e in load_json("task2.json").items():
    L.append(f"| {m} | {e['train']:.3f} | {e['val']:.3f} | {e['test']:.3f} |")

cls_table("Task 3: 1-hidden-layer AE codes", t3)
cls_table("Task 4: 3-hidden-layer AE codes", t4)

t5 = load_json("task5.json")
L.append(f"\n### Task 5: denoising AEs (k={t5['k']}, Gaussian noise)\n")
L.append("| model | recon train | recon val | recon test | " + " | ".join(f"val {a}" for a in FCNN_ARCHS)
         + " | " + " | ".join(f"test {a}" for a in FCNN_ARCHS) + " |")
L.append("|---" * (4 + 2 * len(FCNN_ARCHS)) + "|")
for tag in ["dae20", "dae40"]:
    r = t5[tag]; e = r["recon_error"]; c = r["classification"]
    L.append(f"| {tag} | {e['train']:.3f} | {e['val']:.3f} | {e['test']:.3f} | "
             + " | ".join(f"{c['val_acc'][a]:.4f}" for a in FCNN_ARCHS) + " | "
             + " | ".join(f"{c['test_acc_all'][a]:.4f}" for a in FCNN_ARCHS) + " |")
L.append(f"\nTask-3 best arch for k={t5['k']}: {t5['task3_best_arch']}")

L.append("\n### Overall comparison (best test accuracy per method)\n")
L.append("| method | best dim | test acc |\n|---|---|---|")
L.append(f"| Assignment 3 (raw 784-d) | 784 | {A3_BEST_TEST_ACC if A3_BEST_TEST_ACC is not None else 'FILL IN'} |")
for name, r in [("PCA (Task 1)", t1), ("1-hidden AE (Task 3)", t3), ("3-hidden AE (Task 4)", t4)]:
    L.append(f"| {name} | {r['best_dim']} | {r['per_dim'][str(r['best_dim'])]['test_acc']:.4f} |")
for tag in ["dae20", "dae40"]:
    L.append(f"| {tag} (Task 5, Task-3 arch) | {t5['k']} | {t5[tag]['classification']['test_acc_task3_arch']:.4f} |")

open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
