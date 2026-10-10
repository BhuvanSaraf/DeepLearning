"""Task 3: classification on bottleneck codes of the 1-hidden-layer autoencoders."""
from common import *

Xtr, ytr, Xva, yva, Xte, yte, classes = load_all()
results = {}
for k in DIMS:
    m = AE(k, 1).to(DEVICE); m.load_state_dict(torch.load(f"{OUT}/models/ae1_{k}.pt"))
    results[k] = classify_representation(encode(m, Xtr), ytr, encode(m, Xva), yva, encode(m, Xte), yte,
                                         classes, f"task3_ae1_{k}")
best_k = max(results, key=lambda k: results[k]["test_acc"])
print(f"Best 1-hidden AE dimension: {best_k} (test acc {results[best_k]['test_acc']:.4f})")
save_json({"per_dim": results, "best_dim": best_k}, "task3.json")
