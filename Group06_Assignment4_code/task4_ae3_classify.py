"""Task 4: classification on bottleneck codes of the 3-hidden-layer (400-k-400) autoencoders."""
from common import *

Xtr, ytr, Xva, yva, Xte, yte, classes = load_all()
results = {}
for k in DIMS:
    m = AE(k, 3).to(DEVICE); m.load_state_dict(torch.load(f"{OUT}/models/ae3_{k}.pt"))
    results[k] = classify_representation(encode(m, Xtr), ytr, encode(m, Xva), yva, encode(m, Xte), yte,
                                         classes, f"task4_ae3_{k}")
best_k = max(results, key=lambda k: results[k]["test_acc"])
print(f"Best 3-hidden AE dimension: {best_k} (test acc {results[best_k]['test_acc']:.4f})")
save_json({"per_dim": results, "best_dim": best_k}, "task4.json")
