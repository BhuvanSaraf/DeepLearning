"""Task 6: plot encoder weights (input -> bottleneck) as images for the best 1-hidden AE and both DAEs.
For a linear unit h_i = w_i.x + b_i, the unit-norm input that maximally activates it is x* = w_i / ||w_i||."""
import math
from common import *

k = int(load_json("task3.json")["best_dim"])

def plot_weights(path_pt, title, out_png):
    m = AE(k, 1).to(DEVICE); m.load_state_dict(torch.load(path_pt))
    W = m.encoder[0].weight.detach().cpu().numpy()            # (k, 784)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    n = math.ceil(math.sqrt(k))
    fig, axes = plt.subplots(n, n, figsize=(n * 0.7, n * 0.7))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < k:
            a = np.abs(W[i]).max()
            ax.imshow(W[i].reshape(28, 28), cmap="gray", vmin=-a, vmax=a)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)

plot_weights(f"{OUT}/models/ae1_{k}.pt", f"1-hidden AE (k={k}) encoder weights", f"{OUT}/weights_ae1_{k}.png")
plot_weights(f"{OUT}/models/dae20_{k}.pt", f"Denoising AE 20% (k={k}) encoder weights", f"{OUT}/weights_dae20_{k}.png")
plot_weights(f"{OUT}/models/dae40_{k}.pt", f"Denoising AE 40% (k={k}) encoder weights", f"{OUT}/weights_dae40_{k}.png")
print("saved weight plots")
