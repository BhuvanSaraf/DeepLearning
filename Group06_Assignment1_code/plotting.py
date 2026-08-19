"""
plotting.py
-----------
All plots required by the assignment:

Classification:
  - plot_error_curve
  - plot_decision_region (pairwise and combined, superimposed with
    training data only)

Regression:
  - plot_error_curve (reused)
  - plot_model_vs_target_1d   (Dataset 1: univariate)
  - plot_model_vs_target_2d   (Dataset 2: bivariate, 3D surface-ish scatter)
  - plot_scatter_target_vs_pred
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Common
# ----------------------------------------------------------------------
def plot_error_curve(error_history, title="Average Error vs Epochs",
                      ax=None, save_path=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(error_history) + 1), error_history, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Error")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
    return ax


# ----------------------------------------------------------------------
# Classification: decision regions
# ----------------------------------------------------------------------
def plot_decision_region(X, y, predict_fn, title="Decision Region",
                          class_labels=None, ax=None, save_path=None,
                          resolution=300, margin=1.0):
    """
    X, y : training data ONLY (per assignment instructions) used both
           to size the grid and to superimpose as scatter points.
    predict_fn : callable(X_grid) -> predicted class labels for grid points.
                 This lets it work for a single pairwise classifier OR
                 a combined one-against-one voting classifier.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    # Build an explicit grid of cells (not just sample points). Each grid
    # point is classified individually using OUR OWN trained perceptron
    # (predict_fn) -- no plotting/ML library decides the boundary.
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                          np.linspace(y_min, y_max, resolution))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = predict_fn(grid).reshape(xx.shape)  # class label for every cell, computed by us

    # pcolormesh paints each grid cell exactly the color of its computed
    # label -- no smoothing/interpolation of the boundary, unlike contourf.
    # This keeps matplotlib doing pure rendering, not boundary computation.
    classes_sorted = sorted(np.unique(y))
    cmap = plt.get_cmap("tab10")
    class_to_int = {c: i for i, c in enumerate(classes_sorted)}
    Z_int = np.vectorize(class_to_int.get)(Z)
    ax.pcolormesh(xx, yy, Z_int, alpha=0.25, cmap="tab10",
                  vmin=0, vmax=max(len(classes_sorted) - 1, 1), shading="auto")

    classes = sorted(np.unique(y))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        mask = y == cls
        label = class_labels[cls] if class_labels else f"Class {cls}"
        ax.scatter(X[mask, 0], X[mask, 1], s=15, color=cmap(i),
                   edgecolor="k", linewidth=0.3, label=label)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
    return ax


# ----------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------
def plot_model_vs_target_1d(x, y_target, y_model, title="Model vs Target",
                             ax=None, save_path=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(x.ravel())
    ax.scatter(x, y_target, s=12, color="tab:blue", label="Target", alpha=0.7)
    ax.scatter(x, y_model, s=12, color="tab:red", marker="x", label="Model output")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
    return ax


def plot_model_vs_target_2d(x1, x2, y_target, y_model,
                             title="Model vs Target (Bivariate)",
                             save_path=None):
    """3D scatter: x1, x2 axes, y-axis is target/model output."""
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, y_target, s=12, color="tab:blue", label="Target", alpha=0.7)
    ax.scatter(x1, x2, y_model, s=12, color="tab:red", marker="x", label="Model output")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_scatter_target_vs_pred(y_target, y_pred, title="Target vs Model Output",
                                 ax=None, save_path=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_target, y_pred, s=14, alpha=0.7, color="tab:purple")
    lo = min(np.min(y_target), np.min(y_pred))
    hi = max(np.max(y_target), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Ideal (y=x)")
    ax.set_xlabel("Target output")
    ax.set_ylabel("Model output")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
    return ax


if __name__ == "__main__":
    # smoke test with synthetic data
    err = [1.0 / (i + 1) for i in range(50)]
    fig = plot_error_curve(err, save_path="/tmp/test_error.png")
    plt.close(fig)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    fig = plot_decision_region(X, y, lambda g: (g[:, 0] + g[:, 1] > 0).astype(int),
                                save_path="/tmp/test_region.png")
    plt.close(fig)

    x = np.linspace(0, 10, 40)
    yt = 2 * x + 1 + rng.normal(scale=0.5, size=40)
    ym = 2 * x + 0.9
    fig = plot_model_vs_target_1d(x, yt, ym, save_path="/tmp/test_1d.png")
    plt.close(fig)

    fig = plot_scatter_target_vs_pred(yt, ym, save_path="/tmp/test_scatter.png")
    plt.close(fig)
    print("all plotting smoke tests passed")
