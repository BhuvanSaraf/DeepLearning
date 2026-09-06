import numpy as np
import matplotlib.pyplot as plt


def plot_error_curve(error_history, title="Average Error vs Epochs", ax=None, save_path=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = None

    ax.plot(range(1, len(error_history) + 1), error_history, color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Error')
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
    return ax


def plot_decision_region(X, y, predict_fn, title="Decision Region", resolution=300, margin=1.0, save_path=None):

    fig, ax = plt.subplots(figsize=(6, 5))

    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                          np.linspace(y_min, y_max, resolution))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = predict_fn(grid).reshape(xx.shape)   

    classes = sorted(np.unique(y))
    class_to_int = {c: i for i, c in enumerate(classes)}
    Z_int = np.vectorize(class_to_int.get)(Z)

    ax.pcolormesh(xx, yy, Z_int, alpha=0.25, cmap='tab10',
                  vmin=0, vmax=max(len(classes) - 1, 1), shading='auto')

    cmap = plt.get_cmap('tab10')
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], s=15, color=cmap(i),
                   edgecolor='k', linewidth=0.3, label=f'Class {cls}')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_model_vs_target_1d(x, y_target, y_model, title="Model vs Target", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y_target, s=12, color='tab:blue', label='Target', alpha=0.7)
    ax.scatter(x, y_model, s=12, color='tab:red', marker='x', label='Model output')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_model_vs_target_2d(x1, x2, y_target, y_model, title="Model vs Target (Bivariate)", save_path=None):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y_target, s=12, color='tab:blue', label='Target', alpha=0.7)
    ax.scatter(x1, x2, y_model, s=12, color='tab:red', marker='x', label='Model output')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_scatter_target_vs_pred(y_target, y_pred, title="Target vs Model Output", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_target, y_pred, s=14, alpha=0.7, color='tab:purple')

    lo = min(np.min(y_target), np.min(y_pred))
    hi = max(np.max(y_target), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='Ideal (y=x)')

    ax.set_xlabel('Target output')
    ax.set_ylabel('Model output')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_node_outputs(model, X, input_dim, title_prefix, out_dir, split_name, y_labels=None):
    # Plots the output (activation) of every hidden node and every
    # output node in the FCNN, against the input variable(s), for the
    # given data split (train/val/test). Required by the assignment:
    # x (and y, for 2D input) are the input variables, z is the node's
    # output.
    #
    # y_labels: optional array of class labels (classification only).
    # When given, points are colored by class so it's easy to see how
    # each node separates the classes. Left as None for regression.
    import os
    os.makedirs(out_dir, exist_ok=True)

    model.forward(X)               # populate model.A for this X
    layer_activations = model.A[1:]   # skip the raw input layer

    if y_labels is not None:
        classes = sorted(np.unique(y_labels))
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i, cls in enumerate(classes)]
        point_colors = np.array([colors[classes.index(lab)] for lab in y_labels])
    else:
        point_colors = 'tab:blue'

    n_layers = len(layer_activations)
    for layer_idx, A_layer in enumerate(layer_activations):
        is_output_layer = (layer_idx == n_layers - 1)
        layer_label = 'output' if is_output_layer else f'hidden{layer_idx + 1}'
        n_nodes = A_layer.shape[1]

        for node_idx in range(n_nodes):
            z_vals = A_layer[:, node_idx]

            if input_dim == 1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(X[:, 0], z_vals, s=8, alpha=0.7, c=point_colors)
                ax.set_xlabel('x')
                ax.set_ylabel('node output')
                ax.set_title(f'{title_prefix} {layer_label} node {node_idx} ({split_name})')
                if y_labels is not None:
                    handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=colors[i], markersize=7, label=f'Class {cls}')
                               for i, cls in enumerate(classes)]
                    ax.legend(handles=handles, fontsize=7)
                fig.tight_layout()
                fig.savefig(f'{out_dir}/{layer_label}_node{node_idx}_{split_name}.png', dpi=120)
                plt.close(fig)
            else:
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X[:, 0], X[:, 1], z_vals, s=6, alpha=0.7, c=point_colors)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_zlabel('node output')
                ax.set_title(f'{title_prefix} {layer_label} node {node_idx} ({split_name})')
                if y_labels is not None:
                    handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=colors[i], markersize=7, label=f'Class {cls}')
                               for i, cls in enumerate(classes)]
                    ax.legend(handles=handles, fontsize=7)
                fig.tight_layout()
                fig.savefig(f'{out_dir}/{layer_label}_node{node_idx}_{split_name}.png', dpi=120)
                plt.close(fig)


if __name__ == '__main__':
    err = [1.0 / (i + 1) for i in range(50)]
    fig = plot_error_curve(err, save_path='./tmp/test_error.png')
    plt.close(fig)

    np.random.seed(0)
    X = np.random.randn(60, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    fig = plot_decision_region(X, y, lambda g: (g[:, 0] + g[:, 1] > 0).astype(int),
                                save_path='./tmp/test_region.png')
    plt.close(fig)

    x = np.linspace(0, 10, 40)
    yt = 2 * x + 1 + np.random.normal(scale=0.5, size=40)
    ym = 2 * x + 0.9
    fig = plot_model_vs_target_1d(x, yt, ym, save_path='./tmp/test_1d.png')
    plt.close(fig)

    fig = plot_scatter_target_vs_pred(yt, ym, save_path='./tmp/test_scatter.png')
    plt.close(fig)
    print('all plotting smoke tests passed')
 