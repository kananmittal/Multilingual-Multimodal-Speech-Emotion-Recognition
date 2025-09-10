import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.manifold import TSNE
from utils import load_embeddings

def plot_tsne(X, y, results_dir, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    X_2d = tsne.fit_transform(X)

    unique_labels = np.unique(y)
    num_classes = len(unique_labels)

    # Use only the first K tab10 colors for K classes and a discrete normalization
    tab10 = plt.get_cmap("tab10")
    cmap_discrete = colors.ListedColormap(tab10.colors[:num_classes])
    bounds = np.arange(-0.5, num_classes, 1.0)
    norm = colors.BoundaryNorm(bounds, cmap_discrete.N)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_discrete, norm=norm, alpha=0.7
    )
    cbar = plt.colorbar(scatter, ticks=np.arange(num_classes))
    cbar.set_label("Emotion Label")
    plt.title("t-SNE visualization of embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "tsne_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] t-SNE plot saved to {plot_path}")

def main(npz_path, results_dir):
    X, y, _ = load_embeddings(npz_path)
    plot_tsne(X, y, results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/embeddings_crema_test_fused.npz")
    parser.add_argument("--results", type=str, default="../results/plots")
    args = parser.parse_args()

    main(args.data, args.results)
