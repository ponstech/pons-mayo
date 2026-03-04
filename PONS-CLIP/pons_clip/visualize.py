"""
Embedding visualization: t-SNE and UMAP by class for the TEST SET only.

Uses embeddings (and labels) produced by evaluate.py on the test split.
Optional: plot keyword positions in the same 2D space if keyword_embeddings.npy exists.
Outputs clustering metrics (silhouette, etc.) for comparing B-mode vs multi-feature.
"""

import argparse
import json
import os

from .warnings_config import setup_warnings
setup_warnings()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .constants import BREAST_US_CLASSES, KEYWORDS


def load_embeddings_and_labels(embeddings_dir: str):
    embeds = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    labels = np.load(os.path.join(embeddings_dir, "labels.npy"))
    return embeds, labels


def load_keyword_embeddings(embeddings_dir: str):
    path = os.path.join(embeddings_dir, "keyword_embeddings.npy")
    if os.path.isfile(path):
        return np.load(path)
    return None


def run_tsne(embeds: np.ndarray, perplexity: float = 30, max_iter: int = 1000) -> np.ndarray:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42, init="pca")
    return tsne.fit_transform(embeds)


def run_umap(embeds: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeds)


def compute_clustering_metrics(embeds: np.ndarray, labels: np.ndarray, class_names: list) -> dict:
    """Compute metrics in original embedding space (no retraining)."""
    from sklearn.metrics import silhouette_score
    n = len(labels)
    n_classes = len(class_names)
    sil = silhouette_score(embeds, labels, metric="cosine") if n > n_classes else None
    # Mean within-class distance vs mean to other class
    within, between = [], []
    for i in range(n_classes):
        mask_i = labels == i
        if mask_i.sum() < 2:
            continue
        ci = embeds[mask_i].mean(axis=0, keepdims=True)
        within.append(np.mean(1 - (embeds[mask_i] @ ci.T).flatten()))
        for j in range(n_classes):
            if j == i:
                continue
            mask_j = labels == j
            if mask_j.sum() == 0:
                continue
            cj = embeds[mask_j].mean(axis=0, keepdims=True)
            between.append(1 - (ci @ cj.T).item())
    return {
        "n_samples": int(n),
        "n_classes": n_classes,
        "silhouette_score": float(sil) if sil is not None else None,
        "mean_within_class_cosine_distance": float(np.mean(within)) if within else None,
        "mean_between_class_cosine_distance": float(np.mean(between)) if between else None,
        "note": "Higher silhouette = better separation. Lower within-class distance = tighter clusters.",
    }


def plot_embeddings(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    title: str,
    path: str,
    keyword_coords: np.ndarray = None,
    keyword_names: list = None,
    n_test: int = None,
) -> None:
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", n_colors=len(class_names))
    for i, cls_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[i]],
            label=cls_name,
            alpha=0.6,
            s=20,
            zorder=1,
        )
    if keyword_coords is not None and keyword_names is not None and len(keyword_names) == len(keyword_coords):
        plt.scatter(
            keyword_coords[:, 0],
            keyword_coords[:, 1],
            marker="*",
            s=400,
            c="black",
            edgecolors="yellow",
            linewidths=1.5,
            zorder=2,
            label="Keywords",
        )
        for k, name in enumerate(keyword_names):
            plt.annotate(
                name,
                (keyword_coords[k, 0], keyword_coords[k, 1]),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
            )
    title_str = title
    if n_test is not None:
        title_str += f" — Test set (n={n_test})"
    plt.title(title_str)
    plt.legend(loc="best", fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description="Visualize test-set embeddings; optionally show keywords and clustering metrics.")
    p.add_argument("--embeddings_dir", type=str, required=True, help="Dir with embeddings.npy, labels.npy (from evaluate.py on test set)")
    p.add_argument("--output_dir", type=str, default="./plots")
    p.add_argument("--method", type=str, default="both", choices=["tsne", "umap", "both"])
    p.add_argument("--prefix", type=str, default="", help="Filename prefix (e.g. bmode or multifeature)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    embeds, labels = load_embeddings_and_labels(args.embeddings_dir)
    keyword_embeds = load_keyword_embeddings(args.embeddings_dir)
    n_test = len(embeds)
    class_names = BREAST_US_CLASSES
    prefix = f"{args.prefix}_" if args.prefix else ""

    # Clustering metrics (in original 512-d space) — no retraining
    metrics = compute_clustering_metrics(embeds, labels, class_names)
    metrics_path = os.path.join(args.embeddings_dir, "clustering_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Clustering metrics (test set, n={n_test}):")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {metrics_path}")

    # Combine image + keyword embeddings for same 2D space
    if keyword_embeds is not None and len(KEYWORDS) == keyword_embeds.shape[0]:
        combined = np.vstack([embeds, keyword_embeds])
    else:
        combined = embeds
        keyword_embeds = None

    if args.method in ("tsne", "both"):
        coords_all = run_tsne(combined)
        coords_tsne = coords_all[:n_test]
        kw_coords_tsne = coords_all[n_test:] if keyword_embeds is not None else None
        plot_embeddings(
            coords_tsne,
            labels,
            class_names,
            title="Embedding visualization (t-SNE)",
            path=os.path.join(args.output_dir, f"{prefix}tsne_by_class.png"),
            keyword_coords=kw_coords_tsne,
            keyword_names=KEYWORDS if kw_coords_tsne is not None else None,
            n_test=n_test,
        )
        np.save(os.path.join(args.output_dir, f"{prefix}tsne_coords.npy"), coords_tsne)

    if args.method in ("umap", "both"):
        coords_all = run_umap(combined)
        coords_umap = coords_all[:n_test]
        kw_coords_umap = coords_all[n_test:] if keyword_embeds is not None else None
        plot_embeddings(
            coords_umap,
            labels,
            class_names,
            title="Embedding visualization (UMAP)",
            path=os.path.join(args.output_dir, f"{prefix}umap_by_class.png"),
            keyword_coords=kw_coords_umap,
            keyword_names=KEYWORDS if kw_coords_umap is not None else None,
            n_test=n_test,
        )
        np.save(os.path.join(args.output_dir, f"{prefix}umap_coords.npy"), coords_umap)

    print("Done. Visualizations use TEST SET embeddings only.")


if __name__ == "__main__":
    main()
