import warnings
warnings.filterwarnings("ignore")

import math
import torch
import matplotlib.pyplot as plt

from pathlib import Path

NN_RESULT_DIR = Path("../nn_jf/results")
RFM_RESULT_DIR = Path("../rfm/results")
OUTPUT_ROOT = Path("./visualizations")

MAX_LAYERS: int | None = 1
MAX_GRID_COLS = 3

def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def feature_labels(size: int) -> list[str]:
    """Return 1-indexed feature labels."""
    return [str(i) for i in range(1, size + 1)]

def sorted_layers(payload: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    """Extract layer tensors sorted by layer index."""
    layers = [
        (f"Layer {int(key.split('_')[-1]) + 1}", value.detach().cpu())
        for key, value in payload.items()
        if key.startswith("layer_") and isinstance(value, torch.Tensor)
    ]
    layers = sorted(layers, key=lambda x: int(x[0].split()[-1]))
    if MAX_LAYERS is not None:
        layers = layers[:MAX_LAYERS]
    return layers

def plot_heatmap(matrix: torch.Tensor, title: str, save_path: Path) -> None:
    """Save one matrix as a heatmap."""
    matrix = matrix.detach().cpu().numpy()
    nrows, ncols = matrix.shape

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(feature_labels(ncols))
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(feature_labels(nrows))
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    ax.set_title(title)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_layer_grid(
    layers: list[tuple[str, torch.Tensor]],
    title: str,
    save_path: Path,
) -> None:
    """Save layerwise matrices as a grid of heatmaps."""
    if not layers:
        return

    nplots = len(layers)
    ncols = min(MAX_GRID_COLS, nplots)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for i, (label, matrix) in enumerate(layers):
        ax = axes[i // ncols][i % ncols]
        matrix = matrix.detach().cpu().numpy()
        h, w = matrix.shape

        im = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(w))
        ax.set_xticklabels(feature_labels(w))
        ax.set_yticks(range(h))
        ax.set_yticklabels(feature_labels(h))
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(nplots, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def visualize_dataset(dataset_name: str) -> None:
    """Create all NN and RFM plots for one dataset into results/visualizations/<dataset_name>."""
    specs = [
        (
            NN_RESULT_DIR / dataset_name / "avg_layerwise_agop.pt",
            lambda out_dir, p: plot_layer_grid(
                sorted_layers(torch.load(p, map_location="cpu")),
                f"{dataset_name}: NN Average Layerwise AGOP",
                out_dir / "nn_avg_layerwise_agop_heatmaps.png",
            ),
        ),
        (
            NN_RESULT_DIR / dataset_name / "avg_layerwise_nfm.pt",
            lambda out_dir, p: plot_layer_grid(
                sorted_layers(torch.load(p, map_location="cpu")),
                f"{dataset_name}: NN Average Layerwise NFM",
                out_dir / "nn_avg_layerwise_nfm_heatmaps.png",
            ),
        ),
        (
            RFM_RESULT_DIR / dataset_name / "avg_final_M.pt",
            lambda out_dir, p: plot_heatmap(
                torch.load(p, map_location="cpu")["M"],
                f"{dataset_name}: RFM Average Final M",
                out_dir / "rfm_avg_final_M_heatmap.png",
            ),
        ),
    ]

    existing_specs = [(path, fn) for path, fn in specs if path.is_file()]
    if not existing_specs:
        return

    out_dir = ensure_dir(OUTPUT_ROOT / dataset_name)

    for path, fn in existing_specs:
        fn(out_dir, path)

def main() -> None:
    """Create shared visualization outputs under results/visualizations."""
    ensure_dir(OUTPUT_ROOT)

    dataset_names = sorted({
        path.name for path in NN_RESULT_DIR.iterdir() if path.is_dir()
    } | {
        path.name for path in RFM_RESULT_DIR.iterdir() if path.is_dir()
    })

    for dataset_name in dataset_names:
        visualize_dataset(dataset_name)

if __name__ == "__main__":
    main()