import warnings
warnings.filterwarnings("ignore")

import math
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

NN_RESULT_DIR = Path("../nn_jf/results")
OUTPUT_ROOT = Path("./visualizations")

CLASSIFIER_DIRS: dict[str, Path] = {
    "LAPLACE-RFM": Path("../rfm/results/laplace"),
    "GAUSSIAN-RFM": Path("../rfm/results/gaussian"),
    "NTK-RFM": Path("../rfm/results/ntk"),
}

MAX_LAYERS: int | None = 1
MAX_GRID_COLS = 3


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    """Save one matrix as a heatmap without feature ticks or axis labels."""
    matrix_np = matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix_np, aspect="auto")

    ax.set_xticks([])
    ax.set_yticks([])
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
    """Save layerwise matrices as a grid of heatmaps without feature ticks or axis labels."""
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
        matrix_np = matrix.detach().cpu().numpy()

        im = ax.imshow(matrix_np, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(nplots, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_nn(dataset_name: str, out_dir: Path) -> None:
    """Create NN AGOP and NFM plots for one dataset."""
    agop_path = NN_RESULT_DIR / dataset_name / "avg_layerwise_agop.pt"
    if agop_path.is_file():
        plot_layer_grid(
            sorted_layers(torch.load(agop_path, map_location="cpu")),
            f"{dataset_name}: NN AGOP",
            out_dir / "nn_avg_layerwise_agop_heatmaps.png",
        )

    nfm_path = NN_RESULT_DIR / dataset_name / "avg_layerwise_nfm.pt"
    if nfm_path.is_file():
        plot_layer_grid(
            sorted_layers(torch.load(nfm_path, map_location="cpu")),
            f"{dataset_name}: NN NFM",
            out_dir / "nn_avg_layerwise_nfm_heatmaps.png",
        )


def visualize_rfm(dataset_name: str, out_dir: Path) -> None:
    """Create one final-M heatmap for each RFM classifier for one dataset."""
    for classifier_name, classifier_dir in CLASSIFIER_DIRS.items():
        matrix_path = classifier_dir / dataset_name / "avg_final_M.pt"
        if not matrix_path.is_file():
            continue

        payload = torch.load(matrix_path, map_location="cpu")
        matrix = payload["M"]

        filename = f"{classifier_name.lower()}_avg_final_M_heatmap.png".replace("-", "_")
        plot_heatmap(
            matrix,
            # f"{dataset_name}: {classifier_name} M",
        "",
            out_dir / filename,
        )


def all_dataset_names() -> list[str]:
    """Collect dataset names appearing in either NN or RFM result directories."""
    names: set[str] = set()

    if NN_RESULT_DIR.is_dir():
        names |= {path.name for path in NN_RESULT_DIR.iterdir() if path.is_dir()}

    for classifier_dir in CLASSIFIER_DIRS.values():
        if classifier_dir.is_dir():
            names |= {path.name for path in classifier_dir.iterdir() if path.is_dir()}

    return sorted(names)


def visualize_dataset(dataset_name: str) -> None:
    """Create all NN and RFM plots for one dataset."""
    out_dir = ensure_dir(OUTPUT_ROOT / dataset_name)
    visualize_nn(dataset_name, out_dir)
    visualize_rfm(dataset_name, out_dir)


def main() -> None:
    """Create visualization outputs under OUTPUT_ROOT."""
    ensure_dir(OUTPUT_ROOT)

    for dataset_name in tqdm(all_dataset_names()):
        visualize_dataset(dataset_name)


if __name__ == "__main__":
    main()