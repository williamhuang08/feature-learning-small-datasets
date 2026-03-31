import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def visualize_agop(agop, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.imshow(agop, cmap="viridis")
    plt.colorbar()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", default="nn/results", type=str, help="Directory containing dataset_* result folders",)
    args = parser.parse_args()
    datadir = os.path.abspath(args.dir)

    for entry in sorted(os.listdir(datadir)):
        base = os.path.join(datadir, entry)
        if not os.path.isdir(base):
            continue
        agop_path = os.path.join(base, "matrices", "agop.npy")
        if not os.path.isfile(agop_path):
            continue
        agop = np.load(agop_path)
        out_path = os.path.join(base, "visualizations", "agop.png")
        visualize_agop(agop, out_path)
        print(f"saved {out_path}")
