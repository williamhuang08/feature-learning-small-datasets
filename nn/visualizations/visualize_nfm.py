import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def visualize_nfm(nfm, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.imshow(nfm, cmap="viridis")
    plt.colorbar()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", default="nn/results", type=str, help="Directory containing result folders",)
    args = parser.parse_args()
    datadir = os.path.abspath(args.dir)

    for entry in sorted(os.listdir(datadir)):
        base = os.path.join(datadir, entry)
        if not os.path.isdir(base):
            continue
        nfm_path = os.path.join(base, "matrices", "nfm.npy")
        if not os.path.isfile(nfm_path):
            continue
        nfm = np.load(nfm_path)
        out_path = os.path.join(base, "visualizations", "nfm.png")
        visualize_nfm(nfm, out_path)
        print(f"saved {out_path}")
