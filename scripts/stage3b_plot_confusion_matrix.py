#!/usr/bin/env python3
"""
Stage 3b: render the honest (no sttl/dttl) UNSW-NB15 confusion matrix as a
row-normalized heatmap, same convention as Stage 3's plot.

Usage:
    python scripts/stage3b_plot_confusion_matrix.py [data/processed/stage3b]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/stage3b")
    cm = pd.read_csv(out_root / "confusion_matrix_honest_without_ttl.csv", index_col=0)

    row_sums = cm.sum(axis=1).replace(0, np.nan)
    cm_norm = cm.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage 3b (UNSW-NB15, honest model) confusion matrix\n(row-normalized)")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm.values[i, j]
            if np.isnan(val) or val < 0.01:
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val > 0.5 else "black", fontsize=7)

    fig.colorbar(im, ax=ax, label="fraction of true-class row")
    fig.tight_layout()
    out_path = out_root / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
