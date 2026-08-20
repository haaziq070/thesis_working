#!/usr/bin/env python3
"""
Stage 3: render the confusion matrix saved by stage3_train_identifier.py as
a readable heatmap. Row-normalized (each row sums to 1.0) rather than raw
counts, because class sizes here range from 4 test examples (Heartbleed) to
104,773 (DoS_Hulk) -- a raw-count heatmap would just show one bright row and
nothing else. Row-normalized means each cell reads as "of all true examples
of this class, what fraction got predicted as that class" -- i.e. per-class
recall broken down by exactly where the errors go, which is what you want
for defending individual class weaknesses (e.g. Web_XSS) in the viva.

Usage:
    python scripts/stage3_plot_confusion_matrix.py [data/processed/stage3]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/stage3")
    cm = pd.read_csv(out_root / "confusion_matrix_full.csv", index_col=0)

    row_sums = cm.sum(axis=1).replace(0, np.nan)
    cm_norm = cm.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage 3 confusion matrix (row-normalized, i.e. per-class recall breakdown)")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm.values[i, j]
            if np.isnan(val):
                continue
            if val >= 0.01:  # skip clutter from near-zero cells
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if val > 0.5 else "black", fontsize=7)

    fig.colorbar(im, ax=ax, label="fraction of true-class row")
    fig.tight_layout()
    out_path = out_root / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # also flag, directly from the plotted data, any class whose recall (diagonal)
    # is suspiciously perfect alongside large support -- same discipline as the
    # training script's own near-perfect check, applied here as a second look.
    diag = pd.Series(np.diag(cm_norm.values), index=cm.index)
    support = cm.sum(axis=1)
    for cls in cm.index:
        if diag[cls] >= 0.99 and support[cls] >= 100:
            print(f"NOTE: {cls} has row-normalized recall {diag[cls]:.3f} on {int(support[cls])} "
                  f"test examples -- already investigated in the training run's own near-perfect "
                  f"check; see that report before treating this as a new finding.")


if __name__ == "__main__":
    main()
