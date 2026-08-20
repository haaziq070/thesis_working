#!/usr/bin/env python3
"""
Stage 4: plot the DQN training curves (validation F1/precision/recall, reward,
epsilon, training loss) so the honest trajectory is visible, not just the
final numbers.

Usage:
    python scripts/stage4_plot_training_curves.py [data/processed/stage4]
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/stage4")
    hist = pd.read_csv(out_root / "training_history.csv")
    eval_hist = pd.read_csv(out_root / "eval_history.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(eval_hist["episode"], eval_hist["val_f1"], marker="o", markersize=3, label="val F1")
    ax.plot(eval_hist["episode"], eval_hist["val_precision"], marker="o", markersize=3, label="val precision")
    ax.plot(eval_hist["episode"], eval_hist["val_recall"], marker="o", markersize=3, label="val recall")
    ax.set_xlabel("episode")
    ax.set_ylabel("score")
    ax.set_title("Validation precision / recall / F1 over training")
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axes[0, 1]
    ax.plot(eval_hist["episode"], eval_hist["val_reward"])
    ax.set_xlabel("episode")
    ax.set_ylabel("total val reward (both val campaigns)")
    ax.set_title("Validation reward over training")

    ax = axes[1, 0]
    ax.plot(hist["episode"], hist["epsilon"])
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax.set_title("Epsilon-greedy exploration decay")

    ax = axes[1, 1]
    loss = hist["avg_loss"].dropna()
    ax.plot(hist.loc[loss.index, "episode"], loss, linewidth=0.5)
    # rolling mean to see the trend under the per-episode noise
    roll = hist["avg_loss"].rolling(window=50, min_periods=10).mean()
    ax.plot(hist["episode"], roll, linewidth=2, color="darkred", label="50-episode rolling mean")
    ax.set_xlabel("episode")
    ax.set_ylabel("TD loss (per training step, avg per episode)")
    ax.set_title("Training loss")
    ax.legend()

    fig.tight_layout()
    out_path = out_root / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
