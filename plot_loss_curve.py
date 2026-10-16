"""Draw the full training loss from the metrics CSVs.

This script does not train and does not touch the live monitor plot.
Run it whenever you want a fresh loss_curve.png:

    python plot_loss_curve.py
"""

import csv
import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_logger import moving_average

CURRENT_CSV = "training_metrics.csv"
HISTORY_CSV = "training_metrics_pre_weighted.csv"
OUT_PATH = "loss_curve.png"
SMOOTH_WINDOW = 50
SIGMA_MIN = 0.001
SIGMA_MAX = 0.75
NUM_STEPS = 3000


def load_series(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_epoch = OrderedDict()
    for row in rows:
        by_epoch[int(row["epoch"])] = (
            float(row["train_loss"]),
            float(row["val_loss"]),
        )
    epochs = np.array(list(by_epoch), dtype=int)
    train = np.array([by_epoch[e][0] for e in epochs])
    val = np.array([by_epoch[e][1] for e in epochs])
    return epochs, train, val


def weighted_scale():
    """Unweighted MSE is about mean(sigma^2) times the sigma-weighted MSE."""
    sigma = np.linspace(SIGMA_MIN, SIGMA_MAX, NUM_STEPS)
    return float(np.mean(sigma ** 2))


def main():
    new_e, new_train, new_val = load_series(CURRENT_CSV)
    if os.path.exists(HISTORY_CSV):
        old_e, old_train, old_val = load_series(HISTORY_CSV)
        split = int(new_e[0])
        keep = old_e < split
        old_e, old_train, old_val = old_e[keep], old_train[keep], old_val[keep]
        scale = weighted_scale()
        epochs = np.concatenate([old_e, new_e])
        train = np.concatenate([old_train / scale, new_train])
        val = np.concatenate([old_val / scale, new_val])
    else:
        epochs, train, val = new_e, new_train, new_val

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, color="blue", alpha=0.2, linewidth=1)
    plt.plot(epochs, val, color="orange", alpha=0.2, linewidth=1)
    train_x, train_ma = moving_average(train, SMOOTH_WINDOW, epochs)
    val_x, val_ma = moving_average(val, SMOOTH_WINDOW, epochs)
    plt.plot(train_x, train_ma, color="blue", linewidth=2, label=f"Train (MA-{SMOOTH_WINDOW})")
    plt.plot(val_x, val_ma, color="darkorange", linewidth=2, label=f"Val (MA-{SMOOTH_WINDOW})")

    last = int(epochs[-1])
    y_lo, y_hi = np.quantile(np.concatenate([train, val]), [0.01, 0.99])
    pad = 0.15 * (y_hi - y_lo)
    plt.xlim(0, last)
    plt.ylim(y_lo - pad, y_hi + pad)
    plt.xlabel("Epochs")
    plt.ylabel("sigma-weighted MSE")
    plt.title("Training loss, all epochs")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PATH)
    plt.close()
    print(
        f"Wrote {OUT_PATH} for epochs 0–{last}. "
        f"Earlier losses divided by mean(sigma^2)={scale:.4f} so they share this scale."
    )


if __name__ == "__main__":
    main()
