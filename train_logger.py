import csv
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(values, window, epochs=None):
    if values is None or len(values) == 0:
        return np.array([]), np.array([])
    x = np.asarray(values, dtype=float)
    w = min(max(1, int(window)), len(x))
    kernel = np.ones(w) / w
    smoothed = np.convolve(x, kernel, mode="valid")
    if epochs is None:
        epochs = np.arange(1, len(x) + 1)
    else:
        epochs = np.asarray(epochs)
    return epochs[w - 1 :], smoothed


class TrainLogger:
    """Local stand-in for W&B: CSV + stdout + a live loss-curve PNG."""

    def __init__(self, csv_path, plot_path, plot_interval=10, resume=False, smooth_window=50):
        self.csv_path = csv_path
        self.plot_path = plot_path
        self.plot_interval = max(1, int(plot_interval))
        self.smooth_window = max(1, int(smooth_window))
        self.t0 = time.time()
        self.epoch_t0 = time.time()

        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        write_header = (not resume) or (not os.path.exists(csv_path))
        if write_header:
            with open(csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "train_loss_ma",
                        "val_loss_ma",
                        "best_val_loss",
                        "lr",
                        "epoch_seconds",
                        "elapsed_seconds",
                    ]
                )

    def log(
        self,
        epoch,
        num_epochs,
        train_loss,
        val_loss,
        best_val_loss,
        lr,
        train_losses,
        val_losses,
    ):
        now = time.time()
        epoch_s = now - self.epoch_t0
        elapsed = now - self.t0
        self.epoch_t0 = now

        _, train_ma = moving_average(train_losses, self.smooth_window)
        _, val_ma = moving_average(val_losses, self.smooth_window)
        train_ma_now = train_ma[-1] if len(train_ma) else train_loss
        val_ma_now = val_ma[-1] if len(val_ma) else val_loss

        with open(self.csv_path, "a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{val_loss:.8f}",
                    f"{train_ma_now:.8f}",
                    f"{val_ma_now:.8f}",
                    f"{best_val_loss:.8f}",
                    f"{lr:.6e}",
                    f"{epoch_s:.2f}",
                    f"{elapsed:.1f}",
                ]
            )

        remaining_h = ((num_epochs - epoch) * epoch_s) / 3600.0
        print(
            f"Epoch {epoch:05d}/{num_epochs} | "
            f"train {train_loss:.6f} (ma {train_ma_now:.6f}) | "
            f"val {val_loss:.6f} (ma {val_ma_now:.6f}) | "
            f"best {best_val_loss:.6f} | {epoch_s:.1f}s/epoch | "
            f"~{remaining_h:.1f}h left",
            flush=True,
        )

        if epoch == 1 or epoch % self.plot_interval == 0 or epoch == num_epochs:
            self._plot(train_losses, val_losses, epoch=epoch)

    def _plot(self, train_losses, val_losses, epoch=None):
        n = len(train_losses)
        last_epoch = int(epoch) if epoch is not None else n
        epochs = np.arange(last_epoch - n + 1, last_epoch + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, color="blue", alpha=0.25, linewidth=1, label="Train (raw)")
        plt.plot(epochs, val_losses, color="orange", alpha=0.25, linewidth=1, label="Val (raw)")

        train_x, train_ma = moving_average(train_losses, self.smooth_window, epochs)
        val_x, val_ma = moving_average(val_losses, self.smooth_window, epochs)
        if len(train_ma):
            plt.plot(train_x, train_ma, color="blue", linewidth=2, label=f"Train (MA-{self.smooth_window})")
        if len(val_ma):
            plt.plot(val_x, val_ma, color="darkorange", linewidth=2, label=f"Val (MA-{self.smooth_window})")

        y_lo, y_hi, use_log = _monitor_ylim(train_losses, val_losses)
        plt.xlim(float(epochs[0]), float(epochs[-1]))
        if use_log:
            plt.yscale("log")
        plt.ylim(y_lo, y_hi)
        plt.xlabel("Epochs")
        plt.ylabel("sigma-weighted MSE")
        plt.title("Training monitor")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()


def _monitor_ylim(train_losses, val_losses, recent=150):
    """Fit the axis to recent epochs. Use a log axis while the range is still wide."""
    train = np.asarray(train_losses, dtype=float)[-recent:]
    val = np.asarray(val_losses, dtype=float)[-recent:]
    values = np.concatenate([train, val])
    values = values[np.isfinite(values) & (values > 0)]
    lo, hi = np.quantile(values, [0.05, 0.95])
    lo, hi = float(lo), float(hi)
    if hi / max(lo, 1e-8) > 8:
        return lo / 1.4, hi * 1.4, True
    span = max(hi - lo, 0.05)
    pad = 0.2 * span
    return max(lo - pad, 0.0), hi + pad, False
