"""Render measured confusion matrices and an explanatory pipeline figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = Path("docs/images")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, name in zip(
        axes.flat, ("svm_hamming", "svm_hann", "random_forest_hamming", "random_forest_hann")
    ):
        report = json.loads(Path(f"reports/{name}/metrics.json").read_text(encoding="utf-8"))
        matrix = np.array(report["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=4)
        for row in range(10):
            for col in range(10):
                ax.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > 2 else "black",
                    fontsize=9,
                )
        ax.set(
            xticks=range(10),
            yticks=range(10),
            xlabel="Predicted digit",
            ylabel="True digit",
            title=f"{name}: {report['correct']}/{report['total']}",
        )
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    ax.set(xlim=(0, 12), ylim=(0, 3))
    ax.axis("off")
    labels = [
        "WAV input\n16 kHz mono",
        "Pre-emphasis\nFrames + window",
        "13 MFCC\n78-d pooling",
        "SVM / RF\nDigit 0-9",
    ]
    for index, label in enumerate(labels):
        x = index * 3 + 1.5
        ax.text(
            x,
            1.5,
            label,
            ha="center",
            va="center",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.8", fc="#e8f1ff", ec="#3164a3"),
        )
        if index < 3:
            ax.annotate(
                "",
                xy=(x + 1.8, 1.5),
                xytext=(x + 1.15, 1.5),
                arrowprops=dict(arrowstyle="->", color="#3164a3", lw=2),
            )
    ax.set_title("DigitASR: training and inference share the same feature configuration")
    fig.savefig(output / "pipeline.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
