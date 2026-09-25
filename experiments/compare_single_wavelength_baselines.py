"""
Compare single-wavelength baseline models.

Expected folders:
    outputs/baseline_440_unweighted/
    outputs/baseline_555_unweighted/
    outputs/baseline_740_unweighted/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt


def main():

    model_dirs = {
        "440 nm": Path("outputs/baseline_440_unweighted"),
        "555 nm": Path("outputs/baseline_555_unweighted"),
        "740 nm": Path("outputs/baseline_740_unweighted"),
    }

    all_metrics = []

    for model_name, model_dir in model_dirs.items():

        metrics_path = model_dir / "metrics.csv"

        if not metrics_path.exists():
            print(f"Missing metrics file: {metrics_path}")
            continue

        metrics = pd.read_csv(metrics_path)
        metrics["input"] = model_name

        all_metrics.append(metrics)

    if len(all_metrics) == 0:
        raise FileNotFoundError(
            "No metrics.csv files found. Please evaluate the models first."
        )

    comparison = pd.concat(
        all_metrics,
        ignore_index=True
    )

    output_dir = Path("outputs/comparison_single_wavelength")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        output_dir / "single_wavelength_metrics.csv",
        index=False
    )

    print()
    print("=" * 60)
    print("Single-Wavelength Baseline Comparison")
    print("=" * 60)
    print(comparison)

    # ========================================================
    # Plot test RMSE
    # ========================================================

    test_metrics = comparison[
        comparison["split"] == "test"
    ].copy()

    plt.figure(figsize=(7, 5))

    plt.bar(
        test_metrics["input"],
        test_metrics["RMSE"]
    )

    plt.xlabel("Input wavelength")
    plt.ylabel("Test RMSE")
    plt.title("Single-Wavelength Baseline Test RMSE")

    plt.tight_layout()

    plt.savefig(
        output_dir / "test_rmse_comparison.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # Plot test R2
    # ========================================================

    plt.figure(figsize=(7, 5))

    plt.bar(
        test_metrics["input"],
        test_metrics["R2"]
    )

    plt.xlabel("Input wavelength")
    plt.ylabel("Test R²")
    plt.title("Single-Wavelength Baseline Test R²")

    plt.tight_layout()

    plt.savefig(
        output_dir / "test_r2_comparison.png",
        dpi=300
    )

    plt.close()

    print()
    print("Saved comparison results to:")
    print(output_dir)


if __name__ == "__main__":

    main()