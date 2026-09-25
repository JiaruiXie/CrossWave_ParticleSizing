"""
Compare baseline models.

Models:
    440-only
    555-only
    740-only
    three-wavelength fusion
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
        "Fusion": Path("outputs/baseline_fusion_unweighted"),
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

    output_dir = Path("outputs/comparison_baselines")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        output_dir / "baseline_metrics.csv",
        index=False
    )

    print()
    print("=" * 60)
    print("Baseline Model Comparison")
    print("=" * 60)
    print(comparison)

    # ========================================================
    # Test metrics only
    # ========================================================

    test_metrics = comparison[
        comparison["split"] == "test"
    ].copy()

    # Keep desired order
    order = ["440 nm", "555 nm", "740 nm", "Fusion"]

    test_metrics["input"] = pd.Categorical(
        test_metrics["input"],
        categories=order,
        ordered=True
    )

    test_metrics = test_metrics.sort_values("input")

    # ========================================================
    # Plot test RMSE
    # ========================================================

    plt.figure(figsize=(7, 5))

    plt.bar(
        test_metrics["input"].astype(str),
        test_metrics["RMSE"]
    )

    plt.xlabel("Input")
    plt.ylabel("Test RMSE")
    plt.title("Baseline Test RMSE Comparison")

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
        test_metrics["input"].astype(str),
        test_metrics["R2"]
    )

    plt.xlabel("Input")
    plt.ylabel("Test R²")
    plt.title("Baseline Test R² Comparison")

    plt.tight_layout()

    plt.savefig(
        output_dir / "test_r2_comparison.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # Save compact test-only table
    # ========================================================

    compact_test = test_metrics[
        ["input", "MAE", "RMSE", "R2"]
    ].copy()

    compact_test.to_csv(
        output_dir / "baseline_test_metrics_compact.csv",
        index=False
    )

    print()
    print("=" * 60)
    print("Compact Test Metrics")
    print("=" * 60)
    print(compact_test)

    print()
    print("Saved comparison results to:")
    print(output_dir)


if __name__ == "__main__":

    main()