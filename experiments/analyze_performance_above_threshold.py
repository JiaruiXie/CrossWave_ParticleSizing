"""
Analyze model performance only for particles with cube-root Cy3 max intensity > threshold.

Default:
    threshold = 5

This excludes very low-intensity particles from evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(
        np.abs(y_pred - y_true)
    )

    rmse = np.sqrt(
        np.mean((y_pred - y_true) ** 2)
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def analyze_model(
    model_name,
    prediction_path,
    threshold=5.0
):

    df = pd.read_csv(
        prediction_path
    )

    original_count = len(df)

    df_filtered = df[
        df["target"] > threshold
    ].copy()

    filtered_count = len(df_filtered)

    removed_count = original_count - filtered_count

    metrics = compute_metrics(
        df_filtered["target"],
        df_filtered["prediction"]
    )

    row = {
        "model": model_name,
        "threshold": threshold,
        "condition": f"target > {threshold}",
        "original_samples": original_count,
        "evaluated_samples": filtered_count,
        "removed_samples": removed_count,
        "removed_percentage": removed_count / original_count * 100,
        **metrics
    }

    return row, df_filtered


def plot_filtered_distribution(
    df,
    model_name,
    output_dir,
    threshold=5.0
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    y_true = df["target"].values
    y_pred = df["prediction"].values

    min_value = min(
        y_true.min(),
        y_pred.min()
    )

    max_value = max(
        y_true.max(),
        y_pred.max()
    )

    bins = np.linspace(
        min_value,
        max_value,
        80
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        y_true,
        bins=bins,
        alpha=0.5,
        label="Ground truth",
        edgecolor="black"
    )

    plt.hist(
        y_pred,
        bins=bins,
        alpha=0.5,
        label="Prediction",
        edgecolor="black"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {threshold}"
    )

    plt.xlabel("Cube-root Cy3 max intensity")
    plt.ylabel("Frequency")
    plt.title(f"{model_name}: Prediction Distribution, Target > {threshold}")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{model_name}_distribution_target_gt_{threshold:g}.png",
        dpi=300
    )

    plt.close()


def plot_filtered_scatter(
    df,
    model_name,
    metrics,
    output_dir,
    threshold=5.0
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    y_true = df["target"].values
    y_pred = df["prediction"].values

    min_value = min(
        y_true.min(),
        y_pred.min()
    )

    max_value = max(
        y_true.max(),
        y_pred.max()
    )

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.25,
        s=8
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=1.5
    )

    text = (
        f"R² = {metrics['R2']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.4f}\n"
        f"MAE = {metrics['MAE']:.4f}"
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            alpha=0.8
        )
    )

    plt.xlabel("True cube-root Cy3 max intensity")
    plt.ylabel("Predicted cube-root Cy3 max intensity")
    plt.title(f"{model_name}: Predicted vs True, Target > {threshold}")

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{model_name}_scatter_target_gt_{threshold:g}.png",
        dpi=300
    )

    plt.close()


def main():

    threshold = 5.0

    models = {
        "fusion_unweighted": Path(
            "outputs/baseline_fusion_unweighted/predictions/test_predictions.csv"
        ),
        "fusion_weighted_clipped_20": Path(
            "outputs/baseline_fusion_weighted_clipped_20/predictions/test_predictions.csv"
        ),
        "fusion_weighted_clipped_10": Path(
            "outputs/baseline_fusion_weighted_clipped_10/predictions/test_predictions.csv"
        ),
    }

    output_dir = Path(
        "outputs/performance_target_gt_5"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for model_name, prediction_path in models.items():

        if not prediction_path.exists():
            print(f"Missing prediction file: {prediction_path}")
            continue

        row, df_filtered = analyze_model(
            model_name=model_name,
            prediction_path=prediction_path,
            threshold=threshold
        )

        rows.append(row)

        metrics = {
            "MAE": row["MAE"],
            "RMSE": row["RMSE"],
            "R2": row["R2"]
        }

        plot_filtered_distribution(
            df=df_filtered,
            model_name=model_name,
            output_dir=output_dir,
            threshold=threshold
        )

        plot_filtered_scatter(
            df=df_filtered,
            model_name=model_name,
            metrics=metrics,
            output_dir=output_dir,
            threshold=threshold
        )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        output_dir / "performance_target_gt_5.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("Performance for Target > 5")
    print("=" * 80)
    print(summary)

    # Plot RMSE comparison
    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["model"],
        summary["RMSE"]
    )

    plt.xlabel("Model")
    plt.ylabel("Test RMSE")
    plt.title("Test RMSE for Particles with Cube-root Cy3 > 5")
    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        output_dir / "rmse_comparison_target_gt_5.png",
        dpi=300
    )

    plt.close()

    print()
    print("Saved results to:")
    print(output_dir)


if __name__ == "__main__":

    main()