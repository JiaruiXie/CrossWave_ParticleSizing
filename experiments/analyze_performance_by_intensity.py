"""
Analyze model performance by target intensity range.

Compares:
    - baseline_fusion_unweighted
    - baseline_fusion_weighted

Target scale:
    Cube-root transformed Cy3 max intensity
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(np.abs(y_pred - y_true))

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return mae, rmse, r2


def analyze_one_model(model_name, prediction_path, split):

    df = pd.read_csv(prediction_path)

    # Quantile-based bins on target intensity
    df["intensity_group"] = pd.qcut(
        df["target"],
        q=[0.0, 0.5, 0.9, 1.0],
        labels=["low_0_50%", "medium_50_90%", "high_90_100%"],
        duplicates="drop"
    )

    rows = []

    for group_name, group_df in df.groupby("intensity_group", observed=True):

        mae, rmse, r2 = compute_metrics(
            group_df["target"],
            group_df["prediction"]
        )

        rows.append(
            {
                "model": model_name,
                "split": split,
                "intensity_group": group_name,
                "num_samples": len(group_df),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            }
        )

    return pd.DataFrame(rows)


def main():

    models = {
    "fusion_unweighted": Path(
        "outputs/baseline_fusion_unweighted/predictions/test_predictions.csv"
    ),
    "fusion_weighted": Path(
        "outputs/baseline_fusion_weighted/predictions/test_predictions.csv"
    ),
    "fusion_weighted_clipped_20": Path(
        "outputs/baseline_fusion_weighted_clipped_20/predictions/test_predictions.csv"
    ),
    "fusion_weighted_clipped_10": Path(
        "outputs/baseline_fusion_weighted_clipped_10/predictions/test_predictions.csv"
    ),
}

    output_dir = Path(
        "outputs/performance_by_intensity"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    for model_name, prediction_path in models.items():

        if not prediction_path.exists():
            print(f"Missing prediction file: {prediction_path}")
            continue

        result = analyze_one_model(
            model_name=model_name,
            prediction_path=prediction_path,
            split="test"
        )

        all_results.append(result)

    comparison = pd.concat(
        all_results,
        ignore_index=True
    )

    comparison.to_csv(
        output_dir / "fusion_performance_by_intensity.csv",
        index=False
    )

    print()
    print("=" * 60)
    print("Performance by Intensity Group")
    print("=" * 60)
    print(comparison)

    # Plot RMSE by group
    pivot_rmse = comparison.pivot(
        index="intensity_group",
        columns="model",
        values="RMSE"
    )

    plt.figure(figsize=(8, 5))

    pivot_rmse.plot(
        kind="bar"
    )

    plt.xlabel("Intensity group")
    plt.ylabel("Test RMSE")
    plt.title("Fusion Model RMSE by Intensity Group")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        output_dir / "fusion_rmse_by_intensity.png",
        dpi=300
    )

    plt.close()

    print()
    print("Saved results to:")
    print(output_dir)


if __name__ == "__main__":

    main()