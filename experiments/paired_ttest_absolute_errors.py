"""
Paired t-test on absolute errors between two models.

This compares whether Model A has significantly different absolute error
from Model B on the same test samples.

Example:

python experiments/paired_ttest_absolute_errors.py ^
    --model_a_csv "outputs/optuna/baseline_440_unweighted_soft_sampler_alpha_0.3/trial_025/paired_ttest_440_test/predictions_for_paired_ttest.csv" ^
    --model_b_csv "outputs/optuna/baseline_555_unweighted_soft_sampler_alpha_0.3/trial_XXX/paired_ttest_555_test/predictions_for_paired_ttest.csv" ^
    --model_a_name baseline_440 ^
    --model_b_name baseline_555 ^
    --scale cube_root
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():

    parser = argparse.ArgumentParser(
        description="Paired t-test on absolute errors between two models."
    )

    parser.add_argument(
        "--model_a_csv",
        type=str,
        required=True,
        help="Prediction CSV for Model A."
    )

    parser.add_argument(
        "--model_b_csv",
        type=str,
        required=True,
        help="Prediction CSV for Model B."
    )

    parser.add_argument(
        "--model_a_name",
        type=str,
        default="model_a"
    )

    parser.add_argument(
        "--model_b_name",
        type=str,
        default="model_b"
    )

    parser.add_argument(
        "--scale",
        type=str,
        default="cube_root",
        choices=["cube_root", "original"],
        help="Compare errors on cube-root or original intensity scale."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None
    )

    return parser.parse_args()


def get_columns(scale):

    if scale == "cube_root":

        return {
            "target": "target_cube_root",
            "prediction": "prediction_cube_root",
            "abs_error": "abs_error_cube_root"
        }

    if scale == "original":

        return {
            "target": "target_original",
            "prediction": "prediction_original",
            "abs_error": "abs_error_original"
        }

    raise ValueError(
        f"Unknown scale: {scale}"
    )


def compute_metrics(y_true, y_pred):

    mae = np.mean(
        np.abs(y_pred - y_true)
    )

    mse = np.mean(
        (y_pred - y_true) ** 2
    )

    rmse = np.sqrt(
        mse
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
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def main():

    args = parse_args()

    model_a_csv = Path(args.model_a_csv)
    model_b_csv = Path(args.model_b_csv)

    if not model_a_csv.exists():
        raise FileNotFoundError(
            f"Cannot find Model A CSV: {model_a_csv}"
        )

    if not model_b_csv.exists():
        raise FileNotFoundError(
            f"Cannot find Model B CSV: {model_b_csv}"
        )

    if args.output_dir is None:

        output_dir = Path(
            "outputs"
        ) / "statistical_tests" / f"{args.model_a_name}_vs_{args.model_b_name}_{args.scale}"

    else:

        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df_a = pd.read_csv(
        model_a_csv
    )

    df_b = pd.read_csv(
        model_b_csv
    )

    columns = get_columns(
        args.scale
    )

    merge_keys = [
        "particle_id",
        "image_index"
    ]

    df = df_a[
        merge_keys + [
            columns["target"],
            columns["prediction"],
            columns["abs_error"]
        ]
    ].merge(
        df_b[
            merge_keys + [
                columns["target"],
                columns["prediction"],
                columns["abs_error"]
            ]
        ],
        on=merge_keys,
        suffixes=("_a", "_b")
    )

    # Check that the true targets match between the two model prediction files.
    target_diff = np.max(
        np.abs(
            df[f"{columns['target']}_a"].values
            - df[f"{columns['target']}_b"].values
        )
    )

    if target_diff > 1e-5:

        raise ValueError(
            f"Targets do not match between two CSVs. Max target difference = {target_diff}"
        )

    y_true = df[f"{columns['target']}_a"].values

    pred_a = df[f"{columns['prediction']}_a"].values
    pred_b = df[f"{columns['prediction']}_b"].values

    abs_error_a = df[f"{columns['abs_error']}_a"].values
    abs_error_b = df[f"{columns['abs_error']}_b"].values

    error_difference = abs_error_a - abs_error_b

    # Paired t-test on absolute errors.
    paired_test = stats.ttest_rel(
        abs_error_a,
        abs_error_b
    )

    # Equivalent one-sample t-test on the paired error difference.
    one_sample_test = stats.ttest_1samp(
        error_difference,
        popmean=0.0
    )

    n = len(error_difference)

    mean_abs_error_a = np.mean(
        abs_error_a
    )

    mean_abs_error_b = np.mean(
        abs_error_b
    )

    mean_difference = np.mean(
        error_difference
    )

    std_difference = np.std(
        error_difference,
        ddof=1
    )

    se_difference = std_difference / np.sqrt(
        n
    )

    dfree = n - 1

    confidence_level = 0.95

    t_critical = stats.t.ppf(
        1.0 - (1.0 - confidence_level) / 2.0,
        df=dfree
    )

    ci_lower = mean_difference - t_critical * se_difference
    ci_upper = mean_difference + t_critical * se_difference

    cohen_dz = (
        mean_difference / std_difference
        if std_difference != 0
        else np.nan
    )

    percent_change = (
        mean_difference / mean_abs_error_b * 100.0
        if mean_abs_error_b != 0
        else np.nan
    )

    metrics_a = compute_metrics(
        y_true=y_true,
        y_pred=pred_a
    )

    metrics_b = compute_metrics(
        y_true=y_true,
        y_pred=pred_b
    )

    summary = {
        "scale": args.scale,
        "model_a": args.model_a_name,
        "model_b": args.model_b_name,
        "n": n,

        "MAE_model_a": metrics_a["MAE"],
        "MAE_model_b": metrics_b["MAE"],
        "RMSE_model_a": metrics_a["RMSE"],
        "RMSE_model_b": metrics_b["RMSE"],
        "R2_model_a": metrics_a["R2"],
        "R2_model_b": metrics_b["R2"],

        "mean_absolute_error_a": mean_abs_error_a,
        "mean_absolute_error_b": mean_abs_error_b,
        "mean_absolute_error_difference_a_minus_b": mean_difference,
        "std_difference": std_difference,
        "se_difference": se_difference,

        "t_statistic_paired": paired_test.statistic,
        "p_value_paired": paired_test.pvalue,

        "t_statistic_one_sample_difference": one_sample_test.statistic,
        "p_value_one_sample_difference": one_sample_test.pvalue,

        "df": dfree,
        "confidence_level": confidence_level,
        "ci_lower_mean_difference": ci_lower,
        "ci_upper_mean_difference": ci_upper,
        "cohen_dz": cohen_dz,
        "percent_change_relative_to_model_b": percent_change
    }

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    df["abs_error_difference_a_minus_b"] = error_difference

    df.to_csv(
        output_dir / "paired_absolute_error_differences.csv",
        index=False
    )

    summary_df.to_csv(
        output_dir / "paired_ttest_absolute_error_results.csv",
        index=False
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        200
    )

    print()
    print("=" * 80)
    print("Paired t-test on absolute errors")
    print("=" * 80)
    print(summary_df)

    print()
    print("Interpretation:")
    print(
        f"Difference = absolute_error({args.model_a_name}) - absolute_error({args.model_b_name})"
    )

    if mean_difference < 0:

        print(
            f"{args.model_a_name} has lower absolute error on average."
        )

    elif mean_difference > 0:

        print(
            f"{args.model_b_name} has lower absolute error on average."
        )

    else:

        print(
            "Both models have the same mean absolute error."
        )

    print()
    print(f"Saved paired differences to: {output_dir / 'paired_absolute_error_differences.csv'}")
    print(f"Saved summary results to: {output_dir / 'paired_ttest_absolute_error_results.csv'}")


if __name__ == "__main__":

    main()