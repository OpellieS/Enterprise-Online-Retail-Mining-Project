"""Professional, non-misleading figures for the CLV report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _capture_curve(actual: pd.Series, predicted: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-predicted.to_numpy(dtype=float))
    actual_sorted = actual.to_numpy(dtype=float)[order]
    total = actual_sorted.sum()
    y = np.cumsum(actual_sorted) / total if total > 0 else np.zeros(len(actual_sorted))
    x = np.arange(1, len(actual_sorted) + 1) / len(actual_sorted)
    return np.insert(x, 0, 0), np.insert(y, 0, 0)


def generate_all_figures(
    invoices: pd.DataFrame,
    scored: pd.DataFrame,
    bg_validation: pd.DataFrame,
    lightgbm_test: pd.DataFrame,
    feature_importance: pd.DataFrame,
    retention: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    color = "#176B87"
    accent = "#D95F59"
    files: list[Path] = []

    monthly = invoices.assign(Month=invoices["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
    monthly_summary = monthly.groupby("Month").agg(
        Revenue=("OrderRevenue", "sum"),
        ActiveCustomers=("CustomerID", "nunique"),
        Orders=("InvoiceNo", "nunique"),
    )
    monthly_summary["AverageOrderValue"] = monthly_summary["Revenue"] / monthly_summary["Orders"]

    path = figure_dir / "01_monthly_revenue.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_summary.index, monthly_summary["Revenue"], marker="o", color=color)
    ax.set(title="Observed Monthly Positive-Purchase Revenue", xlabel="Month", ylabel="Revenue (£)")
    ax.ticklabel_format(style="plain", axis="y")
    _finish(fig, path); files.append(path)

    path = figure_dir / "02_monthly_active_customers.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_summary.index, monthly_summary["ActiveCustomers"], marker="o", color=accent)
    ax.set(title="Monthly Active Customers", xlabel="Month", ylabel="Distinct customers")
    _finish(fig, path); files.append(path)

    path = figure_dir / "03_recency_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(scored["days_since_last_purchase"], bins=40, color=color, ax=ax)
    ax.set(title="Days Since Last Purchase", xlabel="Days since last purchase", ylabel="Customers")
    _finish(fig, path); files.append(path)

    path = figure_dir / "04_frequency_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(np.log1p(scored["frequency"]), bins=40, color=color, ax=ax)
    ax.set(title="Repeat-Purchase Frequency (log1p scale)", xlabel="log(1 + repeat invoices)", ylabel="Customers")
    _finish(fig, path); files.append(path)

    path = figure_dir / "05_recency_frequency_behavior.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    points = ax.scatter(
        scored["days_since_last_purchase"],
        scored["frequency"] + 1,
        c=np.log1p(scored["historical_revenue"]),
        cmap="viridis",
        alpha=0.55,
        s=22,
    )
    ax.set_yscale("log")
    ax.set(title="Recency × Frequency Customer Behavior", xlabel="Days since last purchase (lower is more recent)", ylabel="Historical invoices (log scale)")
    cbar = fig.colorbar(points, ax=ax); cbar.set_label("log(1 + historical revenue £)")
    _finish(fig, path); files.append(path)

    path = figure_dir / "06_bgnbd_holdout_validation.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(bg_validation["predicted_holdout_orders"], bg_validation["actual_holdout_orders"], alpha=0.35, s=16, color=color)
    upper = max(bg_validation["predicted_holdout_orders"].max(), bg_validation["actual_holdout_orders"].max())
    axes[0].plot([0, upper], [0, upper], linestyle="--", color="black")
    axes[0].set(title="Predicted vs observed", xlabel="Predicted 90-day invoices", ylabel="Observed 90-day invoices")
    residual = bg_validation["actual_holdout_orders"] - bg_validation["predicted_holdout_orders"]
    sns.histplot(residual, bins=40, color=accent, ax=axes[1])
    axes[1].set(title="Holdout residuals", xlabel="Observed − predicted invoices")
    grouped = bg_validation.assign(
        FrequencyGroup=pd.cut(
            bg_validation["frequency"],
            bins=[-0.1, 0.5, 1.5, 2.5, 4.5, 9.5, np.inf],
            labels=["0", "1", "2", "3–4", "5–9", "10+"],
        )
    ).groupby("FrequencyGroup", observed=True).agg(
        Actual=("actual_holdout_orders", "mean"),
        Predicted=("predicted_holdout_orders", "mean"),
    )
    grouped.plot(kind="bar", ax=axes[2], color=[accent, color])
    axes[2].set(title="Mean orders by calibration frequency", xlabel="Calibration repeat invoices", ylabel="Mean 90-day invoices")
    axes[2].tick_params(axis="x", rotation=0)
    _finish(fig, path); files.append(path)

    path = figure_dir / "07_probability_alive_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(scored["probability_alive"], bins=40, color=color, ax=ax)
    ax.axvline(0.50, color=accent, linestyle="--", label="Primary risk threshold = 0.50")
    ax.set(title="BG/NBD Latent Probability Alive", xlabel="Probability Alive (not observed churn)", ylabel="Customers")
    ax.legend()
    _finish(fig, path); files.append(path)

    path = figure_dir / "08_predicted_clv_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(np.log1p(scored["discounted_clv_12m"]), bins=45, color=color, ax=ax)
    ax.set(title="Predicted Discounted 12-Month CLV", xlabel="log(1 + discounted CLV £)", ylabel="Customers")
    _finish(fig, path); files.append(path)

    path = figure_dir / "09_clv_pareto_curve.png"
    ranked = scored.sort_values("discounted_clv_12m", ascending=False)
    x = np.arange(1, len(ranked) + 1) / len(ranked)
    y = ranked["discounted_clv_12m"].cumsum() / ranked["discounted_clv_12m"].sum()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, color=color, linewidth=2, label="Observed model concentration")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Uniform value")
    ax.axvline(0.20, color=accent, linestyle="--", alpha=0.8)
    ax.axhline(0.80, color=accent, linestyle="--", alpha=0.8)
    ax.set(title="Cumulative Contribution of Predicted CLV", xlabel="Cumulative customer share", ylabel="Cumulative predicted CLV share", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    _finish(fig, path); files.append(path)

    path = figure_dir / "10_ltv_risk_matrix.png"
    tier_order = ["Whales", "High Value", "Mid Value", "Long Tail / Low Value"]
    risk_order = ["High latent risk", "Low latent risk"]
    matrix = scored.pivot_table(index="ltv_tier", columns="risk_tier", values="CustomerID", aggfunc="count", fill_value=0).reindex(index=tier_order, columns=risk_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", ax=ax)
    ax.set(title="Customer Count: Predicted LTV Tier × Latent Activity Risk", xlabel="Risk proxy", ylabel="Predicted LTV tier")
    _finish(fig, path); files.append(path)

    path = figure_dir / "11_lightgbm_actual_vs_predicted.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.log1p(lightgbm_test["lightgbm_prediction"]), np.log1p(lightgbm_test["future_90d_revenue"]), alpha=0.35, s=18, color=color)
    upper = max(np.log1p(lightgbm_test["lightgbm_prediction"]).max(), np.log1p(lightgbm_test["future_90d_revenue"]).max())
    ax.plot([0, upper], [0, upper], color="black", linestyle="--")
    ax.set(title="LightGBM Final 90-Day Revenue Test", xlabel="log(1 + predicted revenue £)", ylabel="log(1 + actual revenue £)")
    _finish(fig, path); files.append(path)

    path = figure_dir / "12_lightgbm_feature_importance.png"
    top = feature_importance.head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["Feature"], top["Importance"], color=color)
    ax.set(title="LightGBM Feature Importance", xlabel="Split importance", ylabel="Feature")
    _finish(fig, path); files.append(path)

    path = figure_dir / "13_top_decile_capture_curves.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, column, line_color in (
        ("LightGBM", "lightgbm_prediction", color),
        ("Historic-rate baseline", "simple_historic_rate_prediction", accent),
    ):
        curve_x, curve_y = _capture_curve(lightgbm_test["future_90d_revenue"], lightgbm_test[column])
        ax.plot(curve_x, curve_y, label=label, color=line_color, linewidth=2)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random expectation")
    ax.axvline(0.10, color="black", linestyle="--", alpha=0.5)
    ax.set(title="Actual Holdout Spend Capture", xlabel="Cumulative customer share ranked by prediction", ylabel="Cumulative actual spend share", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    _finish(fig, path); files.append(path)

    path = figure_dir / "14_retention_sensitivity.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(retention["OrderUplift"].map(lambda x: f"{x:.0%}"), retention["DiscountedIncrementalMarginGBP"], color=color)
    ax.set(title="Retention Scenario: Incremental Discounted Margin", xlabel="Hypothetical order uplift (not causal)", ylabel="Discounted incremental margin (£)")
    ax.ticklabel_format(style="plain", axis="y")
    _finish(fig, path); files.append(path)

    path = figure_dir / "15_empirical_repeat_rate_by_recency.png"
    empirical = bg_validation.assign(
        RecencyBin=pd.qcut(
            bg_validation["days_since_last_purchase"], q=10, duplicates="drop"
        ),
        Repeated=bg_validation["actual_holdout_orders"].gt(0),
    ).groupby("RecencyBin", observed=True).agg(
        MeanRecency=("days_since_last_purchase", "mean"),
        RepeatRate=("Repeated", "mean"),
        Customers=("CustomerID", "size"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(empirical["MeanRecency"], empirical["RepeatRate"], marker="o", color=color)
    ax.set(
        title="Empirical 90-Day Repeat Rate by Prior Recency",
        xlabel="Mean days since last purchase in recency decile",
        ylabel="Observed share purchasing in holdout",
        ylim=(0, max(0.05, empirical["RepeatRate"].max() * 1.1)),
    )
    _finish(fig, path); files.append(path)

    return files
