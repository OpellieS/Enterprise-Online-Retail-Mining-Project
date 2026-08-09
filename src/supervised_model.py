"""Leakage-safe LightGBM benchmark for future 90-day customer revenue."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .clv_models import regression_metrics
from .config import RANDOM_STATE
from .feature_engineering import build_customer_features


NUMERIC_FEATURES = [
    "days_since_last_purchase",
    "transaction_frequency",
    "tenure_days",
    "total_historical_revenue",
    "mean_order_value",
    "median_order_value",
    "std_order_value",
    "max_order_value",
    "revenue_last_30d",
    "revenue_last_60d",
    "revenue_last_90d",
    "orders_last_30d",
    "orders_last_60d",
    "orders_last_90d",
    "units_last_30d",
    "units_last_60d",
    "units_last_90d",
    "spend_30_vs_previous_30",
    "spend_60_vs_previous_60",
    "order_frequency_momentum",
    "unique_products",
    "average_unique_products_per_order",
    "average_units_per_order",
    "active_purchase_days",
    "average_days_between_orders",
]


def _window_aggregate(
    invoices: pd.DataFrame, cutoff: pd.Timestamp, days: int
) -> pd.DataFrame:
    start = cutoff - pd.Timedelta(days=days)
    window = invoices.loc[
        (invoices["InvoiceDate"] > start) & (invoices["InvoiceDate"] <= cutoff)
    ]
    return (
        window.groupby("CustomerID", observed=True)
        .agg(
            **{
                f"revenue_last_{days}d": ("OrderRevenue", "sum"),
                f"orders_last_{days}d": ("InvoiceNo", "nunique"),
                f"units_last_{days}d": ("TotalUnits", "sum"),
            }
        )
        .reset_index()
    )


def engineer_snapshot(
    invoices: pd.DataFrame,
    clean_lines: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """Compute every feature at/before cutoff and the target strictly after cutoff."""
    cutoff = pd.Timestamp(cutoff)
    history = invoices.loc[invoices["InvoiceDate"] <= cutoff]
    base = build_customer_features(invoices, cutoff).rename(
        columns={
            "historical_orders": "transaction_frequency",
            "historical_revenue": "total_historical_revenue",
        }
    )

    for days in (30, 60, 90):
        base = base.merge(_window_aggregate(history, cutoff, days), on="CustomerID", how="left")

    previous_30 = history.loc[
        (history["InvoiceDate"] > cutoff - pd.Timedelta(days=60))
        & (history["InvoiceDate"] <= cutoff - pd.Timedelta(days=30))
    ].groupby("CustomerID", observed=True)["OrderRevenue"].sum()
    previous_60 = history.loc[
        (history["InvoiceDate"] > cutoff - pd.Timedelta(days=120))
        & (history["InvoiceDate"] <= cutoff - pd.Timedelta(days=60))
    ].groupby("CustomerID", observed=True)["OrderRevenue"].sum()
    previous_orders_90 = history.loc[
        (history["InvoiceDate"] > cutoff - pd.Timedelta(days=180))
        & (history["InvoiceDate"] <= cutoff - pd.Timedelta(days=90))
    ].groupby("CustomerID", observed=True)["InvoiceNo"].nunique()

    base = base.set_index("CustomerID")
    base["spend_30_vs_previous_30"] = base["revenue_last_30d"].fillna(0) - previous_30.reindex(base.index).fillna(0)
    base["spend_60_vs_previous_60"] = base["revenue_last_60d"].fillna(0) - previous_60.reindex(base.index).fillna(0)
    base["order_frequency_momentum"] = base["orders_last_90d"].fillna(0) - previous_orders_90.reindex(base.index).fillna(0)

    product_history = clean_lines.loc[clean_lines["InvoiceDate"] <= cutoff]
    unique_products = product_history.groupby("CustomerID", observed=True)["StockCode"].nunique()
    base["unique_products"] = unique_products.reindex(base.index).fillna(0)
    base["average_unique_products_per_order"] = (
        history.groupby("CustomerID", observed=True)["UniqueProducts"].mean().reindex(base.index)
    )
    base["average_units_per_order"] = (
        history.groupby("CustomerID", observed=True)["TotalUnits"].mean().reindex(base.index)
    )

    target_end = cutoff + pd.Timedelta(days=horizon_days)
    future = invoices.loc[
        (invoices["InvoiceDate"] > cutoff) & (invoices["InvoiceDate"] <= target_end)
    ]
    target = future.groupby("CustomerID", observed=True)["OrderRevenue"].sum()
    base["future_90d_revenue"] = target.reindex(base.index).fillna(0)
    base["cutoff"] = cutoff
    base["target_start"] = cutoff + pd.Timedelta(microseconds=1)
    base["target_end"] = target_end
    base = base.reset_index()

    rolling_cols = [
        f"{metric}_last_{days}d"
        for days in (30, 60, 90)
        for metric in ("revenue", "orders", "units")
    ]
    base[rolling_cols] = base[rolling_cols].fillna(0.0)
    base["average_days_between_orders"] = base["average_days_between_orders"].fillna(
        base["tenure_days"].clip(lower=0)
    )
    return base


def chronological_cutoffs(
    invoices: pd.DataFrame, horizon_days: int = 90
) -> tuple[list[pd.Timestamp], pd.Timestamp, pd.Timestamp]:
    """Return monthly training cutoffs plus non-overlapping validation and test cutoffs."""
    first = pd.Timestamp(invoices["InvoiceDate"].min())
    observation_end = pd.Timestamp(invoices["InvoiceDate"].max())
    test_cutoff = observation_end - pd.Timedelta(days=horizon_days)
    validation_cutoff = test_cutoff - pd.Timedelta(days=horizon_days)
    latest_train_cutoff = validation_cutoff - pd.Timedelta(days=horizon_days)
    start = (first + pd.offsets.MonthBegin(1)).normalize()
    training = list(pd.date_range(start=start, end=latest_train_cutoff, freq="MS"))
    if len(training) < 2:
        raise ValueError("Dataset does not span enough time for chronological snapshots")
    return training, validation_cutoff, test_cutoff


def build_snapshot_dataset(
    invoices: pd.DataFrame,
    clean_lines: pd.DataFrame,
    horizon_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_cutoffs, validation_cutoff, test_cutoff = chronological_cutoffs(
        invoices, horizon_days
    )
    frames = []
    design_rows = []
    for split, cutoffs in (
        ("train", training_cutoffs),
        ("validation", [validation_cutoff]),
        ("test", [test_cutoff]),
    ):
        for cutoff in cutoffs:
            snapshot = engineer_snapshot(invoices, clean_lines, cutoff, horizon_days)
            snapshot["split"] = split
            frames.append(snapshot)
            design_rows.append(
                {
                    "Split": split,
                    "Cutoff": cutoff,
                    "FeatureWindowLatest": cutoff,
                    "TargetWindowStartExclusive": cutoff,
                    "TargetWindowEndInclusive": cutoff + pd.Timedelta(days=horizon_days),
                    "Customers": len(snapshot),
                }
            )
    design = pd.DataFrame(design_rows)
    train_target_end = design.loc[design["Split"].eq("train"), "TargetWindowEndInclusive"].max()
    validation_cutoff = design.loc[design["Split"].eq("validation"), "Cutoff"].iloc[0]
    validation_target_end = design.loc[
        design["Split"].eq("validation"), "TargetWindowEndInclusive"
    ].iloc[0]
    test_cutoff = design.loc[design["Split"].eq("test"), "Cutoff"].iloc[0]
    if train_target_end > validation_cutoff or validation_target_end > test_cutoff:
        raise AssertionError("Chronological split has overlapping future information")
    return pd.concat(frames, ignore_index=True), design


def _design_matrix(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric = snapshots[NUMERIC_FEATURES].astype(float)
    country = pd.get_dummies(
        snapshots["country"].fillna("Unknown"), prefix="country", dtype=float
    )
    matrix = pd.concat([numeric, country], axis=1)
    return matrix, list(matrix.columns)


def train_lightgbm_benchmark(
    snapshots: pd.DataFrame, model_path: Path | None = None
):
    """Validate chronologically, refit through validation, and evaluate final test."""
    import lightgbm as lgb

    X, feature_names = _design_matrix(snapshots)
    y = snapshots["future_90d_revenue"].astype(float)
    train_mask = snapshots["split"].eq("train")
    validation_mask = snapshots["split"].eq("validation")
    test_mask = snapshots["split"].eq("test")

    initial = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    initial.fit(
        X.loc[train_mask],
        y.loc[train_mask],
        eval_set=[(X.loc[validation_mask], y.loc[validation_mask])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    validation_prediction = np.clip(initial.predict(X.loc[validation_mask]), 0, None)
    best_iterations = int(initial.best_iteration_ or 300)

    final_model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=best_iterations,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    development_mask = train_mask | validation_mask
    final_model.fit(X.loc[development_mask], y.loc[development_mask])
    test_prediction = np.clip(final_model.predict(X.loc[test_mask]), 0, None)

    validation = snapshots.loc[validation_mask, ["CustomerID", "cutoff", "future_90d_revenue"]].copy()
    validation["lightgbm_prediction"] = validation_prediction
    test = snapshots.loc[test_mask, ["CustomerID", "cutoff", "future_90d_revenue", "total_historical_revenue", "tenure_days"]].copy()
    test["lightgbm_prediction"] = test_prediction
    test["simple_historic_rate_prediction"] = (
        test["total_historical_revenue"] / test["tenure_days"].clip(lower=30.0) * 90.0
    )

    metrics = [
        regression_metrics(
            validation["future_90d_revenue"],
            validation["lightgbm_prediction"],
            "LightGBM validation revenue",
        ),
        regression_metrics(
            test["future_90d_revenue"],
            test["simple_historic_rate_prediction"],
            "Historic-rate baseline test revenue",
        ),
        regression_metrics(
            test["future_90d_revenue"],
            test["lightgbm_prediction"],
            "LightGBM final test revenue",
        ),
    ]
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": final_model.feature_importances_}
    ).sort_values("Importance", ascending=False, ignore_index=True)
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": final_model, "features": feature_names}, model_path
        )
    return final_model, validation, test, pd.DataFrame(metrics), importance

