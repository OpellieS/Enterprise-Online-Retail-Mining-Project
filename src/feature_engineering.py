"""Customer-level RFM, RFM-T, and temporal validation features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 86_400.0


def _score_1_to_5(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Tie-safe percentile score; avoids qcut failures on repeated values."""
    pct = values.rank(method="average", pct=True, ascending=higher_is_better)
    return np.ceil(pct * 5).clip(1, 5).astype("int64")


def build_customer_features(
    invoices: pd.DataFrame,
    observation_end: pd.Timestamp | None = None,
    gross_margin_rate: float = 0.30,
) -> pd.DataFrame:
    """Create invoice-event customer features using only events through observation_end."""
    if observation_end is None:
        observation_end = pd.Timestamp(invoices["InvoiceDate"].max())
    observation_end = pd.Timestamp(observation_end)
    history = invoices.loc[invoices["InvoiceDate"] <= observation_end].copy()
    if history.empty:
        raise ValueError("No invoice events exist on or before the observation end")
    history = history.sort_values(["CustomerID", "InvoiceDate", "InvoiceNo"])
    history["order_number"] = history.groupby("CustomerID", observed=True).cumcount()

    customer = (
        history.groupby("CustomerID", observed=True)
        .agg(
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
            historical_orders=("InvoiceNo", "nunique"),
            historical_revenue=("OrderRevenue", "sum"),
            mean_order_value=("OrderRevenue", "mean"),
            median_order_value=("OrderRevenue", "median"),
            std_order_value=("OrderRevenue", "std"),
            max_order_value=("OrderRevenue", "max"),
            total_units=("TotalUnits", "sum"),
            active_purchase_days=("InvoiceDate", lambda s: s.dt.normalize().nunique()),
            country=("Country", "first"),
        )
        .reset_index()
    )
    repeat_value = (
        history.loc[history["order_number"] > 0]
        .groupby("CustomerID", observed=True)["OrderRevenue"]
        .mean()
        .rename("monetary_value")
    )
    customer = customer.merge(repeat_value, on="CustomerID", how="left")
    customer["monetary_value"] = customer["monetary_value"].fillna(0.0)
    customer["std_order_value"] = customer["std_order_value"].fillna(0.0)
    customer["frequency"] = customer["historical_orders"] - 1
    customer["bgnbd_recency"] = _days(customer["last_purchase"] - customer["first_purchase"])
    customer["T"] = _days(observation_end - customer["first_purchase"])
    customer["days_since_last_purchase"] = _days(observation_end - customer["last_purchase"])
    customer["tenure_days"] = customer["T"]
    customer["average_days_between_orders"] = np.where(
        customer["frequency"] > 0,
        customer["bgnbd_recency"] / customer["frequency"],
        np.nan,
    )
    customer["historic_clv_annual_margin"] = (
        customer["historical_revenue"]
        * gross_margin_rate
        / customer["tenure_days"].clip(lower=1.0)
        * 365.0
    )

    customer["R_score"] = _score_1_to_5(customer["days_since_last_purchase"], higher_is_better=False)
    customer["F_score"] = _score_1_to_5(customer["historical_orders"], higher_is_better=True)
    customer["M_score"] = _score_1_to_5(customer["historical_revenue"], higher_is_better=True)
    customer["RFM_score"] = (
        customer["R_score"].astype(str)
        + customer["F_score"].astype(str)
        + customer["M_score"].astype(str)
    )

    numeric = ["frequency", "bgnbd_recency", "T", "days_since_last_purchase"]
    if customer[numeric].isna().any().any():
        raise AssertionError("RFM-T features contain missing timeline values")
    if not (
        customer["frequency"].ge(0).all()
        and customer["bgnbd_recency"].ge(0).all()
        and customer["T"].ge(0).all()
        and customer["bgnbd_recency"].le(customer["T"] + 1e-9).all()
    ):
        raise AssertionError("RFM-T invariants failed")
    return customer.sort_values("CustomerID", ignore_index=True)


def make_calibration_holdout(
    invoices: pd.DataFrame,
    holdout_days: int = 90,
    gross_margin_rate: float = 0.30,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Build a leakage-safe calibration table and observed 90-day outcomes."""
    observation_end = pd.Timestamp(invoices["InvoiceDate"].max())
    calibration_end = observation_end - pd.Timedelta(days=holdout_days)
    calibration = build_customer_features(invoices, calibration_end, gross_margin_rate)
    holdout = invoices.loc[
        (invoices["InvoiceDate"] > calibration_end)
        & (invoices["InvoiceDate"] <= observation_end)
        & (invoices["CustomerID"].isin(calibration["CustomerID"]))
    ]
    outcomes = (
        holdout.groupby("CustomerID", observed=True)
        .agg(
            actual_holdout_orders=("InvoiceNo", "nunique"),
            actual_holdout_revenue=("OrderRevenue", "sum"),
        )
        .reset_index()
    )
    calibration = calibration.merge(outcomes, on="CustomerID", how="left")
    calibration[["actual_holdout_orders", "actual_holdout_revenue"]] = calibration[
        ["actual_holdout_orders", "actual_holdout_revenue"]
    ].fillna(0.0)
    return calibration, calibration_end, observation_end

