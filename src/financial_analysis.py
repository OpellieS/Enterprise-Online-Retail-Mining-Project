"""Discounted CLV, capital-allocation thresholds, and scenario analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clv_models import _model_inputs


def calculate_discounted_clv(
    bg_model,
    scored: pd.DataFrame,
    gross_margin_rate: float = 0.30,
    annual_discount_rate: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate incremental month-by-month expected value for 12 months."""
    result = scored.copy()
    frequency, recency, tenure = _model_inputs(result)
    effective_monthly_rate = (1 + annual_discount_rate) ** (1 / 12) - 1
    prior_orders = np.zeros(len(result), dtype=float)
    monthly_rows = []
    discounted_margin = np.zeros(len(result), dtype=float)
    discounted_margin_6m = np.zeros(len(result), dtype=float)
    revenue_12m = np.zeros(len(result), dtype=float)
    margin_12m = np.zeros(len(result), dtype=float)

    for month in range(1, 13):
        horizon_days = month * 365.0 / 12.0
        cumulative_orders = np.asarray(
            bg_model.conditional_expected_number_of_purchases_up_to_time(
                horizon_days, frequency, recency, tenure
            ),
            dtype=float,
        )
        incremental_orders = np.clip(cumulative_orders - prior_orders, 0, None)
        incremental_revenue = incremental_orders * result["expected_monetary_value"].to_numpy()
        incremental_margin = incremental_revenue * gross_margin_rate
        discount_factor = 1 / ((1 + effective_monthly_rate) ** month)
        present_margin = incremental_margin * discount_factor

        revenue_12m += incremental_revenue
        margin_12m += incremental_margin
        discounted_margin += present_margin
        if month <= 6:
            discounted_margin_6m += present_margin
        monthly_rows.append(
            {
                "Month": month,
                "HorizonDays": horizon_days,
                "ExpectedOrders": incremental_orders.sum(),
                "ExpectedRevenueGBP": incremental_revenue.sum(),
                "ExpectedGrossMarginGBP": incremental_margin.sum(),
                "DiscountFactor": discount_factor,
                "DiscountedGrossMarginGBP": present_margin.sum(),
            }
        )
        prior_orders = cumulative_orders

    result["expected_orders_12m"] = prior_orders
    result["expected_revenue_12m"] = revenue_12m
    result["expected_margin_12m"] = margin_12m
    result["discounted_clv_12m"] = discounted_margin
    result["max_cac_6m_payback"] = discounted_margin_6m
    result["max_cac_3x"] = discounted_margin / 3.0
    result["max_cac_1_5x"] = discounted_margin / 1.5
    return result, pd.DataFrame(monthly_rows)


def add_value_tiers_and_actions(
    scored: pd.DataFrame, risk_threshold: float = 0.50
) -> pd.DataFrame:
    """Create mutually exclusive CLV tiers and a transparent latent-risk action matrix."""
    result = scored.copy()
    percentile = result["discounted_clv_12m"].rank(
        method="first", ascending=False
    ) / len(result)
    result["ltv_tier"] = np.select(
        [percentile <= 0.10, percentile <= 0.20, percentile <= 0.60],
        ["Whales", "High Value", "Mid Value"],
        default="Long Tail / Low Value",
    )
    result["risk_tier"] = np.where(
        result["probability_alive"] < risk_threshold,
        "High latent risk",
        "Low latent risk",
    )

    high_value = result["ltv_tier"].isin(["Whales", "High Value"])
    mid_value = result["ltv_tier"].eq("Mid Value")
    high_risk = result["risk_tier"].eq("High latent risk")
    result["recommended_action"] = np.select(
        [
            high_value & high_risk,
            high_value & ~high_risk,
            mid_value & high_risk,
            mid_value & ~high_risk,
            ~high_value & ~mid_value & high_risk,
        ],
        [
            "Priority targeted retention; test a reversible offer",
            "VIP service and loyalty perks; avoid blanket discount",
            "Low-cost automated re-engagement",
            "Targeted cross-sell through owned channels",
            "No paid retention; passive/organic communication only",
        ],
        default="Low-cost organic nurture",
    )
    return result


def concentration_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Measure observed revenue and predicted CLV concentration without forcing 80/20."""
    n = len(scored)
    ranked_clv = scored.sort_values("discounted_clv_12m", ascending=False)
    ranked_revenue = scored.sort_values("historical_revenue", ascending=False)
    n10 = max(1, int(np.ceil(n * 0.10)))
    n20 = max(1, int(np.ceil(n * 0.20)))
    clv_total = ranked_clv["discounted_clv_12m"].sum()
    revenue_total = ranked_revenue["historical_revenue"].sum()
    cumulative = ranked_clv["discounted_clv_12m"].cumsum() / clv_total
    customers_to_80 = int(np.searchsorted(cumulative.to_numpy(), 0.80) + 1)
    top20_share = ranked_clv.head(n20)["discounted_clv_12m"].sum() / clv_total
    rows = [
        ("Observed historical revenue share from top 10%", ranked_revenue.head(n10)["historical_revenue"].sum() / revenue_total),
        ("Predicted CLV share from top 10%", ranked_clv.head(n10)["discounted_clv_12m"].sum() / clv_total),
        ("Predicted CLV share from top 20%", top20_share),
        ("Customers required for 80% of predicted CLV", customers_to_80),
        ("Customer share required for 80% of predicted CLV", customers_to_80 / n),
        ("Strict 80/20 pattern (top 20% >= 80% of CLV)", bool(top20_share >= 0.80)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def risk_threshold_sensitivity(scored: pd.DataFrame) -> pd.DataFrame:
    high_value = scored["ltv_tier"].isin(["Whales", "High Value"])
    rows = []
    for threshold in (0.30, 0.50, 0.70):
        high_risk = scored["probability_alive"] < threshold
        rows.append(
            {
                "ProbabilityAliveThreshold": threshold,
                "AllHighRiskCustomers": int(high_risk.sum()),
                "HighValueHighRiskCustomers": int((high_value & high_risk).sum()),
                "CLVAtRiskGBP": float(scored.loc[high_value & high_risk, "discounted_clv_12m"].sum()),
            }
        )
    return pd.DataFrame(rows)


def ltv_cac_tables(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer = scored[
        [
            "CustomerID",
            "ltv_tier",
            "discounted_clv_12m",
            "max_cac_3x",
            "max_cac_1_5x",
            "max_cac_6m_payback",
        ]
    ].copy()
    summary = (
        customer.groupby("ltv_tier", observed=True)
        .agg(
            Customers=("CustomerID", "size"),
            TotalDiscountedCLVGBP=("discounted_clv_12m", "sum"),
            MeanMaxCAC3xGBP=("max_cac_3x", "mean"),
            MedianMaxCAC3xGBP=("max_cac_3x", "median"),
            MeanKillLevelCAC1_5xGBP=("max_cac_1_5x", "mean"),
            MeanSixMonthMaxCACGBP=("max_cac_6m_payback", "mean"),
        )
        .reset_index()
    )
    tier_order = ["Whales", "High Value", "Mid Value", "Long Tail / Low Value"]
    summary["ltv_tier"] = pd.Categorical(summary["ltv_tier"], tier_order, ordered=True)
    return customer, summary.sort_values("ltv_tier")


def retention_sensitivity(scored: pd.DataFrame) -> pd.DataFrame:
    """Scenario analysis only: no causal uplift claim is made."""
    target = scored.loc[
        scored["ltv_tier"].isin(["Whales", "High Value"])
        & scored["risk_tier"].eq("High latent risk")
    ]
    rows = []
    for uplift in (0.05, 0.10, 0.15):
        incremental_orders = target["expected_orders_12m"].sum() * uplift
        incremental_revenue = target["expected_revenue_12m"].sum() * uplift
        incremental_margin = target["expected_margin_12m"].sum() * uplift
        discounted_incremental_margin = target["discounted_clv_12m"].sum() * uplift
        rows.append(
            {
                "ScenarioLabel": "SCENARIO ANALYSIS — NOT A CAUSAL ESTIMATE",
                "OrderUplift": uplift,
                "TargetCustomers": len(target),
                "IncrementalExpectedOrders": incremental_orders,
                "IncrementalRevenueGBP": incremental_revenue,
                "IncrementalGrossMarginGBP": incremental_margin,
                "DiscountedIncrementalMarginGBP": discounted_incremental_margin,
                "BreakEvenSpendPerTargetCustomerGBP": (
                    discounted_incremental_margin / len(target) if len(target) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def financial_summary(scored: pd.DataFrame) -> pd.DataFrame:
    target = scored.loc[
        scored["ltv_tier"].isin(["Whales", "High Value"])
        & scored["risk_tier"].eq("High latent risk")
    ]
    return pd.DataFrame(
        [
            ("Total predicted 12-month revenue run-rate", scored["expected_revenue_12m"].sum(), "MODEL ESTIMATE"),
            ("Total predicted 12-month gross-margin run-rate", scored["expected_margin_12m"].sum(), "MODEL ESTIMATE + 30% COURSE ASSUMPTION"),
            ("Total discounted 12-month CLV", scored["discounted_clv_12m"].sum(), "MODEL ESTIMATE + COURSE ASSUMPTIONS"),
            ("High-LTV/high-risk customer count", len(target), "MODEL-DERIVED SEGMENT"),
            ("Discounted CLV in high-LTV/high-risk segment", target["discounted_clv_12m"].sum(), "MODEL ESTIMATE"),
        ],
        columns=["Metric", "Value", "Classification"],
    )

