"""Run the complete Online Retail CLV analysis and write reproducible artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.clv_models import (
    bgnbd_penalizer_sensitivity,
    fit_bgnbd,
    fit_gamma_gamma,
    model_parameters,
    monetary_assumption_diagnostic,
    score_bgnbd,
    score_gamma_gamma,
    validate_bgnbd_gamma,
)
from src.config import (
    ANNUAL_DISCOUNT_RATE,
    BGNBD_PENALIZER,
    COURSE_BGNBD_PENALIZER,
    DATA_DIR,
    FIGURE_DIR,
    GAMMA_GAMMA_PENALIZER,
    GROSS_MARGIN_RATE,
    HOLDOUT_DAYS,
    MODEL_DIR,
    RISK_THRESHOLD,
    TABLE_DIR,
)
from src.data_processing import (
    aggregate_invoices,
    clean_purchase_lines,
    dataset_audit,
    dataset_metadata,
    load_raw_data,
    locate_csv,
)
from src.feature_engineering import build_customer_features, make_calibration_holdout
from src.financial_analysis import (
    add_value_tiers_and_actions,
    calculate_discounted_clv,
    concentration_metrics,
    financial_summary,
    ltv_cac_tables,
    retention_sensitivity,
    risk_threshold_sensitivity,
)
from src.supervised_model import build_snapshot_dataset, train_lightgbm_benchmark
from src.visualization import generate_all_figures


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    frame.to_csv(path, index=False)
    return path


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_full_analysis() -> dict:
    for directory in (TABLE_DIR, FIGURE_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    csv_path = locate_csv(DATA_DIR)
    raw = load_raw_data(csv_path)
    audit = dataset_audit(raw)
    metadata = dataset_metadata(raw, csv_path)
    clean_lines, cleaning_audit, excluded = clean_purchase_lines(raw)
    invoices = aggregate_invoices(clean_lines)
    observation_end = pd.Timestamp(invoices["InvoiceDate"].max())
    customers = build_customer_features(invoices, observation_end, GROSS_MARGIN_RATE)

    _write(audit, "dataset_audit.csv")
    _write(metadata, "dataset_metadata.csv")
    _write(cleaning_audit, "data_cleaning_audit.csv")
    _write(
        excluded["ExclusionReasons"].value_counts().rename_axis("ExclusionReasons").reset_index(name="Rows"),
        "excluded_row_reason_combinations.csv",
    )
    _write(invoices, "invoice_purchase_events.csv")
    _write(customers, "customer_rfmt.csv")

    calibration, calibration_end, observation_end = make_calibration_holdout(
        invoices, HOLDOUT_DAYS, GROSS_MARGIN_RATE
    )
    penalizer_sensitivity = bgnbd_penalizer_sensitivity(calibration)
    (
        calibration_bg,
        calibration_gg,
        bg_validation,
        bg_order_metrics,
        bg_revenue_metrics,
        calibration_monetary_diagnostic,
        calibration_warnings,
    ) = validate_bgnbd_gamma(
        calibration,
        HOLDOUT_DAYS,
        BGNBD_PENALIZER,
        GAMMA_GAMMA_PENALIZER,
    )

    full_bg, full_bg_warnings = fit_bgnbd(customers, BGNBD_PENALIZER)
    scored = score_bgnbd(full_bg, customers)
    full_gg, eligible, full_gg_warnings = fit_gamma_gamma(
        customers, GAMMA_GAMMA_PENALIZER
    )
    scored = score_gamma_gamma(full_gg, scored)
    monetary_diagnostic = monetary_assumption_diagnostic(eligible)
    scored, monthly_clv = calculate_discounted_clv(
        full_bg,
        scored,
        GROSS_MARGIN_RATE,
        ANNUAL_DISCOUNT_RATE,
    )
    scored = add_value_tiers_and_actions(scored, RISK_THRESHOLD)

    snapshots, temporal_design = build_snapshot_dataset(
        invoices, clean_lines, HOLDOUT_DAYS
    )
    (
        lightgbm_model,
        lightgbm_validation,
        lightgbm_test,
        lightgbm_metrics,
        feature_importance,
    ) = train_lightgbm_benchmark(
        snapshots, MODEL_DIR / "lightgbm_90d_revenue.joblib"
    )

    holdout_predictions = lightgbm_test.merge(
        bg_validation[
            [
                "CustomerID",
                "actual_holdout_orders",
                "predicted_holdout_orders",
                "actual_holdout_revenue",
                "predicted_holdout_revenue_bgnbd_gg",
            ]
        ],
        on="CustomerID",
        how="left",
    )
    if not np.allclose(
        holdout_predictions["future_90d_revenue"],
        holdout_predictions["actual_holdout_revenue"],
    ):
        raise AssertionError("LightGBM and BG/NBD holdout targets are not aligned")

    model_metrics = pd.concat(
        [
            pd.DataFrame([bg_order_metrics, bg_revenue_metrics]),
            lightgbm_metrics,
        ],
        ignore_index=True,
    )
    concentration = concentration_metrics(scored)
    risk_sensitivity = risk_threshold_sensitivity(scored)
    cac_customer, cac_summary = ltv_cac_tables(scored)
    retention = retention_sensitivity(scored)
    finance = financial_summary(scored)
    parameters = model_parameters(full_bg, full_gg)

    final_columns = [
        "CustomerID",
        "historical_orders",
        "days_since_last_purchase",
        "frequency",
        "bgnbd_recency",
        "T",
        "historical_revenue",
        "mean_order_value",
        "probability_alive",
        "expected_orders_90d",
        "expected_orders_365d",
        "expected_monetary_value",
        "expected_revenue_12m",
        "expected_margin_12m",
        "discounted_clv_12m",
        "ltv_tier",
        "risk_tier",
        "recommended_action",
        "max_cac_3x",
        "max_cac_1_5x",
        "max_cac_6m_payback",
        "monetary_value_source",
    ]
    final_scores = scored[final_columns].sort_values(
        "discounted_clv_12m", ascending=False, ignore_index=True
    )
    numeric_final = final_scores.select_dtypes(include=[np.number])
    if not np.isfinite(numeric_final.to_numpy()).all():
        raise AssertionError("Final customer scores contain unexpected NaN/inf")
    if not final_scores["discounted_clv_12m"].ge(0).all():
        raise AssertionError("Final customer scores contain negative CLV")

    action_matrix = final_scores[
        [
            "CustomerID",
            "discounted_clv_12m",
            "ltv_tier",
            "probability_alive",
            "risk_tier",
            "recommended_action",
        ]
    ]
    top_customers = final_scores.head(50)

    recency_repeat = bg_validation.assign(
        RecencyBin=pd.qcut(
            bg_validation["days_since_last_purchase"],
            q=10,
            duplicates="drop",
        ),
        RepeatedInHoldout=bg_validation["actual_holdout_orders"].gt(0),
    )
    empirical_repeat = (
        recency_repeat.groupby("RecencyBin", observed=True)
        .agg(
            Customers=("CustomerID", "size"),
            MeanDaysSinceLastPurchase=("days_since_last_purchase", "mean"),
            Observed90dRepeatRate=("RepeatedInHoldout", "mean"),
        )
        .reset_index()
    )
    empirical_repeat["RecencyBin"] = empirical_repeat["RecencyBin"].astype(str)

    assumptions = pd.DataFrame(
        [
            ("Gross margin rate", GROSS_MARGIN_RATE, "COURSE / SCENARIO ASSUMPTION; cost is absent"),
            ("Annual discount rate", ANNUAL_DISCOUNT_RATE, "COURSE / SCENARIO ASSUMPTION"),
            ("Effective monthly discount rate", (1 + ANNUAL_DISCOUNT_RATE) ** (1 / 12) - 1, "Derived from annual assumption"),
            ("BG/NBD penalizer course baseline", COURSE_BGNBD_PENALIZER, "COURSE BASELINE; numerically invalid for long-horizon scoring on this dataset"),
            ("BG/NBD penalizer operational", BGNBD_PENALIZER, "MODEL IMPLEMENTATION — smallest tested positive penalty with finite 30–365 day predictions"),
            ("Gamma-Gamma penalizer", GAMMA_GAMMA_PENALIZER, "COURSE BASELINE"),
            ("Holdout days", HOLDOUT_DAYS, "TEMPORAL VALIDATION DESIGN"),
            ("Probability Alive risk threshold", RISK_THRESHOLD, "MODEL-DERIVED PROXY; not observed churn"),
            ("Observed CAC", np.nan, "UNAVAILABLE — never fabricated"),
        ],
        columns=["Assumption", "Value", "Classification"],
    )

    _write(bg_validation, "bgnbd_holdout_predictions.csv")
    _write(calibration_monetary_diagnostic, "calibration_gamma_gamma_diagnostic.csv")
    _write(monetary_diagnostic, "gamma_gamma_assumption_diagnostic.csv")
    _write(model_metrics, "model_validation_metrics.csv")
    _write(parameters, "model_parameters.csv")
    _write(penalizer_sensitivity, "bgnbd_penalizer_sensitivity.csv")
    _write(monthly_clv, "monthly_clv_runrate.csv")
    _write(final_scores, "customer_clv_scores.csv")
    _write(action_matrix, "customer_action_matrix.csv")
    _write(top_customers, "top_clv_customers.csv")
    _write(cac_customer, "ltv_cac_thresholds.csv")
    _write(cac_summary, "ltv_cac_summary_by_tier.csv")
    _write(retention, "retention_sensitivity.csv")
    _write(finance, "financial_run_rate_summary.csv")
    _write(concentration, "clv_concentration.csv")
    _write(risk_sensitivity, "risk_threshold_sensitivity.csv")
    _write(temporal_design, "temporal_snapshot_design.csv")
    _write(feature_importance, "lightgbm_feature_importance.csv")
    _write(lightgbm_validation, "lightgbm_validation_predictions.csv")
    _write(holdout_predictions, "holdout_revenue_predictions.csv")
    _write(empirical_repeat, "empirical_repeat_rate_by_recency.csv")
    _write(assumptions, "analysis_assumptions.csv")

    figures = generate_all_figures(
        invoices,
        scored,
        bg_validation,
        lightgbm_test,
        feature_importance,
        retention,
        FIGURE_DIR,
    )

    metric_lookup = model_metrics.set_index("Model")
    concentration_lookup = concentration.set_index("Metric")["Value"]
    retention_lookup = retention.set_index("OrderUplift")
    target_mask = scored["ltv_tier"].isin(["Whales", "High Value"]) & scored[
        "risk_tier"
    ].eq("High latent risk")
    summary = {
        "dataset": {
            "source_file": csv_path.name,
            "sha256": metadata.loc[metadata["Field"].eq("sha256"), "Value"].iloc[0],
            "raw_rows": len(raw),
            "raw_columns": raw.shape[1],
            "clean_rows": len(clean_lines),
            "excluded_rows": len(excluded),
            "invoice_events": len(invoices),
            "date_min": raw["InvoiceDate"].iloc[0],
            "observation_end": observation_end,
        },
        "customers": {
            "usable": len(scored),
            "one_time": int(scored["frequency"].eq(0).sum()),
            "repeat": int(scored["frequency"].gt(0).sum()),
            "one_time_share": float(scored["frequency"].eq(0).mean()),
            "high_ltv_high_risk": int(target_mask.sum()),
        },
        "validation": {
            "calibration_end": calibration_end,
            "holdout_end": observation_end,
            "bgnbd_order_mae": metric_lookup.loc["BG/NBD holdout orders", "MAE"],
            "bgnbd_order_rmse": metric_lookup.loc["BG/NBD holdout orders", "RMSE"],
            "bgnbd_gg_revenue_mae": metric_lookup.loc["BG/NBD + Gamma-Gamma holdout revenue", "MAE"],
            "bgnbd_gg_revenue_rmse": metric_lookup.loc["BG/NBD + Gamma-Gamma holdout revenue", "RMSE"],
            "bgnbd_gg_top_decile_lift": metric_lookup.loc["BG/NBD + Gamma-Gamma holdout revenue", "TopDecileLift"],
            "lightgbm_mae": metric_lookup.loc["LightGBM final test revenue", "MAE"],
            "lightgbm_rmse": metric_lookup.loc["LightGBM final test revenue", "RMSE"],
            "lightgbm_top_decile_lift": metric_lookup.loc["LightGBM final test revenue", "TopDecileLift"],
            "baseline_mae": metric_lookup.loc["Historic-rate baseline test revenue", "MAE"],
            "baseline_rmse": metric_lookup.loc["Historic-rate baseline test revenue", "RMSE"],
            "baseline_top_decile_lift": metric_lookup.loc["Historic-rate baseline test revenue", "TopDecileLift"],
        },
        "gamma_gamma": {
            "eligible_repeat_customers": len(eligible),
            "pearson": monetary_diagnostic.iloc[0]["Correlation"],
            "spearman": monetary_diagnostic.iloc[1]["Correlation"],
        },
        "financial": {
            "predicted_revenue_12m": scored["expected_revenue_12m"].sum(),
            "predicted_margin_12m": scored["expected_margin_12m"].sum(),
            "discounted_clv_12m": scored["discounted_clv_12m"].sum(),
            "top_10_clv_share": concentration_lookup["Predicted CLV share from top 10%"],
            "top_20_clv_share": concentration_lookup["Predicted CLV share from top 20%"],
            "customer_share_for_80_clv": concentration_lookup["Customer share required for 80% of predicted CLV"],
            "high_ltv_high_risk_clv": scored.loc[target_mask, "discounted_clv_12m"].sum(),
            "retention_5_margin": retention_lookup.loc[0.05, "DiscountedIncrementalMarginGBP"],
            "retention_10_margin": retention_lookup.loc[0.10, "DiscountedIncrementalMarginGBP"],
            "retention_15_margin": retention_lookup.loc[0.15, "DiscountedIncrementalMarginGBP"],
            "retention_10_break_even_per_customer": retention_lookup.loc[0.10, "BreakEvenSpendPerTargetCustomerGBP"],
        },
        "top_customers": top_customers.head(10)[
            ["CustomerID", "discounted_clv_12m", "probability_alive", "ltv_tier", "risk_tier"]
        ].to_dict(orient="records"),
        "warnings": {
            "calibration": calibration_warnings,
            "full_bgnbd": full_bg_warnings,
            "full_gamma_gamma": full_gg_warnings,
            "same_timestamp_repeat_customers": int(
                ((customers["frequency"] > 0) & customers["bgnbd_recency"].eq(0)).sum()
            ),
        },
        "artifacts": {
            "tables": len(list(TABLE_DIR.glob("*.csv"))),
            "figures": len(figures),
        },
    }
    summary = json.loads(json.dumps(summary, default=_json_value))
    summary_path = TABLE_DIR / "executive_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_full_analysis()
    print(json.dumps(result, indent=2))
