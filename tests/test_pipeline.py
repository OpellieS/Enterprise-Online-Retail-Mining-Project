from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, TABLE_DIR
from src.data_processing import (
    aggregate_invoices,
    clean_purchase_lines,
    load_raw_data,
    locate_csv,
)
from src.feature_engineering import build_customer_features


def test_cleaning_and_invoice_aggregation_on_actual_source():
    raw = load_raw_data(locate_csv(DATA_DIR))
    clean, audit, _ = clean_purchase_lines(raw)
    invoices = aggregate_invoices(clean)

    assert clean["CustomerID"].notna().all()
    assert clean["Quantity"].gt(0).all()
    assert clean["UnitPrice"].gt(0).all()
    assert clean["LineRevenue"].gt(0).all()
    assert not clean["InvoiceNo"].str.startswith(("C", "c"), na=False).any()
    assert not invoices.duplicated(["InvoiceNo", "CustomerID"]).any()
    assert invoices["OrderRevenue"].gt(0).all()
    assert "Any exclusion rule (deduplicated union)" in set(audit["Rule"])


def test_rfmt_invariants_on_actual_source():
    raw = load_raw_data(locate_csv(DATA_DIR))
    clean, _, _ = clean_purchase_lines(raw)
    customers = build_customer_features(aggregate_invoices(clean))

    assert customers["frequency"].ge(0).all()
    assert customers["bgnbd_recency"].ge(0).all()
    assert customers["T"].ge(0).all()
    assert customers["bgnbd_recency"].le(customers["T"] + 1e-9).all()
    assert (customers["frequency"] == customers["historical_orders"] - 1).all()
    assert customers[["R_score", "F_score", "M_score"]].isin(range(1, 6)).all().all()


def test_generated_customer_scores_are_financially_valid():
    path = TABLE_DIR / "customer_clv_scores.csv"
    assert path.exists(), "Run run_analysis.py before the integration tests"
    scored = pd.read_csv(path, dtype={"CustomerID": "string"})
    numeric = scored.select_dtypes(include=[np.number])

    assert np.isfinite(numeric.to_numpy()).all()
    assert scored["probability_alive"].between(0, 1).all()
    assert scored["expected_orders_90d"].ge(0).all()
    assert scored["expected_orders_365d"].ge(0).all()
    assert scored["expected_monetary_value"].gt(0).all()
    assert scored["discounted_clv_12m"].ge(0).all()
    assert scored["max_cac_3x"].ge(0).all()
    assert scored["max_cac_6m_payback"].ge(0).all()


def test_supervised_temporal_windows_do_not_leak():
    path = TABLE_DIR / "temporal_snapshot_design.csv"
    assert path.exists(), "Run run_analysis.py before the integration tests"
    design = pd.read_csv(
        path,
        parse_dates=[
            "Cutoff",
            "FeatureWindowLatest",
            "TargetWindowStartExclusive",
            "TargetWindowEndInclusive",
        ],
    )
    assert (design["FeatureWindowLatest"] <= design["Cutoff"]).all()
    assert (design["TargetWindowStartExclusive"] == design["Cutoff"]).all()

    train_target_end = design.loc[
        design["Split"].eq("train"), "TargetWindowEndInclusive"
    ].max()
    validation_cutoff = design.loc[design["Split"].eq("validation"), "Cutoff"].iloc[0]
    validation_target_end = design.loc[
        design["Split"].eq("validation"), "TargetWindowEndInclusive"
    ].iloc[0]
    test_cutoff = design.loc[design["Split"].eq("test"), "Cutoff"].iloc[0]
    assert train_target_end <= validation_cutoff
    assert validation_target_end <= test_cutoff


def test_all_required_tables_exist():
    required = {
        "data_cleaning_audit.csv",
        "customer_rfmt.csv",
        "customer_clv_scores.csv",
        "customer_action_matrix.csv",
        "model_validation_metrics.csv",
        "top_clv_customers.csv",
        "ltv_cac_thresholds.csv",
        "retention_sensitivity.csv",
    }
    assert required.issubset({path.name for path in TABLE_DIR.glob("*.csv")})

