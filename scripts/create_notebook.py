"""Build the narrative notebook from verified pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "outputs/tables/executive_summary.json").read_text())
d = SUMMARY["dataset"]
c = SUMMARY["customers"]
v = SUMMARY["validation"]
g = SUMMARY["gamma_gamma"]
f = SUMMARY["financial"]
top_ids = ", ".join(item["CustomerID"] for item in SUMMARY["top_customers"][:5])


def money(value: float) -> str:
    return f"£{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def python(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = []


def section(title: str, body: str, code_text: str | None = None) -> None:
    cells.append(markdown(f"## {title}\n\n{body}"))
    if code_text:
        cells.append(python(code_text))


cells.append(
    markdown(
        f"""
# Enterprise Online Retail CLV Modeling

## 1. Executive Summary

The validated dataset supports **{c['usable']:,} customers** and **{d['invoice_events']:,} invoice-level purchase events**. The model estimates **{money(f['predicted_revenue_12m'])}** of 12-month revenue, **{money(f['predicted_margin_12m'])}** of gross margin under the explicit 30% course assumption, and **{money(f['discounted_clv_12m'])}** of discounted CLV.

The top 20% contributes **{pct(f['top_20_clv_share'])}**, so this is not an 80/20 dataset. At `p_alive < 0.70`, **{c['high_ltv_high_risk']} high-LTV customers** enter priority retention. A hypothetical 10% order uplift creates **{money(f['retention_10_margin'])}** discounted incremental margin and supports at most **{money(f['retention_10_break_even_per_customer'])} per targeted customer**. This is scenario analysis, not causal evidence.

LightGBM improves MAE versus a historic-rate baseline but worsens RMSE. BG/NBD + Gamma-Gamma is slightly stronger on the final holdout and produces **{v['bgnbd_gg_top_decile_lift']:.2f}×** top-decile lift. The five highest predicted-CLV CustomerIDs are **{top_ids}**.
"""
    )
)

cells.append(
    python(
        """
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from IPython.display import Image, display
from run_analysis import run_full_analysis

pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# Recompute all outputs: this notebook is executable, not a stale template.
summary = run_full_analysis()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

def show_table(name, rows=None):
    frame = pd.read_csv(TABLE_DIR / name)
    display(frame if rows is None else frame.head(rows))
    return frame

def show_figures(*names):
    for name in names:
        alt = name.removesuffix(".png").replace("_", " ")
        display(Image(filename=str(FIGURE_DIR / name), width=900, alt=alt))

print(
    f"Pipeline complete: {summary['dataset']['raw_rows']:,} raw rows; "
    f"{summary['customers']['usable']:,} scored customers; "
    f"{summary['artifacts']['tables']} tables; {summary['artifacts']['figures']} figures."
)
"""
    )
)

section(
    "2. Business Problem and Course Framework",
    """
This is Customer Lifetime Value analysis—not association-rule mining. The decision chain is **Customer Acquisition → Customer Retention → Customer Expansion**. Revenue is observed; cost, margin, CAC, contractual churn, and campaign treatment are absent. Assumptions and model estimates are labelled separately from observed data.

The primary scenario uses 30% gross margin and 10% annual discount. The effective monthly rate is `(1 + 0.10)^(1/12) - 1`. Holdout validation is chronological over 90 days.
""",
    'show_table("analysis_assumptions.csv")',
)

section(
    "3. Dataset and Data Audit",
    f"""
The local CSV is the source of truth and remains unchanged outside Git. It contains **{d['raw_rows']:,} rows and {d['raw_columns']} columns**. Extreme-value flags are diagnostics only; positive extreme purchases are not silently removed.
""",
    'show_table("dataset_metadata.csv"); show_table("dataset_audit.csv")',
)

section(
    "4. Data Cleaning",
    f"""
Positive-purchase hygiene requires a valid customer, invoice and date, `Quantity > 0`, `UnitPrice > 0`, no C/c cancellation prefix, and first occurrence of an exact duplicate. Rules overlap; their counts must not be added. The union excludes **{d['excluded_rows']:,} rows**, leaving **{d['clean_rows']:,} line items**. SKU rows are aggregated to **{d['invoice_events']:,} `InvoiceNo × CustomerID` events**.
""",
    'show_table("data_cleaning_audit.csv"); show_table("invoice_purchase_events.csv", 5)',
)

section(
    "5. Exploratory Data Analysis",
    """
The plots cover monthly revenue, active customers, recency, frequency, customer behavior, and empirical holdout repeat rate. Log transformations are labelled. The repeat-rate curve declines gradually; it does not support inventing a single defection cliff.
""",
    """
show_figures(
    "01_monthly_revenue.png", "02_monthly_active_customers.png",
    "03_recency_distribution.png", "04_frequency_distribution.png",
    "05_recency_frequency_behavior.png", "15_empirical_repeat_rate_by_recency.png"
)
show_table("empirical_repeat_rate_by_recency.csv")
""",
)

section(
    "6. Historic CLV and RFM Baselines",
    """
Historic annualized margin is `(Historical Revenue × 30% / max(Tenure Days, 1)) × 365`; 30% is assumed. Tie-safe percentile ranks produce 1–5 R, F and M scores. RFM ranks customers but is not a probabilistic future valuation.
""",
    """
rfmt = show_table("customer_rfmt.csv")
display(rfmt[["historic_clv_annual_margin", "R_score", "F_score", "M_score"]].describe())
display(rfmt[["CustomerID", "historic_clv_annual_margin", "RFM_score"]].head(10))
""",
)

section(
    "7. RFM-T Feature Engineering",
    f"""
`frequency` is invoice events minus one; `bgnbd_recency` is first-to-last elapsed time; `T` is first purchase to observation end; `days_since_last_purchase` is marketing recency and is not mixed with the BG/NBD definition. All **{c['usable']:,} customers** satisfy `frequency ≥ 0` and `0 ≤ recency ≤ T`. One-time buyers are **{c['one_time']:,} ({pct(c['one_time_share'])})**; repeat buyers are **{c['repeat']:,}**.
""",
    """
display(rfmt[["frequency", "bgnbd_recency", "T", "days_since_last_purchase", "monetary_value"]].describe())
assert (rfmt["bgnbd_recency"] <= rfmt["T"] + 1e-9).all()
print("RFM-T invariants: PASS")
""",
)

section(
    "8. BG/NBD Transaction Model",
    """
The required `penalizer_coef=0.01` is tested. It drives `a+b < 1`, and `lifetimes 0.11.3` then returns NaN for some frequency-zero long-horizon predictions. A 0.0001 operational penalty is the smallest tested positive value with finite 30–365 day predictions. This deviation is numerical—not metric-driven. Four same-timestamp repeat customers use a one-minute scoring floor while reported invoice frequency remains unchanged.
""",
    'show_table("model_parameters.csv"); show_table("bgnbd_penalizer_sensitivity.csv")',
)

section(
    "9. Gamma-Gamma Monetary Model",
    f"""
The repeat-customer diagnostic has Pearson **{g['pearson']:.3f}** and Spearman **{g['spearman']:.3f}** correlation. Linear dependence is weak, but mild monotonic dependence remains; neither statistic proves independence. At the required 0.01 penalty, repeat-customer predictions are finite but `q ≤ 1` makes the unconditional population mean undefined. One-time buyers therefore receive an explicit observed cohort-mean invoice fallback, not fabricated individual behavior.
""",
    'show_table("gamma_gamma_assumption_diagnostic.csv")',
)

section(
    "10. Temporal Calibration / Holdout Validation",
    f"""
Calibration ends **{v['calibration_end'][:10]}** and holdout ends **{v['holdout_end'][:10]}**. BG/NBD order MAE is **{v['bgnbd_order_mae']:.3f}** and RMSE **{v['bgnbd_order_rmse']:.3f}**. No random row split is used.
""",
    'show_table("temporal_snapshot_design.csv"); show_table("model_validation_metrics.csv"); show_figures("06_bgnbd_holdout_validation.png")',
)

section(
    "11. 12-Month Discounted CLV",
    f"""
Each month's expected orders are the difference between consecutive cumulative BG/NBD predictions. Revenue is multiplied by the assumed 30% margin and discounted monthly. Totals are **{money(f['predicted_revenue_12m'])} revenue**, **{money(f['predicted_margin_12m'])} margin**, and **{money(f['discounted_clv_12m'])} discounted CLV**.
""",
    'show_table("monthly_clv_runrate.csv")',
)

section(
    "12. Customer Value and Latent-Risk Matrix",
    f"""
Whales are the top 10%, High Value the next 10%, Mid Value the next 40%, and Long Tail the bottom 40%. Probability Alive is latent activity—not observed churn. Thresholds 0.30 and 0.50 identify no high-value risk; 0.70 flags only 42 customers overall and **{c['high_ltv_high_risk']} high-LTV customers**, so the operational scope remains conservative.
""",
    'show_table("risk_threshold_sensitivity.csv"); show_figures("07_probability_alive_distribution.png", "08_predicted_clv_distribution.png", "09_clv_pareto_curve.png", "10_ltv_risk_matrix.png")',
)

section(
    "13. LightGBM 90-Day Spend Benchmark",
    """
Every feature is recomputed at or before its cutoff; the target is strictly `(cutoff, cutoff + 90 days]`. Training target windows end before validation features, and validation targets end before final-test features. Tweedie regression is an implementation extension suited to a non-negative, zero-heavy target.
""",
    'show_figures("11_lightgbm_actual_vs_predicted.png", "12_lightgbm_feature_importance.png"); show_table("lightgbm_feature_importance.csv", 15)',
)

section(
    "14. Model Performance",
    f"""
LightGBM final-test MAE is **{money(v['lightgbm_mae'])}**, better than baseline **{money(v['baseline_mae'])}**. Its RMSE is **{money(v['lightgbm_rmse'])}**, worse than baseline **{money(v['baseline_rmse'])}**, so it does not dominate. Top-decile lift is **{v['lightgbm_top_decile_lift']:.2f}×** for LightGBM and **{v['bgnbd_gg_top_decile_lift']:.2f}×** for BG/NBD + Gamma-Gamma; the 3.5× lecture benchmark was not forced.
""",
    'show_table("model_validation_metrics.csv"); show_figures("13_top_decile_capture_curves.png")',
)

section(
    "15. Financial Run-Rate Impact",
    f"""
The model-estimated annual revenue run-rate is **{money(f['predicted_revenue_12m'])}**. Applying the 30% course margin assumption gives **{money(f['predicted_margin_12m'])}**; discounting gives **{money(f['discounted_clv_12m'])}**. The high-LTV/high-risk cell contains **{money(f['high_ltv_high_risk_clv'])}** of discounted CLV.
""",
    'show_table("financial_run_rate_summary.csv"); show_table("clv_concentration.csv")',
)

section(
    "16. LTV/CAC Capital Allocation",
    """
Observed CAC is unavailable and never fabricated. Customer ceilings are `CLV/3` for the target 3× ratio, `CLV/1.5` as the kill-level threshold, and discounted expected margin in months 1–6 for payback. These are allowable maxima, not historical CAC.
""",
    'show_table("ltv_cac_summary_by_tier.csv"); show_table("ltv_cac_thresholds.csv", 10)',
)

section(
    "17. Retention Scenario Analysis",
    f"""
**SCENARIO ANALYSIS — NOT CAUSAL ESTIMATES.** Targeting only {c['high_ltv_high_risk']} high-LTV/high-risk customers, +5%, +10%, and +15% hypothetical order uplift yields **{money(f['retention_5_margin'])}**, **{money(f['retention_10_margin'])}**, and **{money(f['retention_15_margin'])}** discounted incremental margin.
""",
    'show_table("retention_sensitivity.csv"); show_figures("14_retention_sensitivity.png")',
)

section(
    "18. Type 1 vs Type 2 Recommendations",
    """
**Type 2—test first:** targeted email, capped digital voucher, app notification, temporary VIP perk, and A/B-tested messaging. **Type 1—require stronger evidence:** permanent VIP infrastructure, long-term SLA changes, physical process redesign, or durable capital reallocation. Observational CLV alone does not justify irreversible spending.
""",
)

section(
    "19. Limitations",
    """
The history spans roughly one year. Gross margin, cost, CAC, marketing exposure, and intervention treatment are absent. Probability Alive is latent. Gamma-Gamma shows mild monotonic dependence and needs an observed fallback for one-time buyers. The course BG/NBD penalty is numerically invalid for some long-horizon scores. LightGBM improves MAE but not RMSE. Retention uplift is hypothetical and requires an experiment for causal validation.
""",
)

section(
    "20. Final Executive Recommendations",
    f"""
1. Usable customers: **{c['usable']:,}**.
2. One-time vs repeat: **{c['one_time']:,} ({pct(c['one_time_share'])})** vs **{c['repeat']:,} ({pct(1-c['one_time_share'])})**.
3. Highest CLV CustomerIDs: **{top_ids}**.
4. High-value / low Probability Alive: **{c['high_ltv_high_risk']} customers** at 0.70; IDs/actions are in `customer_action_matrix.csv`.
5. BG/NBD order holdout: **MAE {v['bgnbd_order_mae']:.3f}; RMSE {v['bgnbd_order_rmse']:.3f}**.
6. Gamma-Gamma assumption: Pearson **{g['pearson']:.3f}**, Spearman **{g['spearman']:.3f}**; independence is not proven.
7. Predicted revenue: **{money(f['predicted_revenue_12m'])}**.
8. Predicted gross margin: **{money(f['predicted_margin_12m'])}** under 30% assumed margin.
9. Discounted CLV: **{money(f['discounted_clv_12m'])}**.
10. Top-10% / top-20% CLV share: **{pct(f['top_10_clv_share'])} / {pct(f['top_20_clv_share'])}**.
11. 80/20: **No**; **{pct(f['customer_share_for_80_clv'])}** of customers are needed for 80% of CLV.
12. Top-decile lift: **{v['bgnbd_gg_top_decile_lift']:.2f}× BG/GG; {v['lightgbm_top_decile_lift']:.2f}× LightGBM**.
13. LightGBM: better MAE, worse RMSE than baseline; no clean win.
14. CAC: use CLV/3, CLV/1.5, and six-month discounted margin ceilings in the customer table.
15. Fund retention for the **{c['high_ltv_high_risk']} high-LTV/high-risk customers** through capped Type 2 tests.
16. Use no paid rescue for low-LTV/high-risk customers; keep communication organic.
17. Scenario margin at +5%/+10%/+15%: **{money(f['retention_5_margin'])} / {money(f['retention_10_margin'])} / {money(f['retention_15_margin'])}**.
18. Ten-percent scenario break-even spend: **{money(f['retention_10_break_even_per_customer'])} per target**.
19. Type 1: permanent infrastructure, SLA, physical-process, and durable capital changes.
20. Type 2: targeted digital offers, temporary perks, and randomized tests.

**Management action:** do not spread discounts evenly. Protect healthy whales with service benefits, test retention only in the small high-value/high-risk cell, avoid paid rescue of the low-value tail, and demand causal evidence before scaling permanent spend. The model prioritizes decisions; it is not a crystal ball wearing a tie.
""",
)

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
)
output = ROOT / "notebooks" / "Enterprise_Online_Retail_CLV_Report.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, output)
print(output)
