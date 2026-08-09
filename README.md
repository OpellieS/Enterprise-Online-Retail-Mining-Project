# Enterprise Online Retail Mining Project

Customer Lifetime Value modeling and financial run-rate analysis on the UCI Online Retail dataset. The project converts 541,909 SKU-level rows into validated invoice events, estimates non-contractual customer value with BG/NBD and Gamma-Gamma, benchmarks future revenue with chronological LightGBM snapshots, and translates the scores into acquisition, retention, and expansion decisions.

This is a CLV project. It deliberately does not turn the assignment into association-rule mining.

## Executive results

| Result | Value |
|---|---:|
| Raw rows | 541,909 |
| Positive-purchase rows after cleaning | 392,692 |
| Invoice-level purchase events | 18,532 |
| Usable customers | 4,338 |
| One-time / repeat customers | 1,493 (34.4%) / 2,845 (65.6%) |
| Predicted 12-month revenue | £10,369,750 |
| Predicted 12-month gross margin | £3,110,925 |
| Discounted 12-month CLV | £2,955,975 |
| Top-10% / top-20% CLV share | 54.8% / 67.7% |
| Customer share needed for 80% of CLV | 35.4% |
| High-LTV/high-latent-risk customers | 3 |
| BG/NBD 90-day order MAE / RMSE | 1.063 / 2.104 invoices |
| BG/NBD + Gamma-Gamma top-decile lift | 5.23× |
| LightGBM top-decile lift | 5.16× |

The dataset does not exhibit a strict 80/20 pattern. LightGBM improves revenue MAE over the historic-rate baseline (£652 vs £721) but worsens RMSE (£4,363 vs £4,102), so it is not an across-the-board replacement for the simpler models.

## Business framework

The analytical story follows:

**Customer Acquisition → Customer Retention → Customer Expansion**

The model supports two economic goals:

1. Increase revenue by focusing expansion and retention on customers with high expected value.
2. Reduce cost by avoiding blanket discounts, low-value paid rescue, and acquisition spend that cannot clear the required LTV/CAC hurdle.

Recommended actions:

- Preserve healthy whales with service and loyalty benefits rather than unnecessary discounts.
- Test capped, reversible retention offers for the three high-LTV/high-risk customers.
- Use low-cost automated re-engagement for mid-value/high-risk customers.
- Use no paid retention for low-value/high-risk customers.
- Require experimental evidence before committing permanent infrastructure or long-term capital.

## Dataset and cleaning

The local file is the primary source of truth:

- Source file: `data/raw/Online_Retail.csv`
- SHA256: `5c1b5517919301b1da060b3dc486614f487da43515a9b2a52709e2b04d5da575`
- Date range: 1 December 2010 to 9 December 2011
- Raw customers: 4,372
- Raw invoices: 25,900
- Missing CustomerID: 135,080 rows
- Exact duplicates: 5,268 rows
- Cancellation rows: 9,288
- Negative quantity rows: 10,624
- Zero/negative price rows: 2,517

Core CLV purchase hygiene requires:

```text
CustomerID is valid
Quantity > 0
UnitPrice > 0
InvoiceNo is valid and does not start with C/c
InvoiceDate is valid
Exact duplicate line is removed after its first occurrence
```

Cleaning rules overlap. The deduplicated exclusion union is 149,217 rows, not the sum of individual rule counts. Positive lines are aggregated to `InvoiceNo × CustomerID`; SKU rows are never treated as independent purchases.

## Methodology

### Historic CLV and RFM

Historic annualized margin provides a conventional baseline:

```text
Historic CLV = Historical Revenue × Gross Margin Rate / max(Tenure Days, 1) × 365
```

R, F, and M use tie-safe percentile scoring from 1 to 5. RFM ranks customers but does not estimate a probabilistic future monetary value.

### BG/NBD

The customer timeline uses:

- `frequency`: invoice events minus one
- `bgnbd_recency`: first-to-last purchase time
- `T`: first purchase to observation end
- `days_since_last_purchase`: marketing recency, kept separate from BG/NBD recency

The required `BetaGeoFitter(penalizer_coef=0.01)` is fitted and reported as the course baseline. On this dataset it yields `a + b < 1`, and `lifetimes 0.11.3` returns NaN for some frequency-zero long-horizon predictions. Penalizer sensitivity demonstrates that 0.0001 is the smallest tested positive value with finite 30-, 90-, 180-, and 365-day scores. The operational model therefore uses 0.0001. This deviation is numerical, not chosen to improve validation metrics.

Four customers have multiple invoices at the exact first timestamp. A one-minute floor is used only inside BG/NBD scoring to avoid zero elapsed time with positive frequency; reported invoice counts and timestamps are not rewritten.

`probability_alive` is a latent activity probability, not observed churn. Sensitivity at 0.30, 0.50, and 0.70 shows that 0.50 produces no high-value risk cell. The operational 0.70 threshold still flags fewer than 1% of customers and identifies three high-value candidates.

### Gamma-Gamma

Gamma-Gamma is fitted at the required 0.01 penalty to customers with positive repeat frequency and monetary value. Frequency/value diagnostics are:

- Pearson: 0.011, p = 0.570
- Spearman: 0.209, p < 0.001

The Pearson result does not prove independence; the Spearman result indicates mild monotonic dependence. At the required penalty, `q ≤ 1`, so the unconditional Gamma-Gamma population mean is undefined. Repeat-customer estimates remain finite. One-time buyers receive the observed cohort mean invoice value as a clearly labelled fallback rather than fabricated individual monetary behavior.

### Month-by-month CLV

For month `m = 1…12`:

```text
Incremental Orders_m = Cumulative Expected Orders_m - Cumulative Expected Orders_(m-1)
Revenue_m = Incremental Orders_m × Expected Monetary Value
Margin_m = Revenue_m × 30%
Discounted Margin_m = Margin_m / (1 + Effective Monthly Discount Rate)^m
CLV_12M = sum(Discounted Margin_m)
```

The effective monthly rate is derived from the 10% annual scenario: `(1 + 0.10)^(1/12) - 1`.

### Temporal validation and LightGBM

Transactions are never randomly split. The final calibration cutoff is 10 September 2011 and the holdout ends 9 December 2011. LightGBM snapshots obey:

```text
features <= cutoff
target = revenue in (cutoff, cutoff + 90 days]
training target end <= validation cutoff
validation target end <= final-test cutoff
```

The Tweedie LightGBM objective is an implementation extension for a non-negative, zero-heavy, skewed target.

| Model | MAE | RMSE | Top-decile lift |
|---|---:|---:|---:|
| BG/NBD holdout orders | 1.063 | 2.104 | 3.95× |
| BG/NBD + Gamma-Gamma revenue | £636 | £4,125 | 5.23× |
| Historic-rate revenue baseline | £721 | £4,102 | 4.92× |
| LightGBM final-test revenue | £652 | £4,363 | 5.16× |

## Financial assumptions and capital allocation

| Input | Classification |
|---|---|
| Revenue and transaction history | Observed data |
| BG/NBD, Gamma-Gamma, and LightGBM outputs | Model estimates |
| 30% gross margin | Course/scenario assumption |
| 10% annual discount | Course/scenario assumption |
| CAC | Unavailable; never fabricated |
| +5%, +10%, +15% retention uplift | Scenario assumption; not causal |

Allowable CAC is derived rather than observed:

```text
Max CAC for LTV/CAC = 3.0  → discounted CLV / 3
Kill-level CAC at 1.5      → discounted CLV / 1.5
Six-month max CAC          → discounted expected margin in months 1–6
```

For the three high-LTV/high-risk customers, hypothetical +5%, +10%, and +15% order uplift yields £1,030, £2,060, and £3,089 of incremental discounted margin. At +10%, break-even retention spend is £687 per targeted customer. These are sensitivity outputs, not campaign-effect estimates.

## Type 1 and Type 2 decisions

- **Type 2, test first:** email, app notification, capped voucher, temporary VIP perk, and randomized retention offer.
- **Type 1, require stronger validation:** permanent VIP fulfillment infrastructure, long-term SLA changes, physical process redesign, and durable capital reallocation.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── run_analysis.py
├── data/
│   ├── README.md
│   └── raw/                         # CSV ignored by Git
├── notebooks/
│   └── Enterprise_Online_Retail_CLV_Report.ipynb
├── reports/
│   └── Enterprise_Online_Retail_CLV_Report.html
├── scripts/
│   └── create_notebook.py
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── clv_models.py
│   ├── supervised_model.py
│   ├── financial_analysis.py
│   └── visualization.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── model_artifacts/
└── tests/
    └── test_pipeline.py
```

## Setup and execution

Python 3.13 is verified with the pinned environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Apple Silicon macOS, the LightGBM wheel requires OpenMP:

```bash
brew install libomp
```

Download the source only if it is absent:

```bash
curl -L \
  https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv \
  -o data/raw/Online_Retail.csv
```

Run the pipeline and tests:

```bash
.venv/bin/python run_analysis.py
.venv/bin/pytest -q
```

Rebuild and execute the notebook:

```bash
.venv/bin/python scripts/create_notebook.py
.venv/bin/jupyter nbconvert \
  --to notebook --execute notebooks/Enterprise_Online_Retail_CLV_Report.ipynb \
  --inplace --ExecutePreprocessor.timeout=1800
```

## Main artifacts

- Executed notebook: `notebooks/Enterprise_Online_Retail_CLV_Report.ipynb`
- HTML report: `reports/Enterprise_Online_Retail_CLV_Report.html`
- Final customer scores: `outputs/tables/customer_clv_scores.csv`
- Customer action matrix: `outputs/tables/customer_action_matrix.csv`
- Model validation: `outputs/tables/model_validation_metrics.csv`
- CAC thresholds: `outputs/tables/ltv_cac_thresholds.csv`
- Retention sensitivity: `outputs/tables/retention_sensitivity.csv`
- Figures: `outputs/figures/`

## Limitations

- The data span roughly one year; longer-horizon stationarity is untested.
- Gross margin, product cost, CAC, marketing exposure, and experimental treatment are absent.
- Probability Alive is latent, not contractual churn.
- Gamma-Gamma has mild monotonic frequency/value dependence and no defined one-time-buyer population mean at the course penalty.
- The course BG/NBD penalty is numerically invalid for some long-horizon one-time-customer scores; the operational deviation is documented.
- LightGBM improves MAE but worsens RMSE.
- Financial uplift is scenario analysis. Causal campaign value requires a randomized or strong quasi-experimental design.
