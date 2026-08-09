"""Explicit model and financial assumptions used across the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "model_artifacts"

FALLBACK_DATA_URL = (
    "https://raw.githubusercontent.com/guipsamora/pandas_exercises/"
    "master/07_Visualization/Online_Retail/Online_Retail.csv"
)

# COURSE / SCENARIO ASSUMPTIONS — not observed in the transaction data.
GROSS_MARGIN_RATE = 0.30
ANNUAL_DISCOUNT_RATE = 0.10
HOLDOUT_DAYS = 90
BGNBD_PENALIZER = 0.01
GAMMA_GAMMA_PENALIZER = 0.01
RISK_THRESHOLD = 0.50
RANDOM_STATE = 42

