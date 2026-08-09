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
# The required course baseline is tested and reported. On this dataset it drives
# a+b below 1 and the lifetimes 0.11.3 conditional-expectation implementation
# returns NaN for one-time buyers at longer horizons. A minimal positive penalty
# is therefore used operationally; see bgnbd_penalizer_sensitivity.csv.
COURSE_BGNBD_PENALIZER = 0.01
BGNBD_PENALIZER = 0.0001
GAMMA_GAMMA_PENALIZER = 0.01
# The course starting point (0.50) flags only 15/4,338 customers and no
# high-value customers. 0.70 remains conservative (~1% flagged) while making
# the high-value risk quadrant operational; 0.30/0.50/0.70 remain in sensitivity.
RISK_THRESHOLD = 0.70
RANDOM_STATE = 42
