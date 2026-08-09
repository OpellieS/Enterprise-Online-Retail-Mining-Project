"""Load, audit, clean, and aggregate Online Retail line items."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}
DATE_FORMAT = "%m/%d/%y %H:%M"


def locate_csv(data_dir: Path) -> Path:
    """Return the only/most likely transaction CSV in the raw-data directory."""
    candidates = sorted(data_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {data_dir}")
    preferred = [p for p in candidates if "online" in p.name.lower() and "retail" in p.name.lower()]
    return preferred[0] if preferred else candidates[0]


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the source without altering values; tolerate the UCI text encoding."""
    try:
        frame = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin1")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typed_columns(raw: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    quantity = pd.to_numeric(raw["Quantity"], errors="coerce")
    price = pd.to_numeric(raw["UnitPrice"], errors="coerce")
    dates = pd.to_datetime(raw["InvoiceDate"], format=DATE_FORMAT, errors="coerce")
    return quantity, price, dates


def dataset_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a formal, long-form audit of the unmodified source data."""
    quantity, price, dates = _typed_columns(raw)
    invoice_text = raw["InvoiceNo"].astype("string")
    positive_qty = quantity[quantity > 0]
    positive_price = price[price > 0]
    qty_threshold = float(positive_qty.quantile(0.999))
    price_threshold = float(positive_price.quantile(0.999))

    records = [
        ("raw_row_count", len(raw), "OBSERVED DATA"),
        ("column_count", raw.shape[1], "OBSERVED DATA"),
        ("date_min", dates.min().isoformat(), "OBSERVED DATA"),
        ("date_max", dates.max().isoformat(), "OBSERVED DATA"),
        ("unique_invoices", raw["InvoiceNo"].nunique(dropna=True), "OBSERVED DATA"),
        ("unique_customers", raw["CustomerID"].nunique(dropna=True), "OBSERVED DATA"),
        ("unique_products", raw["StockCode"].nunique(dropna=True), "OBSERVED DATA"),
        ("countries", raw["Country"].nunique(dropna=True), "OBSERVED DATA"),
        ("missing_CustomerID", raw["CustomerID"].isna().sum(), "OBSERVED DATA"),
        ("missing_Description", raw["Description"].isna().sum(), "OBSERVED DATA"),
        ("missing_InvoiceNo", raw["InvoiceNo"].isna().sum(), "OBSERVED DATA"),
        ("missing_InvoiceDate", raw["InvoiceDate"].isna().sum(), "OBSERVED DATA"),
        ("malformed_dates", dates.isna().sum(), "OBSERVED DATA"),
        ("non_numeric_quantities", quantity.isna().sum() - raw["Quantity"].isna().sum(), "OBSERVED DATA"),
        ("non_numeric_prices", price.isna().sum() - raw["UnitPrice"].isna().sum(), "OBSERVED DATA"),
        ("zero_quantities", (quantity == 0).sum(), "OBSERVED DATA"),
        ("negative_quantities", (quantity < 0).sum(), "OBSERVED DATA"),
        ("zero_prices", (price == 0).sum(), "OBSERVED DATA"),
        ("negative_prices", (price < 0).sum(), "OBSERVED DATA"),
        ("cancelled_invoice_rows", invoice_text.str.startswith(("C", "c"), na=False).sum(), "OBSERVED DATA"),
        ("duplicate_rows", raw.duplicated().sum(), "OBSERVED DATA"),
        (
            f"quantity_above_positive_p99.9_{qty_threshold:g}",
            (quantity.abs() > qty_threshold).sum(),
            "DATA-DRIVEN DIAGNOSTIC; not automatically removed",
        ),
        (
            f"price_above_positive_p99.9_{price_threshold:g}",
            (price > price_threshold).sum(),
            "DATA-DRIVEN DIAGNOSTIC; not automatically removed",
        ),
    ]
    return pd.DataFrame(records, columns=["Metric", "Value", "Classification"])


def dataset_metadata(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Record source provenance, checksum, schema, and inferred dtypes."""
    rows = [
        ("source_file", path.name),
        ("source_path", str(path.resolve())),
        ("sha256", file_sha256(path)),
        ("rows", str(len(raw))),
        ("columns", str(raw.shape[1])),
        ("column_names", " | ".join(map(str, raw.columns))),
        ("pandas_dtypes", " | ".join(f"{c}:{t}" for c, t in raw.dtypes.items())),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def clean_purchase_lines(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return validated positive purchases, an overlapping-rule audit, and excluded rows."""
    work = raw.copy(deep=True)
    work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce")
    work["UnitPrice"] = pd.to_numeric(work["UnitPrice"], errors="coerce")
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], format=DATE_FORMAT, errors="coerce")
    work["InvoiceNo"] = work["InvoiceNo"].astype("string").str.strip()
    customer_numeric = pd.to_numeric(work["CustomerID"], errors="coerce")

    rules = {
        "Missing CustomerID": customer_numeric.isna(),
        "Invalid CustomerID": customer_numeric.notna()
        & ((customer_numeric <= 0) | ~np.isclose(customer_numeric % 1, 0)),
        "Missing/blank InvoiceNo": work["InvoiceNo"].isna() | work["InvoiceNo"].eq(""),
        "Cancellation invoice (C/c prefix)": work["InvoiceNo"].str.startswith(("C", "c"), na=False),
        "Malformed/missing InvoiceDate": work["InvoiceDate"].isna(),
        "Non-positive or non-numeric Quantity": work["Quantity"].isna() | work["Quantity"].le(0),
        "Non-positive or non-numeric UnitPrice": work["UnitPrice"].isna() | work["UnitPrice"].le(0),
        "Exact duplicate line": raw.duplicated(keep="first"),
    }
    meanings = {
        "Missing CustomerID": "Cannot link the purchase event to a customer.",
        "Invalid CustomerID": "Customer identifier is non-positive or non-integral.",
        "Missing/blank InvoiceNo": "Cannot construct an invoice-level purchase event.",
        "Cancellation invoice (C/c prefix)": "Reversal/cancellation, not a positive purchase event.",
        "Malformed/missing InvoiceDate": "Cannot place the event on the customer timeline.",
        "Non-positive or non-numeric Quantity": "Return, reversal, or invalid unit quantity.",
        "Non-positive or non-numeric UnitPrice": "No valid positive economic value.",
        "Exact duplicate line": "Repeated source record; retain the first occurrence only.",
    }
    excluded_mask = pd.Series(False, index=work.index)
    for mask in rules.values():
        excluded_mask |= mask

    audit_rows = [
        {
            "Rule": rule,
            "RowsAffected": int(mask.sum()),
            "PercentOfRawRows": float(mask.mean() * 100),
            "BusinessMeaning": meanings[rule],
        }
        for rule, mask in rules.items()
    ]
    audit_rows.append(
        {
            "Rule": "Any exclusion rule (deduplicated union)",
            "RowsAffected": int(excluded_mask.sum()),
            "PercentOfRawRows": float(excluded_mask.mean() * 100),
            "BusinessMeaning": "Rows excluded from positive-purchase CLV modeling.",
        }
    )
    cleaning_audit = pd.DataFrame(audit_rows)

    reason_frame = pd.DataFrame(rules)
    excluded = work.loc[excluded_mask].copy()
    excluded["ExclusionReasons"] = reason_frame.loc[excluded_mask].apply(
        lambda row: " | ".join(row.index[row].tolist()), axis=1
    )

    clean = work.loc[~excluded_mask].copy()
    clean["CustomerID"] = customer_numeric.loc[~excluded_mask].astype("int64").astype("string")
    clean["StockCode"] = clean["StockCode"].astype("string").str.strip()
    clean["Country"] = clean["Country"].astype("string").str.strip()
    clean["LineRevenue"] = clean["Quantity"] * clean["UnitPrice"]
    if not clean["LineRevenue"].gt(0).all():
        raise AssertionError("Cleaning left non-positive LineRevenue rows")
    return clean, cleaning_audit, excluded


def aggregate_invoices(clean_lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SKU rows to the required InvoiceNo × CustomerID purchase event."""
    invoices = (
        clean_lines.groupby(["InvoiceNo", "CustomerID"], as_index=False, observed=True)
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            OrderRevenue=("LineRevenue", "sum"),
            TotalUnits=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
            InvoiceLines=("StockCode", "size"),
            Country=("Country", "first"),
        )
        .sort_values(["InvoiceDate", "InvoiceNo"], ignore_index=True)
    )
    if invoices.duplicated(["InvoiceNo", "CustomerID"]).any():
        raise AssertionError("Invoice aggregation did not produce unique purchase events")
    if not invoices["OrderRevenue"].gt(0).all():
        raise AssertionError("Invoice table contains non-positive orders")
    return invoices

