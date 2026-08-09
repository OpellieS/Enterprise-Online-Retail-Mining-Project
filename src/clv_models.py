"""Probabilistic purchase and monetary models plus validation metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _model_inputs(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Use a one-minute floor only for split invoices sharing the first timestamp."""
    frequency = frame["frequency"].astype(float)
    recency = frame["bgnbd_recency"].astype(float).copy()
    recency = recency.mask((frequency > 0) & (recency == 0), 1.0 / 1_440.0)
    tenure = frame["T"].astype(float).clip(lower=1.0 / 1_440.0)
    recency = np.minimum(recency, tenure)
    return frequency, recency, tenure


def fit_bgnbd(frame: pd.DataFrame, penalizer_coef: float = 0.01):
    """Fit the course-standard BG/NBD implementation."""
    from lifetimes import BetaGeoFitter

    frequency, recency, tenure = _model_inputs(frame)
    model = BetaGeoFitter(penalizer_coef=penalizer_coef)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(frequency, recency, tenure)
    params = model.params_.astype(float)
    if not np.isfinite(params).all() or not params.gt(0).all():
        raise RuntimeError(f"Invalid BG/NBD parameters: {params.to_dict()}")
    return model, [str(w.message) for w in caught]


def score_bgnbd(
    model,
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (30, 90, 180, 365),
) -> pd.DataFrame:
    """Add latent activity probability and expected transaction counts."""
    scored = frame.copy()
    frequency, recency, tenure = _model_inputs(scored)
    scored["probability_alive"] = model.conditional_probability_alive(
        frequency, recency, tenure
    )
    for days in horizons:
        scored[f"expected_orders_{days}d"] = model.conditional_expected_number_of_purchases_up_to_time(
            days, frequency, recency, tenure
        )
    prediction_cols = ["probability_alive"] + [f"expected_orders_{d}d" for d in horizons]
    if not np.isfinite(scored[prediction_cols].to_numpy()).all():
        raise AssertionError("BG/NBD produced NaN or infinite predictions")
    if not scored["probability_alive"].between(0, 1).all():
        raise AssertionError("Probability Alive is outside [0, 1]")
    if not scored[[f"expected_orders_{d}d" for d in horizons]].ge(0).all().all():
        raise AssertionError("BG/NBD produced negative transaction predictions")
    return scored


def fit_gamma_gamma(frame: pd.DataFrame, penalizer_coef: float = 0.01):
    """Fit Gamma-Gamma to repeat customers with positive repeat-order value."""
    from lifetimes import GammaGammaFitter

    eligible = frame.loc[(frame["frequency"] > 0) & (frame["monetary_value"] > 0)].copy()
    if eligible.empty:
        raise ValueError("No customers are eligible for Gamma-Gamma")
    model = GammaGammaFitter(penalizer_coef=penalizer_coef)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(eligible["frequency"], eligible["monetary_value"])
    params = model.params_.astype(float)
    if not np.isfinite(params).all() or not params.gt(0).all():
        raise RuntimeError(f"Invalid Gamma-Gamma parameters: {params.to_dict()}")
    if params.get("q", 0) <= 1:
        raise RuntimeError("Gamma-Gamma population mean is undefined because q <= 1")
    return model, eligible, [str(w.message) for w in caught]


def score_gamma_gamma(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Score repeat customers and use the model population mean for one-time buyers."""
    scored = frame.copy()
    scored["expected_monetary_value"] = model.conditional_expected_average_profit(
        scored["frequency"].astype(float), scored["monetary_value"].astype(float)
    )
    scored["monetary_value_source"] = np.where(
        scored["frequency"] > 0,
        "MODEL ESTIMATE — individual Gamma-Gamma shrinkage",
        "MODEL ESTIMATE — Gamma-Gamma population expectation for one-time buyer",
    )
    if not (
        np.isfinite(scored["expected_monetary_value"]).all()
        and scored["expected_monetary_value"].gt(0).all()
    ):
        raise AssertionError("Gamma-Gamma produced invalid monetary predictions")
    return scored


def monetary_assumption_diagnostic(eligible: pd.DataFrame) -> pd.DataFrame:
    """Report simple association diagnostics, not a proof of independence."""
    x = eligible["frequency"].astype(float)
    y = eligible["monetary_value"].astype(float)
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return pd.DataFrame(
        [
            {
                "Diagnostic": "Pearson frequency vs monetary value",
                "Correlation": pearson.statistic,
                "PValue": pearson.pvalue,
                "Interpretation": "Linear-association diagnostic only; does not prove independence.",
            },
            {
                "Diagnostic": "Spearman frequency vs monetary value",
                "Correlation": spearman.statistic,
                "PValue": spearman.pvalue,
                "Interpretation": "Monotonic-association diagnostic only; does not prove independence.",
            },
        ]
    )


def regression_metrics(actual, predicted, model_name: str) -> dict[str, float | str]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.clip(np.asarray(predicted, dtype=float), 0, None)
    errors = actual_values - predicted_values
    count = max(1, int(np.ceil(len(actual_values) * 0.10)))
    top_idx = np.argsort(-predicted_values)[:count]
    total_actual = actual_values.sum()
    capture = actual_values[top_idx].sum() / total_actual if total_actual > 0 else np.nan
    return {
        "Model": model_name,
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "TopDecileSpendCapture": float(capture),
        "TopDecileLift": float(capture / 0.10),
        "CustomersEvaluated": len(actual_values),
    }


def validate_bgnbd_gamma(
    calibration: pd.DataFrame,
    holdout_days: int = 90,
    bgnbd_penalizer: float = 0.01,
    gamma_penalizer: float = 0.01,
):
    """Fit on calibration history only and score observed holdout outcomes."""
    bg_model, bg_warnings = fit_bgnbd(calibration, bgnbd_penalizer)
    scored = score_bgnbd(bg_model, calibration, horizons=(holdout_days,))
    gg_model, eligible, gg_warnings = fit_gamma_gamma(calibration, gamma_penalizer)
    scored = score_gamma_gamma(gg_model, scored)
    scored["predicted_holdout_orders"] = scored[f"expected_orders_{holdout_days}d"]
    scored["predicted_holdout_revenue_bgnbd_gg"] = (
        scored["predicted_holdout_orders"] * scored["expected_monetary_value"]
    )
    order_metrics = regression_metrics(
        scored["actual_holdout_orders"],
        scored["predicted_holdout_orders"],
        "BG/NBD holdout orders",
    )
    revenue_metrics = regression_metrics(
        scored["actual_holdout_revenue"],
        scored["predicted_holdout_revenue_bgnbd_gg"],
        "BG/NBD + Gamma-Gamma holdout revenue",
    )
    diagnostics = monetary_assumption_diagnostic(eligible)
    warnings_out = {"bgnbd": bg_warnings, "gamma_gamma": gg_warnings}
    return bg_model, gg_model, scored, order_metrics, revenue_metrics, diagnostics, warnings_out


def model_parameters(bg_model, gg_model) -> pd.DataFrame:
    rows = []
    for name, value in bg_model.params_.items():
        rows.append({"Model": "BG/NBD", "Parameter": name, "Value": float(value)})
    for name, value in gg_model.params_.items():
        rows.append({"Model": "Gamma-Gamma", "Parameter": name, "Value": float(value)})
    return pd.DataFrame(rows)

