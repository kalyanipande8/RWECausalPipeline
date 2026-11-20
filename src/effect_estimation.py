"""Five real confounding-adjustment methods for estimating the treatment
effect on the survival outcome, plus survival-curve comparison via
lifelines (Kaplan-Meier + Cox proportional hazards)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LinearRegression

from .propensity import estimate_propensity, nn_match, iptw_weights, stratify, balance_table


def unadjusted_effect(df: pd.DataFrame) -> float:
    cph = CoxPHFitter()
    cph.fit(df[["time", "event", "treatment"]], duration_col="time", event_col="event")
    return float(cph.params_["treatment"])


def psm_effect(df: pd.DataFrame, covariates: list) -> tuple[float, pd.DataFrame]:
    ps = estimate_propensity(df, covariates)
    matched = nn_match(df, ps)
    cph = CoxPHFitter()
    cph.fit(matched[["time", "event", "treatment"]], duration_col="time", event_col="event")
    return float(cph.params_["treatment"]), matched


def iptw_effect(df: pd.DataFrame, covariates: list) -> tuple[float, np.ndarray]:
    ps = estimate_propensity(df, covariates)
    w = iptw_weights(df["treatment"].values, ps)
    cph = CoxPHFitter()
    d = df[["time", "event", "treatment"]].copy()
    d["weight"] = w
    cph.fit(d, duration_col="time", event_col="event", weights_col="weight", robust=True)
    return float(cph.params_["treatment"]), w


def stratification_effect(df: pd.DataFrame, covariates: list, n_strata: int = 5) -> float:
    ps = estimate_propensity(df, covariates)
    strata = stratify(df, ps, n_strata)
    d = df.copy()
    d["stratum"] = strata
    cph = CoxPHFitter()
    cph.fit(d[["time", "event", "treatment", "stratum"]], duration_col="time", event_col="event",
            strata=["stratum"])
    return float(cph.params_["treatment"])


def aipw_effect(df: pd.DataFrame, covariates: list) -> float:
    """Doubly-robust AIPW estimator on a linear outcome-risk proxy
    (log of (time+eps) as a continuous proxy outcome, since lifelines
    doesn't expose a direct AIPW API)."""
    ps = estimate_propensity(df, covariates)
    ps = np.clip(ps, 0.02, 0.98)
    y = np.log(df["time"].values + 1e-3)
    t = df["treatment"].values
    X = df[covariates].values

    m1 = LinearRegression().fit(X[t == 1], y[t == 1])
    m0 = LinearRegression().fit(X[t == 0], y[t == 0])
    mu1 = m1.predict(X)
    mu0 = m0.predict(X)

    aipw = (
        mu1 - mu0
        + t * (y - mu1) / ps
        - (1 - t) * (y - mu0) / (1 - ps)
    )
    return float(aipw.mean())


def run_sensitivity_analyses(df: pd.DataFrame, covariates: list) -> dict:
    unadj = unadjusted_effect(df)
    psm, matched = psm_effect(df, covariates)
    iptw, weights = iptw_effect(df, covariates)
    strat = stratification_effect(df, covariates)
    aipw = aipw_effect(df, covariates)

    balance_before = balance_table(df, covariates)
    balance_after_psm = balance_table(matched, covariates)
    balance_after_iptw = balance_table(df, covariates, weights=weights)

    return {
        "unadjusted_log_hr": round(unadj, 4),
        "psm_log_hr": round(psm, 4),
        "iptw_log_hr": round(iptw, 4),
        "stratification_log_hr": round(strat, 4),
        "aipw_effect": round(aipw, 4),
        "balance_before_smd": {k: round(v, 4) for k, v in balance_before.items()},
        "balance_after_psm_smd": {k: round(v, 4) for k, v in balance_after_psm.items()},
        "balance_after_iptw_smd": {k: round(v, 4) for k, v in balance_after_iptw.items()},
        "n_matched_pairs": len(matched) // 2,
    }
