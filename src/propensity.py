"""Real propensity-score estimation, matching, IPTW, stratification, and
balance diagnostics (standardized mean difference, SMD)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def estimate_propensity(df: pd.DataFrame, covariates: list) -> np.ndarray:
    X = df[covariates].values
    y = df["treatment"].values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def smd(x_treat: np.ndarray, x_control: np.ndarray) -> float:
    m1, m0 = x_treat.mean(), x_control.mean()
    s1, s0 = x_treat.std(ddof=1), x_control.std(ddof=1)
    pooled = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
    if pooled == 0:
        return 0.0
    return abs(m1 - m0) / pooled


def balance_table(df: pd.DataFrame, covariates: list, weights: np.ndarray | None = None) -> dict:
    treat = df["treatment"].values == 1
    out = {}
    for c in covariates:
        x = df[c].values
        if weights is None:
            out[c] = smd(x[treat], x[~treat])
        else:
            w = weights
            m1 = np.average(x[treat], weights=w[treat])
            m0 = np.average(x[~treat], weights=w[~treat])
            s1 = np.sqrt(np.average((x[treat] - m1) ** 2, weights=w[treat]))
            s0 = np.sqrt(np.average((x[~treat] - m0) ** 2, weights=w[~treat]))
            pooled = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
            out[c] = abs(m1 - m0) / pooled if pooled > 0 else 0.0
    return out


def nn_match(df: pd.DataFrame, ps: np.ndarray, caliper: float = 0.05) -> pd.DataFrame:
    """1:1 nearest-neighbor propensity matching without replacement."""
    d = df.copy()
    d["ps"] = ps
    treated = d[d["treatment"] == 1].reset_index(drop=True)
    control = d[d["treatment"] == 0].reset_index(drop=True)

    nn = NearestNeighbors(n_neighbors=1).fit(control[["ps"]].values)
    dist, idx = nn.kneighbors(treated[["ps"]].values)

    matched_control_idx = []
    matched_treated_rows = []
    used = set()
    for i, (dd, ii) in enumerate(zip(dist[:, 0], idx[:, 0])):
        if dd <= caliper and ii not in used:
            used.add(ii)
            matched_control_idx.append(ii)
            matched_treated_rows.append(i)

    matched = pd.concat([
        treated.iloc[matched_treated_rows].assign(match_group=range(len(matched_treated_rows))),
        control.iloc[matched_control_idx].assign(match_group=range(len(matched_control_idx))),
    ], ignore_index=True)
    return matched.drop(columns=["ps"])


def iptw_weights(treatment: np.ndarray, ps: np.ndarray) -> np.ndarray:
    ps = np.clip(ps, 0.01, 0.99)
    return np.where(treatment == 1, 1 / ps, 1 / (1 - ps))


def stratify(df: pd.DataFrame, ps: np.ndarray, n_strata: int = 5) -> pd.Series:
    return pd.qcut(ps, n_strata, labels=False, duplicates="drop")
