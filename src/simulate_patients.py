"""Simulate observational patient records with confounding by indication.

Treatment assignment depends on covariates (age, severity, comorbidity_score),
and those same covariates independently affect the outcome, creating real
confounding that a naive comparison would get wrong. The TRUE treatment
effect is fixed here, independently of any later estimation method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_PATIENTS = 820
TRUE_ATE = -0.35  # true treatment effect on log-hazard of the outcome event
SEED = 2025


def simulate(n: int = N_PATIENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(62, 10, n)
    severity = rng.normal(0, 1, n)
    comorbidity_score = rng.poisson(2, n).astype(float)
    sex = rng.integers(0, 2, n)

    # propensity for treatment depends on covariates (confounding by indication)
    logit_ps = (
        -1.0
        + 0.03 * (age - 62)
        + 0.8 * severity
        - 0.25 * comorbidity_score
        + 0.1 * sex
    )
    true_ps = 1 / (1 + np.exp(-logit_ps))
    treatment = rng.binomial(1, true_ps)

    # true outcome model: covariates affect risk independently of treatment
    log_hazard = (
        -2.0
        + 0.02 * (age - 62)
        + 0.5 * severity
        + 0.15 * comorbidity_score
        + TRUE_ATE * treatment
    )
    hazard = np.exp(log_hazard)
    time_to_event = rng.exponential(1.0 / hazard)
    censor_time = rng.uniform(0.5, 5.0, n)
    time = np.minimum(time_to_event, censor_time)
    event = (time_to_event <= censor_time).astype(int)

    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": age,
        "severity": severity,
        "comorbidity_score": comorbidity_score,
        "sex": sex,
        "true_propensity": true_ps,
        "treatment": treatment,
        "time": time,
        "event": event,
    })
    return df


COVARIATES = ["age", "severity", "comorbidity_score", "sex"]
