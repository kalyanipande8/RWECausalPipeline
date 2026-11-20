import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_patients import simulate, COVARIATES, TRUE_ATE
from src.effect_estimation import run_sensitivity_analyses, unadjusted_effect, psm_effect


def test_at_least_800_patients():
    df = simulate()
    assert len(df) >= 800


def test_five_adjustment_methods_present():
    df = simulate()
    r = run_sensitivity_analyses(df, COVARIATES)
    methods = ["unadjusted_log_hr", "psm_log_hr", "iptw_log_hr", "stratification_log_hr", "aipw_effect"]
    assert all(m in r for m in methods)
    assert len(methods) == 5


def test_adjustment_improves_balance():
    df = simulate()
    r = run_sensitivity_analyses(df, COVARIATES)
    before = sum(r["balance_before_smd"].values())
    after = sum(r["balance_after_psm_smd"].values())
    assert after < before


def test_adjusted_estimate_closer_to_truth_than_unadjusted():
    df = simulate()
    r = run_sensitivity_analyses(df, COVARIATES)
    unadj_err = abs(r["unadjusted_log_hr"] - TRUE_ATE)
    psm_err = abs(r["psm_log_hr"] - TRUE_ATE)
    assert psm_err < unadj_err


def test_matching_produces_pairs():
    df = simulate()
    eff, matched = psm_effect(df, COVARIATES)
    assert len(matched) > 0
    assert len(matched) % 2 == 0
