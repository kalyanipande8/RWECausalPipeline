import json
import os

from src.simulate_patients import simulate, COVARIATES, TRUE_ATE
from src.effect_estimation import run_sensitivity_analyses


def main():
    df = simulate()
    results = run_sensitivity_analyses(df, COVARIATES)
    results["n_patients"] = len(df)
    results["true_ate_log_hazard"] = TRUE_ATE

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
