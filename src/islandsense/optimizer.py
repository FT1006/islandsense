"""
M2.5 Tiny Prescriptive Optimiser (global weekly policy)

- One (x, y) pair for the whole week:
    x = forward fraction (e.g. 0.10 → "bring forward 10%")
    y = air-lift fraction (e.g. 0.05 → "air-lift 5%")

- For each category c ∈ {fresh, fuel}:

    alpha_eff[c]      = beta_f[c] * x + beta_a[c] * y
    penalty_avoided[c] = penalty_per_unit[c] * E_loss_total[c] * alpha_eff[c]
    action_cost[c]     = (cost_forward[c] * x + cost_air[c] * y) * total_exposure[c]
    NetBenefit[c]      = penalty_avoided[c] - action_cost[c]

- We choose (x, y) on a small grid to maximise:
    NetBenefit_total = Σ_c NetBenefit[c]

This file provides:
    - compute_net_benefit(x, y, category_data, config)
    - grid_search(category_data, config)
    - optimize_scenarios(predictions_df, config)
    - CLI entrypoint: python -m islandsense.optimizer
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from islandsense.config import get_config

CATEGORIES = ["fresh", "fuel"]


def build_category_data(predictions_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute weekly aggregates needed by the optimiser.

    For each category c:
        E_loss_total[c]  = Σ_s p_sail[s] * exposure[c,s]
        total_exposure[c] = Σ_s exposure[c,s]

    Returns:
        {
          "fresh": {"E_loss_total": ..., "total_exposure": ...},
          "fuel":  {"E_loss_total": ..., "total_exposure": ...},
        }
    """
    data: dict[str, dict[str, float]] = {}

    for c in CATEGORIES:
        units_col = f"{c}_units"
        if units_col not in predictions_df.columns:
            msg = f"Missing column '{units_col}' in per_sailing_predictions.csv"
            raise ValueError(msg)

        exposure = predictions_df[units_col]
        e_loss_total = float((predictions_df["p_sail"] * exposure).sum())
        total_exposure = float(exposure.sum())

        data[c] = {
            "E_loss_total": e_loss_total,
            "total_exposure": total_exposure,
        }

    return data


def compute_net_benefit(
    x: float,
    y: float,
    category_data: dict[str, dict[str, float]],
    config: Any,
) -> tuple[float, dict[str, dict[str, float]], dict[str, float]]:
    """
    For one global (x, y) policy, compute per-category net benefits.

    Args:
        x: forward fraction (0 ≤ x ≤ 1)
        y: air-lift fraction (0 ≤ y ≤ 1)
        category_data:
            {
              "fresh": {"E_loss_total": float, "total_exposure": float},
              "fuel":  {"E_loss_total": float, "total_exposure": float},
            }
        config: config object with .optimizer dict-like attribute

    Returns:
        total_net_benefit: float
        per_category: {
            c: {
                "alpha_eff": float,
                "penalty_avoided": float,
                "action_cost": float,
                "net_benefit": float,
            }
        }
        alpha_eff: {c: alpha_eff[c]}
    """
    opt_cfg = config.optimizer

    per_category: dict[str, dict[str, float]] = {}
    alpha_eff: dict[str, float] = {}
    total_net_benefit = 0.0

    for c in CATEGORIES:
        e_loss_total = category_data[c]["E_loss_total"]
        total_exposure = category_data[c]["total_exposure"]

        if total_exposure <= 0 or e_loss_total <= 0:
            # Nothing at risk => zero effect
            per_category[c] = {
                "alpha_eff": 0.0,
                "penalty_avoided": 0.0,
                "action_cost": 0.0,
                "net_benefit": 0.0,
            }
            alpha_eff[c] = 0.0
            continue

        penalty_per_unit = opt_cfg["penalty_per_unit"][c]
        cost_forward = opt_cfg["cost_forward"][c]
        cost_air = opt_cfg["cost_air"][c]
        beta_f = opt_cfg["beta_forward"][c]
        beta_a = opt_cfg["beta_air"][c]

        # Effective reduction in at-risk exposure for category c
        a_eff = beta_f * x + beta_a * y

        penalty_avoided = penalty_per_unit * e_loss_total * a_eff
        action_cost = (cost_forward * x + cost_air * y) * total_exposure
        net_benefit = penalty_avoided - action_cost

        per_category[c] = {
            "alpha_eff": a_eff,
            "penalty_avoided": penalty_avoided,
            "action_cost": action_cost,
            "net_benefit": net_benefit,
        }
        alpha_eff[c] = a_eff
        total_net_benefit += net_benefit

    return total_net_benefit, per_category, alpha_eff


def grid_search(
    category_data: dict[str, dict[str, float]],
    config: Any,
) -> dict[str, Any]:
    """
    Grid search over (x, y) to maximize total net benefit.

    - x ∈ optimizer.grid.forward_fractions
    - y ∈ optimizer.grid.air_fractions
    - Skip if x > shift_limits[c].max_forward_fraction for any category
    - Skip if y > shift_limits[c].max_air_fraction for any category

    Returns:
        {
          "forward_frac": x,
          "air_frac": y,
          "net_benefit_total": float,
          "per_category": {...},
          "alpha_eff": {...},
        }
    """
    opt_cfg = config.optimizer

    grid_forward = opt_cfg["grid"]["forward_fractions"]
    grid_air = opt_cfg["grid"]["air_fractions"]

    best_result: dict[str, Any] | None = None
    best_net_benefit = float("-inf")

    for x in grid_forward:
        for y in grid_air:
            # Check per-category shift limits
            valid = True
            for c in CATEGORIES:
                limits = opt_cfg["shift_limits"][c]
                if x > limits["max_forward_fraction"] or y > limits["max_air_fraction"]:
                    valid = False
                    break
            if not valid:
                continue

            total, per_category, alpha_eff = compute_net_benefit(
                x, y, category_data, config
            )

            if total > best_net_benefit:
                best_net_benefit = total
                best_result = {
                    "forward_frac": x,
                    "air_frac": y,
                    "net_benefit_total": total,
                    "per_category": per_category,
                    "alpha_eff": alpha_eff,
                }

    # Fallback: no valid point → treat as (0, 0)
    if best_result is None:
        _, per_category_zero, alpha_eff_zero = compute_net_benefit(
            0.0, 0.0, category_data, config
        )
        best_result = {
            "forward_frac": 0.0,
            "air_frac": 0.0,
            "net_benefit_total": 0.0,
            "per_category": per_category_zero,
            "alpha_eff": alpha_eff_zero,
        }

    return best_result


def optimize_scenarios(predictions_df: pd.DataFrame, config: Any) -> dict[str, Any]:
    """
    Main entry: compute category_data, run grid search, return results.

    Returns:
        {
          "forward_frac": x,
          "air_frac": y,
          "alpha_eff": {category: float},
          "per_category": {...},
          "net_benefit_total": float,
        }
    """
    category_data = build_category_data(predictions_df)
    best = grid_search(category_data, config)
    return best


def main() -> None:
    """CLI entry point for the optimizer."""
    parser = argparse.ArgumentParser(
        description="Tiny prescriptive optimiser for IslandSense"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="per_sailing_predictions.csv",
        help="Path to per_sailing_predictions.csv",
    )
    args = parser.parse_args()

    config = get_config()
    opt_cfg = config.optimizer

    if not opt_cfg or not opt_cfg.get("enabled", False):
        print("Optimizer disabled (optimizer.enabled is false or missing).")
        return

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        msg = f"Predictions file not found: {predictions_path}"
        raise SystemExit(msg)

    print("=" * 70)
    print("IslandSense Tiny Prescriptive Optimiser")
    print("=" * 70)
    print(f"Loading predictions from {predictions_path}...")
    df = pd.read_csv(predictions_path)

    # Basic weekly totals for context
    category_data = build_category_data(df)
    print("\nWeekly totals:")
    for c in CATEGORIES:
        cd = category_data[c]
        print(
            f"  {c.capitalize():5s}  "
            f"E_loss_total={cd['E_loss_total']:8.2f}  "
            f"total_exposure={cd['total_exposure']:8.2f}"
        )

    print("\nRunning grid search over policies...")
    best = grid_search(category_data, config)

    x = best["forward_frac"]
    y = best["air_frac"]

    print("\nBest global policy (weekly):")
    print(f"  forward_frac (x) = {x:.2f}  → 'bring forward {x * 100:.0f}%'")
    print(f"  air_frac     (y) = {y:.2f}  → 'air-lift {y * 100:.0f}%'")
    print(
        f"  NetBenefit_total = {best['net_benefit_total']:.2f} (penalty avoided - cost)"
    )

    print("\nPer-category breakdown:")
    for c in CATEGORIES:
        res = best["per_category"][c]
        print(
            f"  {c.capitalize():5s}  "
            f"alpha_eff={res['alpha_eff']:.3f}  "
            f"penalty_avoided={res['penalty_avoided']:.2f}  "
            f"action_cost={res['action_cost']:.2f}  "
            f"net_benefit={res['net_benefit']:.2f}"
        )

    print("\nEffective alpha per category (to plug into scenario A):")
    for c, a in best["alpha_eff"].items():
        print(f"  {c}: alpha_eff = {a:.3f}")

    print("\nYou can now either:")
    print("  - Copy these alpha_eff values into config.scenarios[scenario_A].alpha, or")
    print(
        "  - Wire aggregate.py to call optimize_scenarios() and override "
        "Scenario A dynamically."
    )


if __name__ == "__main__":
    main()
