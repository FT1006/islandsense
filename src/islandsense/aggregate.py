"""M3 Aggregation: Per-sailing predictions → Daily Risk → Weekly Risk + Scenarios.

This module converts per_sailing_predictions.csv into:
- daily_risk.csv: Daily risk score per category (Fresh/Fuel) for D0..D6
- weekly_risk.csv: Weekly risk baseline + scenario risk scores per category
- scenario_impact.csv: Impact metrics (hours_avoided, trailers_avoided)
- sailing_contrib.csv: Per-sailing contributions for drilldown
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from islandsense.config import get_config


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load per_sailing_predictions.csv and parse timestamps.

    Args:
        predictions_path: Path to per_sailing_predictions.csv

    Returns:
        DataFrame with etd_iso parsed to datetime
    """
    df = pd.read_csv(predictions_path)

    # Parse etd_iso to datetime (timezone-naive for date comparison)
    df["etd_dt"] = pd.to_datetime(df["etd_iso"])

    return df


def compute_day_index(
    df: pd.DataFrame, now: datetime, horizon_days: int = 7
) -> pd.DataFrame:
    """Add day_index column (D0..D6) and filter to horizon.

    Args:
        df: DataFrame with etd_dt column
        now: Reference datetime (usually UTC now)
        horizon_days: Number of days in forecast (default: 7)

    Returns:
        DataFrame with day_index column, filtered to D0..D(horizon_days-1)
    """
    df = df.copy()

    # Get reference date (Python datetime handles tz-aware correctly)
    now_date = now.date()

    # Compute day index: (etd_date - now_date).days
    df["day_index"] = df["etd_dt"].dt.date.apply(lambda d: (d - now_date).days)

    # Filter to D0..D(horizon_days-1)
    mask = (df["day_index"] >= 0) & (df["day_index"] < horizon_days)
    df = df[mask].copy()

    # Add formatted date column
    df["date"] = df["etd_dt"].dt.strftime("%Y-%m-%d")

    print(f"  ✓ Computed day indices: {len(df)} sailings in D0..D{horizon_days - 1}")
    print(f"    Reference date: {now_date}")

    return df


def compute_daily_e_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily expected loss per category.

    E_loss[c,d] = Σ p_sail[s] * exposure[c,s]

    Args:
        df: DataFrame with day_index, p_sail, fresh_units, fuel_units

    Returns:
        DataFrame with columns: day_index, date, category, E_loss
    """
    # Group by day and compute E_loss for each category
    daily_grouped = (
        df.groupby(["day_index", "date"])
        .agg(
            E_loss_fresh=("contrib_fresh", "sum"),
            E_loss_fuel=("contrib_fuel", "sum"),
        )
        .reset_index()
    )

    # Melt to long format (one row per category per day)
    fresh_df = daily_grouped[["day_index", "date", "E_loss_fresh"]].copy()
    fresh_df["category"] = "fresh"
    fresh_df = fresh_df.rename(columns={"E_loss_fresh": "E_loss"})

    fuel_df = daily_grouped[["day_index", "date", "E_loss_fuel"]].copy()
    fuel_df["category"] = "fuel"
    fuel_df = fuel_df.rename(columns={"E_loss_fuel": "E_loss"})

    e_loss_df = pd.concat([fresh_df, fuel_df], ignore_index=True)
    e_loss_df = e_loss_df.sort_values(["day_index", "category"]).reset_index(drop=True)

    return e_loss_df


def ensure_full_grid(
    e_loss_df: pd.DataFrame, now_date, horizon_days: int = 7
) -> pd.DataFrame:
    """Ensure all (day_index, category) pairs exist, filling missing with E_loss=0.

    This guarantees 7×2=14 rows for daily_risk.csv and consistent weekly averaging.

    Args:
        e_loss_df: DataFrame with day_index, date, category, E_loss
        now_date: Reference date for computing missing dates
        horizon_days: Number of days in horizon (default: 7)

    Returns:
        DataFrame with all day/category combinations filled
    """
    # Build complete grid of (day_index, category) pairs
    all_rows = []
    for d in range(horizon_days):
        # Compute date string for this day_index
        day_date = now_date + pd.Timedelta(days=d)
        date_str = day_date.strftime("%Y-%m-%d")
        for c in ["fresh", "fuel"]:
            all_rows.append({"day_index": d, "category": c, "date_default": date_str})

    full_grid = pd.DataFrame(all_rows)

    # Merge with existing E_loss data
    merged = full_grid.merge(e_loss_df, on=["day_index", "category"], how="left")

    # Fill missing E_loss with 0.0
    merged["E_loss"] = merged["E_loss"].fillna(0.0)

    # Use existing date if present, otherwise use computed date
    merged["date"] = merged["date"].fillna(merged["date_default"])
    merged = merged.drop(columns=["date_default"])

    # Sort and return
    merged = merged.sort_values(["day_index", "category"]).reset_index(drop=True)

    return merged


def compute_risk_score(
    e_loss: float, expected_loss_min: float, expected_loss_max: float
) -> int:
    """Scale E_loss to risk score [0, 100].

    risk = round(clamp((E_loss - min) / (max - min), 0, 1) * 100)

    Args:
        e_loss: Expected loss value
        expected_loss_min: Minimum expected loss (maps to risk 0)
        expected_loss_max: Maximum expected loss (maps to risk 100)

    Returns:
        Risk score as integer [0, 100]
    """
    if expected_loss_max <= expected_loss_min:
        return 0

    e_norm = (e_loss - expected_loss_min) / (expected_loss_max - expected_loss_min)
    e_norm = np.clip(e_norm, 0.0, 1.0)
    risk = int(round(e_norm * 100))

    return risk


def get_band(risk: int, bands_config: Dict) -> str:
    """Determine risk band (green/amber/red) from config thresholds.

    Args:
        risk: Risk score [0, 100]
        bands_config: Band configuration from config.yaml

    Returns:
        Band name: "green", "amber", or "red"
    """
    for band_name, band_info in bands_config.items():
        low, high = band_info["range"]
        if low <= risk <= high:
            return band_name

    # Default fallback
    return "red"


def add_risk_columns(e_loss_df: pd.DataFrame, config) -> pd.DataFrame:
    """Add risk_baseline and band columns to E_loss DataFrame.

    Args:
        e_loss_df: DataFrame with day_index, date, category, E_loss
        config: Config object

    Returns:
        DataFrame with risk_baseline and band columns added
    """
    df = e_loss_df.copy()

    # Compute risk score for each row
    risk_values = []
    bands = []

    for _, row in df.iterrows():
        category = row["category"]
        e_loss = row["E_loss"]

        # Get per-category scaling parameters
        loss_min = config.risk_expected_loss_min(category)
        loss_max = config.risk_expected_loss_max(category)

        risk = compute_risk_score(e_loss, loss_min, loss_max)
        band = get_band(risk, config.risk_bands)

        risk_values.append(risk)
        bands.append(band)

    df["risk_baseline"] = risk_values
    df["band"] = bands

    return df


def compute_weekly_risk(daily_risk_df: pd.DataFrame) -> Dict[str, int]:
    """Compute weekly baseline risk per category (mean of daily risk scores).

    Args:
        daily_risk_df: DataFrame with category, risk_baseline

    Returns:
        Dict mapping category to weekly risk baseline
    """
    weekly = {}

    for category in ["fresh", "fuel"]:
        cat_df = daily_risk_df[daily_risk_df["category"] == category]
        mean_val = cat_df["risk_baseline"].mean()
        # Handle edge case where no data exists
        if pd.isna(mean_val):
            weekly_risk = 0
        else:
            weekly_risk = int(round(mean_val))
        weekly[category] = weekly_risk

    return weekly


def apply_scenario(
    e_loss_df: pd.DataFrame, scenario_config: Dict, config
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply scenario alpha scaling to E_loss and compute scenario risk scores.

    E_loss_k[c,d] = E_loss[c,d] * (1 - alpha_k[c])

    Args:
        e_loss_df: DataFrame with day_index, date, category, E_loss
        scenario_config: Scenario definition from config (id, name, alpha)
        config: Config object

    Returns:
        Tuple of (daily_scenario_df, weekly_risk_dict)
    """
    alphas = scenario_config["alpha"]

    df = e_loss_df.copy()

    # Apply alpha scaling
    def apply_alpha(row):
        category = row["category"]
        alpha = alphas.get(category, 0.0)
        return row["E_loss"] * (1 - alpha)

    df["E_loss_scenario"] = df.apply(apply_alpha, axis=1)

    # Compute risk score for scenario E_loss
    risk_values = []
    for _, row in df.iterrows():
        category = row["category"]
        e_loss_scenario = row["E_loss_scenario"]

        loss_min = config.risk_expected_loss_min(category)
        loss_max = config.risk_expected_loss_max(category)

        risk = compute_risk_score(e_loss_scenario, loss_min, loss_max)
        risk_values.append(risk)

    df["risk_scenario"] = risk_values

    # Compute weekly risk for this scenario
    weekly_risk = {}
    for category in ["fresh", "fuel"]:
        cat_df = df[df["category"] == category]
        weekly_risk[category] = int(round(cat_df["risk_scenario"].mean()))

    return df, weekly_risk


def compute_impact(
    e_loss_df: pd.DataFrame, scenario_df: pd.DataFrame, config
) -> Dict[str, Dict[str, float]]:
    """Compute impact metrics (hours_avoided for Fresh, trailers_avoided for Fuel).

    delta_E_loss[c] = Σ_d (E_loss[c,d] - E_loss_k[c,d])
    hours_avoided = delta_E_loss * k_hours_per_unit
    trailers_avoided = delta_E_loss / units_per_trailer

    Args:
        e_loss_df: Baseline E_loss DataFrame
        scenario_df: Scenario E_loss DataFrame with E_loss_scenario column
        config: Config object

    Returns:
        Dict mapping category to impact metrics
    """
    # Merge baseline and scenario E_loss
    merged = e_loss_df.merge(
        scenario_df[["day_index", "category", "E_loss_scenario"]],
        on=["day_index", "category"],
    )

    impact = {}

    for category in ["fresh", "fuel"]:
        cat_df = merged[merged["category"] == category]

        # Sum of delta E_loss across all days
        delta_e_loss = (cat_df["E_loss"] - cat_df["E_loss_scenario"]).sum()

        if category == "fresh":
            hours_avoided = round(delta_e_loss * config.k_hours_per_unit, 1)
            trailers_avoided = 0.0
        else:  # fuel
            hours_avoided = 0.0
            trailers_avoided = round(delta_e_loss / config.units_per_trailer, 1)

        impact[category] = {
            "hours_avoided": hours_avoided,
            "trailers_avoided": trailers_avoided,
            "delta_e_loss": float(delta_e_loss),  # for tuning / logging
        }

    return impact


def compute_sailing_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sailing contributions for drilldown.

    contrib_fresh[s] = p_sail[s] * fresh_units[s]
    contrib_fuel[s] = p_sail[s] * fuel_units[s]

    Args:
        df: Predictions DataFrame with day_index

    Returns:
        DataFrame with per-sailing contributions
    """
    contrib_df = df[
        [
            "sailing_id",
            "day_index",
            "date",
            "route",
            "vessel",
            "etd_iso",
            "p_sail",
            "fresh_units",
            "fuel_units",
            "contrib_fresh",
            "contrib_fuel",
        ]
    ].copy()

    # Sort by total contribution (Fresh + Fuel) for honest "top risk" ranking
    contrib_df["contrib_total"] = (
        contrib_df["contrib_fresh"] + contrib_df["contrib_fuel"]
    )
    contrib_df = contrib_df.sort_values(
        ["day_index", "contrib_total"], ascending=[True, False]
    ).reset_index(drop=True)
    contrib_df = contrib_df.drop(columns=["contrib_total"])

    return contrib_df


def count_red_days(daily_risk_df: pd.DataFrame) -> Dict[str, int]:
    """Count number of days with red band (risk >= 70) per category.

    Args:
        daily_risk_df: DataFrame with category, band columns

    Returns:
        Dict mapping category to count of red days
    """
    red_counts = {}
    for category in ["fresh", "fuel"]:
        cat_df = daily_risk_df[daily_risk_df["category"] == category]
        red_count = (cat_df["band"] == "red").sum()
        red_counts[category] = int(red_count)
    return red_counts


def compute_sailing_scenario_deltas(
    sailing_contrib_df: pd.DataFrame, scenario_config: Dict
) -> pd.DataFrame:
    """Compute per-sailing contribution deltas under a scenario.

    Baseline contrib = p_sail * units
    Scenario contrib = p_sail * units * (1 - alpha)
    Delta = baseline - scenario = p_sail * units * alpha

    Args:
        sailing_contrib_df: DataFrame with contrib_fresh, contrib_fuel
        scenario_config: Scenario definition with alpha per category

    Returns:
        DataFrame with delta_fresh, delta_fuel columns added
    """
    df = sailing_contrib_df.copy()
    alphas = scenario_config["alpha"]

    # Delta = baseline contrib * alpha (reduction due to scenario)
    df["delta_fresh"] = df["contrib_fresh"] * alphas.get("fresh", 0.0)
    df["delta_fuel"] = df["contrib_fuel"] * alphas.get("fuel", 0.0)

    return df


def aggregate(
    predictions_path: Path = Path("per_sailing_predictions.csv"),
    output_dir: Path = Path("data"),
    now: Optional[datetime] = None,
) -> Dict[str, pd.DataFrame]:
    """Main aggregation pipeline: predictions → risk scores + scenarios + impact.

    Args:
        predictions_path: Path to per_sailing_predictions.csv
        output_dir: Directory to write output CSVs
        now: Reference datetime (default: current UTC time)

    Returns:
        Dict of output DataFrames
    """
    if now is None:
        now = datetime.now(timezone.utc)

    print("=" * 70)
    print("IslandSense M3 Aggregation Pipeline")
    print("=" * 70)

    # Load config
    config = get_config()
    print("\nConfig loaded:")
    print(f"  Horizon: {config.horizon_days} days")
    print(f"  Scenarios: {[s['id'] for s in config.scenarios]}")

    # Load predictions
    print(f"\nLoading predictions from {predictions_path}...")
    df = load_predictions(predictions_path)
    print(f"  ✓ Loaded {len(df)} sailing predictions")

    # Validate required columns
    required_cols = {
        "sailing_id",
        "route",
        "vessel",
        "etd_iso",
        "p_sail",
        "fresh_units",
        "fuel_units",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {missing}")
    print("  ✓ Required columns validated")

    # Compute per-sailing contributions
    print("\nComputing per-sailing contributions...")
    df["contrib_fresh"] = df["p_sail"] * df["fresh_units"]
    df["contrib_fuel"] = df["p_sail"] * df["fuel_units"]

    # Compute day indices
    print("\nComputing day indices...")
    df = compute_day_index(df, now, config.horizon_days)

    if len(df) == 0:
        print("\n⚠️  No sailings in forecast window!")
        return {}

    # Compute daily E_loss
    print("\nComputing daily expected loss...")
    e_loss_df = compute_daily_e_loss(df)
    print(f"  ✓ Computed E_loss for {len(e_loss_df)} (category, day) pairs")

    # Ensure full 7-day grid (fill missing days with E_loss=0)
    print("\nEnsuring full 7-day grid...")
    e_loss_df = ensure_full_grid(e_loss_df, now.date(), config.horizon_days)
    print(f"  ✓ Grid now has {len(e_loss_df)} rows (7 days × 2 categories)")

    # Add risk score and band columns
    print("\nScaling to risk baseline...")
    daily_risk_df = add_risk_columns(e_loss_df, config)

    # Compute weekly baseline
    print("\nComputing weekly baseline risk...")
    weekly_baseline = compute_weekly_risk(daily_risk_df)
    print(f"  Fresh: {weekly_baseline['fresh']}")
    print(f"  Fuel: {weekly_baseline['fuel']}")

    # Process each scenario
    print("\nProcessing scenarios...")
    scenario_results = {}
    weekly_risks = {"baseline": weekly_baseline}
    impacts = {}
    red_days_by_scenario = {"baseline": count_red_days(daily_risk_df)}
    sailing_deltas_by_scenario = {}

    for scenario_config in config.scenarios:
        scenario_id = scenario_config["id"]
        print(f"\n  Scenario: {scenario_id} ({scenario_config['name']})")

        # Apply scenario
        scenario_df, weekly_risk = apply_scenario(
            daily_risk_df, scenario_config, config
        )
        print(f"    Fresh risk: {weekly_baseline['fresh']} → {weekly_risk['fresh']}")
        print(f"    Fuel risk: {weekly_baseline['fuel']} → {weekly_risk['fuel']}")

        # Compute impact
        impact = compute_impact(daily_risk_df, scenario_df, config)
        print(f"    Fresh hours avoided: {impact['fresh']['hours_avoided']:.1f}")
        print(f"    Fuel trailers avoided: {impact['fuel']['trailers_avoided']:.1f}")
        print(
            f"    Fresh ΔE_loss: {impact['fresh']['delta_e_loss']:.2f} units "
            f"(before k_hours_per_unit)"
        )
        print(
            f"    Fuel  ΔE_loss: {impact['fuel']['delta_e_loss']:.2f} units "
            f"(before units_per_trailer)"
        )

        # Count red days under scenario
        red_days_scenario = count_red_days(
            scenario_df.assign(
                band=lambda x: x.apply(
                    lambda row: get_band(row["risk_scenario"], config.risk_bands),
                    axis=1,
                )
            )
        )
        red_days_by_scenario[scenario_id] = red_days_scenario
        print(
            f"    Red days Fresh: {red_days_by_scenario['baseline']['fresh']} → {red_days_scenario['fresh']}"
        )
        print(
            f"    Red days Fuel: {red_days_by_scenario['baseline']['fuel']} → {red_days_scenario['fuel']}"
        )

        # Compute per-sailing deltas
        sailing_deltas = compute_sailing_scenario_deltas(
            compute_sailing_contributions(df), scenario_config
        )
        sailing_deltas_by_scenario[scenario_id] = sailing_deltas

        scenario_results[scenario_id] = scenario_df
        weekly_risks[scenario_id] = weekly_risk
        impacts[scenario_id] = impact

    # Build output DataFrames
    print("\n" + "=" * 70)
    print("GENERATING OUTPUT CSVs")
    print("=" * 70)

    # 1. daily_risk.csv
    daily_risk_output = daily_risk_df[
        ["day_index", "date", "category", "E_loss", "risk_baseline", "band"]
    ].copy()
    daily_risk_output = daily_risk_output.sort_values(["day_index", "category"])
    print(f"\n1. daily_risk.csv: {len(daily_risk_output)} rows")

    # 2. weekly_risk.csv
    weekly_risk_rows = []
    for category in ["fresh", "fuel"]:
        row = {
            "category": category,
            "weekly_risk_baseline": weekly_baseline[category],
        }
        for scenario_config in config.scenarios:
            scenario_id = scenario_config["id"]
            col_name = f"weekly_risk_{scenario_id}"
            row[col_name] = weekly_risks[scenario_id][category]
        weekly_risk_rows.append(row)

    weekly_risk_df = pd.DataFrame(weekly_risk_rows)
    print(f"2. weekly_risk.csv: {len(weekly_risk_df)} rows")

    # 3. scenario_impact.csv
    impact_rows = []
    for scenario_config in config.scenarios:
        scenario_id = scenario_config["id"]
        for category in ["fresh", "fuel"]:
            row = {
                "scenario_id": scenario_id,
                "category": category,
                "weekly_risk_baseline": weekly_baseline[category],
                "weekly_risk_scenario": weekly_risks[scenario_id][category],
                "delta_risk": weekly_baseline[category]
                - weekly_risks[scenario_id][category],
                "hours_avoided": impacts[scenario_id][category]["hours_avoided"],
                "trailers_avoided": impacts[scenario_id][category]["trailers_avoided"],
                "red_days_baseline": red_days_by_scenario["baseline"][category],
                "red_days_scenario": red_days_by_scenario[scenario_id][category],
            }
            impact_rows.append(row)

    scenario_impact_df = pd.DataFrame(impact_rows)
    print(f"3. scenario_impact.csv: {len(scenario_impact_df)} rows")

    # 4. sailing_contrib.csv
    sailing_contrib_df = compute_sailing_contributions(df)
    print(f"4. sailing_contrib.csv: {len(sailing_contrib_df)} rows")

    # 5. sailing_scenario_deltas.csv (per-sailing deltas for each scenario)
    sailing_delta_rows = []
    for scenario_config in config.scenarios:
        scenario_id = scenario_config["id"]
        deltas_df = sailing_deltas_by_scenario[scenario_id]
        deltas_df = deltas_df.copy()
        deltas_df["scenario_id"] = scenario_id
        sailing_delta_rows.append(deltas_df)

    sailing_scenario_deltas_df = pd.concat(sailing_delta_rows, ignore_index=True)
    # Reorder columns to put scenario_id first
    cols = ["scenario_id"] + [
        c for c in sailing_scenario_deltas_df.columns if c != "scenario_id"
    ]
    sailing_scenario_deltas_df = sailing_scenario_deltas_df[cols]
    print(f"5. sailing_scenario_deltas.csv: {len(sailing_scenario_deltas_df)} rows")

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_risk_output.to_csv(output_dir / "daily_risk.csv", index=False)
    weekly_risk_df.to_csv(output_dir / "weekly_risk.csv", index=False)
    scenario_impact_df.to_csv(output_dir / "scenario_impact.csv", index=False)
    sailing_contrib_df.to_csv(output_dir / "sailing_contrib.csv", index=False)
    sailing_scenario_deltas_df.to_csv(
        output_dir / "sailing_scenario_deltas.csv", index=False
    )

    print(f"\n✓ All CSVs written to {output_dir}/")

    return {
        "daily_risk": daily_risk_output,
        "weekly_risk": weekly_risk_df,
        "scenario_impact": scenario_impact_df,
        "sailing_contrib": sailing_contrib_df,
        "sailing_scenario_deltas": sailing_scenario_deltas_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-sailing predictions into risk metrics"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="data/per_sailing_predictions.csv",
        help="Path to per_sailing_predictions.csv (default: data/per_sailing_predictions.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for CSVs (default: data)",
    )
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="Reference datetime (ISO format, default: current UTC)",
    )

    args = parser.parse_args()

    # Parse now if provided
    if args.now:
        now = datetime.fromisoformat(args.now.replace("Z", "+00:00"))
    else:
        now = datetime.now(timezone.utc)

    # Run aggregation
    results = aggregate(
        predictions_path=Path(args.predictions),
        output_dir=Path(args.output_dir),
        now=now,
    )

    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("Daily risk sample:")
        print(results["daily_risk"].head(4).to_string(index=False))
        print("\nWeekly risk:")
        print(results["weekly_risk"].to_string(index=False))
        print("\nScenario Impact:")
        print(results["scenario_impact"].to_string(index=False))
        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
