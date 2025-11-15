"""Per-sailing prediction script for IslandSense M2→M3 handoff.

Generates predictions for upcoming sailings within a time horizon.

Usage:
    python predict.py --all              # Predict all future sailings in data
    python predict.py --horizon-hours 72 # Predict sailings in next 72 hours (from now)
    python predict.py --horizon-hours 48 # Predict sailings in next 48 hours

Output:
    Writes per_sailing_predictions.csv with schema:
    sailing_id,route,vessel,etd_iso,etd_bin,p_sail_uncal,p_sail,fresh_units,fuel_units
"""

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from islandsense.config import get_config
from islandsense.features import compute_features
from islandsense.schema import SailingColumns, ExposureColumns


def load_models(models_dir: Path = Path("models")):
    """Load trained model and calibrator.

    Returns:
        Tuple of (model, calibrator)
    """
    model_path = models_dir / "model.pkl"
    calibrator_path = models_dir / "calibrator.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(calibrator_path, "rb") as f:
        calibrator = pickle.load(f)

    print(f"✓ Loaded model from {model_path}")
    print(f"✓ Loaded calibrator from {calibrator_path}")

    return model, calibrator


def load_data(config):
    """Load all required data files.

    Returns:
        Tuple of (sailings_df, metocean_df, tides_df, status_df, exposure_df)
    """
    print("\nLoading data...")

    sailings_df = pd.read_csv(config.sailings_file)
    metocean_df = pd.read_csv(config.metocean_file)
    tides_df = pd.read_csv(config.tides_file)
    status_df = pd.read_csv(config.status_file)
    exposure_df = pd.read_csv(config.exposure_file)

    print(f"  ✓ {len(sailings_df)} sailings")
    print(f"  ✓ {len(metocean_df)} metocean records")
    print(f"  ✓ {len(tides_df)} tide records")
    print(f"  ✓ {len(status_df)} status records")
    print(f"  ✓ {len(exposure_df)} exposure records")

    return sailings_df, metocean_df, tides_df, status_df, exposure_df


def filter_sailings_by_horizon(
    sailings_df: pd.DataFrame,
    horizon_hours: Optional[int] = None,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    """Filter sailings by ETD within time horizon.

    Args:
        sailings_df: DataFrame with sailings
        horizon_hours: Hours ahead to predict (None = all sailings)
        now: Reference time (default: current UTC time)

    Returns:
        Filtered DataFrame
    """
    if horizon_hours is None:
        print("\n✓ Predicting ALL sailings (no time filter)")
        return sailings_df.copy()

    if now is None:
        now = datetime.now(timezone.utc)

    # Convert to pandas Timestamp (timezone-naive for comparison with data)
    now_ts = pd.Timestamp(now).tz_localize(None)

    sailings_df = sailings_df.copy()
    sailings_df["etd"] = pd.to_datetime(sailings_df[SailingColumns.ETD_ISO])

    # Filter: ETD in [now, now + horizon_hours]
    future_mask = (sailings_df["etd"] >= now_ts) & (
        sailings_df["etd"] <= now_ts + pd.Timedelta(hours=horizon_hours)
    )

    filtered_df = sailings_df[future_mask].copy()

    print(f"\n✓ Filtered to {len(filtered_df)} sailings in next {horizon_hours}h")
    print(f"  Reference time: {now.isoformat()}")
    if len(filtered_df) > 0:
        print(f"  First sailing: {filtered_df['etd'].min()}")
        print(f"  Last sailing: {filtered_df['etd'].max()}")

    return filtered_df


def compute_etd_bin(etd_iso: str, bin_hours: int) -> str:
    """Bin ETD timestamp to nearest bin_hours interval.

    Args:
        etd_iso: ISO timestamp string
        bin_hours: Bin size in hours (e.g., 6)

    Returns:
        Binned timestamp as ISO string
    """
    etd = pd.to_datetime(etd_iso)

    # Floor to nearest bin_hours interval
    hour_offset = (etd.hour // bin_hours) * bin_hours
    binned = etd.replace(hour=hour_offset, minute=0, second=0, microsecond=0)

    return binned.isoformat()


def predict(
    sailings_df: pd.DataFrame,
    metocean_df: pd.DataFrame,
    tides_df: pd.DataFrame,
    status_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    model,
    calibrator,
    config,
) -> pd.DataFrame:
    """Generate predictions for sailings.

    Args:
        sailings_df: Sailings to predict
        metocean_df: Metocean data
        tides_df: Tide data
        status_df: Historical status data (for prior_24h_delay)
        exposure_df: Exposure data
        model: Trained model
        calibrator: Fitted calibrator
        config: Config object

    Returns:
        DataFrame with schema: sailing_id,route,vessel,etd_iso,etd_bin,p_sail_uncal,p_sail,fresh_units,fuel_units
    """
    if len(sailings_df) == 0:
        print("\n⚠️  No sailings to predict")
        return pd.DataFrame()

    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    # Compute features
    features_df = compute_features(sailings_df, metocean_df, tides_df, status_df)

    # Prepare feature matrix (must match training order)
    feature_cols = [
        "WOTDI",
        "BSEF",
        "gust_max_3h",
        "tide_gate_margin",
        "prior_24h_delay",
        "day_of_week",
        "month",
    ]

    X = features_df[feature_cols].values

    # Run inference
    print("\nRunning inference...")
    print(f"  Model: XGBoost with {len(feature_cols)} features")

    # Get uncalibrated probabilities
    y_prob_uncal = model.predict_proba(X)[:, 1]  # Probability of class 1 (disruption)

    # Apply calibration
    y_prob_cal = calibrator.transform(y_prob_uncal)

    print(f"  ✓ Generated {len(y_prob_cal)} predictions")
    print(
        f"    Uncalibrated p_sail range: [{y_prob_uncal.min():.3f}, {y_prob_uncal.max():.3f}]"
    )
    print(
        f"    Calibrated p_sail range: [{y_prob_cal.min():.3f}, {y_prob_cal.max():.3f}]"
    )

    # Build output DataFrame
    predictions_df = sailings_df[
        [
            SailingColumns.SAILING_ID,
            SailingColumns.ROUTE,
            SailingColumns.VESSEL,
            SailingColumns.ETD_ISO,
        ]
    ].copy()

    # Add etd_bin
    predictions_df["etd_bin"] = predictions_df[SailingColumns.ETD_ISO].apply(
        lambda x: compute_etd_bin(x, config.bin_hours)
    )

    # Add predictions
    predictions_df["p_sail_uncal"] = y_prob_uncal
    predictions_df["p_sail"] = y_prob_cal

    # Merge exposure
    predictions_df = predictions_df.merge(
        exposure_df[
            [
                ExposureColumns.SAILING_ID,
                ExposureColumns.FRESH_UNITS,
                ExposureColumns.FUEL_UNITS,
            ]
        ],
        on=SailingColumns.SAILING_ID,
        how="left",
    )

    # Handle missing exposure (fill with 0.0)
    predictions_df[ExposureColumns.FRESH_UNITS] = predictions_df[
        ExposureColumns.FRESH_UNITS
    ].fillna(0.0)
    predictions_df[ExposureColumns.FUEL_UNITS] = predictions_df[
        ExposureColumns.FUEL_UNITS
    ].fillna(0.0)

    # Rename columns to match output schema
    predictions_df = predictions_df.rename(
        columns={
            SailingColumns.SAILING_ID: "sailing_id",
            SailingColumns.ROUTE: "route",
            SailingColumns.VESSEL: "vessel",
            SailingColumns.ETD_ISO: "etd_iso",
            ExposureColumns.FRESH_UNITS: "fresh_units",
            ExposureColumns.FUEL_UNITS: "fuel_units",
        }
    )

    # Select final columns in order
    output_cols = [
        "sailing_id",
        "route",
        "vessel",
        "etd_iso",
        "etd_bin",
        "p_sail_uncal",
        "p_sail",
        "fresh_units",
        "fuel_units",
    ]

    predictions_df = predictions_df[output_cols]

    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-sailing predictions for M3 handoff"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Predict all sailings in data (no time filter)",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=72,
        help="Hours ahead to predict (default: 72)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="per_sailing_predictions.csv",
        help="Output CSV path (default: per_sailing_predictions.csv)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("IslandSense M2 Per-Sailing Prediction Pipeline")
    print("=" * 70)

    # Load config
    config = get_config()

    # Load models
    model, calibrator = load_models()

    # Load data
    sailings_df, metocean_df, tides_df, status_df, exposure_df = load_data(config)

    # Filter sailings by horizon
    if args.all:
        filtered_sailings = filter_sailings_by_horizon(sailings_df, horizon_hours=None)
    else:
        filtered_sailings = filter_sailings_by_horizon(
            sailings_df, horizon_hours=args.horizon_hours
        )

    # Generate predictions
    predictions_df = predict(
        filtered_sailings,
        metocean_df,
        tides_df,
        status_df,
        exposure_df,
        model,
        calibrator,
        config,
    )

    # Save output
    if len(predictions_df) > 0:
        output_path = Path(args.output)
        predictions_df.to_csv(output_path, index=False)

        print("\n" + "=" * 70)
        print("OUTPUT")
        print("=" * 70)
        print(f"✓ Saved {len(predictions_df)} predictions to {output_path}")
        print(f"\nSchema: {','.join(predictions_df.columns)}")
        print("\nSample (first 3 rows):")
        print(predictions_df.head(3).to_string(index=False))
    else:
        print("\n⚠️  No predictions generated (empty result)")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
