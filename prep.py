"""Data preparation pipeline: load CSVs -> features -> labels -> training table.

This script orchestrates the full M1 pipeline:
1. Load synthetic CSVs (from M0)
2. Validate schemas
3. Compute features (from features.py)
4. Create labels
5. Join into training table
6. Validate feature parity and sanity checks
"""

import pandas as pd
from islandsense.config import get_config
from islandsense.schema import (
    validate_sailings,
    validate_status,
    validate_metocean,
    validate_tides,
    validate_exposure,
)
from islandsense.features import compute_features, create_label, validate_features


def main():
    """Run the full data preparation pipeline."""
    print("=" * 80)
    print("IslandSense Data Preparation Pipeline (M1)")
    print("=" * 80)

    config = get_config()

    # Step 1: Load CSVs
    print("\n[Step 1/6] Loading CSVs...")
    sailings_df = pd.read_csv(config.sailings_file)
    status_df = pd.read_csv(config.status_file)
    metocean_df = pd.read_csv(config.metocean_file)
    tides_df = pd.read_csv(config.tides_file)
    exposure_df = pd.read_csv(config.exposure_file)

    print(f"  Loaded {len(sailings_df)} sailings")
    print(f"  Loaded {len(status_df)} status records")
    print(f"  Loaded {len(metocean_df)} metocean records")
    print(f"  Loaded {len(tides_df)} tide records")
    print(f"  Loaded {len(exposure_df)} exposure records")

    # Step 2: Validate schemas
    print("\n[Step 2/6] Validating schemas...")
    validate_sailings(sailings_df)
    validate_status(status_df)
    validate_metocean(metocean_df)
    validate_tides(tides_df)
    validate_exposure(exposure_df)
    print("  [OK] All schemas valid")

    # Step 3: Compute features
    print("\n[Step 3/6] Computing features...")
    features_df = compute_features(sailings_df, metocean_df, tides_df, status_df)

    # Step 4: Create labels
    print("\n[Step 4/6] Creating labels...")
    labels = create_label(
        status_df, disruption_delay_minutes=config.disruption_delay_minutes
    )
    print(f"  [OK] Labels created: {len(labels)}")
    print("  Label distribution:")
    print(f"    Disrupted (1): {labels.sum()} ({labels.mean():.1%})")
    print(f"    Normal (0): {(1 - labels).sum()} ({(1 - labels.mean()):.1%})")

    # Step 5: Join into training table
    print("\n[Step 5/6] Assembling training table...")
    train_df = (
        sailings_df[["sailing_id", "route", "vessel", "etd_iso"]]
        .merge(features_df, on="sailing_id")
        .merge(labels.to_frame(), left_on="sailing_id", right_index=True)
        .merge(exposure_df, on="sailing_id")
    )

    print(f"  [OK] Training table shape: {train_df.shape}")
    print(f"  Columns: {list(train_df.columns)}")

    # Step 6: Validate features
    print("\n[Step 6/6] Validating features...")
    validate_features(features_df)

    # Summary
    print("\n" + "=" * 80)
    print("Data Preparation Complete")
    print("=" * 80)
    print(f"Training table: {train_df.shape[0]} rows x {train_df.shape[1]} columns")
    print(f"Feature columns: {features_df.shape[1] - 1}")  # Exclude sailing_id
    print(f"Disruption rate: {train_df['disruption'].mean():.1%}")
    print("\nReady for M2 model training!")
    print("=" * 80)

    return train_df


if __name__ == "__main__":
    train_df = main()
