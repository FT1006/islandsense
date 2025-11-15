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
from islandsense.features import (
    compute_features,
    create_label,
    validate_features,
    validate_features_comprehensive,
)


def main():
    """Run the full data preparation pipeline."""
    print("=" * 80)
    print("IslandSense Data Preparation Pipeline (M1)")
    print("=" * 80)

    config = get_config()

    # Step 1: Load CSVs
    print("\n[Step 1/9] Loading CSVs...")
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
    print("\n[Step 2/9] Validating schemas...")
    validate_sailings(sailings_df)
    validate_status(status_df)
    validate_metocean(metocean_df)
    validate_tides(tides_df)
    validate_exposure(exposure_df)
    print("  [OK] All schemas valid")

    # Step 3: Compute features
    print("\n[Step 3/9] Computing features...")
    features_df = compute_features(sailings_df, metocean_df, tides_df, status_df)

    # Step 4: Create labels
    print("\n[Step 4/9] Creating labels...")
    labels = create_label(
        status_df, disruption_delay_minutes=config.disruption_delay_minutes
    )
    print(f"  [OK] Labels created: {len(labels)}")
    print("  Label distribution:")
    print(f"    Disrupted (1): {labels.sum()} ({labels.mean():.1%})")
    print(f"    Normal (0): {(1 - labels).sum()} ({(1 - labels.mean()):.1%})")

    # Step 5: Join into training table
    print("\n[Step 5/9] Assembling training table...")
    train_df = (
        sailings_df[["sailing_id", "route", "vessel", "etd_iso"]]
        .merge(features_df, on="sailing_id")
        .merge(labels.to_frame(), left_on="sailing_id", right_index=True)
        .merge(exposure_df, on="sailing_id")
    )

    print(f"  [OK] Training table shape: {train_df.shape}")
    print(f"  Columns: {list(train_df.columns)}")

    # Step 5b: Add train/calib/test split (date-based, not random)
    # THREE-WAY SPLIT to avoid calibration leakage
    print("\n[Step 5b] Adding train/calib/test split (date-based)...")
    train_df["etd"] = pd.to_datetime(train_df["etd_iso"])
    train_df = train_df.sort_values("etd").reset_index(drop=True)

    # 70/15/15 split based on chronological order
    train_idx = int(len(train_df) * 0.70)
    calib_idx = int(len(train_df) * 0.85)  # 70% + 15% = 85%

    train_df["split"] = "train"
    train_df.loc[train_idx:calib_idx, "split"] = "calib"
    train_df.loc[calib_idx:, "split"] = "test"

    train_date = (
        train_df.loc[train_idx, "etd"]
        if train_idx < len(train_df)
        else train_df.iloc[-1]["etd"]
    )
    calib_date = (
        train_df.loc[calib_idx, "etd"]
        if calib_idx < len(train_df)
        else train_df.iloc[-1]["etd"]
    )

    train_count = (train_df["split"] == "train").sum()
    calib_count = (train_df["split"] == "calib").sum()
    test_count = (train_df["split"] == "test").sum()

    print(
        f"  Train: {train_count} sailings ({train_count / len(train_df):.1%}) - up to {train_date.date()}"
    )
    print(
        f"  Calib: {calib_count} sailings ({calib_count / len(train_df):.1%}) - {train_date.date()} to {calib_date.date()}"
    )
    print(
        f"  Test:  {test_count} sailings ({test_count / len(train_df):.1%}) - from {calib_date.date()} onwards"
    )
    print(
        f"  Train disruption rate: {train_df[train_df['split'] == 'train']['disruption'].mean():.1%}"
    )
    print(
        f"  Calib disruption rate: {train_df[train_df['split'] == 'calib']['disruption'].mean():.1%}"
    )
    print(
        f"  Test disruption rate:  {train_df[train_df['split'] == 'test']['disruption'].mean():.1%}"
    )

    # Step 6: Validate features
    print("\n[Step 6/9] Validating features...")
    validate_features(features_df)

    # Step 7: Comprehensive physics validation
    print("\n[Step 7/9] Cross-validating features vs source data...")
    validate_features_comprehensive(features_df, sailings_df, metocean_df)

    # Step 8: Baseline heuristic evaluation
    print("\n[Step 8/9] Evaluating baseline heuristic (physics rule without ML)...")
    from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

    # Get TEST set (never used for training or calibration)
    test_df = train_df[train_df["split"] == "test"].copy()
    y_true = test_df["disruption"].values

    # Baseline heuristic: Disrupted if (BSEF > threshold) OR (gust_max_3h > threshold)
    # Tune thresholds on train set to be reasonable (not optimized, just sensible)
    BSEF_THRESHOLD = 2.0  # High beam-sea exposure
    GUST_THRESHOLD = 40.0  # High gust speed (kts)

    # Simple rule: predict disruption = 1 if either threshold exceeded
    y_pred_baseline = (
        (test_df["BSEF"] > BSEF_THRESHOLD) | (test_df["gust_max_3h"] > GUST_THRESHOLD)
    ).astype(int)

    # For probabilistic metrics, assume 0.8 confidence when predicting 1, 0.2 when predicting 0
    # (simple heuristic doesn't give probabilities, so we use fixed confidence levels)
    y_prob_baseline = y_pred_baseline * 0.8 + (1 - y_pred_baseline) * 0.2

    # Compute metrics
    baseline_brier = brier_score_loss(y_true, y_prob_baseline)
    baseline_logloss = log_loss(y_true, y_prob_baseline)
    baseline_acc = accuracy_score(y_true, y_pred_baseline)
    baseline_disruption_rate = y_pred_baseline.mean()

    print(
        f"  Baseline rule: (BSEF > {BSEF_THRESHOLD}) OR (gust_max_3h > {GUST_THRESHOLD} kts)"
    )
    print(f"  Predicted disruption rate: {baseline_disruption_rate:.1%}")
    print(f"  Accuracy:    {baseline_acc:.3f}")
    print(f"  Brier score: {baseline_brier:.3f}")
    print(f"  Log loss:    {baseline_logloss:.3f}")
    print(
        f"\n  [BASELINE FLOOR] Model must beat Brier < {baseline_brier:.3f} to be worthwhile"
    )

    # Step 9: Save dataset
    print("\n[Step 9/9] Saving dataset...")
    output_path = config.data_dir / "train_dataset.csv"
    train_df.to_csv(output_path, index=False)
    print(f"  [OK] Dataset saved to {output_path}")
    print(f"  Columns: {list(train_df.columns)}")

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
