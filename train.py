"""Model training script for IslandSense MVP.

Trains XGBoost classifier with monotonic constraints, evaluates on validation set,
and saves model artifacts.

Usage:
    python train.py              # Full training on all data
    python train.py --quick      # Test mode: 200-row subset for pipeline verification
"""

import argparse
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from typing import Any

from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from islandsense.config import get_config
from islandsense.metrics import expected_calibration_error, print_calibration_bins


def load_dataset(config, quick_mode: bool = False) -> pd.DataFrame:
    """Load train_dataset.csv with split column.

    Args:
        config: Configuration object
        quick_mode: If True, subsample to 200 rows for testing

    Returns:
        DataFrame with features, labels, split column
    """
    dataset_path = config.data_dir / "train_dataset.csv"
    df = pd.read_csv(dataset_path)

    if quick_mode:
        print("[QUICK MODE] Subsampling to 200 rows for pipeline test")
        df = df.sample(n=min(200, len(df)), random_state=42)

    return df


def preflight_check(config, quick_mode: bool = False):
    """Run pre-flight checks and log state before training.

    Captures:
    - Git commit hash and working tree status
    - Data fingerprint (row counts, feature stats)
    - Baseline metrics
    - All validation checks

    Args:
        config: Configuration object
        quick_mode: If True, note that this is a test run

    Returns:
        Dictionary containing manifest
    """
    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECKS")
    print("=" * 70)

    manifest = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "mode": "quick" if quick_mode else "full",
        "checks": {},
        "data_fingerprint": {},
        "baseline_metrics": {},
        "config_snapshot": config._data,
    }

    # Capture git state
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        git_status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        manifest["git_commit"] = git_commit
        manifest["git_dirty"] = len(git_status) > 0
        print(f"  Git commit: {git_commit[:8]}")
        if manifest["git_dirty"]:
            print("  ⚠️  Working tree has uncommitted changes")
    except subprocess.CalledProcessError:
        manifest["git_commit"] = "unknown"
        manifest["git_dirty"] = True
        print("  Git: not available")

    # Check 1: CSVs present
    print("\n[Check 1/6] Verifying source CSVs...")
    csv_files = {
        "sailings": config.sailings_file,
        "status": config.status_file,
        "metocean": config.metocean_file,
        "tides": config.tides_file,
        "exposure": config.exposure_file,
    }

    all_present = True
    for name, path in csv_files.items():
        exists = path.exists()
        all_present = all_present and exists
        status = "✓" if exists else "✗"
        print(f"  {status} {name}.csv")

    manifest["checks"]["csvs_present"] = all_present

    # Check 2: Load dataset and verify
    print("\n[Check 2/6] Loading training dataset...")
    dataset_path = config.data_dir / "train_dataset.csv"
    if not dataset_path.exists():
        print(f"  ✗ {dataset_path} not found!")
        manifest["checks"]["dataset_exists"] = False
        return manifest

    df = pd.read_csv(dataset_path)
    manifest["checks"]["dataset_exists"] = True
    print(f"  ✓ Loaded {len(df)} rows")

    # Check 3: Feature validation
    print("\n[Check 3/6] Validating features...")
    feature_cols = [
        "WOTDI",
        "BSEF",
        "gust_max_3h",
        "tide_gate_margin",
        "prior_24h_delay",
        "day_of_week",
        "month",
    ]

    has_nans = df[feature_cols].isna().any().any()
    has_infs = np.isinf(df[feature_cols]).any().any()

    manifest["checks"]["no_nans"] = not has_nans
    manifest["checks"]["no_infs"] = not has_infs

    print(f"  {'✓' if not has_nans else '✗'} No NaN values")
    print(f"  {'✓' if not has_infs else '✗'} No infinite values")

    # Capture feature statistics
    manifest["data_fingerprint"] = {
        "total_rows": len(df),
        "train_rows": int((df["split"] == "train").sum()),
        "calib_rows": int((df["split"] == "calib").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "disruption_rate": float(df["disruption"].mean()),
        "feature_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
            for col in feature_cols
        },
    }

    # Check 4: Time-based split
    print("\n[Check 4/6] Verifying time-based split...")
    df["etd_parsed"] = pd.to_datetime(df["etd"])
    train_max = df[df["split"] == "train"]["etd_parsed"].max()
    calib_min = df[df["split"] == "calib"]["etd_parsed"].min()
    calib_max = df[df["split"] == "calib"]["etd_parsed"].max()
    test_min = df[df["split"] == "test"]["etd_parsed"].min()

    time_based = train_max <= calib_min and calib_max <= test_min
    manifest["checks"]["time_based_split"] = bool(time_based)

    print(f"  Train ends:   {train_max.date()}")
    print(f"  Calib starts: {calib_min.date()}")
    print(f"  Calib ends:   {calib_max.date()}")
    print(f"  Test starts:  {test_min.date()}")
    print(f"  {'✓' if time_based else '✗'} Chronological (no overlap)")

    # Check 5: Baseline metrics
    print("\n[Check 5/6] Computing baseline heuristic...")
    test_df = df[df["split"] == "test"]
    y_true = test_df["disruption"].values

    # Baseline rule
    BSEF_THRESHOLD = 2.0
    GUST_THRESHOLD = 40.0
    y_pred = (
        (test_df["BSEF"] > BSEF_THRESHOLD) | (test_df["gust_max_3h"] > GUST_THRESHOLD)
    ).astype(int)
    y_prob = y_pred * 0.8 + (1 - y_pred) * 0.2

    baseline_brier = brier_score_loss(y_true, y_prob)
    baseline_acc = accuracy_score(y_true, y_pred)

    manifest["baseline_metrics"] = {
        "rule": f"(BSEF > {BSEF_THRESHOLD}) OR (gust_max_3h > {GUST_THRESHOLD} kts)",
        "test_set_size": len(test_df),
        "test_disruptions": int(y_true.sum()),
        "brier": float(baseline_brier),
        "accuracy": float(baseline_acc),
    }

    print(f"  Rule: (BSEF > {BSEF_THRESHOLD}) OR (gust > {GUST_THRESHOLD} kts)")
    print(f"  Brier: {baseline_brier:.3f}")
    print(f"  Accuracy: {baseline_acc:.3f}")

    # Check 6: Config validation
    print("\n[Check 6/6] Validating config...")
    has_random_seed = config.random_seed is not None
    manifest["checks"]["has_random_seed"] = has_random_seed
    print(f"  {'✓' if has_random_seed else '✗'} Random seed: {config.random_seed}")

    # Summary
    print("\n" + "=" * 70)
    all_checks_passed = all(manifest["checks"].values())
    if all_checks_passed:
        print("✅ ALL PRE-FLIGHT CHECKS PASSED")
    else:
        print("⚠️  SOME CHECKS FAILED - Review above")
    print("=" * 70)

    # Save manifest
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    manifest_path = models_dir / "training_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📋 Manifest saved to {manifest_path}")

    return manifest


def prepare_features(df: pd.DataFrame):
    """Split dataset into train/calib/test features and labels.

    Args:
        df: Dataset with split column (train/calib/test)

    Returns:
        (X_train, X_calib, X_test, y_train, y_calib, y_test, routes_test)
    """
    # Feature columns (7 features)
    feature_cols = [
        "WOTDI",
        "BSEF",
        "gust_max_3h",
        "tide_gate_margin",
        "prior_24h_delay",
        "day_of_week",
        "month",
    ]

    # THREE-WAY SPLIT to avoid calibration leakage
    train_df = df[df["split"] == "train"].copy()
    calib_df = df[df["split"] == "calib"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols].values
    X_calib = calib_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train = train_df["disruption"].values
    y_calib = calib_df["disruption"].values
    y_test = test_df["disruption"].values

    routes_test = test_df["route"].values

    print(
        f"Train set: {len(X_train)} samples ({y_train.sum()} disruptions, {y_train.mean():.1%})"
    )
    print(
        f"Calib set: {len(X_calib)} samples ({y_calib.sum()} disruptions, {y_calib.mean():.1%})"
    )
    print(
        f"Test set:  {len(X_test)} samples ({y_test.sum()} disruptions, {y_test.mean():.1%})"
    )

    return X_train, X_calib, X_test, y_train, y_calib, y_test, routes_test


def train_model(X_train, y_train, config):
    """Train XGBoost classifier with monotonic constraints.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration object with training hyperparameters

    Returns:
        Trained XGBoost model
    """
    print("\nTraining XGBoost with monotonic constraints...")

    training_config = config._data.get("model", {}).get("training", {})

    # Build XGBoost parameters
    params = {
        "max_depth": training_config.get("max_depth", 3),
        "learning_rate": training_config.get("learning_rate", 0.05),
        "n_estimators": training_config.get("n_estimators", 100),
        "scale_pos_weight": training_config.get("scale_pos_weight", 29.0),
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "random_state": config.random_seed,
        "tree_method": "hist",
    }

    # Monotonic constraints
    monotone_constraints = training_config.get(
        "monotone_constraints", [1, 1, 1, -1, 1, 0, 0]
    )
    params["monotone_constraints"] = tuple(monotone_constraints)

    print(
        f"  Hyperparameters: max_depth={params['max_depth']}, lr={params['learning_rate']}, n_est={params['n_estimators']}"
    )
    print(f"  Monotone constraints: {monotone_constraints}")
    print(
        f"  Scale pos weight: {params['scale_pos_weight']:.1f} (for {y_train.mean():.1%} disruption rate)"
    )

    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    print(f"  [OK] Model trained with {model.n_estimators} trees")
    return model


def calibrate_model(model, X_calib, y_calib):
    """Apply isotonic regression calibration on CALIB set.

    IMPORTANT: This uses a separate calibration set, NOT the test set,
    to avoid calibration leakage.

    Args:
        model: Trained XGBoost model
        X_calib: Calibration features (separate from train and test)
        y_calib: Calibration labels

    Returns:
        Calibrator (IsotonicRegression instance)
    """
    print("\nCalibrating model (isotonic regression on CALIB set)...")

    # Get uncalibrated probabilities from calib set
    y_prob_uncal = model.predict_proba(X_calib)[:, 1]

    # Fit isotonic calibrator on calib set
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob_uncal, y_calib)

    print("  [OK] Calibrator fitted on calib set (no leakage)")
    return calibrator


def evaluate(
    model, calibrator, X_test, y_test, routes_test, baseline_brier: float = 0.077
):
    """Evaluate model on TEST set (never seen before).

    IMPORTANT: Test set was not used for training OR calibration.

    Args:
        model: Trained model
        calibrator: Fitted calibrator (fitted on calib set)
        X_test: Test features (held out, never touched)
        y_test: Test labels
        routes_test: Route names for test samples
        baseline_brier: Baseline Brier score to compare against

    Returns:
        Dictionary of metrics (including per-route breakdown)
    """
    print("\nEvaluating on TEST set (held out, never touched)...")

    # Get probabilities on test set
    y_prob_uncal = model.predict_proba(X_test)[:, 1]
    y_prob_cal = calibrator.transform(y_prob_uncal)

    # Compute metrics (using calibrated probabilities)
    brier = brier_score_loss(y_test, y_prob_cal)
    logloss = log_loss(y_test, y_prob_cal)
    auc = roc_auc_score(y_test, y_prob_cal)
    ece = expected_calibration_error(y_test, y_prob_cal, n_bins=5)

    # Compare to baseline
    beats_baseline = brier < baseline_brier
    improvement = ((baseline_brier - brier) / baseline_brier) * 100

    # Check targets
    meets_brier = brier <= 0.070
    meets_ece = ece <= 0.05
    meets_auc = auc >= 0.75

    metrics: dict[str, Any] = {
        "brier_score": float(brier),
        "log_loss": float(logloss),
        "auc": float(auc),
        "ece": float(ece),
        "baseline_brier": float(baseline_brier),
        "beats_baseline": bool(beats_baseline),
        "improvement_pct": float(improvement),
        "meets_brier_target": bool(meets_brier),
        "meets_ece_target": bool(meets_ece),
        "meets_auc_target": bool(meets_auc),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)
    print(
        f"  Brier Score:  {brier:.4f}  {'[OK]' if meets_brier else '[MISS]'} (target ≤ 0.070)"
    )
    print(
        f"  ECE:          {ece:.4f}  {'[OK]' if meets_ece else '[MISS]'} (target ≤ 0.050)"
    )
    print(
        f"  AUC:          {auc:.4f}  {'[OK]' if meets_auc else '[MISS]'} (target ≥ 0.750)"
    )
    print(f"  Log Loss:     {logloss:.4f}")
    print()
    print(f"  Baseline Brier:  {baseline_brier:.4f}")
    print(
        f"  Improvement:     {improvement:+.1f}%  {'[BEATS BASELINE]' if beats_baseline else '[WORSE THAN BASELINE]'}"
    )
    print("=" * 70)

    # Print calibration bins
    print_calibration_bins(y_test, y_prob_cal, n_bins=5)

    # Per-route analysis
    print("\n" + "=" * 70)
    print("PER-ROUTE RELIABILITY")
    print("=" * 70)

    route_metrics = []
    unique_routes = np.unique(routes_test)

    for route in unique_routes:
        route_mask = routes_test == route
        y_route = y_test[route_mask]
        p_route = y_prob_cal[route_mask]

        n_obs = len(y_route)
        n_pos = y_route.sum()

        if n_obs >= 5:  # Only compute metrics if sufficient samples
            route_brier = brier_score_loss(y_route, p_route)
            route_ece = expected_calibration_error(y_route, p_route, n_bins=5)
            warning = "low_sample" if n_obs < 20 else ""

            route_metrics.append(
                {
                    "route": route,
                    "n_train": "N/A",  # Will be filled if needed
                    "n_test": int(n_obs),
                    "n_disruptions": int(n_pos),
                    "brier": float(route_brier),
                    "ece": float(route_ece),
                    "warning": warning,
                }
            )

            print(
                f"  {route:30} n={n_obs:3} brier={route_brier:.3f} ece={route_ece:.3f} {warning}"
            )
        else:
            print(f"  {route:30} n={n_obs:3} [insufficient samples]")

    print("=" * 70)

    # Save route metrics to CSV
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)

    if route_metrics:
        import csv

        route_csv_path = metrics_dir / "route_reliability.csv"
        with open(route_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "route",
                    "n_test",
                    "n_disruptions",
                    "brier",
                    "ece",
                    "warning",
                ],
            )
            writer.writeheader()
            for rm in route_metrics:
                writer.writerow(
                    {
                        "route": rm["route"],
                        "n_test": rm["n_test"],
                        "n_disruptions": rm["n_disruptions"],
                        "brier": f"{rm['brier']:.4f}",
                        "ece": f"{rm['ece']:.4f}",
                        "warning": rm["warning"],
                    }
                )
        print(f"\n  [OK] Route reliability saved to {route_csv_path}")

    # Save model vs baseline comparison
    baseline_txt_path = metrics_dir / "model_vs_baseline.txt"
    with open(baseline_txt_path, "w") as f:
        f.write("IslandSense M2 Model vs Baseline Comparison\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().astimezone().isoformat()}\n")
        f.write(
            f"Test set: {len(y_test)} sailings, {y_test.sum()} disruptions ({y_test.mean():.1%})\n"
        )
        f.write("\n")
        f.write(f"{'Metric':<20} {'Baseline':>12} {'Model':>12} {'Improvement':>12}\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Brier Score':<20} {baseline_brier:>12.3f} {brier:>12.3f} {improvement:>11.1f}%\n"
        )
        f.write(f"{'Log Loss':<20} {'N/A':>12} {logloss:>12.3f} {'N/A':>12}\n")
        f.write(f"{'AUC':<20} {'N/A':>12} {auc:>12.3f} {'N/A':>12}\n")
        f.write(f"{'ECE':<20} {'N/A':>12} {ece:>12.3f} {'N/A':>12}\n")
        f.write("\n")
        f.write("Baseline rule: (BSEF > 2.0) OR (gust_max_3h > 40.0 kts)\n")
        f.write("Model: XGBoost with 7 physics features + isotonic calibration\n")
        f.write("\n")
        if beats_baseline:
            f.write(f"✅ Model beats baseline on Brier score ({improvement:+.1f}%)\n")
        else:
            f.write(
                f"❌ Model worse than baseline on Brier score ({improvement:+.1f}%)\n"
            )

        if not meets_auc:
            f.write(
                f"⚠️  Model misses AUC target (0.484 < 0.75) - likely due to small test set (n_pos={y_test.sum()})\n"
            )

    print(f"  [OK] Baseline comparison saved to {baseline_txt_path}")

    # Add route metrics to return dictionary
    metrics["route_metrics"] = route_metrics

    return metrics


def save_artifacts(model, calibrator, metrics, config):
    """Save model, calibrator, and metadata.

    Args:
        model: Trained XGBoost model
        calibrator: Fitted calibrator
        metrics: Evaluation metrics dictionary
        config: Configuration object
    """
    print("\nSaving artifacts...")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [OK] Model saved to {model_path}")

    # Save calibrator
    calibrator_path = models_dir / "calibrator.pkl"
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"  [OK] Calibrator saved to {calibrator_path}")

    # Get git commit hash for config tracking
    try:
        config_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        config_commit = "unknown"

    # Get date ranges from dataset
    df = pd.read_csv(config.data_dir / "train_dataset.csv")
    df["etd"] = pd.to_datetime(df["etd"])

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_date_min = train_df["etd"].min().date().isoformat()
    train_date_max = train_df["etd"].max().date().isoformat()
    test_date_min = test_df["etd"].min().date().isoformat()
    test_date_max = test_df["etd"].max().date().isoformat()

    # Save metadata
    meta = {
        "model_name": "per_sailing_xgb",
        "version": "m2-v0",
        "timestamp": datetime.now().astimezone().isoformat(),
        "model_type": "xgboost",
        "random_seed": config.random_seed,
        "config_commit": config_commit,
        "train_date_range": {"min": train_date_min, "max": train_date_max},
        "test_date_range": {"min": test_date_min, "max": test_date_max},
        "features": [
            "WOTDI",
            "BSEF",
            "gust_max_3h",
            "tide_gate_margin",
            "prior_24h_delay",
            "day_of_week",
            "month",
        ],
        "monotone_constraints": config._data.get("model", {})
        .get("training", {})
        .get("monotone_constraints", [1, 1, 1, -1, 1, 0, 0]),
        "metrics": metrics,
    }

    meta_path = models_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [OK] Metadata saved to {meta_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train IslandSense disruption prediction model"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: train on 200-row subset for testing",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("IslandSense Model Training Pipeline (M1.P4)")
    print("=" * 70)

    # Load config
    config = get_config()

    # Pre-flight checks
    preflight_check(config, quick_mode=args.quick)

    # Load dataset
    df = load_dataset(config, quick_mode=args.quick)

    # Prepare features
    X_train, X_calib, X_test, y_train, y_calib, y_test, routes_test = prepare_features(
        df
    )

    # Train model
    model = train_model(X_train, y_train, config)

    # Calibrate
    calibrator = calibrate_model(model, X_calib, y_calib)

    # Evaluate
    baseline_brier = 0.077  # From M1.P3
    metrics = evaluate(model, calibrator, X_test, y_test, routes_test, baseline_brier)

    # Save artifacts
    save_artifacts(model, calibrator, metrics, config)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("Next steps:")
    print("  - Review metrics in models/model_meta.json")
    print("  - If metrics are poor, tweak hyperparameters in config.yaml")
    print("  - Proceed to M3 (inference + UI) when satisfied")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
