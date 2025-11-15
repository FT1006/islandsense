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
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

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


def evaluate(model, calibrator, X_test, y_test, baseline_brier: float = 0.077):
    """Evaluate model on TEST set (never seen before).

    IMPORTANT: Test set was not used for training OR calibration.

    Args:
        model: Trained model
        calibrator: Fitted calibrator (fitted on calib set)
        X_test: Test features (held out, never touched)
        y_test: Test labels
        baseline_brier: Baseline Brier score to compare against

    Returns:
        Dictionary of metrics
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

    metrics = {
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

    # Save metadata
    meta = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model_type": "xgboost",
        "random_seed": config.random_seed,
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
    metrics = evaluate(model, calibrator, X_test, y_test, baseline_brier)

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
