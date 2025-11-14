"""Per-sailing disruption model training and prediction.

Trains a gradient-boosted tree classifier (XGBoost/LightGBM).
This is the ONLY ML component in the system.
To be implemented in M2.
"""


def train_model(features_df, labels_df, config):
    """
    Train per-sailing disruption classifier.

    To implement (M2):
    - Train/validation split by route
    - XGBoost or LightGBM classifier
    - Compute Brier score, ECE
    - Save model to models/model.pkl

    Returns:
        Trained model object
    """
    raise NotImplementedError("To be implemented in M2")


def predict(model, features_df):
    """
    Generate per-sailing disruption probabilities.

    Returns:
        pd.DataFrame with sailing_id, p_sail
    """
    raise NotImplementedError("To be implemented in M2")


def evaluate(model, features_df, labels_df):
    """
    Evaluate model reliability (Brier, ECE).

    Returns:
        Dict with metrics per route
    """
    raise NotImplementedError("To be implemented in M2")
