"""Feature engineering for IslandSense MVP.

Computes physics-based features (WOTDI, BSEF, etc.) from raw sailing data.
To be implemented in M1.
"""


def compute_features(sailings_df, metocean_df, tides_df):
    """
    Compute per-sailing feature matrix.

    Features to implement (M1):
    - WOTDI: wind-tide misalignment index
    - BSEF: beam-sea exposure factor
    - gust_max_3h: max gust in prior 3 hours
    - tide_gate_margin: time to next low tide
    - day_of_week, month: temporal features
    - prior_24h_delay: historical context

    Returns:
        pd.DataFrame with features per sailing
    """
    raise NotImplementedError("To be implemented in M1")


def create_label(status_df, disruption_delay_minutes=120):
    """
    Create binary disruption label from status.

    disruption = 1 if (status == "cancelled") OR (delay_min > threshold)

    Returns:
        pd.Series with disruption labels
    """
    raise NotImplementedError("To be implemented in M1")
