"""Decision window aggregation and JDI computation.

Converts per-sailing predictions into category-level JDI scores.
All math, no ML.
To be implemented in M3.
"""


def compute_windows(sailings_df, config):
    """
    Assign sailings to decision windows (e.g., "Next 72h", "Today", "Tomorrow").

    Returns:
        pd.DataFrame with sailing_id, window_id mappings
    """
    raise NotImplementedError("To be implemented in M3")


def compute_category_jdi(sailings_df, predictions_df, exposure_df, windows_df, config):
    """
    Compute JDI per category per window.

    For each (category, window):
    - E_loss = Σ exposure[c,s] * p_sail(s)
    - JDI = linear scale E_loss to [0, 100]
    - Band = Green/Amber/Red

    Returns:
        pd.DataFrame with window, category, E_loss, JDI, band
    """
    raise NotImplementedError("To be implemented in M3")
