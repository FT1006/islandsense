"""Action recommendation and impact calculation.

Recommends shift actions (do nothing, bring forward, partial air) based on JDI.
Estimates hours avoided (Fresh) or trailers avoided (Fuel).
All heuristic algebra, no ML.
To be implemented in M3.
"""


def recommend_actions(category_jdi_df, config):
    """
    Recommend action per (category, window) based on JDI band.

    Rules:
    - Green → do nothing
    - Amber → bring forward
    - Red → partial air

    Returns:
        pd.DataFrame with window, category, action, shift_fraction
    """
    raise NotImplementedError("To be implemented in M3")


def compute_impact(actions_df, category_jdi_df, config):
    """
    Compute impact estimates (hours avoided, trailers avoided).

    Fresh: hours_avoided = E_loss * shift_fraction * k_hours_per_unit
    Fuel: trailers_avoided = (E_loss * shift_fraction) / units_per_trailer

    Returns:
        pd.DataFrame with window, category, action, impact_hours, impact_trailers
    """
    raise NotImplementedError("To be implemented in M3")


def allocate_to_sailings(actions_df, sailings_df, exposure_df):
    """
    Push window-level actions back down to per-sailing shift fractions.

    Proportional allocation: y_s ∝ exposure[c,s]

    Returns:
        pd.DataFrame with sailing_id, window, category, y_s
    """
    raise NotImplementedError("To be implemented in M3")
