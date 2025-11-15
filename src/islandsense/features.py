"""Feature engineering for IslandSense MVP.

Computes physics-based features (WOTDI, BSEF, etc.) from raw sailing data.
All formulas match synth.py to ensure feature parity with label generation.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional

from islandsense.schema import (
    SailingColumns,
    StatusColumns,
    MetoceanColumns,
    TideColumns,
)


# ============================================================================
# Helper Functions
# ============================================================================


def wrap_angle(deg: float) -> float:
    """Wrap angle to [-180, 180] range."""
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def _index_by_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Convert timestamp column to datetime and index for fast lookups."""
    df_indexed = df.copy()
    df_indexed["ts"] = pd.to_datetime(df_indexed[ts_col])
    df_indexed = df_indexed.set_index("ts").sort_index()
    return df_indexed


def _get_nearest_row(etd: pd.Timestamp, df_indexed: pd.DataFrame) -> pd.Series:
    """Get nearest row from indexed DataFrame (nearest hour)."""
    nearest_idx = df_indexed.index.get_indexer([etd], method="nearest")[0]
    return df_indexed.iloc[nearest_idx]


# ============================================================================
# Physics Features (matching synth.py formulas exactly)
# ============================================================================


def compute_BSEF(metocean_df: pd.DataFrame, sailings_df: pd.DataFrame) -> pd.Series:
    """
    Compute Beam-Sea Exposure Factor per sailing.

    Formula (from synth.py:273-274):
        rel_wave_deg = wrap_angle(wave_dir_deg - head_deg)
        bsef = abs(sin(radians(rel_wave_deg))) * hs_m

    Higher when waves are perpendicular to vessel heading (beam sea).

    Returns:
        pd.Series indexed by sailing_id
    """
    metocean_indexed = _index_by_timestamp(metocean_df, MetoceanColumns.TS_ISO)

    bsef_values = []
    sailing_ids = []

    for _, sailing in sailings_df.iterrows():
        etd = pd.to_datetime(sailing[SailingColumns.ETD_ISO])
        head_deg = sailing[SailingColumns.HEAD_DEG]

        metocean_row = _get_nearest_row(etd, metocean_indexed)

        wave_dir_deg = metocean_row[MetoceanColumns.WAVE_DIR_DEG]
        hs_m = metocean_row[MetoceanColumns.HS_M]

        # BSEF formula
        rel_wave_deg = wrap_angle(wave_dir_deg - head_deg)
        bsef = abs(np.sin(np.radians(rel_wave_deg))) * hs_m

        bsef_values.append(bsef)
        sailing_ids.append(sailing[SailingColumns.SAILING_ID])

    return pd.Series(bsef_values, index=sailing_ids, name="BSEF")


def compute_WOTDI(metocean_df: pd.DataFrame, sailings_df: pd.DataFrame) -> pd.Series:
    """
    Compute Wind-Tide Directional Index (simplified as wind-heading misalignment).

    Formula (from synth.py:277-278):
        rel_wind_deg = wrap_angle(wind_dir_deg - head_deg)
        wotdi = abs(sin(radians(rel_wind_deg))) * (wind_kts / 20.0)

    Note: True tide flow direction not available; using wind-heading proxy.

    Returns:
        pd.Series indexed by sailing_id
    """
    metocean_indexed = _index_by_timestamp(metocean_df, MetoceanColumns.TS_ISO)

    wotdi_values = []
    sailing_ids = []

    for _, sailing in sailings_df.iterrows():
        etd = pd.to_datetime(sailing[SailingColumns.ETD_ISO])
        head_deg = sailing[SailingColumns.HEAD_DEG]

        metocean_row = _get_nearest_row(etd, metocean_indexed)

        wind_dir_deg = metocean_row[MetoceanColumns.WIND_DIR_DEG]
        wind_kts = metocean_row[MetoceanColumns.WIND_KTS]

        # WOTDI formula
        rel_wind_deg = wrap_angle(wind_dir_deg - head_deg)
        wotdi = abs(np.sin(np.radians(rel_wind_deg))) * (wind_kts / 20.0)

        wotdi_values.append(wotdi)
        sailing_ids.append(sailing[SailingColumns.SAILING_ID])

    return pd.Series(wotdi_values, index=sailing_ids, name="WOTDI")


def compute_gust_max_3h(
    metocean_df: pd.DataFrame, sailings_df: pd.DataFrame
) -> pd.Series:
    """
    Compute maximum gust speed in prior 3 hours before departure.

    Returns:
        pd.Series indexed by sailing_id
    """
    metocean_indexed = _index_by_timestamp(metocean_df, MetoceanColumns.TS_ISO)

    gust_max_values = []
    sailing_ids = []

    for _, sailing in sailings_df.iterrows():
        etd = pd.to_datetime(sailing[SailingColumns.ETD_ISO])

        # Query 3-hour window: [ETD - 3h, ETD]
        start_time = etd - timedelta(hours=3)
        window_data = metocean_indexed[
            (metocean_indexed.index >= start_time) & (metocean_indexed.index <= etd)
        ]

        if len(window_data) == 0:
            # Fallback: use nearest point if no data in window
            metocean_row = _get_nearest_row(etd, metocean_indexed)
            gust_max = metocean_row[MetoceanColumns.GUST_KTS]
        else:
            gust_max = window_data[MetoceanColumns.GUST_KTS].max()

        gust_max_values.append(gust_max)
        sailing_ids.append(sailing[SailingColumns.SAILING_ID])

    return pd.Series(gust_max_values, index=sailing_ids, name="gust_max_3h")


# ============================================================================
# Temporal Features
# ============================================================================


def compute_tide_gate_margin(
    tides_df: pd.DataFrame, sailings_df: pd.DataFrame
) -> pd.Series:
    """
    Compute minutes until next low tide after departure.

    Returns:
        pd.Series indexed by sailing_id (continuous, in minutes)
    """
    tides_indexed = _index_by_timestamp(tides_df, TideColumns.TS_ISO)

    margin_values = []
    sailing_ids = []

    for _, sailing in sailings_df.iterrows():
        etd = pd.to_datetime(sailing[SailingColumns.ETD_ISO])

        # Filter tides after ETD (next 24h)
        future_tides = tides_indexed[
            (tides_indexed.index > etd)
            & (tides_indexed.index <= etd + timedelta(hours=24))
        ]

        if len(future_tides) < 2:
            # Not enough data to find low tide
            margin_minutes = 1440  # Sentinel: 24h
        else:
            # Find local minima in tide_m (low tides)
            tide_heights = future_tides[TideColumns.TIDE_M].values
            tide_times = future_tides.index

            # Simple approach: find first global minimum in next 24h
            min_idx = np.argmin(tide_heights)
            next_low_tide_time = tide_times[min_idx]

            # Calculate minutes to low tide
            margin_minutes = (next_low_tide_time - etd).total_seconds() / 60

        margin_values.append(margin_minutes)
        sailing_ids.append(sailing[SailingColumns.SAILING_ID])

    return pd.Series(margin_values, index=sailing_ids, name="tide_gate_margin")


def compute_prior_24h_delay(
    status_df: pd.DataFrame, sailings_df: pd.DataFrame
) -> pd.Series:
    """
    Compute mean delay of sailings on same route in prior 24 hours.

    Returns:
        pd.Series indexed by sailing_id
    """
    # Prepare status with sailing info
    sailings_with_status = sailings_df.merge(
        status_df, on=SailingColumns.SAILING_ID, how="left"
    )
    sailings_with_status["etd"] = pd.to_datetime(
        sailings_with_status[SailingColumns.ETD_ISO]
    )

    prior_delay_values = []
    sailing_ids = []

    for _, sailing in sailings_df.iterrows():
        etd = pd.to_datetime(sailing[SailingColumns.ETD_ISO])
        route = sailing[SailingColumns.ROUTE]
        sailing_id = sailing[SailingColumns.SAILING_ID]

        # Filter: same route, ETD in [current_ETD - 24h, current_ETD)
        prior_sailings = sailings_with_status[
            (sailings_with_status[SailingColumns.ROUTE] == route)
            & (sailings_with_status["etd"] >= etd - timedelta(hours=24))
            & (sailings_with_status["etd"] < etd)
            & (sailings_with_status[SailingColumns.SAILING_ID] != sailing_id)
        ]

        if len(prior_sailings) == 0:
            # No prior sailings on this route
            mean_delay = 0.0
        else:
            mean_delay = prior_sailings[StatusColumns.DELAY_MIN].mean()
            if pd.isna(mean_delay):
                mean_delay = 0.0

        prior_delay_values.append(mean_delay)
        sailing_ids.append(sailing_id)

    return pd.Series(prior_delay_values, index=sailing_ids, name="prior_24h_delay")


# ============================================================================
# Main Feature Computation
# ============================================================================


def compute_features(
    sailings_df: pd.DataFrame,
    metocean_df: pd.DataFrame,
    tides_df: pd.DataFrame,
    status_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute per-sailing feature matrix with all physics and temporal features.

    Features:
    - WOTDI: wind-heading misalignment index
    - BSEF: beam-sea exposure factor
    - gust_max_3h: max gust in prior 3 hours
    - tide_gate_margin: minutes to next low tide
    - prior_24h_delay: mean delay on same route in prior 24h (if status_df provided)
    - day_of_week: 0=Monday, 6=Sunday
    - month: 1-12

    Returns:
        pd.DataFrame with sailing_id + all feature columns
    """
    print("Computing features...")

    # Compute physics features
    print("  [1/7] Computing WOTDI...")
    wotdi = compute_WOTDI(metocean_df, sailings_df)

    print("  [2/7] Computing BSEF...")
    bsef = compute_BSEF(metocean_df, sailings_df)

    print("  [3/7] Computing gust_max_3h...")
    gust_max_3h = compute_gust_max_3h(metocean_df, sailings_df)

    print("  [4/7] Computing tide_gate_margin...")
    tide_margin = compute_tide_gate_margin(tides_df, sailings_df)

    # Compute temporal features
    print("  [5/7] Computing calendar features...")
    sailings_df["etd"] = pd.to_datetime(sailings_df[SailingColumns.ETD_ISO])
    day_of_week = sailings_df["etd"].dt.dayofweek
    month = sailings_df["etd"].dt.month

    # Compute historical features
    print("  [6/7] Computing prior_24h_delay...")
    if status_df is not None:
        prior_delay = compute_prior_24h_delay(status_df, sailings_df)
    else:
        prior_delay = pd.Series(
            0.0, index=sailings_df[SailingColumns.SAILING_ID], name="prior_24h_delay"
        )

    # Combine into DataFrame
    print("  [7/7] Assembling feature matrix...")
    features_df = pd.DataFrame(
        {
            "sailing_id": sailings_df[SailingColumns.SAILING_ID].values,
            "WOTDI": wotdi.values,
            "BSEF": bsef.values,
            "gust_max_3h": gust_max_3h.values,
            "tide_gate_margin": tide_margin.values,
            "prior_24h_delay": prior_delay.values,
            "day_of_week": day_of_week.values,
            "month": month.values,
        }
    )

    # Fail-fast: check for NaN values
    nan_count = features_df.isna().sum().sum()
    if nan_count > 0:
        print(f"\n[ERROR] Features contain {nan_count} NaN values:")
        print(features_df.isna().sum())
        raise ValueError("Features contain NaN values. Check data quality.")

    print(f"[OK] Features computed: {features_df.shape}")
    return features_df


def create_label(
    status_df: pd.DataFrame, disruption_delay_minutes: int = 120
) -> pd.Series:
    """
    Create binary disruption label from status.

    disruption = 1 if (status == "cancelled") OR (delay_min > threshold)

    Returns:
        pd.Series with sailing_id index, values 0/1
    """
    disrupted = (status_df[StatusColumns.STATUS] == "cancelled") | (
        status_df[StatusColumns.DELAY_MIN] > disruption_delay_minutes
    )

    label = disrupted.astype(int)

    return pd.Series(
        label.values, index=status_df[SailingColumns.SAILING_ID], name="disruption"
    )


# ============================================================================
# Feature Validation
# ============================================================================


def validate_features(features_df: pd.DataFrame) -> None:
    """Print sanity checks for computed features."""
    print("\n" + "=" * 80)
    print("Feature Validation Summary")
    print("=" * 80)

    for col in features_df.columns:
        if col == "sailing_id":
            continue
        print(f"\n{col}:")
        print(f"  min:  {features_df[col].min():.3f}")
        print(f"  mean: {features_df[col].mean():.3f}")
        print(f"  max:  {features_df[col].max():.3f}")
        print(f"  null: {features_df[col].isna().sum()}")

    # Specific physics checks
    print("\n" + "=" * 80)
    print("Physics Sanity Checks:")
    print("=" * 80)

    # BSEF should be [0, ~5] (can't exceed max wave height * 1.0)
    bsef_max = features_df["BSEF"].max()
    if bsef_max > 10:
        print(f"[WARNING] BSEF max = {bsef_max:.2f} > 10 (check formula)")
    else:
        print(f"[OK] BSEF in reasonable range [0, {bsef_max:.2f}]")

    # WOTDI should be [0, ~2.5] (max wind ~50kts / 20 = 2.5)
    wotdi_max = features_df["WOTDI"].max()
    if wotdi_max > 5:
        print(f"[WARNING] WOTDI max = {wotdi_max:.2f} > 5 (check formula)")
    else:
        print(f"[OK] WOTDI in reasonable range [0, {wotdi_max:.2f}]")

    # gust_max_3h should be >= 0
    if (features_df["gust_max_3h"] >= 0).all():
        print("[OK] gust_max_3h >= 0 for all sailings")
    else:
        print("[WARNING] gust_max_3h has negative values")

    # tide_gate_margin should be [0, 1440] minutes
    tide_min = features_df["tide_gate_margin"].min()
    tide_max = features_df["tide_gate_margin"].max()
    if tide_min < 0:
        print(f"[WARNING] tide_gate_margin min = {tide_min:.1f} < 0")
    if tide_max > 1440:
        print(f"[WARNING] tide_gate_margin max = {tide_max:.1f} > 1440 (24h sentinel)")
    print(f"[OK] tide_gate_margin range: [{tide_min:.1f}, {tide_max:.1f}] minutes")

    # prior_24h_delay should be >= 0
    if (features_df["prior_24h_delay"] >= 0).all():
        print("[OK] prior_24h_delay >= 0 for all sailings")
    else:
        print("[WARNING] prior_24h_delay has negative values")

    print("=" * 80)


def validate_features_comprehensive(
    features_df: pd.DataFrame,
    sailings_df: pd.DataFrame,
    metocean_df: pd.DataFrame,
) -> None:
    """Cross-validate features against source data for physics sanity.

    Checks:
    1. BSEF near 0 for head-on/astern, large for beam seas
    2. gust_max_3h >= wind_kts on average
    3. prior_24h_delay higher preceding storms (high BSEF/WOTDI)
    """
    print("\n" + "=" * 80)
    print("Comprehensive Physics Validation")
    print("=" * 80)

    # Prepare metocean indexed by time
    metocean_indexed = _index_by_timestamp(metocean_df, MetoceanColumns.TS_ISO)

    # Build joined data for analysis
    validation_data = []
    for _, sailing in sailings_df.iterrows():
        sailing_id = sailing[SailingColumns.SAILING_ID]
        etd = pd.to_datetime(sailing[SailingColumns.ETD_ISO])
        head_deg = sailing[SailingColumns.HEAD_DEG]

        metocean_row = _get_nearest_row(etd, metocean_indexed)
        wave_dir_deg = metocean_row[MetoceanColumns.WAVE_DIR_DEG]
        wind_kts = metocean_row[MetoceanColumns.WIND_KTS]

        # Get features for this sailing
        feature_row = features_df[features_df["sailing_id"] == sailing_id].iloc[0]

        # Compute relative wave angle
        rel_wave_deg = wrap_angle(wave_dir_deg - head_deg)
        abs_rel_wave = abs(rel_wave_deg)

        validation_data.append(
            {
                "sailing_id": sailing_id,
                "rel_wave_deg": rel_wave_deg,
                "abs_rel_wave": abs_rel_wave,
                "BSEF": feature_row["BSEF"],
                "WOTDI": feature_row["WOTDI"],
                "wind_kts": wind_kts,
                "gust_max_3h": feature_row["gust_max_3h"],
                "prior_24h_delay": feature_row["prior_24h_delay"],
            }
        )

    val_df = pd.DataFrame(validation_data)

    # Check 1: BSEF correlation with beam sea angle
    print("\n[Check 1] BSEF vs wave angle:")
    # Beam seas: abs(rel_wave) near 90 deg should have high BSEF
    # Head/astern: abs(rel_wave) near 0 or 180 should have low BSEF
    beam_seas = val_df[val_df["abs_rel_wave"].between(70, 110)]  # ±20 deg around 90
    head_astern = val_df[(val_df["abs_rel_wave"] < 20) | (val_df["abs_rel_wave"] > 160)]

    if len(beam_seas) > 0 and len(head_astern) > 0:
        mean_bsef_beam = beam_seas["BSEF"].mean()
        mean_bsef_head = head_astern["BSEF"].mean()
        ratio = mean_bsef_beam / mean_bsef_head if mean_bsef_head > 0 else float("inf")

        print(
            f"  Beam seas (70-110°): mean BSEF = {mean_bsef_beam:.3f} (n={len(beam_seas)})"
        )
        print(
            f"  Head/astern (0-20°, 160-180°): mean BSEF = {mean_bsef_head:.3f} (n={len(head_astern)})"
        )

        if ratio > 2.0:
            print(
                f"[OK] Beam BSEF {ratio:.1f}x higher than head/astern (physics valid)"
            )
        elif ratio > 1.2:
            print(f"[WARNING] Beam BSEF only {ratio:.1f}x higher (expected >2x)")
        else:
            print(f"[ERROR] BSEF physics broken: ratio = {ratio:.1f} (should be >2x)")
    else:
        print("[WARNING] Insufficient data to validate BSEF vs wave angle")

    # Check 2: gust_max_3h >= wind_kts on average
    print("\n[Check 2] gust_max_3h vs wind_kts:")
    mean_gust = val_df["gust_max_3h"].mean()
    mean_wind = val_df["wind_kts"].mean()
    gust_exceeds = (val_df["gust_max_3h"] >= val_df["wind_kts"]).mean()

    print(f"  Mean gust_max_3h: {mean_gust:.1f} kts")
    print(f"  Mean wind_kts (at ETD): {mean_wind:.1f} kts")
    print(f"  Gust >= wind in {gust_exceeds:.1%} of sailings")

    if mean_gust > mean_wind and gust_exceeds > 0.7:
        print(f"[OK] Gust captures peaks (mean {mean_gust / mean_wind:.2f}x wind)")
    elif gust_exceeds > 0.5:
        print(f"[WARNING] Gust only {gust_exceeds:.1%} >= wind (expected >70%)")
    else:
        print(f"[ERROR] Gust physics broken: only {gust_exceeds:.1%} >= wind")

    # Check 3: prior_24h_delay higher preceding storms
    print("\n[Check 3] prior_24h_delay vs weather severity:")
    # Define "severe weather" as high BSEF or WOTDI
    val_df["severity"] = val_df["BSEF"] + val_df["WOTDI"]
    severe_threshold = val_df["severity"].quantile(0.75)  # Top 25% severity

    severe = val_df[val_df["severity"] >= severe_threshold]
    mild = val_df[val_df["severity"] < val_df["severity"].quantile(0.25)]

    if len(severe) > 0 and len(mild) > 0:
        mean_delay_severe = severe["prior_24h_delay"].mean()
        mean_delay_mild = mild["prior_24h_delay"].mean()

        print(
            f"  Severe weather (top 25%): mean prior_24h_delay = {mean_delay_severe:.1f} min"
        )
        print(
            f"  Mild weather (bottom 25%): mean prior_24h_delay = {mean_delay_mild:.1f} min"
        )

        if mean_delay_severe > mean_delay_mild * 1.5:
            print(
                f"[OK] Severe weather shows {mean_delay_severe / mean_delay_mild:.1f}x higher prior delays"
            )
        elif mean_delay_severe > mean_delay_mild:
            print(
                f"[WARNING] Weak correlation: only {mean_delay_severe / mean_delay_mild:.1f}x higher"
            )
        else:
            print("[WARNING] No correlation between weather severity and prior delays")
    else:
        print("[WARNING] Insufficient data to validate delay correlation")

    print("=" * 80)
