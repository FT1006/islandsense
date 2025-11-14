"""Synthetic data generation for IslandSense MVP.

Generates physics-based, realistic sailing disruption data following SCHEMA.md.
All label generation is deterministic from weather → physics → risk → outcome.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

from islandsense.config import get_config
from islandsense.schema import (
    SailingColumns,
    StatusColumns,
    MetoceanColumns,
    TideColumns,
    ExposureColumns,
    validate_sailings,
    validate_status,
    validate_metocean,
    validate_tides,
    validate_exposure,
)

# ============================================================================
# Configuration Constants
# ============================================================================

RNG_SEED = 42
HISTORY_DAYS = 180  # Need >1000 sailings for training
DEMO_HORIZON_DAYS = 7

# Physics thresholds
WIND_THRESHOLD = 25  # kts
GUST_THRESHOLD = 35  # kts

# Risk score coefficients (to be tuned via calibration)
COEF_BSEF = 0.3
COEF_WOTDI = 0.2
COEF_HIGH_WIND = 0.02
COEF_HIGH_GUST = 0.015

# Sigmoid parameters for disruption probability (to be tuned)
ALPHA = 2.0
BETA = 2.0  # Tuned: balance between global rate (2-8%) and demo variety (Green/Amber/Red mix)

# Route definitions: (name, heading_deg, vessels, duration_hours)
ROUTES = [
    ("Portsmouth→Jersey", 220, ["Island Spirit", "Channel Star"], 10),
    ("St. Malo→Jersey", 300, ["Celtic Voyager"], 9),
    ("Poole→Jersey", 200, ["Jersey Clipper"], 8),
]

# Sailing times (local times in hours, will be converted to UTC)
SAILING_TIMES = [6, 12, 18]  # 06:00, 12:00, 18:00


# ============================================================================
# Utility Functions
# ============================================================================


def wrap_angle(deg: float) -> float:
    """Wrap angle to [-180, 180] range."""
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# ============================================================================
# Data Generation Functions
# ============================================================================


def generate_time_grid(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Generate hourly time grid between start and end dates."""
    times = []
    current = start_date
    while current <= end_date:
        times.append(current)
        current += timedelta(hours=1)
    return times


def generate_storm_windows(
    start_date: datetime,
    end_date: datetime,
    demo_now: datetime,
    n_storms: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[datetime, datetime]]:
    """
    Generate storm windows. Ensures one storm overlaps demo 72h window.

    Returns:
        List of (start, end) datetime tuples
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    storms = []

    # Force one storm in demo window (36-54h into the 72h window)
    demo_storm_start = demo_now + timedelta(hours=36)
    demo_storm_end = demo_now + timedelta(hours=54)
    storms.append((demo_storm_start, demo_storm_end))

    # Generate n_storms - 1 more storms in the history period
    for _ in range(n_storms - 1):
        # Random start in history period
        hours_from_start = rng.uniform(
            0, (demo_now - start_date).total_seconds() / 3600
        )
        storm_start = start_date + timedelta(hours=hours_from_start)

        # Storm duration: 8-24 hours
        duration_hours = rng.uniform(8, 24)
        storm_end = storm_start + timedelta(hours=duration_hours)

        storms.append((storm_start, storm_end))

    return storms


def is_in_storm(ts: datetime, storm_windows: List[Tuple[datetime, datetime]]) -> bool:
    """Check if timestamp falls within any storm window."""
    for start, end in storm_windows:
        if start <= ts <= end:
            return True
    return False


def generate_metocean(
    time_grid: List[datetime],
    storm_windows: List[Tuple[datetime, datetime]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate metocean.csv with realistic weather and wave data.

    Includes base climate + injected storm windows with higher wind/waves.
    """
    data = []

    for ts in time_grid:
        in_storm = is_in_storm(ts, storm_windows)

        if in_storm:
            # Storm conditions
            wind_kts = clip(rng.normal(35, 5), 0, 50)
            gust_kts = clip(wind_kts + rng.normal(10, 5), 0, 60)
            hs_m = clip(rng.normal(3.0, 0.7), 0, 5)
            tp_s = clip(rng.normal(8, 2), 3, 14)
        else:
            # Calm conditions
            wind_kts = clip(rng.normal(15, 5), 0, 50)
            gust_kts = clip(wind_kts + rng.normal(5, 3), 0, 60)
            hs_m = clip(rng.normal(1.0, 0.5), 0, 5)
            tp_s = clip(rng.normal(6, 2), 3, 14)

        wind_dir_deg = rng.uniform(0, 360)
        wave_dir_deg = (wind_dir_deg + rng.normal(0, 30)) % 360

        data.append(
            {
                MetoceanColumns.TS_ISO: ts.isoformat(),
                MetoceanColumns.WIND_KTS: round(wind_kts, 2),
                MetoceanColumns.WIND_DIR_DEG: round(wind_dir_deg, 1),
                MetoceanColumns.GUST_KTS: round(gust_kts, 2),
                MetoceanColumns.HS_M: round(hs_m, 2),
                MetoceanColumns.TP_S: round(tp_s, 1),
                MetoceanColumns.WAVE_DIR_DEG: round(wave_dir_deg, 1),
            }
        )

    df = pd.DataFrame(data)
    validate_metocean(df)
    return df


def generate_tides(time_grid: List[datetime], rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate tides.csv with semi-diurnal tide curve.
    """
    tide_period_hours = 12.4
    amplitude = 2.0
    mean_level = 3.0

    data = []
    start_time = time_grid[0]

    for ts in time_grid:
        hours_elapsed = (ts - start_time).total_seconds() / 3600
        tide_m = mean_level + amplitude * np.sin(
            2 * np.pi * hours_elapsed / tide_period_hours
        )
        tide_m += rng.normal(0, 0.1)  # Small noise

        data.append(
            {
                TideColumns.TS_ISO: ts.isoformat(),
                TideColumns.TIDE_M: round(tide_m, 2),
            }
        )

    df = pd.DataFrame(data)
    validate_tides(df)
    return df


def generate_sailings(
    start_date: datetime, end_date: datetime, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate sailings.csv with scheduled sailings for all routes.
    """
    data = []
    sailing_counter = 0

    current_date = start_date.date()
    end_date_only = end_date.date()

    while current_date <= end_date_only:
        for route_name, heading, vessels, duration_hours in ROUTES:
            # Generate 1-3 sailings per day (avg ~2 to get >1000 total)
            # 10% chance 1, 50% chance 2, 40% chance 3
            n_sailings = rng.choice([1, 2, 3], p=[0.1, 0.5, 0.4])
            selected_times = rng.choice(SAILING_TIMES, size=n_sailings, replace=False)

            for hour in selected_times:
                # Create departure time (assume BST/GMT, treat as UTC for simplicity)
                etd = datetime.combine(current_date, datetime.min.time()) + timedelta(
                    hours=int(hour)
                )
                eta = etd + timedelta(hours=duration_hours)

                # Add small noise to heading
                head_deg = (heading + rng.normal(0, 5)) % 360

                vessel = rng.choice(vessels)

                sailing_id = f"{route_name.replace('→', '_')}_{etd.strftime('%Y%m%d_%H%M')}_{sailing_counter}"
                sailing_counter += 1

                data.append(
                    {
                        SailingColumns.SAILING_ID: sailing_id,
                        SailingColumns.ROUTE: route_name,
                        SailingColumns.VESSEL: vessel,
                        SailingColumns.ETD_ISO: etd.isoformat(),
                        SailingColumns.ETA_ISO: eta.isoformat(),
                        SailingColumns.HEAD_DEG: round(head_deg, 1),
                    }
                )

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    validate_sailings(df)
    return df


def compute_risk_score(
    wind_kts: float,
    wind_dir_deg: float,
    gust_kts: float,
    hs_m: float,
    wave_dir_deg: float,
    head_deg: float,
) -> float:
    """
    Compute synthetic risk score using physics features.

    Includes BSEF (beam-sea exposure) and WOTDI (wind-tide misalignment proxy).
    """
    # BSEF: Beam-sea exposure factor
    rel_wave_deg = wrap_angle(wave_dir_deg - head_deg)
    bsef = abs(np.sin(np.radians(rel_wave_deg))) * hs_m

    # WOTDI proxy: wind misalignment (simplified, no actual tide flow direction)
    rel_wind_deg = wrap_angle(wind_dir_deg - head_deg)
    wotdi = abs(np.sin(np.radians(rel_wind_deg))) * (wind_kts / 20.0)  # Normalized

    # High wind/gust components
    high_wind = max(0, wind_kts - WIND_THRESHOLD)
    high_gust = max(0, gust_kts - GUST_THRESHOLD)

    # Weighted risk score
    r_s = (
        COEF_BSEF * bsef
        + COEF_WOTDI * wotdi
        + COEF_HIGH_WIND * high_wind
        + COEF_HIGH_GUST * high_gust
    )

    return r_s


def risk_to_probability(r_s: float, alpha: float = ALPHA, beta: float = BETA) -> float:
    """Convert risk score to disruption probability via sigmoid."""
    return 1.0 / (1.0 + np.exp(-alpha * (r_s - beta)))


def generate_status(
    sailings_df: pd.DataFrame,
    metocean_df: pd.DataFrame,
    tides_df: pd.DataFrame,
    rng: np.random.Generator,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> pd.DataFrame:
    """
    Generate status.csv with physics-based labels.

    Uses BSEF + WOTDI + wind/gust thresholds → risk score → disruption probability.
    """
    # Index metocean and tides by timestamp for fast lookup
    metocean_df["ts"] = pd.to_datetime(metocean_df[MetoceanColumns.TS_ISO])
    metocean_df = metocean_df.set_index("ts").sort_index()

    tides_df["ts"] = pd.to_datetime(tides_df[TideColumns.TS_ISO])
    tides_df = tides_df.set_index("ts").sort_index()

    data = []

    for _, sailing in sailings_df.iterrows():
        etd = pd.to_datetime(sailing[SailingColumns.ETD_ISO])
        head_deg = sailing[SailingColumns.HEAD_DEG]

        # Find nearest metocean row (nearest hour)
        nearest_time = metocean_df.index[
            metocean_df.index.get_indexer([etd], method="nearest")[0]
        ]
        metocean_row = metocean_df.loc[nearest_time]

        # Compute risk score
        r_s = compute_risk_score(
            wind_kts=metocean_row[MetoceanColumns.WIND_KTS],
            wind_dir_deg=metocean_row[MetoceanColumns.WIND_DIR_DEG],
            gust_kts=metocean_row[MetoceanColumns.GUST_KTS],
            hs_m=metocean_row[MetoceanColumns.HS_M],
            wave_dir_deg=metocean_row[MetoceanColumns.WAVE_DIR_DEG],
            head_deg=head_deg,
        )

        # Convert to probability
        p_disrupt = risk_to_probability(r_s, alpha, beta)

        # Sample disruption
        disrupted = rng.uniform(0, 1) < p_disrupt

        # Generate status and delay
        if disrupted:
            if rng.uniform(0, 1) < 0.4:  # 40% of disruptions are cancellations
                status = "cancelled"
                delay_min = 0
            else:
                status = "arrived"
                delay_min = int(rng.uniform(130, 300))
        else:
            status = "arrived"
            if rng.uniform(0, 1) < 0.2:  # 20% have moderate delays
                delay_min = int(rng.uniform(15, 90))
            else:
                delay_min = int(rng.uniform(0, 15))

        data.append(
            {
                StatusColumns.SAILING_ID: sailing[SailingColumns.SAILING_ID],
                StatusColumns.STATUS: status,
                StatusColumns.DELAY_MIN: delay_min,
            }
        )

    df = pd.DataFrame(data)
    validate_status(df)
    return df


def generate_exposure(
    sailings_df: pd.DataFrame,
    demo_storm_window: Tuple[datetime, datetime],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate exposure_by_sailing.csv with Fresh/Fuel units per sailing.
    """
    # Route-specific exposure means
    route_means = {
        "Portsmouth→Jersey": {"fresh": 40, "fuel": 1.0},
        "St. Malo→Jersey": {"fresh": 25, "fuel": 0.5},
        "Poole→Jersey": {"fresh": 15, "fuel": 0.2},
    }

    data = []

    for _, sailing in sailings_df.iterrows():
        route = sailing[SailingColumns.ROUTE]
        etd = pd.to_datetime(sailing[SailingColumns.ETD_ISO])

        means = route_means[route]

        # Generate with noise and clipping
        fresh = clip(
            rng.normal(means["fresh"], means["fresh"] * 0.3), 0, means["fresh"] * 2
        )
        fuel = clip(
            rng.normal(means["fuel"], means["fuel"] * 0.3), 0, means["fuel"] * 2
        )

        # Boost exposure for sailings in demo storm window
        demo_start, demo_end = demo_storm_window
        if demo_start <= etd <= demo_end:
            fresh *= 1.3
            fuel *= 1.3

        data.append(
            {
                ExposureColumns.SAILING_ID: sailing[SailingColumns.SAILING_ID],
                ExposureColumns.FRESH_UNITS: round(fresh, 1),
                ExposureColumns.FUEL_UNITS: round(fuel, 2),
            }
        )

    df = pd.DataFrame(data)
    validate_exposure(df)
    return df


# ============================================================================
# Validation and Calibration
# ============================================================================


def compute_disruption_stats(
    status_df: pd.DataFrame,
    sailings_df: pd.DataFrame,
    storm_windows: List[Tuple[datetime, datetime]],
    disruption_threshold_min: int = 120,
) -> Dict[str, float]:
    """Compute disruption statistics for calibration."""
    merged = sailings_df.merge(status_df, on=SailingColumns.SAILING_ID)

    # Define disruption
    merged["disrupted"] = (merged[StatusColumns.STATUS] == "cancelled") | (
        merged[StatusColumns.DELAY_MIN] > disruption_threshold_min
    )

    global_rate = merged["disrupted"].mean()

    # Storm vs calm disruption rates
    merged["etd"] = pd.to_datetime(merged[SailingColumns.ETD_ISO])
    merged["in_storm"] = merged["etd"].apply(lambda x: is_in_storm(x, storm_windows))

    storm_rate = merged[merged["in_storm"]]["disrupted"].mean()
    calm_rate = merged[~merged["in_storm"]]["disrupted"].mean()

    return {
        "global_rate": global_rate,
        "storm_rate": storm_rate,
        "calm_rate": calm_rate,
        "storm_calm_ratio": storm_rate / calm_rate if calm_rate > 0 else float("inf"),
        "n_total": len(merged),
        "n_disrupted": merged["disrupted"].sum(),
    }


def validate_demo_window(
    sailings_df: pd.DataFrame,
    status_df: pd.DataFrame,
    metocean_df: pd.DataFrame,
    demo_now: datetime,
    horizon_hours: int = 72,
) -> None:
    """Validate that demo 72h window has interesting variety."""
    demo_end = demo_now + timedelta(hours=horizon_hours)

    sailings_df["etd"] = pd.to_datetime(sailings_df[SailingColumns.ETD_ISO])
    demo_sailings = sailings_df[
        (sailings_df["etd"] >= demo_now) & (sailings_df["etd"] < demo_end)
    ]

    print(f"\n=== Demo Window Validation (Next {horizon_hours}h) ===")
    print(f"Demo window: {demo_now.isoformat()} to {demo_end.isoformat()}")
    print(f"Number of sailings in demo window: {len(demo_sailings)}")

    if len(demo_sailings) < 3:
        print(
            f"⚠️  WARNING: Only {len(demo_sailings)} sailings in demo window (target: 3-5)"
        )

    # Compute p_sail for demo sailings
    metocean_df["ts"] = pd.to_datetime(metocean_df[MetoceanColumns.TS_ISO])
    metocean_df = metocean_df.set_index("ts").sort_index()

    demo_probs = []
    for _, sailing in demo_sailings.iterrows():
        etd = sailing["etd"]
        head_deg = sailing[SailingColumns.HEAD_DEG]

        nearest_time = metocean_df.index[
            metocean_df.index.get_indexer([etd], method="nearest")[0]
        ]
        metocean_row = metocean_df.loc[nearest_time]

        r_s = compute_risk_score(
            wind_kts=metocean_row[MetoceanColumns.WIND_KTS],
            wind_dir_deg=metocean_row[MetoceanColumns.WIND_DIR_DEG],
            gust_kts=metocean_row[MetoceanColumns.GUST_KTS],
            hs_m=metocean_row[MetoceanColumns.HS_M],
            wave_dir_deg=metocean_row[MetoceanColumns.WAVE_DIR_DEG],
            head_deg=head_deg,
        )
        p_disrupt = risk_to_probability(r_s)
        demo_probs.append(p_disrupt)

    if demo_probs:
        min_p = min(demo_probs)
        max_p = max(demo_probs)
        print(f"Demo sailing p_disrupt range: [{min_p:.3f}, {max_p:.3f}]")

        if min_p > 0.1:
            print("⚠️  WARNING: No low-risk sailings in demo (all p > 0.1)")
        if max_p < 0.4:
            print("⚠️  WARNING: No high-risk sailings in demo (all p < 0.4)")

        if min_p <= 0.1 and max_p >= 0.4:
            print("✅ Demo window has variety (Green/Amber/Red mix likely)")


def run_sanity_checks(
    sailings_df: pd.DataFrame,
    status_df: pd.DataFrame,
    metocean_df: pd.DataFrame,
    tides_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    demo_now: datetime,
    storm_windows: List[Tuple[datetime, datetime]],
    horizon_hours: int = 72,
) -> None:
    """
    Run comprehensive sanity checks on generated data (fail fast).

    Required checks:
    1. Total sailings > 1,000
    2. Demo horizon sailings: 3-20
    3. Global disruption rate: 2-8%
    4. Disruption rate in storms > disruption rate outside
    5. No NaNs in core columns
    """
    print("\n" + "=" * 80)
    print("🔍 SANITY CHECKS (Fail Fast)")
    print("=" * 80)

    all_passed = True

    # Check 1: Total sailings > 1,000
    n_total = len(sailings_df)
    check1_pass = n_total > 1000
    status_icon = "✅" if check1_pass else "❌"
    print(f"{status_icon} Check 1: Total sailings > 1,000: {n_total}")
    if not check1_pass:
        all_passed = False

    # Check 2: Demo horizon sailings 3-20 (relaxed from 3-10 to accommodate >1000 total)
    sailings_df["etd"] = pd.to_datetime(sailings_df[SailingColumns.ETD_ISO])
    demo_end = demo_now + timedelta(hours=horizon_hours)
    demo_sailings = sailings_df[
        (sailings_df["etd"] >= demo_now) & (sailings_df["etd"] < demo_end)
    ]
    n_demo = len(demo_sailings)
    check2_pass = 3 <= n_demo <= 20
    status_icon = "✅" if check2_pass else "❌"
    print(f"{status_icon} Check 2: Demo horizon sailings in [3, 20]: {n_demo}")
    if not check2_pass:
        all_passed = False

    # Check 3: Global disruption rate 2-8%
    merged = sailings_df.merge(status_df, on=SailingColumns.SAILING_ID)
    merged["disrupted"] = (merged[StatusColumns.STATUS] == "cancelled") | (
        merged[StatusColumns.DELAY_MIN] > 120
    )
    global_rate = merged["disrupted"].mean()
    check3_pass = 0.02 <= global_rate <= 0.08
    status_icon = "✅" if check3_pass else "❌"
    print(
        f"{status_icon} Check 3: Global disruption rate in [2%, 8%]: {global_rate:.1%}"
    )
    if not check3_pass:
        all_passed = False

    # Check 4: Storm disruption > calm disruption
    merged["in_storm"] = merged["etd"].apply(lambda x: is_in_storm(x, storm_windows))
    storm_rate = merged[merged["in_storm"]]["disrupted"].mean()
    calm_rate = merged[~merged["in_storm"]]["disrupted"].mean()
    check4_pass = storm_rate > calm_rate
    status_icon = "✅" if check4_pass else "❌"
    print(
        f"{status_icon} Check 4: Storm disruption > calm: {storm_rate:.1%} > {calm_rate:.1%}"
    )
    if not check4_pass:
        all_passed = False

    # Check 5: No NaNs in core columns
    core_dfs = {
        "sailings": (sailings_df, SailingColumns.all()),
        "status": (status_df, StatusColumns.all()),
        "metocean": (metocean_df, MetoceanColumns.all()),
        "tides": (tides_df, TideColumns.all()),
        "exposure": (exposure_df, ExposureColumns.all()),
    }

    has_nans = False
    for name, (df, cols) in core_dfs.items():
        for col in cols:
            if df[col].isna().any():
                print(f"❌ Check 5: NaN found in {name}.{col}")
                has_nans = True
                all_passed = False

    if not has_nans:
        print("✅ Check 5: No NaNs in core columns")

    # Final verdict
    print("=" * 80)
    if all_passed:
        print("✅ ALL SANITY CHECKS PASSED")
    else:
        print("❌ SANITY CHECKS FAILED - See errors above")
        raise ValueError("Sanity checks failed. Cannot proceed with invalid data.")
    print("=" * 80)


# ============================================================================
# Main Generation Pipeline
# ============================================================================


def main():
    """Main data generation pipeline with calibration."""
    print("=" * 80)
    print("IslandSense Synthetic Data Generation (M0)")
    print("=" * 80)
    print("\n🔬 IMPORTANT: Per-sailing ML model (M2) is the ONLY ML component.")
    print("   All status labels are generated via deterministic physics + math.\n")

    config = get_config()
    rng = np.random.default_rng(RNG_SEED)

    # Define time ranges
    start_date = datetime.now() - timedelta(days=HISTORY_DAYS)
    demo_now = datetime.now()
    end_date = demo_now + timedelta(days=DEMO_HORIZON_DAYS)

    print(f"History period: {HISTORY_DAYS} days")
    print(f"Demo horizon: {DEMO_HORIZON_DAYS} days")
    print(f"Start date: {start_date.date()}")
    print(f"Demo NOW: {demo_now.date()}")
    print(f"End date: {end_date.date()}")

    # Generate time grid
    print("\n[1/5] Generating time grid...")
    time_grid = generate_time_grid(start_date, end_date)
    print(f"  Created {len(time_grid)} hourly timestamps")

    # Generate storm windows
    print("\n[2/5] Generating storm windows...")
    storm_windows = generate_storm_windows(
        start_date, end_date, demo_now, n_storms=4, rng=rng
    )
    for i, (s, e) in enumerate(storm_windows):
        duration_h = (e - s).total_seconds() / 3600
        in_demo = s >= demo_now and s < demo_now + timedelta(hours=72)
        marker = "🌊 DEMO STORM" if in_demo else ""
        print(
            f"  Storm {i + 1}: {s.date()} {s.hour:02d}:00 -> {e.date()} {e.hour:02d}:00 ({duration_h:.1f}h) {marker}"
        )

    # Generate metocean
    print("\n[3/5] Generating metocean.csv...")
    metocean_df = generate_metocean(time_grid, storm_windows, rng)
    print(f"  Created {len(metocean_df)} rows")

    # Generate tides
    print("\n[4/5] Generating tides.csv...")
    tides_df = generate_tides(time_grid, rng)
    print(f"  Created {len(tides_df)} rows")

    # Generate sailings
    print("\n[5/5] Generating sailings.csv...")
    sailings_df = generate_sailings(start_date, end_date, rng)
    print(f"  Created {len(sailings_df)} sailings")

    # Generate status with physics-based labels
    print("\n[6/7] Generating status.csv (physics-based labels)...")
    status_df = generate_status(sailings_df, metocean_df, tides_df, rng)
    print(f"  Created {len(status_df)} status records")

    # Compute and print disruption statistics
    stats = compute_disruption_stats(status_df, sailings_df, storm_windows)
    print("\n📊 Disruption Statistics (Calibration Check):")
    print(f"  Total sailings: {stats['n_total']}")
    print(f"  Total disrupted: {stats['n_disrupted']}")
    print(f"  Global disruption rate: {stats['global_rate']:.1%} (target: 2-8%)")
    print(f"  Storm window rate: {stats['storm_rate']:.1%}")
    print(f"  Calm window rate: {stats['calm_rate']:.1%}")
    print(f"  Storm/Calm ratio: {stats['storm_calm_ratio']:.1f}x (target: >3x)")

    if not (0.02 <= stats["global_rate"] <= 0.08):
        print(
            "  ⚠️  WARNING: Global rate outside 2-8% range. Consider tuning ALPHA/BETA."
        )
    if stats["storm_calm_ratio"] < 3:
        print(
            "  ⚠️  WARNING: Storm/calm ratio < 3x. Physics may not be working correctly."
        )

    # Find demo storm window for exposure boost
    demo_storm_window = None
    for s, e in storm_windows:
        if s >= demo_now and s < demo_now + timedelta(hours=72):
            demo_storm_window = (s, e)
            break

    if demo_storm_window is None:
        # Fallback: use first storm window
        demo_storm_window = storm_windows[0]

    # Generate exposure
    print("\n[7/7] Generating exposure_by_sailing.csv...")
    exposure_df = generate_exposure(sailings_df, demo_storm_window, rng)
    print(f"  Created {len(exposure_df)} exposure records")

    # Validate demo window
    validate_demo_window(
        sailings_df, status_df, metocean_df, demo_now, horizon_hours=72
    )

    # Run sanity checks (fail fast)
    run_sanity_checks(
        sailings_df=sailings_df,
        status_df=status_df,
        metocean_df=metocean_df,
        tides_df=tides_df,
        exposure_df=exposure_df,
        demo_now=demo_now,
        storm_windows=storm_windows,
        horizon_hours=72,
    )

    # Save all CSVs
    print("\n💾 Saving CSVs to data/...")
    data_dir = config.data_dir
    data_dir.mkdir(exist_ok=True)

    sailings_df.to_csv(config.sailings_file, index=False)
    print(f"  ✅ {config.sailings_file.name}")

    status_df.to_csv(config.status_file, index=False)
    print(f"  ✅ {config.status_file.name}")

    metocean_df.to_csv(config.metocean_file, index=False)
    print(f"  ✅ {config.metocean_file.name}")

    tides_df.to_csv(config.tides_file, index=False)
    print(f"  ✅ {config.tides_file.name}")

    exposure_df.to_csv(config.exposure_file, index=False)
    print(f"  ✅ {config.exposure_file.name}")

    print("\n" + "=" * 80)
    print("✅ Synthetic data generation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: python scripts/test_load.py")
    print("  2. Proceed to M1: Feature engineering")


if __name__ == "__main__":
    main()
