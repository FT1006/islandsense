"""Test script to validate generated synthetic data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from islandsense.config import get_config
from islandsense.schema import (
    validate_sailings,
    validate_status,
    validate_metocean,
    validate_tides,
    validate_exposure,
)


def main():
    """Load and validate all generated CSVs."""
    print("=" * 80)
    print("IslandSense Data Validation (test_load.py)")
    print("=" * 80)

    config = get_config()

    # Load all CSVs
    print("\n📂 Loading CSVs...")

    try:
        sailings_df = pd.read_csv(config.sailings_file)
        print(f"  ✅ sailings.csv: {len(sailings_df)} rows")
    except Exception as e:
        print(f"  ❌ sailings.csv: {e}")
        return 1

    try:
        status_df = pd.read_csv(config.status_file)
        print(f"  ✅ status.csv: {len(status_df)} rows")
    except Exception as e:
        print(f"  ❌ status.csv: {e}")
        return 1

    try:
        metocean_df = pd.read_csv(config.metocean_file)
        print(f"  ✅ metocean.csv: {len(metocean_df)} rows")
    except Exception as e:
        print(f"  ❌ metocean.csv: {e}")
        return 1

    try:
        tides_df = pd.read_csv(config.tides_file)
        print(f"  ✅ tides.csv: {len(tides_df)} rows")
    except Exception as e:
        print(f"  ❌ tides.csv: {e}")
        return 1

    try:
        exposure_df = pd.read_csv(config.exposure_file)
        print(f"  ✅ exposure_by_sailing.csv: {len(exposure_df)} rows")
    except Exception as e:
        print(f"  ❌ exposure_by_sailing.csv: {e}")
        return 1

    # Validate schemas
    print("\n🔍 Validating schemas...")
    try:
        validate_sailings(sailings_df)
        print("  ✅ sailings.csv schema valid")
    except Exception as e:
        print(f"  ❌ sailings.csv schema invalid: {e}")
        return 1

    try:
        validate_status(status_df)
        print("  ✅ status.csv schema valid")
    except Exception as e:
        print(f"  ❌ status.csv schema invalid: {e}")
        return 1

    try:
        validate_metocean(metocean_df)
        print("  ✅ metocean.csv schema valid")
    except Exception as e:
        print(f"  ❌ metocean.csv schema invalid: {e}")
        return 1

    try:
        validate_tides(tides_df)
        print("  ✅ tides.csv schema valid")
    except Exception as e:
        print(f"  ❌ tides.csv schema invalid: {e}")
        return 1

    try:
        validate_exposure(exposure_df)
        print("  ✅ exposure_by_sailing.csv schema valid")
    except Exception as e:
        print(f"  ❌ exposure_by_sailing.csv schema invalid: {e}")
        return 1

    # Print sample sailings
    print("\n📋 Sample Sailings (first 5 rows):")
    print(sailings_df.head(5).to_string(index=False))

    # Summary statistics
    print("\n📊 Summary Statistics:")

    # Disruption stats
    merged = sailings_df.merge(status_df, on="sailing_id")
    merged["disrupted"] = (merged["status"] == "cancelled") | (
        merged["delay_min"] > config.disruption_delay_minutes
    )
    disruption_rate = merged["disrupted"].mean()
    print(f"  Disruption rate: {disruption_rate:.1%}")
    print(f"  Cancelled sailings: {(merged['status'] == 'cancelled').sum()}")
    print(
        f"  Arrived on-time: {((merged['status'] == 'arrived') & (merged['delay_min'] <= 15)).sum()}"
    )

    # Route breakdown
    print("\n  Sailings by route:")
    route_counts = sailings_df["route"].value_counts()
    for route, count in route_counts.items():
        print(f"    {route}: {count}")

    # Exposure stats
    print("\n  Exposure statistics:")
    print(f"    Total Fresh units: {exposure_df['fresh_units'].sum():.1f}")
    print(f"    Total Fuel units: {exposure_df['fuel_units'].sum():.2f}")
    print(f"    Avg Fresh per sailing: {exposure_df['fresh_units'].mean():.1f}")
    print(f"    Avg Fuel per sailing: {exposure_df['fuel_units'].mean():.2f}")

    # Check demo window (next 72h)
    sailings_df["etd"] = pd.to_datetime(sailings_df["etd_iso"])
    now = pd.Timestamp.now()
    demo_sailings = sailings_df[
        (sailings_df["etd"] >= now)
        & (sailings_df["etd"] < now + pd.Timedelta(hours=72))
    ]
    print(f"\n  Sailings in next 72h: {len(demo_sailings)}")

    print("\n" + "=" * 80)
    print("✅ All validation checks passed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
